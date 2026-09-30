# Chapter 8 — Caching and content addressing

Inference is expensive and deterministic. The same model on the same audio with
the same parameters gives the same answer, every time. That is the ideal case
for caching, and the requirement is explicit about the targets: sub-10 ms
tensor retrieval, under 200 ms for a full cached API response.

Caching is also where this project has had some of its nastiest bugs, because a
cache bug does not look like a bug. It looks like a *wrong answer*, arriving
fast, with nothing in the logs.

---

## 8.1 The three failure modes of a cache

1. **Wrong key** — you miss when you should hit. Slow, but correct. The benign
   failure.
2. **Colliding key** — two different requests share a key, so one gets the
   other's answer. Fast and wrong.
3. **Right key, wrong value shape** — the consumer reads a value it cannot
   handle. And this one is *worse than a miss*, for a reason worth stating
   carefully:

```python
"""Canonical cache-key builders and payload shapes (SRS FR4).

Every key family below is consumed by at least one route. The *shape* of the
value stored under a key is part of its contract: a writer that stores the
right key with the wrong shape is worse than a cache miss, because the
consumer will not fall back to recomputation - it will read the value and
fail on it.
"""
```

On a miss, the consumer recomputes and everything works. On a shape mismatch,
the consumer *found* something, so it does not recompute — it proceeds and
crashes. The cache has converted a slow success into a hard failure.

The concrete incident:

```python
"""That is exactly the defect this module exists to prevent. Dataset warmup used
to store the ASR result of ``transcribe_whisper_with_attention`` (a
``{"text", "attention"}`` dict) under the transcript family, whose consumers
all assume ``prediction`` is a plain string; ``/inferences/whisper-accuracy``
then died with ``AttributeError: 'dict' object has no attribute 'lower'``
and the UI sat on a spinner forever.
"""
```

A background warmup wrote a dict where consumers expected a string. The
consumers called `.lower()`. `AttributeError`, spinner forever.

Note where the bug *was* and where it *appeared*. The writer was dataset
warmup; the failure was in an accuracy endpoint, on a different request, some
time later. A cache decouples cause from symptom in both time and space.

---

## 8.2 Three hashes, and which is correct

```python
def path_hash(resolved_path) -> str:
    """``md5`` of the resolved path - the primary key discriminator."""
    return hashlib.md5(str(resolved_path).encode()).hexdigest()
```

Hashes the path *string*. Fast, no file access. Two copies of identical audio at
different paths hash differently (cache miss — benign), and a file **edited in
place** hashes the same (stale result — not benign).

```python
def content_hash(resolved_path) -> str:
    """``md5`` of path + size + mtime, so edits in place invalidate the entry."""
    p = Path(resolved_path)
    st = p.stat()
    return hashlib.md5(f"{str(p)}_{st.st_size}_{st.st_mtime}".encode()).hexdigest()
```

Adds size and modification time, so an in-place edit *usually* invalidates.
Requires a `stat()` — cheap. "Usually", because an edit that preserves both
size and mtime does not.

```python
def content_sha256(resolved_path) -> str:
    """Streamed SHA-256 over the audio bytes themselves (FR4.1).

    This is the identity FR4 specifies: *"a SHA-256 hash of the (audio bytes,
    model identifier, task, parameters) tuple"*. The two hashes above key on
    where a file sits, so the same audio at two paths caches twice and a file
    edited in place can serve a stale result.
    """
    h = hashlib.sha256()
    with open(resolved_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
```

The correct identity. Content-addressed: identical bytes always hash the same
regardless of location, and different bytes always hash differently. Costs a
full file read.

`iter(lambda: fh.read(1024*1024), b"")` is the two-argument `iter` idiom: call
the function repeatedly until it returns the sentinel. It streams in 1 MiB
chunks so memory stays bounded regardless of file size.

### Why MD5 is acceptable here

MD5 is cryptographically broken — you can construct collisions deliberately.
For cache keys derived from a *path string*, that does not matter: an attacker
who can choose your file paths has already won. MD5 is used because it is fast
and the inherited code used it. `content_sha256` uses SHA-256 because it hashes
*content*, where collision resistance genuinely matters.

Knowing *which* properties of a hash your use case needs is what lets you make
that distinction instead of applying a blanket rule.

### The rejected memoisation

```python
    """Deliberately not memoised. A memo keyed on (size, mtime) is the obvious
    optimisation and it is unsound here: two same-length writes milliseconds
    apart share both values on this filesystem even at ``st_mtime_ns``
    resolution, so the memo returned the pre-edit hash - reintroducing exactly
    the staleness this function exists to remove. Measured at ~1.2 ms for a 5 s
    clip, against a request that then runs a model; the trade is not close.
    """
```

The obvious cache key for a content hash — `(size, mtime)` — is exactly the
identity the content hash exists to distrust. And the measurement closes the
argument: 1.2 ms before a multi-second model call is not worth a correctness
risk.

**A comment that records a rejected optimisation is as valuable as one
explaining the code.** Without it, the next reader "improves" this and
reintroduces the bug.

### Reading all three

```python
def both_hashes(resolved_path) -> tuple[str, str, str]:
    """``(content_sha256, path_hash, content_hash)`` — content first.

    Content first so readers prefer the content-addressed key and fall back to
    the two location-derived ones, which keeps every entry written before FR4.1
    readable during the transition. Writers populate all three.

    The name is now a misnomer (three, not both) but it is called from six
    places; renaming belongs in the change that drops the path hashes, not this
    one.
    """
    try:
        content = content_sha256(resolved_path)
    except OSError:
        # An unreadable file still has a usable path identity; degrade rather
        # than lose caching entirely.
        content = ""
    hashes = [content] if content else []
    hashes += [path_hash(resolved_path), content_hash(resolved_path)]
    return tuple(hashes)
```

**Ordering is the migration strategy.** Content first means a reader tries the
correct key, then falls back to the legacy ones. Writers populate all three, so
entries are findable either way. When all legacy entries have expired, the old
hashes can be dropped with no flag day.

**The honest naming note.** `both_hashes` returning three things is wrong, and
the comment says so *and says why it is not being fixed now*: it is called from
six places, and mixing a rename into a behaviour change makes the diff harder
to review. Scoping is a real engineering judgement; recording it beats leaving
a reader to wonder whether the name is a bug.

---

## 8.3 Key families

A **key family** is a (namespace, key-pattern) pair with a documented payload
shape. Each has a builder function returning every spelling a consumer might
look under.

```python
# Bumped when a stored shape changes incompatibly (FR4.1).
CACHE_SCHEMA_VERSION = "v2"

# Namespace used by routes that read predictions without knowing the model.
PREDICTIONS_NS = "predictions"
```

### Transcripts

```python
def transcript_keys(model: str, hashes: tuple[str, ...]) -> list[tuple[str, str]]:
    """ASR transcript. Payload: ``{"prediction": <str>}``.

    Read by ``/inferences/run`` (via ``run_inference``), ``/whisper-accuracy``,
    ``/whisper-batch`` and ``/batch-check``.
    """
    keys: list[tuple[str, str]] = []
    for h in hashes:
        keys.append((model, f"{CACHE_SCHEMA_VERSION}_{model}_{h}"))
        keys.append((model, f"{model}_{h}"))
        keys.append((PREDICTIONS_NS, f"{CACHE_SCHEMA_VERSION}_{model}_{h}"))
    return keys
```

Three hashes × three spellings = nine keys for one payload. That is
deliberate over-writing: nine cheap `SET`s so that any consumer, using any
spelling, finds the entry. The docstring lists the consumers, so you can tell
whether a change to the shape will break anything.

### SER, and a key that was missing the model

```python
def ser_keys(hashes: tuple[str, ...], model: str | None = None) -> list[tuple[str, str]]:
    """SER. Payload: ``{"prediction": <ser dict>}``.

    Keyed on the SER checkpoint. These keys carried no model at all, so a
    custom emotion model read back whatever model had populated the entry
    first - a wrong prediction served from cache, with nothing to distinguish
    it. The default model keeps the unqualified spelling so entries written
    before this change stay readable.
    """
    keys: list[tuple[str, str]] = []
    suffix = "" if model in (None, DEFAULT_SER_MODEL) else f"_{model}"
    for h in hashes:
        keys.append(("wav2vec2", f"wav2vec2_detailed{suffix}_{h}"))
        keys.append(("wav2vec2", f"wav2vec2_detailed_attention_v3{suffix}_{h}"))
    return keys
```

The key omitted the model identity entirely. So the first SER model to run on a
clip owned that clip's cache entry, and every other SER model read its answer.
The model substitution bug from Chapter 6, arriving through the cache instead of
through dispatch.

The fix is careful about backward compatibility: the default model keeps the
*unqualified* spelling, so entries written before the change remain readable,
and only non-default models get a suffix. A cleaner fix would qualify
everything and abandon existing entries — this one costs nothing and keeps a
24-hour cache warm through a deploy.

### The family that once destroyed another family's data

```python
def deepfake_keys(model: str, hashes: tuple[str, ...]) -> list[tuple[str, str]]:
    """ADD. Payload: ``{"prediction": <add dict>}``.

    ``model`` MUST be the deepfake checkpoint that produced the result (e.g.
    ``melody-machine``), never the ASR model the user happens to have selected.
    This family shares the transcript family's key shape because
    ``/inferences/run`` serves every model from one route - so passing a Whisper
    id here writes an ADD dict straight over that model's transcript, and the
    transcript consumers then read a dict and fail on ``.lower()``.
    """
```

The deepfake family and the transcript family use the **same key shape**,
because one route serves every model. Pass a Whisper id to `deepfake_keys` and
you write a deepfake dict over that Whisper model's transcript entry. Every
transcript consumer then reads a dict and dies on `.lower()`.

The `MUST` in that docstring is load-bearing. There is no type-level protection
— both parameters are strings — so the contract lives in prose and in reviewer
attention. That is a genuine design weakness, and the honest response is to
make the requirement impossible to miss when reading the function.

### Schema versions, per family

```python
# Bumped when the acoustic payload gains or loses a field. LIT-248 added
# `spectrogram` without one, so every entry cached before it kept being served
# without a spectrogram for the whole 24 h TTL - a correct computation the UI
# could never see (FR4.1: a shape change must not be serveable under an old key).
ACOUSTIC_SCHEMA_VERSION = "v3"
```

Someone added a field to the acoustic profile payload. The computation was
correct. But old cache entries — without the field — kept being served for 24
hours, so the UI could not see the new data. It looked like the feature had not
shipped.

**A schema version is not documentation. It is a cache invalidation
mechanism.** Adding a field to a cached payload without bumping the version
means up to a full TTL during which your change is invisible. This is
per-family so bumping one does not invalidate everything.

### Separate families for separate shapes

```python
def add_timeline_keys(model: str, hashes: tuple[str, ...]) -> list[tuple[str, str]]:
    """Windowed ADD confidence timeline (FR7.2). Payload: ``{"timeline": [...]}``.

    Its own family, not the clip-level one: the two carry different shapes and
    sharing a key is how an ADD dict once landed on top of an ASR transcript.
    """
    return [(model, f"{model}_add_timeline_{CACHE_SCHEMA_VERSION}_{h}") for h in hashes]
```

The rule that falls out of every incident above: **one shape, one family.** If
two payloads have different shapes, they get different keys, even if they come
from the same model and the same clip.

The remaining families follow the pattern — `attention_keys`, `acoustic_keys`,
`saliency_keys`, `embedding_keys`, `audio_frequency_keys` — each with a
documented payload.

---

## 8.4 Defensive reading

Old entries with old shapes are still in Redis with up to 24 hours of TTL. Two
helpers keep them harmless:

```python
def as_transcript(prediction: Any) -> str:
    """Coerce any historical prediction shape to the plain transcript string.

    Consumers of the transcript family call ``.lower()`` on what they get, so
    they must never receive a dict. Entries written before the shapes were
    separated are still in Redis with a 24 h TTL; this keeps them harmless.
    """
    if isinstance(prediction, str):
        return prediction
    if isinstance(prediction, dict):
        for field in ("text", "transcript", "prediction"):
            value = prediction.get(field)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                return as_transcript(value)
        return ""
    if prediction is None:
        return ""
    return str(prediction)
```

Handles a string, a dict with any of three plausible field names, a nested dict
(recursively), `None`, and anything else. It always returns a string, so
`.lower()` can never fail.

```python
def unwrap_prediction(cached: Any) -> Any:
    """Return the payload a cache entry carries, tolerating both wrappings."""
    if isinstance(cached, dict) and "prediction" in cached:
        return cached["prediction"]
    return cached
```

Some entries are `{"prediction": X}`, some are bare `X`. This tolerates both.

These functions are **migration infrastructure**, and that framing matters. A
cache with a TTL is a rolling window of historical formats — for as long as the
TTL, you are running two versions of your data format simultaneously. Defensive
readers are how you survive that without a maintenance window.

---

## 8.5 The simple cache, and sync/async duplication

`infrastructure/redis.py` holds the store the hot paths use.

```python
def k_sess(sid: str) -> str:  return f"sess:{sid}"
def k_queue(sid: str) -> str: return f"{k_sess(sid)}:queue"
def k_meta(sid: str) -> str:  return f"{k_sess(sid)}:meta"
def k_result(model: str, h: str) -> str: return f"result:{model}:{h}"
```

Key construction in functions, never inline. That is how two call sites cannot
disagree about a key's spelling — the bug that created two invisible progress
channels in Chapter 7.

```python
async def cache_result(model: str, h: str, payload: dict, ttl: int = 6*60*60) -> None:
    try:
        await redis.set(k_result(model, h), json.dumps(payload), ex=ttl)
    except Exception:
        pass


async def get_result(model: str, h: str) -> dict | None:
    try:
        raw = await redis.get(k_result(model, h))
        await metrics_module.arecord_cache(redis, raw is not None)
        return json.loads(raw) if raw else None
    except Exception:
        await metrics_module.arecord_cache(redis, False)
        return None
```

**`except Exception: pass` on a cache write is correct.** A cache is an
optimisation. If Redis is down, the request should still work — slowly. Failing
a request because a cache write failed would convert a performance degradation
into an outage.

This is one of the few places where swallowing an exception is right, and the
test is simple: *if this operation fails, is the caller still correct?* For a
cache write, yes. For a cache read, yes (you recompute). For an inference, no.

Hit/miss is recorded on the read path either way, so the metric reflects
reality including failures.

### Why there are sync and async versions of everything

```python
def cache_result_sync(model: str, h: str, payload: dict, ttl: int = 6*60*60) -> None:
    try:
        from app.infrastructure.rq_connection import get_redis_connection
        conn = get_redis_connection()
        conn.set(k_result(model, h), json.dumps(payload), ex=ttl)
    except Exception:
        pass


def get_result_sync(model: str, h: str) -> dict | None:
    try:
        from app.infrastructure.rq_connection import get_redis_connection
        conn = get_redis_connection()
        raw = conn.get(k_result(model, h))
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        metrics_module.record_cache(conn, raw is not None)
        return json.loads(raw) if raw else None
    except Exception:
        return None
```

Async handlers use `await get_result(...)`. Workers and synchronous routes use
`get_result_sync(...)`. You cannot `await` from synchronous code, and calling
the sync version from an async handler blocks the event loop (§7.8).

The duplication is unavoidable given one async and one sync client. The
`isinstance(raw, bytes)` check appears only in the sync version because the
sync client is configured with `decode_responses=False` (RQ needs raw bytes)
while the async one uses `decode_responses=True`. Two clients, two decoding
behaviours, and the difference has to be handled where it appears.

---

## 8.6 The content-addressed manager

`app/core/redis.py` is a more rigorous cache, used by `results.py`.

```python
def _generate_key(self, audio_bytes, model_id, task, params) -> str:
    """
    Computes a deterministic cache key:
    audiolit: + SHA-256(audio_bytes ‖ model_id ‖ task ‖ canonical_params_json)
    """
    canonical_params = params.copy() if params else {}
    canonical_params["_cache_schema_version"] = CACHE_SCHEMA_VERSION
    params_json = json.dumps(canonical_params, sort_keys=True, separators=(',', ':'))

    sha256_hash = hashlib.sha256()
    sha256_hash.update(audio_bytes)
    sha256_hash.update(model_id.encode('utf-8'))
    sha256_hash.update(task.encode('utf-8'))
    sha256_hash.update(params_json.encode('utf-8'))
    return f"audiolit:tensor:{sha256_hash.hexdigest()}"
```

This is what the requirement actually specifies: a hash over all four
components.

**`sort_keys=True, separators=(',', ':')`** makes the params JSON canonical.
`{"a":1,"b":2}` and `{"b":2,"a":1}` are the same dict but serialise differently
by default — and would hash differently, so logically identical requests would
miss. Sorting the keys and removing whitespace guarantees one canonical
encoding.

**`params.copy()`** before adding the schema version, so the caller's dict is
not mutated. Folding the version into the hashed material means bumping it
invalidates every entry at once.

### NumPy-aware serialisation

```python
def _encode_numpy(self, obj: Any) -> Any:
    """Extensible encoder for msgpack to handle NumPy arrays natively."""
    if isinstance(obj, np.ndarray):
        return {b'__np__': True, b'dtype': obj.dtype.str.encode(),
                b'shape': list(obj.shape), b'data': obj.tobytes()}
    return obj


def _decode_numpy(self, obj: Any) -> Any:
    """Extensible decoder for msgpack to reconstruct NumPy arrays."""
    if isinstance(obj, dict) and obj.get(b'__np__'):
        dtype = obj[b'dtype'].decode()
        shape = tuple(obj[b'shape'])
        return np.frombuffer(obj[b'data'], dtype=dtype).reshape(shape)
    return obj
```

**msgpack** is a binary serialisation format — like JSON but compact and typed.
It has no native array type, so arrays are encoded as a tagged dict carrying
dtype, shape and raw bytes.

This matters enormously for attribution data. A `[128, 3000]` float32 matrix is
384,000 numbers. As JSON text, each is ~20 characters: about 7 MB. As raw bytes:
1.5 MB. Nearly 5× smaller, and no float-parsing on the way back —
`np.frombuffer` is a zero-copy view over the buffer.

The `b'__np__'` marker uses byte keys because msgpack round-trips bytes and
strings distinctly, and using a byte key makes accidental collision with a real
data key essentially impossible.

### Compression with a discriminator

```python
def _serialize(self, value: Any) -> bytes:
    """Serialize with msgpack, apply lz4 compression if > 1MB."""
    packed_data = msgpack.packb(value, default=self._encode_numpy, use_bin_type=True)
    if len(packed_data) > 1024 * 1024:
        return b'LZ4:' + lz4.frame.compress(packed_data)
    return b'RAW:' + packed_data
```

**lz4** is chosen for speed rather than ratio — it compresses and decompresses
at hundreds of MB/s, so the CPU cost is negligible against the memory saved.
gzip would compress better and cost far more time.

**The 1 MB threshold.** Below it, the compression overhead exceeds the benefit.

**The `LZ4:` / `RAW:` prefix** is the important design detail. The reader knows
which path to take by inspecting the first four bytes — no separate metadata
key, no guessing, no "try decompressing and see if it throws". A
self-describing format.

---

## 8.7 Where caches live in the request

```
request
  │
  ├─ resolve audio reference → absolute path
  │
  ├─ hashes = both_hashes(path)          ~1.2 ms
  ├─ keys = <family>_keys(model, hashes)
  │
  ├─ for ns, key in keys:                 ← try every spelling
  │     cached = get_result_sync(ns, key)
  │     if cached: return cached           ← target: <200 ms total
  │
  ├─ compute (model inference)            ← seconds
  │
  ├─ for ns, key in keys:                 ← write every spelling
  │     cache_result_sync(ns, key, result, ttl=86400)
  │
  └─ return result
```

The acoustic route is the clearest example:

```python
# Keys come from cache_keys so this route, the dataset warmup and any future
# writer cannot drift apart. Hand-rolled duplicates are how the transcript
# family ended up holding two incompatible payload shapes.
keys = ck.acoustic_keys(ck.both_hashes(resolved_path))
for ns, key in keys:
    cached = get_result_sync(ns, key)
    if cached:
        return cached
...
prof = extract_acoustic_profile(audio, sr)
for ns, key in keys:
    cache_result_sync(ns, key, prof, ttl=86400)
return prof
```

**Read all spellings, write all spellings, and get both lists from the same
builder.** The comment names the alternative and its consequence.

The saliency route has its own key construction, which predates the module:

```python
SALIENCY_SCHEMA_VERSION = "v3"  # bump to bust stale caches after logic changes
...
file_stat = resolved_path.stat()
file_content_hash = hashlib.md5(
    f"{str(resolved_path)}_{file_stat.st_size}_{file_stat.st_mtime}".encode()
).hexdigest()
cache_key = f"saliency_{SALIENCY_SCHEMA_VERSION}_{request.model}_{request.method}_{file_content_hash}"
```

This is `content_hash` re-implemented inline. It works, and it is exactly the
drift the `cache_keys` module exists to prevent — a second implementation that
must be kept in step by hand. Note also that `method` is in the key: Grad-CAM
and LIME for the same clip and model are different results and must not
collide.

And the `no_cache` escape hatch:

```python
if not request.no_cache:
    cached_result = await get_result("saliency", cache_key)
    if cached_result is not None:
        logger.info(f"Returning cached saliency for {resolved_path}")
        return SaliencyResponse(**cached_result)
```

Necessary when developing attribution code: without it you change the algorithm
and keep seeing the old heatmap for six hours.

---

## 8.8 TTLs

| Data | TTL | Reason |
|---|---|---|
| Transcripts, SER, ADD | 24 h | expensive, fully deterministic |
| Saliency | 6 h | large payloads; algorithms change often |
| Acoustic profile | 24 h | cheap but frequently re-read |
| RQ results | 24 h | matches the analysis window |
| RQ failures | 1 h | long enough to debug, short enough not to accumulate |
| Sessions | 24 h | a working day |
| Mongo analysis records | 24 h | durable tier's own expiry |
| Mongo bias reports | never | research output, deliberately kept |

Saliency's shorter TTL reflects both size (a full spectrogram plus an
attribution matrix) and volatility (the algorithms are still changing). Bias
reports are kept forever because they are the research output — the thing the
project exists to produce.

---

## 8.9 Summary

- Three failure modes: wrong key (benign), colliding key (fast and wrong),
  wrong shape (worse than a miss, because the consumer does not recompute).
- Three hashes: path (fast, stale on in-place edit), path+size+mtime
  (usually right), SHA-256 over content (correct). All three are written;
  content is read first, which is the migration path.
- MD5 is fine for keys derived from paths and wrong for keys derived from
  content. Know which property you need.
- The rejected `(size, mtime)` memo is recorded so nobody re-adds it.
- Key families pair a key pattern with a documented payload shape. One shape,
  one family.
- Per-family schema versions are cache *invalidation*, not documentation — add
  a field without bumping and your change is invisible for a full TTL.
- Defensive readers (`as_transcript`, `unwrap_prediction`) are migration
  infrastructure: a TTL means you run two data formats at once.
- Build keys in functions, never inline, or two call sites will disagree.
- `except Exception: pass` is right for a cache write and wrong for almost
  everything else. The test: would the caller still be correct?
- Sync and async duplicates exist because you cannot await from sync code, and
  the two clients decode differently.
- The content-addressed manager canonicalises params before hashing, encodes
  NumPy arrays as raw bytes (~5× smaller than JSON), and prefixes `LZ4:`/`RAW:`
  so the format is self-describing.

Next: [Chapter 9 — XAI implementation](09-xai-implementation.md).
