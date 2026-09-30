# Chapter 3 — System architecture

Chapters 1 and 2 were about the problem domain. This chapter is about the
software: what the pieces are, why they are separated that way, and what each
boundary buys.

---

## 3.1 The shape of the problem

Every architectural decision here follows from one fact:

> **Running a model takes seconds. A web request must answer in milliseconds.**

Concrete numbers from this project: loading Whisper-base takes about 5 seconds.
Transcribing a short clip takes about 2 seconds. A SHAP attribution takes tens
of seconds on CPU. Meanwhile a browser gives up on an HTTP request after 30–60
seconds, and a user gives up long before that.

The naive design — a route handler that loads a model and returns a prediction
— fails in three ways at once:

1. **It blocks.** A web server has a finite number of worker threads. If each
   holds one for 10 seconds, a handful of users saturate it and everyone else
   waits, including people who only wanted to load the page.
2. **It reloads.** Each request starts fresh, so the 5-second model load is
   paid every time. Measured on this project's own ASR path before it was
   fixed: **5.22 seconds of load against 1.89 seconds of inference — 73% of
   every call was rebuilding a model the process already had.**
3. **It cannot report progress.** HTTP is request/response. A 40-second
   request is a spinner with no information behind it.

The architecture is the answer to those three problems, and almost nothing else.

---

## 3.2 The five layers

```
┌──────────────────────────────────────────────────────────┐
│  PRESENTATION      React 18 + TypeScript + Vite          │
│                    Frontend/src/                          │
└───────────────────────────┬──────────────────────────────┘
                            │  HTTP + WebSocket
┌───────────────────────────▼──────────────────────────────┐
│  APPLICATION       FastAPI gateway — ~15 routers          │
│  (the gateway)     Backend/app/api/routes/                │
│                    NEVER loads an AI model                │
└───────────────────────────┬──────────────────────────────┘
                            │  enqueue
┌───────────────────────────▼──────────────────────────────┐
│  ORCHESTRATION     RQ queues + 5 worker families          │
│                    Backend/app/orchestration/             │
└───────────────────────────┬──────────────────────────────┘
                            │  calls
┌───────────────────────────▼──────────────────────────────┐
│  DOMAIN            models, saliency, DSP, evaluation      │
│                    Backend/app/domain/                    │
└───────────────────────────┬──────────────────────────────┘
                            │  uses
┌───────────────────────────▼──────────────────────────────┐
│  INFRASTRUCTURE    Redis, MongoDB, datasets, settings     │
│                    Backend/app/infrastructure/            │
└──────────────────────────────────────────────────────────┘
```

Each layer depends only on the one below it. That is the rule that keeps a
system with this many moving parts comprehensible.

### Presentation — `Frontend/src/`

React single-page application. Effectively one page: `pages/Index.tsx` renders
`MainLayout`, which composes the workbench panels. Detail in Chapter 15.

### Application — `Backend/app/api/routes/`

FastAPI. Validates input, resolves file references, checks the cache, and
either returns a cached answer or enqueues work and returns a job id.

**The gateway never loads an AI model.** This is the single most important
constraint in the architecture. If it did, the web server would stall every
time a model ran.

There is one deliberate exception, and the reasoning is spelled out where it
happens:

```python
"""Acoustic Wave Profiler exposure (LIT-231, FR10).

Pure DSP - no model load, safe to run synchronously on the request path
(unlike model inference, which is why FR3's async gateway rule doesn't apply
here; SAD §5.1's "gateway never loads AI models" is about AI models
specifically).
"""
```

The rule is about models, not about work. Pitch tracking on a short clip is
fast, deterministic and has no multi-gigabyte state to hold. Running it inline
avoids a whole queue round trip for no benefit. Knowing *why* a rule exists is
what lets you recognise a legitimate exception rather than either breaking the
rule or applying it superstitiously.

### Orchestration — `Backend/app/orchestration/`

The queue fabric. Five worker families, one queue each. Detail in Chapter 7.

### Domain — `Backend/app/domain/`

Where the actual work lives:

| Module | Responsibility |
|---|---|
| `model_registry_service.py` | Safe model download, version pinning, LRU cache |
| `model_loader_service.py` | Inference: ASR, SER, ADD, embeddings |
| `hook_manager_service.py` | Attaching/removing PyTorch hooks safely |
| `saliency_service.py` | All four attribution methods |
| `acoustic_profiler_service.py` | F0, RMS, spectrogram |
| `perturbation_service.py` | Audio mutations, masking |
| `accent_bias_profiler.py` | Cohort batching, per-sample WER |
| `accent_bias_runner.py` | Ranking cohorts into a disparity report |
| `evaluation_service.py` | WER, deletion score, AUC |
| `provenance.py` | The measured/fallback/unavailable contract |

The domain layer has **no knowledge of HTTP, Redis or queues**. Every function
takes plain arguments and returns plain data. That is what makes it testable
without a running stack — and the reason most of the test suite needs no
server.

### Infrastructure — `Backend/app/infrastructure/`

| Module | Responsibility |
|---|---|
| `settings.py` | Configuration via `pydantic-settings` |
| `redis.py` | Async Redis client, session keys, simple result cache |
| `rq_connection.py` | Synchronous Redis connections for RQ |
| `cache_keys.py` | Canonical key builders and payload contracts |
| `metadata_store.py` | MongoDB durable tier |
| `dataset_ingestion.py` | Seven corpus loaders |
| `dataset_service.py` | Resolving a reference to a path on disk |
| `session.py` | Session cookie middleware |
| `logging_config.py` | Structured JSON logging |
| `metrics.py`, `metrics_synthesis.py` | Counters and latency synthesis |

> **`app/services/` does not exist, and its return is a known bug.** Two
> parallel task fabrics were once built because a stale documentation stamp
> pointed at that directory. Chapter 7 tells the story. If you find yourself
> creating it, stop.

---

## 3.3 The request path

```
  browser
     │  POST /inferences/run  {model, file_path}
     ▼
  FastAPI route
     │  1. resolve the audio reference to a real path
     │  2. compute cache keys
     │  3. cache hit?  ──yes──►  return immediately
     │  4. no: enqueue onto the family's queue
     ▼
  returns {job_id, websocket_url}     ← milliseconds
     │
     │  browser opens WS /api/ws/tasks/{job_id}
     ▼
  RQ queue in Redis
     ▼
  worker process (holds its models in memory)
     │  publishes progress → Redis pub/sub channel
     │  audiolit:progress:{job_id}
     ▼
  gateway's WS handler is subscribed, relays each event
     ▼
  browser updates the UI live
     │
  worker finishes → result written to Redis → SUCCESS event
```

Three things to note:

- The **job id is returned immediately**. The HTTP request is short regardless
  of how long the work takes.
- **Progress flows through Redis pub/sub**, not through the worker talking to
  the browser. The worker does not know a browser exists; it publishes to a
  channel. This is what lets a worker run in a different container, or die and
  be replaced, without the frontend needing to care.
- The frontend **falls back to HTTP polling** if the WebSocket fails
  repeatedly. Chapter 15 shows the retry ladder.

---

## 3.4 The five worker families

```python
class WorkerFamily(str, Enum):
    """One worker family per kind of model, plus a CPU-only mutation worker."""
    ASR = "asr"
    SER = "ser"
    ADD = "add"
    XAI = "xai"
    MUTATION = "mutation"
```

One queue and one worker process per family. Two reasons, and they are
different reasons:

**Memory isolation.** Each process only ever holds its own family's models. An
ASR worker holds Whisper; it never holds the emotion or deepfake models. The
governing constraint is that everything running at once must fit in roughly
3–5 GB of GPU memory — a free-tier cloud GPU. One process holding all models
would blow that immediately.

**GPU serialisation.** Model families are GPU-bound and pinned to one worker:

```python
QUEUE_CONFIGS: dict[WorkerFamily, QueueConfig] = {
    WorkerFamily.ASR: QueueConfig(WorkerFamily.ASR, "asr", gpu_bound=True, concurrency=1),
    WorkerFamily.SER: QueueConfig(WorkerFamily.SER, "ser", gpu_bound=True, concurrency=1),
    WorkerFamily.ADD: QueueConfig(WorkerFamily.ADD, "add", gpu_bound=True, concurrency=1),
    WorkerFamily.XAI: QueueConfig(WorkerFamily.XAI, "xai", gpu_bound=True, concurrency=1),
    WorkerFamily.MUTATION: QueueConfig(
        WorkerFamily.MUTATION, "mutation", gpu_bound=False, concurrency=2
    ),
}
```

Two jobs on one GPU do not run twice as fast — they contend for memory and, if
they both allocate at their peak, one gets an out-of-memory error. Serialising
per family is faster *and* more reliable. Mutation is pure CPU signal
processing and may scale out.

Note that the ASR, SER and ADD families can still run **in parallel with each
other** — that is the point of separate queues. What is serialised is two jobs
*within* a family.

---

## 3.5 Fan-out and fan-in

A multi-task analysis runs ASR, SER and ADD on one clip, then combines the
results.

```
 POST /inference/multitask
        │
        ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ asr job │   │ ser job │   │ add job │      (parallel, different queues)
   └────┬────┘   └────┬────┘   └────┬────┘
        └─────────────┼─────────────┘
                      ▼
              ┌───────────────┐
              │ aggregator    │   depends_on all three
              └───────┬───────┘
                      ▼
            combined result + metadata write
```

RQ's `depends_on` holds the aggregator in `DEFERRED` state until every
dependency finishes. The enqueue is pipelined:

```python
with get_redis_connection().pipeline() as pipe:
    for fam in families:
        job = get_queue(fam).enqueue(
            _TASK_FUNCS[fam],
            audio_ref,
            model_ids.get(fam, "default"),
            dict(params.get(fam, {})),
            job_timeout=DEFAULT_JOB_TIMEOUT,
            result_ttl=DEFAULT_RESULT_TTL,
            failure_ttl=DEFAULT_FAILURE_TTL,
            pipeline=pipe,
        )
        family_jobs[fam.value] = job.id
        family_job_objs.append(job)
    # The aggregator's depends_on needs these jobs to exist in Redis, so the
    # pipeline must commit before it is enqueued.
    pipe.execute()
```

A **Redis pipeline** batches commands into one network round trip. Why it
matters, measured:

> *Measured against a live Redis at 10 concurrent users: 22.6 ms median
> sequential against 12.2 ms pipelined, a 46% reduction, with the aggregator
> still correctly DEFERRED on all three dependencies. This matters because a
> loopback Redis round trip is not free: it measured 1.67 ms on Docker Desktop
> for Windows, so the round-trip count, not the work per command, is what the
> enqueue budget is spent on.*

RQ issues about 16 Redis commands per enqueue. Three enqueues is ~48 commands.
At 1.67 ms per round trip, sequential round trips dominate the cost entirely —
the actual work per command is negligible. This is a general lesson about
latency: **count round trips, not operations.**

And note the ordering constraint in the comment. The aggregator's `depends_on`
reads the dependency jobs from Redis, so they must be committed first. The
aggregator is enqueued *after* `pipe.execute()`, deliberately, outside the
pipeline.

The aggregator tolerates partial failure:

```python
for job_id in family_job_ids:
    job = Job.fetch(job_id, connection=conn)
    if job.get_status() != JobStatus.FINISHED:
        combined["tasks"][job_id] = {"status": "failed"}
        continue
    result = job.result or {}
    combined["tasks"][result.get("task", "unknown")] = result
```

One failed sibling never loses the others' results. If SER crashes, you still
get the transcript and the deepfake verdict, and an explicit `failed` marker
for the one that did not make it.

---

## 3.6 Three storage tiers

This is the part people most often get confused about, so it is worth a table
before the prose.

| Tier | Technology | Holds | Survives restart | Required |
|---|---|---|---|---|
| Queue + cache | Redis | job queues, cached results, progress channels, sessions | no (persistence off) | **yes** |
| Durable metadata | MongoDB | analysis records, model provenance, bias reports | yes | no — degrades silently |
| Audio files | local disk (`uploads/`) | uploaded clips | yes, until TTL | yes |

### Redis

Queue broker, result cache and pub/sub bus. Configured with a hard memory cap
and — importantly — a specific eviction policy:

```yaml
command: ["redis-server", "--maxmemory", "2gb", "--maxmemory-policy", "volatile-lru"]
```

Both values are load-bearing, and the comment explains why in detail:

> *This was 256 MB, eight times under the committed budget, which a single
> load-test run exhausted: saliency payloads carry a full base spectrogram plus
> an attribution matrix, so a handful of them fill it. When it filled, writes
> started failing with "OutOfMemoryError: command not allowed when used memory >
> 'maxmemory'" — and because RQ registers and heartbeats workers through the
> same Redis, that killed the workers rather than merely missing a cache entry.*

And on the policy:

> *volatile-lru, not allkeys-lru: this one Redis is also the RQ broker, and
> allkeys-lru evicts *any* key under pressure — RQ's queue lists, its queue
> registry and the per-family worker locks included. When dataset warmup filled
> the cache, queued jobs silently vanished from their queue and a live worker
> lost its lock. volatile-* only evicts keys that carry a TTL: every
> cache/progress/session key does, RQ's queues and registries do not.*

This is worth dwelling on. Sharing one Redis between "cache" and "source of
truth for the job queue" means an eviction policy chosen for the cache can
delete the queue. The fix exploits the fact that every cache key has a TTL and
no RQ structure does, so `volatile-lru` can only ever evict cache entries. It
is a neat solution to a nasty coupling — but the real lesson is that the
coupling existed at all and was only found under load.

Persistence is disabled: **every cache entry is recomputable.** Losing the
cache costs time, not data.

### MongoDB — optional by design

```python
# SRS §3.10 / SAD §9 — durable MongoDB metadata tier. Records must survive
# in MongoDB; nothing here ever touches audio bytes (constraint C4, SR4).
# Empty by default = the tier is "configured off": `get_metadata_store()`
# returns None and every write-through is a logged no-op, so local
# development and CI never need a MongoDB server.
MONGO_URL: str = ""
```

Four collections: `models` (provenance — id, revision, weight digest),
`audio_samples` (paths and metadata, never bytes), `analysis_results` (TTL 24 h),
`bias_reports` (kept permanently — they are research output).

Every write is best-effort:

```python
except Exception as exc:
    logger.warning("metadata.write_failed collection=models: %s", exc)
```

A failed metadata write never fails the operation that triggered it. Graceful
degradation.

> **This has a debugging consequence you must internalise.** A missing feature
> is *never* explained by "Mongo is down", because when Mongo is down the
> system works and simply records nothing. If a prediction is wrong, Mongo is
> not the cause. Conversely, if you expected records and find none, check
> `MONGO_URL` before anything else — the default is off, and compose's `mongo`
> service publishes no host port, so a natively-run API cannot reach it even
> when the container is up.

### Disk

Uploads land in `uploads/` with UUID filenames. Purged on a configurable TTL,
24 hours by default, swept both on each upload and at startup. Chapter 4 has
the code.

---

## 3.7 Two cache key schemes, deliberately

There are two caching systems in this codebase and they coexist on purpose.
Knowing which one you are in is necessary before touching either.

### Scheme 1 — `infrastructure/cache_keys.py` (the hot path)

Inherited from the ECHO baseline. MD5 hashes with three spellings:

```python
def path_hash(resolved_path):
    """``md5`` of the resolved path - the primary key discriminator."""
    return hashlib.md5(str(resolved_path).encode()).hexdigest()

def content_hash(resolved_path):
    """``md5`` of path + size + mtime, so edits in place invalidate the entry."""
    p = Path(resolved_path)
    st = p.stat()
    return hashlib.md5(f"{str(p)}_{st.st_size}_{st.st_mtime}".encode()).hexdigest()

def content_sha256(resolved_path):
    """Streamed SHA-256 over the audio bytes themselves (FR4.1)."""
    h = hashlib.sha256()
    with open(resolved_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
```

`content_sha256` is the correct identity: the same audio at two paths hashes
the same, and a file edited in place hashes differently. The other two are
location-derived and kept for compatibility with entries written earlier.

`both_hashes` returns all three, content first, so readers prefer the
content-addressed key and fall back:

```python
def both_hashes(resolved_path):
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

There is a rejected optimisation documented here that is worth reading, because
the reasoning is subtle:

> *Deliberately not memoised. A memo keyed on (size, mtime) is the obvious
> optimisation and it is unsound here: two same-length writes milliseconds
> apart share both values on this filesystem even at `st_mtime_ns` resolution,
> so the memo returned the pre-edit hash — reintroducing exactly the staleness
> this function exists to remove. Measured at ~1.2 ms for a 5 s clip, against a
> request that then runs a model; the trade is not close.*

The obvious cache key for a hash function is the thing the hash function exists
to distrust. And the measurement settles it: 1.2 ms before a multi-second model
call is not worth a correctness risk.

### Scheme 2 — `core/redis.py` (content-addressed manager)

A more thorough implementation: SHA-256 over `audio_bytes ‖ model_id ‖ task ‖
canonical_params_json`, msgpack serialisation with NumPy support, lz4
compression above 1 MB, deduplication locks.

```python
def _generate_key(self, audio_bytes, model_id, task, params):
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

Two details worth copying. `sort_keys=True, separators=(',',':')` makes the
params JSON **canonical** — the same dict always serialises to the same bytes
regardless of insertion order, so logically identical requests hash
identically. And `_cache_schema_version` is folded into the key, so bumping it
invalidates everything at once.

Serialisation handles NumPy arrays natively:

```python
def _encode_numpy(self, obj):
    if isinstance(obj, np.ndarray):
        return {b'__np__': True, b'dtype': obj.dtype.str.encode(),
                b'shape': list(obj.shape), b'data': obj.tobytes()}
    return obj

def _serialize(self, value) -> bytes:
    packed_data = msgpack.packb(value, default=self._encode_numpy, use_bin_type=True)
    if len(packed_data) > 1024 * 1024:
        return b'LZ4:' + lz4.frame.compress(packed_data)
    return b'RAW:' + packed_data
```

The `LZ4:`/`RAW:` prefix means the reader knows whether to decompress without
guessing or storing metadata elsewhere. Compression only above 1 MB, because
below that the CPU cost exceeds the transfer saving.

Its only production consumer is `results.py`. **Whether it moves or is retired
is an open decision.** Do not resolve it by deleting the module — a route
depends on it.

---

## 3.8 Configuration

`pydantic-settings` reads from environment variables with typed defaults:

```python
class Settings(BaseSettings):
    REDIS_URL: str = "redis://localhost:6379/0"
    LOG_FORMAT: str = "json"
    SESSION_COOKIE_NAME: str = "sid"
    SESSION_TTL_SECONDS: int = 24 * 60 * 60
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"
    COOKIE_DOMAIN: str | None = None
    DEBUG_ENABLED: bool = False
    DATASET_FOOTPRINT_LIMIT_GB: float = 100.0
    DATASET_METADATA_ROW_CAP: int = 2000
    MONGO_URL: str = ""
    MONGO_DB_NAME: str = "audiolit"
    MONGO_ANALYSIS_TTL_HOURS: int = 24
    MONGO_SERVER_SELECTION_TIMEOUT_MS: int = 1500
```

`DEBUG_ENABLED: bool = False` is a security fix, not a convenience:

```python
# the inherited /debug/session diagnostic was reachable unauthenticated and
# echoed cookies + request headers back to any caller. Now disabled unless
# explicitly enabled, and when enabled it returns only the session id.
```

Two lessons. **Default to off** for anything diagnostic. And when you do
re-enable it, return the minimum — "only the session id", not the whole
request context.

Upload limits are plain environment variables rather than settings fields,
because they are read at module import in the route:

```python
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("AUDIOLIT_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
MAX_UPLOAD_DURATION_SECONDS = float(os.getenv("AUDIOLIT_MAX_UPLOAD_SECONDS", str(15 * 60)))
UPLOAD_RETENTION_SECONDS = float(os.getenv("AUDIOLIT_UPLOAD_RETENTION_SECONDS", str(24 * 60 * 60)))
```

---

## 3.9 CORS and sessions

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Cross-Origin Resource Sharing** is the browser rule that a page from origin A
may not read responses from origin B unless B opts in. The frontend runs on
`:8080` and the API on `:8000` — different origins — so the API must opt in.

The regex restricts this to localhost on any port. It is **not** a wildcard,
and that is deliberate: the inherited baseline had `Access-Control-Allow-Origin:
"*"` on file-serving routes, which was removed:

```python
# LIT-223: remove the route-level Access-Control-Allow-Origin: "*" here -
# CORS is the app's CORSMiddleware's job, with a restricted origin allow-list.
```

Two points. A wildcard with `allow_credentials=True` is a genuine
vulnerability. And CORS belongs in one place — middleware — not sprinkled per
route, or you get inconsistent policy nobody can audit.

Sessions are a cookie (`sid`) carrying an opaque id, with state in Redis under
`sess:{sid}`, TTL 24 hours. `SameSite=Lax` means the cookie is sent on
top-level navigations but not on cross-site subrequests. This is why the
compose file is fussy about host names:

> *All messages/options deliberately stay on the 127.0.0.1 host bus: the
> session cookie is SameSite=Lax, and http://127.0.0.1:8080 ->
> http://127.0.0.1:8000 is same-site (ports do not change the site), so the
> browser keeps sending it.*

Mixing `localhost` and `127.0.0.1` between the two makes them cross-site, the
cookie stops being sent, and sessions silently break. Ports do not change the
site; host names do.

---

## 3.10 Deployment

`docker-compose.yml` brings up five services:

| Service | Image | Port | Role |
|---|---|---|---|
| `redis` | redis:7-alpine | 6379 | queue + cache |
| `mongo` | mongo:6 | none published | durable metadata |
| `api` | built from `Backend/Dockerfile` | 8000 | gateway |
| `worker` | same image, different command | none | all five families |
| `web` | built from `Frontend/Dockerfile` | 8080 | nginx serving the Vite build |

Two shared volumes: `model-cache` (`HF_HOME=/models`) so downloaded weights
survive restarts and are shared between API and workers, and `mongo-data`.

Two mount details that bite people:

```yaml
# Must be /app/data, not /data. The loaders resolve the corpus root
# from the module path (parents[2] of app/infrastructure/), which is
# /app inside the image, so a mount at /data is invisible to them.
- ./Backend/data:/app/data
```

The dataset root is derived from the module's own file path:

```python
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
```

That makes it work identically in a container and natively, with no
configuration — but it means the mount point must match where the code lives,
not where it feels natural.

And `pull_policy: build`:

```yaml
# Always rebuild from the working tree on `docker compose up`. Without
# this, compose silently reuses whatever audiolit-backend:latest already
# exists, so backend fixes never reach the containers unless someone
# remembers --build.
```

The worker healthcheck is a good example of a check that means what it says:

```yaml
test: ["CMD", "python", "-c", "... served = {q.split(':')[-1] for w in Worker.all(connection=c) for q in w.queue_names()}; sys.exit(0 if {f.value for f in WorkerFamily} <= served else 1)"]
```

> *Healthy here means every family (asr, ser, add, xai, mutation) has a worker
> registered — "any worker at all" stayed green while a crashed family's queue
> went unserved.*

A healthcheck that can pass while the thing it checks is broken is worse than
none, because it actively reassures you.

---

## 3.11 Directory reference

```
audiolit-workspace/
├── Backend/
│   ├── app/
│   │   ├── main.py                 # app assembly, middleware, 15 routers
│   │   ├── api/
│   │   │   ├── dependencies.py     # get_session_id
│   │   │   └── routes/             # one module per resource
│   │   ├── domain/                 # models, XAI, DSP, evaluation
│   │   ├── orchestration/          # queues, workers, task functions
│   │   ├── infrastructure/         # Redis, Mongo, datasets, settings
│   │   └── core/redis.py           # FR4 cache manager (open decision)
│   ├── tests/                      # pytest, ~40 files
│   ├── data/                       # corpora (gitignored)
│   ├── loadtests/locustfile.py     # on the `testing` branch
│   ├── apitests/                   # Postman collection
│   ├── scripts/evaluate_models.py  # evaluation runner
│   ├── requirements.txt
│   └── Dockerfile
├── Frontend/
│   ├── src/
│   │   ├── pages/Index.tsx
│   │   ├── components/
│   │   │   ├── layout/             # MainLayout, Toolbar
│   │   │   ├── panels/             # the workbench panels
│   │   │   ├── audio/              # upload, waveform, table
│   │   │   ├── visualization/      # canvas overlays, plots
│   │   │   ├── analysis/           # perturbation tools
│   │   │   ├── dataset/            # dataset management
│   │   │   └── ui/                 # shadcn/ui primitives
│   │   ├── contexts/               # Embedding, Playback
│   │   ├── context/                # ModelRegistry  ← note: two dirs
│   │   ├── hooks/                  # useTaskStatus
│   │   └── lib/                    # api.ts, heatmap.ts, warmupJob.ts
│   ├── e2e/                        # Playwright
│   └── package.json
├── docs/
├── docker-compose.yml
└── .github/workflows/ci.yml
```

> **`src/context/` and `src/contexts/` both exist.** `context/` holds
> `ModelRegistryContext`; `contexts/` holds `EmbeddingContext` and
> `PlaybackContext`. This is an accident that has calcified. Check which one you
> mean.

---

## 3.12 Summary

- One fact drives everything: models take seconds, requests must take
  milliseconds.
- Five layers, each depending only on the one below.
- The gateway never loads a model. Pure DSP is a reasoned exception.
- Five worker families, one queue and process each, model families pinned to
  concurrency 1 for GPU memory.
- Fan-out to family queues, fan-in via an RQ dependency; pipelined enqueue
  because round-trip count is the cost.
- Three storage tiers: Redis (required, no persistence, `volatile-lru` so the
  cache cannot evict the queue), MongoDB (optional, silent degradation), disk
  (TTL-purged).
- Two cache-key schemes coexist deliberately; know which one you are in.
- CORS restricted to localhost, never a wildcard; `SameSite=Lax` means host
  names must match between frontend and API.
- Docker Compose runs the whole stack; the dataset mount point is dictated by
  where the code lives.

Next: [Chapter 4 — An audio file's journey](04-upload-path.md).
