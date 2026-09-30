# Chapter 5 — The model registry: loading a model safely

Every model in AudioLIT passes through one module:
`Backend/app/domain/model_registry_service.py`. 409 lines that answer one
question — *how do you let a user type an arbitrary internet identifier and
load whatever it names, without being compromised or silently lied to?*

The requirement is real. Users can paste any Hugging Face repository id. That
is a feature: the whole point is interpreting *your* model, not only the
built-in ones. It also means the input is untrusted.

---

## 5.1 The four ways this can go wrong

1. **Code execution.** `pickle` weights run arbitrary code on load.
2. **Wrong architecture.** A model AudioLIT cannot attach hooks to cannot be
   explained, and pretending otherwise produces fabricated output.
3. **Silent partial load.** Head parameter names that do not match the class
   get randomly initialised — a model that runs and returns noise (§1.8).
4. **Irreproducibility.** A branch name points at different weights next week,
   so a cached result no longer corresponds to the model that produced it.

The registry addresses all four, in that order.

---

## 5.2 Typed rejection, not exceptions-as-strings

```python
UNSUPPORTED_ARCHITECTURE = "UNSUPPORTED_ARCHITECTURE"
UNSAFE_ARTIFACT = "UNSAFE_ARTIFACT"
HUB_UNAVAILABLE = "HUB_UNAVAILABLE"


class ModelRegistryError(Exception):
    """Typed rejection from the registry (unsupported family, unsafe artifact,
    or a Hub outage) — raised instead of attaching partial/fabricated behaviour.
    """
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
```

Three codes, and the distinction between them is not cosmetic. They call for
different responses from the caller and from the user:

| Code | Meaning | What the user should do |
|---|---|---|
| `UNSAFE_ARTIFACT` | no safetensors weights | nothing — this model will never be accepted |
| `UNSUPPORTED_ARCHITECTURE` | not Whisper or Wav2Vec2 | pick a different model |
| `HUB_UNAVAILABLE` | network or Hub problem | retry later |

Only the third is retryable. A string message would force the caller to
pattern-match on prose to find that out. A code makes it a switch.

The docstring names the alternative that was rejected: *"raised instead of
attaching partial/fabricated behaviour."* The registry's whole disposition is
that **refusing is better than approximating**.

---

## 5.3 The supported families

```python
# model_type (from config.json) -> (family label, loader class). Committed
# support per SRS FR1: Whisper and Wav2Vec2 only (LIT-207 scope boundary).
_SUPPORTED_MODEL_TYPES = {
    "whisper": ("whisper", WhisperModel),
    "wav2vec2": ("wav2vec2", Wav2Vec2Model),
}
```

Two families, because hook attachment (§5.8) requires knowing a model's module
tree. There is no generic way to find "the encoder layers" of an arbitrary
transformer — different architectures name and nest them differently. Each
supported family needs a hand-written resolver.

Adding a third family is a well-defined, small task: add an entry here and a
resolver in `hook_manager_service.py`. That is the extension point.

---

## 5.4 Resolution: reject before you download

```python
def resolve_model_id(model_id: str, revision: str = "main", api: Optional[HfApi] = None) -> ResolvedModel:
    """Resolve a model ID to a pinned revision + family, rejecting unsafe/unsupported models.

    Rejects BEFORE any weight download: a repo with no .safetensors file is an
    UNSAFE_ARTIFACT (loading a pickle checkpoint is itself the vulnerability),
    and a model_type outside {whisper, wav2vec2} is UNSUPPORTED_ARCHITECTURE.
    """
```

The ordering in that docstring is the security property. Consider the
alternative: download the weights, then check. You have now written an
attacker-controlled pickle to disk. You have not executed it, so you are
*probably* fine — but the file is there, and "probably fine" is not a security
argument. Checking the file *listing* first means an unsafe repo is rejected
before a single weight byte is transferred.

### Step 1 — ask the Hub what is in the repo

```python
    api = api or HfApi()
    try:
        info = _hub_breaker.guard(api.model_info, model_id, revision=revision, files_metadata=True)
    except HfHubHTTPError as e:
        raise ModelRegistryError(HUB_UNAVAILABLE, f"Could not resolve '{model_id}' from the Hub: {e}")
```

`api.model_info` returns metadata — file names, sizes, and `info.sha`, the
resolved commit hash. No weights.

The `api = api or HfApi()` parameter is a testability decision. Tests inject a
fake `api` object, so the whole resolution path can be tested without network
access. If `HfApi()` were constructed unconditionally inside the function,
every test would need real network or monkeypatching of the module.

### Step 2 — the safetensors gate

```python
    has_safetensors = any(
        sibling.rfilename.endswith(".safetensors") for sibling in (info.siblings or [])
    )
    if not has_safetensors:
        raise ModelRegistryError(
            UNSAFE_ARTIFACT,
            f"'{model_id}' has no .safetensors weights; refusing to load pickle/state-dict artifacts.",
        )
```

`info.siblings` is the file listing. `(info.siblings or [])` guards against
`None`, which some Hub responses return for empty repos.

Recall from §1.10 what is at stake. A `.bin` file is a Python pickle, and
unpickling **executes arbitrary code by design** — `__reduce__` on a malicious
class can run any command. There is no safe way to unpickle untrusted data.
Safetensors has no code path at all: a JSON header of names, shapes and dtypes,
plus raw tensor bytes.

This rejection is *blunt*. A legitimate older model that only publishes `.bin`
is refused. That is the intended trade, and it has a real cost — the inherited
default SER checkpoint was one such model, and replacing it was work (§1.8).
The project accepted that cost rather than adding an "I trust this one" escape
hatch, because a trust flag inevitably gets set.

### Step 3 — architecture check

```python
    try:
        config_path = _hub_breaker.guard(hf_hub_download, model_id, "config.json", revision=info.sha)
    except HfHubHTTPError as e:
        raise ModelRegistryError(HUB_UNAVAILABLE, f"Could not fetch config for '{model_id}': {e}")

    with open(config_path) as f:
        config = json.load(f)
    model_type = config.get("model_type", "")

    supported = _SUPPORTED_MODEL_TYPES.get(model_type)
    if supported is None:
        raise ModelRegistryError(
            UNSUPPORTED_ARCHITECTURE,
            f"'{model_id}' has model_type='{model_type}'; supported families: "
            f"{', '.join(sorted({v[0] for v in _SUPPORTED_MODEL_TYPES.values()}))}.",
        )
    family, _ = supported

    return ResolvedModel(model_id=model_id, revision=info.sha, family=family)
```

Only `config.json` is downloaded — a few kilobytes. Note it is fetched at
`revision=info.sha`, not at the branch name: the config that is checked must be
the config that belongs to the exact commit that will be loaded. Fetching at
`main` and then loading a different commit would let the check pass for one
version and the load happen on another.

The error message lists the supported families, built from the registry dict
rather than hardcoded — so adding a family updates the message automatically.

### Step 4 — the returned identity

```python
@dataclass
class ResolvedModel:
    model_id: str
    revision: str  # resolved commit sha, not a mutable ref like "main"
    family: str
```

The comment is the point. `"main"` is a *pointer*; it moves. A commit SHA is
content-addressed and permanent.

This matters for reproducibility, which is a stated requirement: an analysis
must be reproducible later. If a cached result says "produced by
`org/model@main`", and `main` has since moved, the record is worthless. If it
says `org/model@611e6db8...`, it is exact.

---

## 5.5 The circuit breaker

```python
class _CircuitBreaker:
    """Fails fast after repeated Hub errors instead of piling up slow timeouts.

    Opens after `failure_threshold` failures within `recovery_seconds`; while
    open, `guard()` raises immediately without calling the Hub. Closes again
    once `recovery_seconds` has elapsed since the most recent failure.
    """

    def __init__(self, failure_threshold: int = 5, recovery_seconds: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._failures: deque = deque()

    def _prune(self, now: float) -> None:
        while self._failures and now - self._failures[0] > self.recovery_seconds:
            self._failures.popleft()

    def is_open(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        self._prune(now)
        return len(self._failures) >= self.failure_threshold

    def guard(self, func, *args, **kwargs):
        if self.is_open():
            raise ModelRegistryError(
                HUB_UNAVAILABLE,
                f"Hugging Face Hub circuit open after {self.failure_threshold} recent failures; "
                "not attempting another request.",
            )
        try:
            result = func(*args, **kwargs)
        except HfHubHTTPError:
            self.record_failure()
            raise
        self.record_success()
        return result
```

The **circuit breaker** pattern: after repeated failures against an external
dependency, stop trying for a while.

Without it, if the Hub is down, every request waits for a full HTTP timeout —
maybe 30 seconds. Ten users means ten threads blocked for 30 seconds each. The
dependency's outage becomes your outage, amplified.

Implementation details worth noting:

- **`time.monotonic()`, not `time.time()`.** Monotonic time never goes
  backwards. Wall-clock time can jump — NTP corrections, daylight saving, a
  user changing the clock. A breaker on wall-clock time can be tricked into
  permanently open or permanently closed by a clock adjustment.
- **A sliding window, not a counter.** `_prune` drops failures older than the
  window on every check, so five failures spread over an hour never open the
  breaker. Only five *recent* failures do.
- **`record_success()` clears everything.** One success proves the dependency
  is back.
- **`now` is an injectable parameter.** Tests can advance time without
  sleeping.
- **`except HfHubHTTPError` only.** A `ValueError` from your own code is not
  the Hub's fault and must not count toward opening the breaker. Recording it
  would let a local bug make the Hub look down.

The breaker is a single module-level instance, shared by every call:

```python
_hub_breaker = _CircuitBreaker()
```

Shared state is the point — the breaker must see failures across all callers,
or each caller would need five failures of its own.

---

## 5.6 Download

```python
def download_and_load(resolved, attn_implementation="eager", model_class=None) -> LoadedModel:
    """Download (safetensors-only, version-pinned) and load a resolved model.

    allow_patterns deliberately excludes *.bin — even if a repo has both a
    .safetensors and a legacy pickle checkpoint, only the safe artifact is
    ever fetched.
    """
    try:
        local_dir = _hub_breaker.guard(
            snapshot_download,
            repo_id=resolved.model_id,
            revision=resolved.revision,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*", "vocab*", "merges.txt"],
        )
    except HfHubHTTPError as e:
        raise ModelRegistryError(HUB_UNAVAILABLE, f"Could not download '{resolved.model_id}': {e}")
```

`allow_patterns` is **defence in depth**. §5.4 already rejected repos with no
safetensors. But a repo can contain *both* formats, and `from_pretrained` has
historically preferred whichever it found. By never downloading the `.bin` in
the first place, there is nothing for a later library version's preference
order to pick up.

This is the right shape for a security control: not one check, but a check plus
an arrangement that makes the failure impossible even if the check is bypassed.

The download is pinned to `resolved.revision` — the SHA, again, not a branch.

### Weight digest

```python
def _sha256_of_safetensors(local_dir: str) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(local_dir).rglob("*.safetensors")):
        digest.update(path.read_bytes())
    return digest.hexdigest()
```

A hash over the actual weight bytes. `sorted()` matters: without it, filesystem
enumeration order could vary between machines and produce different digests for
identical weights, which would make the digest useless for comparison.

This digest plus the revision SHA is the model's full identity. Two systems can
compare digests and know whether they ran the same weights.

### Class selection

```python
    if model_class is None:
        _, model_class = _SUPPORTED_MODEL_TYPES[
            next(k for k, v in _SUPPORTED_MODEL_TYPES.items() if v[0] == resolved.family)
        ]
    model = model_class.from_pretrained(local_dir, low_cpu_mem_usage=False,
                                        attn_implementation=attn_implementation)
```

The `model_class` override exists for a specific and important reason:

```python
    """model_class overrides the family's default (bare encoder) class -- e.g.
    a fine-tuned Wav2Vec2ForSequenceClassification checkpoint still resolves
    as family="wav2vec2" (same hook-eligible encoder underneath) but needs
    its own class to keep the classification head instead of loading it as
    a bare Wav2Vec2Model and silently dropping those weights.
    """
```

A fine-tuned emotion classifier has `model_type: "wav2vec2"` in its config, so
it resolves as the wav2vec2 family. But loading it as a bare `Wav2Vec2Model`
would load the encoder and **silently drop the classification head** — the
exact failure from §1.8, arrived at from the other direction. The caller knows
it wants a classifier, so the caller passes the class.

`attn_implementation="eager"` defaults on, because a registry-loaded model may
be used for attribution, and fused attention kernels do not expose attention
weights (§1.6).

`low_cpu_mem_usage=False` is a deliberate choice against the usual advice. The
memory-efficient path creates *meta tensors* — placeholders with shape but no
storage — and moving those to a device requires `to_empty()` rather than `to()`.
That interacts badly with several code paths here, so the straightforward load
is used and the meta case is handled defensively anyway (below).

---

## 5.7 Device placement and the VRAM fallback

```python
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_fallback_reason = None
    try:
        model = model.to(device)
    except Exception as e:
        if _is_vram_exhaustion(e) and device != "cpu":
            # FR1.4: VRAM overflow degrades to CPU rather than failing the load,
            # and says so. A silent retry would leave the user wondering why a
            # model that "loaded fine" is suddenly an order of magnitude slower.
            logger.warning(
                "model_registry: VRAM exhausted loading %s, falling back to CPU: %s",
                resolved.model_id, e,
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                logger.debug("empty_cache failed during VRAM fallback", exc_info=True)
            device = "cpu"
            device_fallback_reason = (
                "GPU memory exhausted while loading %s; running on CPU, which is "
                "substantially slower." % resolved.model_id
            )
            model = model.to(device)
        elif "meta tensor" in str(e):
            model = model.to_empty(device=device)
        else:
            raise
    model.eval()
```

Detecting OOM across torch versions:

```python
def _is_vram_exhaustion(exc: BaseException) -> bool:
    """CUDA OOM, however torch chose to spell it this version."""
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return "out of memory" in str(exc).lower()
```

Belt and braces: the typed exception where it exists, a string match where it
does not. `getattr(getattr(torch, "cuda", None), ...)` handles a torch build
with no CUDA support at all, where `torch.cuda` may be absent.

Three outcomes from the `except`, in priority order: OOM → fall back to CPU;
meta tensor → `to_empty`; anything else → re-raise. **The final `else: raise`
matters.** Swallowing an unknown error here would leave a model in an
undefined device state, and the failure would surface much later somewhere
confusing.

### Making the fallback visible

The requirement is that VRAM overflow triggers a CPU fallback **with a
user-visible warning**. So the flag travels with the model:

```python
@dataclass
class LoadedModel:
    model_id: str
    revision: str
    family: str
    weights_sha256: str
    model: torch.nn.Module
    available_layers: List[str] = field(default_factory=list)
    device: str = "cpu"
    # FR1.4 requires the fallback be *user-visible*, so it travels with the
    # model rather than living only in a log line.
    device_fallback: bool = False
    device_fallback_reason: Optional[str] = None
```

A log line is not user-visible. Nobody watching a spinner reads server logs.
Putting the flag on the returned object means the API can surface it and the UI
can say "this is running on CPU because the GPU ran out of memory" — which
turns an inexplicable 20× slowdown into an explained one.

`model.eval()` is called unconditionally. Forgetting it leaves dropout active
and predictions non-deterministic.

---

## 5.8 Hook attachment, and what "explainable" requires

Attribution needs to read a model's internal activations. PyTorch's mechanism
is **hooks**: callbacks registered on a module that fire when it runs.

`hook_manager_service.py` is a context manager around that.

### Finding the layers

Each family needs a hand-written resolver because module trees differ:

```python
def _resolve_whisper_layers(model: nn.Module) -> List[HookableLayer]:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        encoder = getattr(getattr(model, "model", None), "encoder", None)
    if encoder is None or not hasattr(encoder, "layers"):
        return []

    layers: List[HookableLayer] = []
    for idx, layer in enumerate(encoder.layers):
        layers.append(HookableLayer(f"encoder.layers.{idx}", layer, "encoder"))
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is not None:
            layers.append(HookableLayer(f"encoder.layers.{idx}.self_attn", self_attn, "attention"))
    return layers
```

The two-step encoder lookup handles both shapes: `WhisperModel.encoder` and
`WhisperForConditionalGeneration.model.encoder`. Same architecture, different
nesting, depending on which class you loaded.

Wav2Vec2 differs in two ways — the base is under a `.wav2vec2` attribute on
fine-tuned models, and the attention submodule is called `attention` rather
than `self_attn`:

```python
def _resolve_wav2vec2_layers(model: nn.Module) -> List[HookableLayer]:
    base = getattr(model, "wav2vec2", model)
    encoder = getattr(base, "encoder", None)
    if encoder is None or not hasattr(encoder, "layers"):
        return []
    layers: List[HookableLayer] = []
    for idx, layer in enumerate(encoder.layers):
        layers.append(HookableLayer(f"encoder.layers.{idx}", layer, "encoder"))
        attention = getattr(layer, "attention", None)
        if attention is not None:
            layers.append(HookableLayer(f"encoder.layers.{idx}.attention", attention, "attention"))
    return layers
```

`getattr(model, "wav2vec2", model)` is the idiom for "unwrap if wrapped". These
naming differences are exactly why a generic resolver is not possible.

### Refusing rather than guessing

```python
        if self.family is None or self.family not in _FAMILY_RESOLVERS:
            raise HookRegistrationError(
                UNSUPPORTED_ARCHITECTURE,
                f"No hook support for architecture '{type(model).__name__}'; "
                f"supported families: {', '.join(_FAMILY_RESOLVERS)}.",
            )

        self.layers = _FAMILY_RESOLVERS[self.family](model)
        if not self.layers:
            raise HookRegistrationError(
                UNSUPPORTED_ARCHITECTURE,
                f"Could not resolve encoder/attention layers for '{type(model).__name__}'.",
            )
```

Two rejections: unknown family, and known family whose layers could not be
found. The docstring states the alternative that was refused: *"construction
raises ... rather than attaching partial or guessed hooks."*

Partial hooks would produce partial activations, which would produce an
attribution map over some of the model — presented as an attribution over all
of it.

### The context manager

```python
    def __enter__(self) -> "HookManager":
        config = getattr(self.model, "config", None)
        if config is not None and hasattr(config, "output_attentions"):
            self._prev_output_attentions = config.output_attentions
            config.output_attentions = True

        for layer in self.layers:
            handle = layer.module.register_forward_hook(self._make_hook(layer.name, layer.kind))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

        config = getattr(self.model, "config", None)
        if config is not None and self._prev_output_attentions is not None:
            config.output_attentions = self._prev_output_attentions

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

The docstring states the guarantee:

> *Hooks are only live inside the `with` block; `__exit__` always removes them
> and releases CUDA cache, even if the block raises, so hooks never leak across
> cached-model inferences.*

Three cleanups, all necessary:

- **Remove the handles.** A leaked hook fires on every subsequent forward pass
  through that module — including unrelated requests — capturing tensors into a
  dict that keeps them alive. That is both a memory leak and a correctness bug.
- **Restore `output_attentions`.** The flag was mutated on a *shared cached
  model*. Leaving it `True` makes every later inference compute and return
  attention it does not need, which on a long clip is the 750 MB problem from
  §1.6.
- **`empty_cache()`.** Return freed VRAM to the allocator.

`__exit__` runs on the exception path too. That is the entire reason to use a
context manager rather than paired calls.

### Capturing the right tensor

```python
    @staticmethod
    def _extract_tensor(output: Any, kind: str) -> Optional[torch.Tensor]:
        if torch.is_tensor(output):
            return output
        if isinstance(output, tuple):
            if kind == "attention" and len(output) > 1 and torch.is_tensor(output[1]):
                return output[1]
            if torch.is_tensor(output[0]):
                return output[0]
        return None
```

Transformers modules return tuples whose layout depends on the module. For an
attention module, element 0 is the output states and element 1 is the attention
weights — so `kind` selects which one. Returning `None` for anything
unrecognised means that layer simply is not captured, rather than capturing
something of the wrong meaning under the right name.

And the hook detaches:

```python
    def _make_hook(self, name: str, kind: str):
        def hook(module, inputs, output):
            tensor = self._extract_tensor(output, kind)
            if tensor is not None:
                self.captured[name] = tensor.detach()
        return hook
```

`.detach()` severs the tensor from the autograd graph. Without it, storing the
tensor keeps the entire computation graph alive, and memory grows with every
captured layer.

---

## 5.9 The LRU cache

```python
class ModelRegistry:
    """Lazy-loading, LRU-bounded cache of Whisper/Wav2Vec2 models.

    Nothing loads until get() is actually called for a given model_id — no
    import-time singletons. Evicting the oldest entry beyond max_cache_size
    moves the model off-device and releases CUDA memory before dropping it.
    """

    def __init__(self, max_cache_size: int = 4):
        self.max_cache_size = max_cache_size
        self._cache: "OrderedDict[str, LoadedModel]" = OrderedDict()
        self._active_downloads: dict[str, dict] = {}
        self._lock = threading.Lock()
```

### "No import-time singletons"

This phrase appears throughout the codebase and is worth understanding.

The inherited code did this at module level:

```python
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(...)
emo_model = Wav2Vec2ForSequenceClassification.from_pretrained(...)
```

Consequences: importing the module downloads and loads a model. Every process
that imports it — including the test suite, including a CLI script that only
wanted one function — pays several seconds and several gigabytes. Offline, the
import *fails*, so nothing in the module is usable.

Lazy loading fixes all of that. The cost is that every entry point must call
`ensure_..._loaded()` first, which is a discipline you have to maintain.

### The cache key

```python
            class_suffix = f"#{model_class.__name__}" if model_class else ""
            key = f"{resolved.model_id}@{resolved.revision}{class_suffix}"
```

Three components: id, revision, and class. All three are necessary.

- **Revision** — two commits of the same repo are different models.
- **Class** — the *same weights* loaded as `Wav2Vec2Model` and as
  `Wav2Vec2ForSequenceClassification` are different objects with different
  capabilities. One has a classification head; one does not. Without the class
  in the key, whichever was requested first would be served to both callers,
  and the second would get an object missing the head it needed.

### LRU semantics

```python
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
```

`OrderedDict` plus `move_to_end` gives least-recently-used ordering: a cache
hit moves the entry to the back, so the front is always the least recently
used.

```python
    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.max_cache_size:
            _, evicted = self._cache.popitem(last=False)
            try:
                evicted.model.to("cpu")
            except Exception:
                pass
            del evicted.model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

`popitem(last=False)` takes from the front — the LRU entry.

Eviction is three steps, and skipping any of them leaks GPU memory:

1. **`to("cpu")`** — move tensors off the GPU. Simply dropping the Python
   reference does not free VRAM promptly; the tensors must leave the device.
2. **`del` + `gc.collect()`** — drop the reference and force collection. Python
   frees memory when refcounts hit zero, but PyTorch models are full of
   reference cycles (modules referencing parents), and cycles need the cycle
   collector.
3. **`empty_cache()`** — PyTorch keeps freed blocks in its own allocator pool.
   This returns them to the driver so other processes can use them.

`while`, not `if`, so setting a smaller `max_cache_size` at runtime evicts down
to it in one pass.

### Download cancellation

```python
    def register_download_start(self, model_id: str) -> None:
        with self._lock:
            self._active_downloads[model_id] = {
                "model_id": model_id, "status": "downloading",
                "start_time": time.time(), "cancelled": False,
            }

    def cancel_download(self, model_id: str) -> bool:
        with self._lock:
            if model_id in self._active_downloads:
                self._active_downloads[model_id]["cancelled"] = True
                self._active_downloads[model_id]["status"] = "cancelled"
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
```

A large checkpoint is over a gigabyte. A user who picked the wrong model needs
a way out.

The cancellation is **cooperative**: it sets a flag, and `get()` checks it at
two safe points.

```python
    def get(self, model_id, revision="main", model_class=None) -> LoadedModel:
        self.register_download_start(model_id)
        try:
            resolved = resolve_model_id(model_id, revision=revision)
            if self.is_cancelled(model_id):
                raise ModelRegistryError("CANCELLED", f"Model resolution for '{model_id}' was cancelled by user.")
            ...
            loaded = download_and_load(resolved, model_class=model_class)
            if self.is_cancelled(model_id):
                del loaded
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise ModelRegistryError("CANCELLED", f"Model loading for '{model_id}' was cancelled by user.")

            self._cache[key] = loaded
            self._evict_if_needed()
            return loaded
        finally:
            self.register_download_end(model_id)
```

Cooperative cancellation is the only kind available here, because the download
happens inside `huggingface_hub` code you cannot interrupt. The checks are
placed after each long operation, and the second one explicitly frees the model
that was loaded just before the cancellation was noticed — otherwise a
cancelled load would leave a full model in memory that nothing references but
nothing collects promptly either.

`finally: register_download_end` runs on every path including exceptions, so
the active-downloads dict never accumulates stale entries.

`self._lock` guards `_active_downloads` because it is written by request
threads and read by others. Note the `_cache` itself is *not* under the lock —
a gap the code lives with, since `OrderedDict` operations are individually
atomic under the GIL and the worst case is a duplicate load rather than
corruption.

---

## 5.10 Recording provenance

```python
def _record_model_load(loaded: LoadedModel) -> None:
    """Write-through the loaded model's reproducibility record (LIT-257).

    ``models`` records the resolved revision + weight digest so a model can be
    reproduced exactly later (SRS §3.10); the document maps the tier's fields
    onto the registry's: weight_digest <- weights_sha256, architecture <-
    family, hf_model_id <- model_id, and the unique model_id is
    ``hf_model_id@revision`` (one document per model+revision).

    Never raises: a failed write is logged and the load proceeds (SRS §3.3.1
    graceful degradation; SAD §11.1).
    """
    from ..infrastructure import metadata_store as metadata_store_module

    store = metadata_store_module.get_metadata_store()
    if store is None:
        return
    try:
        store.upsert_model({
            "model_id": f"{loaded.model_id}@{loaded.revision}",
            "name": loaded.model_id,
            "architecture": loaded.family,
            "revision": loaded.revision,
            "weight_digest": loaded.weights_sha256,
            "hf_model_id": loaded.model_id,
        })
    except Exception as exc:
        logger.warning("metadata.write_failed collection=models: %s", exc)
```

Every model load writes a record: which model, which commit, which weight
digest. That is what makes an analysis reproducible six months later.

Two patterns here that recur throughout the codebase:

**`store is None` short-circuits.** The metadata tier is optional (§3.6), so
"not configured" is a normal state, handled before the try block rather than as
an exception.

**The import is function-local.** `from ..infrastructure import metadata_store as metadata_store_module`
inside the function, not at module top. This is deliberate, and the reason is
stated in the project conventions: importing the *module* rather than the
*name*, at call time, means a test that monkeypatches
`metadata_store.get_metadata_store` is seen by this code. A top-level
`from ... import get_metadata_store` would bind the original function object at
import time and never see the patch.

---

## 5.11 Complete trace

A user pastes `myorg/my-whisper-finetune`:

1. `registry.get("myorg/my-whisper-finetune")`
2. `register_download_start` records it as in-flight.
3. `resolve_model_id` → breaker closed → `api.model_info` → file listing +
   `sha = a1b2c3...`
4. Listing contains `model.safetensors` → safe.
5. `config.json` downloaded at `a1b2c3...` → `model_type: "whisper"` →
   supported.
6. Cancellation check: not cancelled.
7. Cache key `myorg/my-whisper-finetune@a1b2c3...` → miss.
8. `snapshot_download` at that SHA, safetensors and JSON only.
9. SHA-256 over the safetensors → weight digest.
10. `WhisperModel.from_pretrained(..., attn_implementation="eager")`.
11. `.to("cuda:0")` → OOM → warn, `empty_cache()`, `.to("cpu")`,
    `device_fallback_reason` set.
12. `.eval()`.
13. `HookManager(...).available_layers()` → 12 names (6 encoder + 6 attention
    for whisper-base).
14. Cancellation check: not cancelled.
15. `LoadedModel` constructed; `_record_model_load` writes to Mongo if
    configured.
16. Cached; evict if over 4 entries.
17. `finally` clears the in-flight record.
18. Returned. The caller can see `device_fallback=True` and tell the user why
    everything is slow.

---

## 5.12 Summary

- One module gates every model load. Untrusted input, four distinct hazards.
- Typed error codes distinguish retryable from permanent.
- Safetensors only, checked from the file *listing* before any weight is
  downloaded, and enforced again by `allow_patterns` so a mixed repo cannot
  surprise a future library version.
- Architecture checked from `config.json` at the pinned SHA.
- Revisions are always resolved to commit SHAs, never left as branch names.
- A circuit breaker on monotonic time with a sliding window stops a Hub outage
  from becoming yours; only Hub errors count toward opening it.
- A weight digest over sorted safetensors files gives models a comparable
  identity.
- `model_class` can override the family default so a fine-tuned checkpoint
  keeps its head.
- VRAM exhaustion falls back to CPU and the reason travels on the returned
  object, because a log line is not user-visible.
- Hooks are attached only inside a context manager; `__exit__` removes them,
  restores mutated config, and frees VRAM even on exceptions.
- Hook resolvers are per-family and refuse rather than guess.
- LRU cache keyed on id + revision + class; eviction moves to CPU, collects
  cycles and empties the allocator.
- Cancellation is cooperative, checked at safe points, and frees a model
  loaded just before the cancel was seen.
- Every load writes a provenance record, best-effort, via a function-local
  module import so tests can patch it.

Next: [Chapter 6 — Inference](06-inference.md).
