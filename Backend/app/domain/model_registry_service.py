import hashlib
import json
import logging
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
from transformers import Wav2Vec2Model, WhisperModel

from .hook_manager_service import HookManager, HookRegistrationError

logger = logging.getLogger(__name__)

UNSUPPORTED_ARCHITECTURE = "UNSUPPORTED_ARCHITECTURE"
UNSAFE_ARTIFACT = "UNSAFE_ARTIFACT"
HUB_UNAVAILABLE = "HUB_UNAVAILABLE"

# model_type (from config.json) -> (family label, loader class). Committed
# support per SRS FR1: Whisper and Wav2Vec2 only (LIT-207 scope boundary).
_SUPPORTED_MODEL_TYPES = {
    "whisper": ("whisper", WhisperModel),
    "wav2vec2": ("wav2vec2", Wav2Vec2Model),
}


class ModelRegistryError(Exception):
    """Typed rejection from the registry (unsupported family, unsafe artifact,
    or a Hub outage) — raised instead of attaching partial/fabricated behaviour.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class ResolvedModel:
    model_id: str
    revision: str  # resolved commit sha, not a mutable ref like "main"
    family: str


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

    def attach_hooks(self) -> HookManager:
        """Fresh HookManager for this model, used as a context manager for
        attribution: `with loaded.attach_hooks() as hooks: ...`. Layer
        resolution already succeeded once at load time (see available_layers
        below) so this is expected to always succeed too.
        """
        return HookManager(self.model, family=self.family)


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

    def record_failure(self, now: Optional[float] = None) -> None:
        self._failures.append(time.monotonic() if now is None else now)

    def record_success(self) -> None:
        self._failures.clear()

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


_hub_breaker = _CircuitBreaker()


def resolve_model_id(model_id: str, revision: str = "main", api: Optional[HfApi] = None) -> ResolvedModel:
    """Resolve a model ID to a pinned revision + family, rejecting unsafe/unsupported models.

    Rejects BEFORE any weight download: a repo with no .safetensors file is an
    UNSAFE_ARTIFACT (loading a pickle checkpoint is itself the vulnerability),
    and a model_type outside {whisper, wav2vec2} is UNSUPPORTED_ARCHITECTURE.
    """
    api = api or HfApi()

    try:
        info = _hub_breaker.guard(api.model_info, model_id, revision=revision, files_metadata=True)
    except HfHubHTTPError as e:
        raise ModelRegistryError(HUB_UNAVAILABLE, f"Could not resolve '{model_id}' from the Hub: {e}")

    has_safetensors = any(
        sibling.rfilename.endswith(".safetensors") for sibling in (info.siblings or [])
    )
    if not has_safetensors:
        raise ModelRegistryError(
            UNSAFE_ARTIFACT,
            f"'{model_id}' has no .safetensors weights; refusing to load pickle/state-dict artifacts.",
        )

    try:
        config_path = _hub_breaker.guard(
            hf_hub_download, model_id, "config.json", revision=info.sha
        )
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


def _sha256_of_safetensors(local_dir: str) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(local_dir).rglob("*.safetensors")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _is_vram_exhaustion(exc: BaseException) -> bool:
    """CUDA OOM, however torch chose to spell it this version."""
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return "out of memory" in str(exc).lower()


def download_and_load(
    resolved: ResolvedModel,
    attn_implementation: str = "eager",
    model_class: Optional[type] = None,
) -> LoadedModel:
    """Download (safetensors-only, version-pinned) and load a resolved model.

    allow_patterns deliberately excludes *.bin — even if a repo has both a
    .safetensors and a legacy pickle checkpoint, only the safe artifact is
    ever fetched.

    model_class overrides the family's default (bare encoder) class -- e.g.
    a fine-tuned Wav2Vec2ForSequenceClassification checkpoint still resolves
    as family="wav2vec2" (same hook-eligible encoder underneath) but needs
    its own class to keep the classification head instead of loading it as
    a bare Wav2Vec2Model and silently dropping those weights.
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

    weights_sha256 = _sha256_of_safetensors(local_dir)

    if model_class is None:
        _, model_class = _SUPPORTED_MODEL_TYPES[
            next(k for k, v in _SUPPORTED_MODEL_TYPES.items() if v[0] == resolved.family)
        ]
    model = model_class.from_pretrained(local_dir, low_cpu_mem_usage=False, attn_implementation=attn_implementation)
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

    try:
        available_layers = HookManager(model, family=resolved.family).available_layers()
    except HookRegistrationError as e:
        # resolve_model_id already gated model_type to a supported family, so
        # this should be rare in practice (an architecture whose config
        # claims whisper/wav2vec2 but whose module tree doesn't match) -- but
        # don't fabricate a layer list if it happens, and don't fail the load
        # over it either, since the model itself still works for inference.
        logger.warning("model_registry: hook layer resolution failed for %s: %s", resolved.model_id, e)
        available_layers = []

    logger.info(
        "model_registry: loaded %s@%s family=%s weights_sha256=%s layers=%d",
        resolved.model_id,
        resolved.revision,
        resolved.family,
        weights_sha256,
        len(available_layers),
    )

    return LoadedModel(
        model_id=resolved.model_id,
        revision=resolved.revision,
        family=resolved.family,
        weights_sha256=weights_sha256,
        model=model,
        available_layers=available_layers,
        device=device,
        device_fallback=device_fallback_reason is not None,
        device_fallback_reason=device_fallback_reason,
    )


import threading

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

    def register_download_start(self, model_id: str) -> None:
        with self._lock:
            self._active_downloads[model_id] = {
                "model_id": model_id,
                "status": "downloading",
                "start_time": time.time(),
                "cancelled": False,
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
        logger.info("model_registry: cancelled download and executed memory cleanup for %s", model_id)
        return True

    def register_download_end(self, model_id: str) -> None:
        with self._lock:
            self._active_downloads.pop(model_id, None)

    def is_cancelled(self, model_id: str) -> bool:
        with self._lock:
            download_info = self._active_downloads.get(model_id)
            return download_info.get("cancelled", False) if download_info else False

    def get_active_downloads(self) -> List[dict]:
        with self._lock:
            return list(self._active_downloads.values())

    def get(self, model_id: str, revision: str = "main", model_class: Optional[type] = None) -> LoadedModel:
        """Get a loaded model, resolving `revision` to a pinned commit sha first."""
        self.register_download_start(model_id)
        try:
            resolved = resolve_model_id(model_id, revision=revision)
            if self.is_cancelled(model_id):
                raise ModelRegistryError("CANCELLED", f"Model resolution for '{model_id}' was cancelled by user.")

            class_suffix = f"#{model_class.__name__}" if model_class else ""
            key = f"{resolved.model_id}@{resolved.revision}{class_suffix}"

            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

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

    def __len__(self) -> int:
        return len(self._cache)


registry = ModelRegistry()

