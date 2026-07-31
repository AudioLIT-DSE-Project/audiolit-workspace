import hashlib
import json
import logging
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
from transformers import Wav2Vec2Model, WhisperModel

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
    model = model_class.from_pretrained(local_dir, attn_implementation=attn_implementation)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    logger.info(
        "model_registry: loaded %s@%s family=%s weights_sha256=%s",
        resolved.model_id,
        resolved.revision,
        resolved.family,
        weights_sha256,
    )

    return LoadedModel(
        model_id=resolved.model_id,
        revision=resolved.revision,
        family=resolved.family,
        weights_sha256=weights_sha256,
        model=model,
    )


class ModelRegistry:
    """Lazy-loading, LRU-bounded cache of Whisper/Wav2Vec2 models.

    Nothing loads until get() is actually called for a given model_id — no
    import-time singletons. Evicting the oldest entry beyond max_cache_size
    moves the model off-device and releases CUDA memory before dropping it.
    """

    def __init__(self, max_cache_size: int = 4):
        self.max_cache_size = max_cache_size
        self._cache: "OrderedDict[str, LoadedModel]" = OrderedDict()

    def get(self, model_id: str, revision: str = "main", model_class: Optional[type] = None) -> LoadedModel:
        """Get a loaded model, resolving `revision` to a pinned commit sha first.

        Always re-resolves (a cheap, circuit-breaker-guarded metadata call) so
        a moving ref like "main" is checked against the cache by its actual
        current commit, not by the literal string the caller passed -- caching
        by the unresolved string would mean a mutable ref never hits the cache
        and silently re-downloads on every call.
        """
        resolved = resolve_model_id(model_id, revision=revision)
        class_suffix = f"#{model_class.__name__}" if model_class else ""
        key = f"{resolved.model_id}@{resolved.revision}{class_suffix}"

        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        loaded = download_and_load(resolved, model_class=model_class)
        self._cache[key] = loaded
        self._evict_if_needed()
        return loaded

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.max_cache_size:
            _, evicted = self._cache.popitem(last=False)
            try:
                evicted.model.to("cpu")
            except Exception:
                pass
            del evicted.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __len__(self) -> int:
        return len(self._cache)


registry = ModelRegistry()
