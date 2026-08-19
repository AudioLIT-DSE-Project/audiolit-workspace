"""Canonical cache-key builders and payload shapes (SRS FR4).

Every key family below is consumed by at least one route. The *shape* of the
value stored under a key is part of its contract: a writer that stores the
right key with the wrong shape is worse than a cache miss, because the
consumer will not fall back to recomputation - it will read the value and
fail on it.

That is exactly the defect this module exists to prevent. Dataset warmup used
to store the ASR result of ``transcribe_whisper_with_attention`` (a
``{"text", "attention"}`` dict) under the transcript family, whose consumers
all assume ``prediction`` is a plain string; ``/inferences/whisper-accuracy``
then died with ``AttributeError: 'dict' object has no attribute 'lower'``
and the UI sat on a spinner forever.

Two hashes are in play, both over the *resolved path*:

* ``path_hash``    - ``md5(str(path))``. What almost every route computes
  (several of them under the misleading local name ``file_content_hash``).
* ``content_hash`` - ``md5(f"{path}_{size}_{mtime}")``. Used by the saliency
  route and as a secondary key elsewhere.

Warm both wherever a family is read under either, so a warmed entry is found
no matter which spelling the consumer uses.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

# Bumped when a stored shape changes incompatibly (FR4.1).
CACHE_SCHEMA_VERSION = "v2"

# Namespace used by routes that read predictions without knowing the model.
PREDICTIONS_NS = "predictions"


def path_hash(resolved_path: str | Path) -> str:
    """``md5`` of the resolved path - the primary key discriminator."""
    return hashlib.md5(str(resolved_path).encode()).hexdigest()


def content_hash(resolved_path: str | Path) -> str:
    """``md5`` of path + size + mtime, so edits in place invalidate the entry."""
    p = Path(resolved_path)
    st = p.stat()
    return hashlib.md5(f"{str(p)}_{st.st_size}_{st.st_mtime}".encode()).hexdigest()


def both_hashes(resolved_path: str | Path) -> tuple[str, str]:
    """``(path_hash, content_hash)`` for the given file."""
    return path_hash(resolved_path), content_hash(resolved_path)


# --------------------------------------------------------------------------- #
# Key families. Each returns [(namespace, key), ...] - every spelling a
# consumer may look under, so callers write all of them for one payload.
# --------------------------------------------------------------------------- #

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


def attention_keys(model: str, hashes: tuple[str, ...]) -> list[tuple[str, str]]:
    """ASR + attention. Payload: ``{"prediction": {"text", "attention"}}``.

    Read by ``/inferences/whisper-attention``.
    """
    return [
        (model, f"{model}_attention_{CACHE_SCHEMA_VERSION}_{h}") for h in hashes
    ]


def ser_keys(hashes: tuple[str, ...]) -> list[tuple[str, str]]:
    """SER. Payload: ``{"prediction": <ser dict>}``.

    ``wav2vec2_detailed_``          -> ``/inferences/wav2vec2-batch``
    ``wav2vec2_detailed_attention_v3_`` -> ``/inferences/wav2vec2-detailed``
    """
    keys: list[tuple[str, str]] = []
    for h in hashes:
        keys.append(("wav2vec2", f"wav2vec2_detailed_{h}"))
        keys.append(("wav2vec2", f"wav2vec2_detailed_attention_v3_{h}"))
    return keys


def deepfake_keys(model: str, hashes: tuple[str, ...]) -> list[tuple[str, str]]:
    """ADD. Payload: ``{"prediction": <add dict>}``. Shares the transcript
    family's key shape because ``/inferences/run`` serves every model."""
    keys: list[tuple[str, str]] = []
    for h in hashes:
        keys.append((model, f"{CACHE_SCHEMA_VERSION}_{model}_{h}"))
        keys.append((model, f"{model}_{h}"))
    return keys


def acoustic_keys(hashes: tuple[str, ...]) -> list[tuple[str, str]]:
    """Acoustic profile. Payload: the profile dict itself, unwrapped."""
    return [("acoustic", f"acoustic_profile_{h}") for h in hashes]


def saliency_keys(model: str, method: str, hashes: tuple[str, ...]) -> list[tuple[str, str]]:
    """Saliency. Payload: the saliency result dict itself, unwrapped."""
    return [
        ("saliency", f"saliency_{CACHE_SCHEMA_VERSION}_{model}_{method}_{h}")
        for h in hashes
    ]


def embedding_keys(model: str, hashes: tuple[str, ...]) -> list[tuple[str, str]]:
    """Latent embedding. Payload: ``{"embedding": [...]}`` (FR11)."""
    return [(model, f"{model}_embeddings_{h}") for h in hashes]


def audio_frequency_keys(hashes: tuple[str, ...]) -> list[tuple[str, str]]:
    """Frequency features. Payload: ``{"features": {...}}``."""
    return [("audio_frequency", f"audio_frequency_{h}") for h in hashes]


# --------------------------------------------------------------------------- #
# Payload normalisation
# --------------------------------------------------------------------------- #

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


def unwrap_prediction(cached: Any) -> Any:
    """Return the payload a cache entry carries, tolerating both wrappings."""
    if isinstance(cached, dict) and "prediction" in cached:
        return cached["prediction"]
    return cached
