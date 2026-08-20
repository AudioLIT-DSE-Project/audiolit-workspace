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


def content_sha256(resolved_path: str | Path) -> str:
    """Streamed SHA-256 over the audio bytes themselves (FR4.1).

    This is the identity FR4 specifies: *"a SHA-256 hash of the (audio bytes,
    model identifier, task, parameters) tuple"*. The two hashes above key on
    where a file sits, so the same audio at two paths caches twice and a file
    edited in place can serve a stale result.

    Deliberately not memoised. A memo keyed on (size, mtime) is the obvious
    optimisation and it is unsound here: two same-length writes milliseconds
    apart share both values on this filesystem even at ``st_mtime_ns``
    resolution, so the memo returned the pre-edit hash - reintroducing exactly
    the staleness this function exists to remove. Measured at ~1.2 ms for a 5 s
    clip, against a request that then runs a model; the trade is not close.
    """
    h = hashlib.sha256()
    with open(resolved_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def both_hashes(resolved_path: str | Path) -> tuple[str, str, str]:
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


DEFAULT_SER_MODEL = "firdhokk/speech-emotion-recognition-with-facebook-wav2vec2-large-xlsr-53"


def ser_keys(hashes: tuple[str, ...], model: str | None = None) -> list[tuple[str, str]]:
    """SER. Payload: ``{"prediction": <ser dict>}``.

    ``wav2vec2_detailed_``          -> ``/inferences/wav2vec2-batch``
    ``wav2vec2_detailed_attention_v3_`` -> ``/inferences/wav2vec2-detailed``

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


def deepfake_keys(model: str, hashes: tuple[str, ...]) -> list[tuple[str, str]]:
    """ADD. Payload: ``{"prediction": <add dict>}``.

    ``model`` MUST be the deepfake checkpoint that produced the result (e.g.
    ``melody-machine``), never the ASR model the user happens to have selected.
    This family shares the transcript family's key shape because
    ``/inferences/run`` serves every model from one route - so passing a Whisper
    id here writes an ADD dict straight over that model's transcript, and the
    transcript consumers then read a dict and fail on ``.lower()``.
    """
    keys: list[tuple[str, str]] = []
    for h in hashes:
        keys.append((model, f"{CACHE_SCHEMA_VERSION}_{model}_{h}"))
        keys.append((model, f"{model}_{h}"))
    return keys


# Bumped when the acoustic payload gains or loses a field. LIT-248 added
# `spectrogram` without one, so every entry cached before it kept being served
# without a spectrogram for the whole 24 h TTL - a correct computation the UI
# could never see (FR4.1: a shape change must not be serveable under an old key).
ACOUSTIC_SCHEMA_VERSION = "v3"


def add_timeline_keys(model: str, hashes: tuple[str, ...]) -> list[tuple[str, str]]:
    """Windowed ADD confidence timeline (FR7.2). Payload: ``{"timeline": [...]}``.

    Its own family, not the clip-level one: the two carry different shapes and
    sharing a key is how an ADD dict once landed on top of an ASR transcript.
    """
    return [(model, f"{model}_add_timeline_{CACHE_SCHEMA_VERSION}_{h}") for h in hashes]


def acoustic_keys(hashes: tuple[str, ...]) -> list[tuple[str, str]]:
    """Acoustic profile. Payload: the profile dict itself, unwrapped."""
    return [
        ("acoustic", f"acoustic_profile_{ACOUSTIC_SCHEMA_VERSION}_{h}") for h in hashes
    ]


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
