"""User-supplied class names for custom classification checkpoints.

Some Hugging Face checkpoints are published without ``id2label``, so
transformers fills in ``LABEL_0 .. LABEL_n`` and every prediction, saliency
target and accuracy comparison carries a placeholder. The real names are not in
the checkpoint; the only honest source is the person who chose the model, so
they are entered once in the Custom Model dialog and stored here.

A JSON file under ``HF_HOME`` rather than a Redis key: the names must outlive
the 24 h cache TTL (every non-RQ key has to carry one under ``volatile-lru``)
and be visible to the API process and every worker, which share that volume
and nothing else on disk that is not purged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path

from huggingface_hub import constants as hf_constants

logger = logging.getLogger("audiolit.infrastructure.model_labels")

#: What transformers generates when a checkpoint ships no ``id2label``.
_PLACEHOLDER = re.compile(r"^LABEL_\d+$")

MAX_LABEL_LENGTH = 40

_write_lock = threading.Lock()
# (mtime_ns, size) -> parsed file, so the per-inference lookup costs one stat().
_cached: tuple[tuple[int, int], dict[str, list[str]]] | None = None


def labels_path() -> Path:
    return Path(hf_constants.HF_HOME) / "audiolit" / "model_labels.json"


def are_placeholders(labels: list[str]) -> bool:
    """True when every name is a transformers-generated ``LABEL_<n>``."""
    return bool(labels) and all(_PLACEHOLDER.match(str(name)) for name in labels)


def normalise_labels(labels: list[str], expected_count: int) -> list[str]:
    """Validate user-entered class names; raises ``ValueError`` with the reason.

    Lower-cased because every other label in the system is (corpus ground
    truth, the default SER checkpoint), and the UI keys colours on them.
    """
    cleaned = [str(name).strip().lower() for name in labels]
    if len(cleaned) != expected_count:
        raise ValueError(f"Expected {expected_count} class names, got {len(cleaned)}.")
    if any(not name for name in cleaned):
        raise ValueError("Every class needs a name.")
    if any(len(name) > MAX_LABEL_LENGTH for name in cleaned):
        raise ValueError(f"Class names are limited to {MAX_LABEL_LENGTH} characters.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Class names must be unique.")
    return cleaned


def _read_all() -> dict[str, list[str]]:
    global _cached
    path = labels_path()
    try:
        st = path.stat()
    except OSError:
        return {}
    stamp = (st.st_mtime_ns, st.st_size)
    if _cached is not None and _cached[0] == stamp:
        return _cached[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        # Unreadable names degrade to the checkpoint's own; inference must not
        # depend on this file.
        logger.warning("model_labels.unreadable: %s", exc)
        return {}
    if not isinstance(data, dict):
        return {}
    _cached = (stamp, data)
    return data


def get_label_override(model_id: str | None) -> list[str] | None:
    """The stored class names for ``model_id`` in index order, or None."""
    if not model_id:
        return None
    labels = _read_all().get(model_id)
    return list(labels) if isinstance(labels, list) and labels else None


def set_label_override(model_id: str, labels: list[str] | None) -> None:
    """Store (or, with an empty value, remove) the class names for ``model_id``."""
    path = labels_path()
    with _write_lock:
        data = dict(_read_all())
        if labels:
            data[model_id] = list(labels)
        else:
            data.pop(model_id, None)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)


def label_fingerprint(model_id: str | None) -> str:
    """Cache-key suffix that changes with the stored names; "" when none are set.

    A result cached before the names were entered (or changed) holds the old
    ones, so it must not be reachable under the same key afterwards.
    """
    labels = get_label_override(model_id)
    if not labels:
        return ""
    return "_L" + hashlib.md5("\x1f".join(labels).encode()).hexdigest()[:8]
