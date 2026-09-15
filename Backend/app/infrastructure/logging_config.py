"""Structured task logging (LIT-259).

One JSON object per line, shape ``{"ts","level","logger","event",...extra}``,
where ``extra`` is everything the emitting call passed as ``LogRecord`` extra
attributes (``job_id``, ``family``, ``queue``, ...). Stdlib-only: a
``logging.Formatter`` subclass, nothing from a structured-logging dependency.

``configure_logging`` is called once at API startup (``app/main.py``) and once
per worker process (``run_worker``). It honours ``LOG_FORMAT`` from the
environment via ``settings``; ``"json"`` (the default, so logs are queryable in
a cluster) is only relaxed to ``"text"`` for a local dev session.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from .settings import settings

logger = logging.getLogger("audiolit.infrastructure.logging_config")


#: The stdlib ``LogRecord`` attributes are structural, not message content; they
#: are already reflected in the fixed ``ts``/``level``/``logger``/``event``
#: fields and must not be re-dumped as ``extra``.
_RESERVED_ATTRIBUTES = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName", "asctime",
}


class JsonFormatter(logging.Formatter):
    """Render a ``LogRecord`` as a single JSON object.

    ``event`` is the record message with ``%``-args applied (so structured
    events log their bare name, e.g. ``task.success``, while human-readable
    messages keep their interpolated text). Exception and stack traces, when
    attached, are serialised under ``exc`` / ``stack`` rather than dropped.
    """

    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRIBUTES and not key.startswith("_"):
                event[key] = value
        if record.exc_info:
            event["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            event["stack"] = self.formatStack(record.stack_info)
        return json.dumps(event, default=str)


_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging() -> None:
    """Attach structured formatting to the ``audiolit`` logger tree.

    Idempotent and non-destructive: if a root handler already exists - uvicorn,
    pytest's caplog - nothing new is attached (so a record still echoes exactly
    once) and the existing handlers are simply reformatted. A bare worker
    process with no handler anywhere gets its own ``StreamHandler`` so JSON
    events still reach stderr.
    """
    use_json = settings.LOG_FORMAT.strip().lower() == "json"
    formatter = JsonFormatter() if use_json else logging.Formatter(_TEXT_FORMAT)

    root = logging.getLogger()
    audiolit = logging.getLogger("audiolit")

    if not root.handlers and not audiolit.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        audiolit.addHandler(handler)
        audiolit.setLevel(logging.INFO)
        # Propagate stays True: with no root handler attached there is nothing
        # above to double-print to, and caplog in the test suite relies on
        # records reaching the root logger.
        return

    for target in (audiolit, root):
        for handler in target.handlers:
            try:
                handler.setFormatter(formatter)
            except Exception:  # pragma: no cover - non-formatter handlers
                logger.debug("handler.format_failed handler=%s", handler, exc_info=True)