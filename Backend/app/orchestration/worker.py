"""
Per-family RQ worker entrypoint (LIT-127, FR3).

Run one worker process per model family so each only ever holds its own model
in memory (SAD §6.1):

    python -m app.orchestration.worker asr
    python -m app.orchestration.worker mutation

Deployment runs ``concurrency`` copies of each (see QUEUE_CONFIGS) — GPU-bound
families are pinned to 1 to respect the VRAM budget (SAD C2), which
``run_worker`` enforces with a per-family Redis lock.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from .task_orchestrator import WorkerFamily, run_worker


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        families = ", ".join(f.value for f in WorkerFamily)
        print(f"usage: python -m app.orchestration.worker <family>\nfamilies: {families}", file=sys.stderr)
        return 2

    try:
        family = WorkerFamily(argv[0])
    except ValueError:
        families = ", ".join(f.value for f in WorkerFamily)
        print(f"unknown family {argv[0]!r}; choose one of: {families}", file=sys.stderr)
        return 2

    run_worker(family)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
