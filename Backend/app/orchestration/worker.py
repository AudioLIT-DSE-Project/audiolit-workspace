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
import multiprocessing
from typing import List, Optional

from ..infrastructure.rq_connection import get_redis_connection
from .task_orchestrator import AudioLITWorker, WorkerFamily, get_queue, run_worker


def _start_family_worker(fam: WorkerFamily):
    """Worker process entrypoint for a single family."""
    run_worker(fam)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        families = ", ".join(f.value for f in WorkerFamily)
        print(f"usage: python -m app.orchestration.worker <family> [family...]\n       python -m app.orchestration.worker all\nfamilies: {families}", file=sys.stderr)
        return 2

    # Support 'all' keyword
    if len(argv) == 1 and argv[0].lower() == "all":
        selected_families = list(WorkerFamily)
    else:
        selected_families = []
        for arg in argv:
            try:
                selected_families.append(WorkerFamily(arg.lower()))
            except ValueError:
                families = ", ".join(f.value for f in WorkerFamily)
                print(f"unknown family {arg!r}; choose from: {families}, all", file=sys.stderr)
                return 2

    if len(selected_families) == 1:
        run_worker(selected_families[0])
    else:
        # Spawn parallel worker processes for each selected family
        processes = []
        print(f"Starting {len(selected_families)} parallel worker processes for queues: {', '.join(f.value for f in selected_families)}")
        for fam in selected_families:
            p = multiprocessing.Process(target=_start_family_worker, args=(fam,))
            p.start()
            processes.append(p)

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("\nShutting down parallel worker processes...")
            for p in processes:
                p.terminate()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
