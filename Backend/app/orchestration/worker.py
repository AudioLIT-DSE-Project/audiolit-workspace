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
import logging

from .task_orchestrator import (
    WORKER_LOCK_PREFIX,
    AudioLITWorker,
    WorkerFamily,
    get_queue,
    run_worker,
)

logger = logging.getLogger("audiolit.orchestration.worker")


def _start_family_worker(fam: WorkerFamily):
    """Worker process entrypoint for a single family."""
    run_worker(fam)


def _cleanup_stale_worker_locks(families: List[WorkerFamily]) -> None:
    """Clear Redis worker locks for families with no *live* worker listening.

    Presence in RQ's registry is not liveness. A worker that dies abnormally -
    killed, or quitting on a broker error - leaves its registry key behind until
    ``worker_ttl`` expires, 420 s on RQ 2.10's defaults. Treating that entry as
    an active worker meant the lock was left in place, and since the lock itself
    is held for 24 h, the next start failed with "GPU family <x> already has a
    worker running" for up to seven minutes after a crash. Reproduced here:
    workers quit on a broker read timeout, and an immediate restart brought up
    only `mutation` - the one non-GPU family, which takes no lock.

    A worker heartbeats every ``job_monitoring_interval``; treating one as live
    only while its heartbeat is recent distinguishes a running worker from a
    corpse that has not expired yet. The grace factor covers a worker that is
    mid-job and slightly late to beat.
    """
    try:
        from datetime import datetime, timezone

        from rq import Worker

        conn = get_redis_connection()
        active_queues = set()
        now = datetime.now(timezone.utc)

        for w in Worker.all(connection=conn):
            heartbeat = getattr(w, "last_heartbeat", None)
            if heartbeat is not None:
                if heartbeat.tzinfo is None:
                    heartbeat = heartbeat.replace(tzinfo=timezone.utc)
                # Two beats of slack before declaring a worker dead.
                grace = max(getattr(w, "worker_ttl", 420), 60)
                if (now - heartbeat).total_seconds() > grace:
                    continue  # stale registry entry, not a live worker
            for q_name in w.queue_names():
                active_queues.add(q_name.replace("audiolit:", ""))

        for fam in families:
            if fam.value not in active_queues:
                # Build the key from the same constant run_worker locks with.
                # This was spelled "audiolit:worker_lock:" here while the lock
                # itself is "audiolit:worker-lock:" - underscore against hyphen -
                # so the purge deleted a key that never existed and this function
                # had never once cleared a lock, despite the README promising it
                # "automatically purges stale Redis locks upon startup". The
                # visible symptom is a worker that crashed being unable to
                # restart: "GPU family <x> already has a worker running", with
                # only `mutation` (the one family that takes no lock) coming up.
                lock_key = f"{WORKER_LOCK_PREFIX}:{fam.value}"
                if conn.exists(lock_key):
                    logger.info("worker.lock.purged family=%s", fam.value)
                    conn.delete(lock_key)
    except Exception:
        pass


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

    _cleanup_stale_worker_locks(selected_families)

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
