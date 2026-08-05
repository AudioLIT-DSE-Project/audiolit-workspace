"""
RQ fan-out/fan-in prototype (LIT-225).

De-risks the async fabric ahead of LIT-127 (RQ broker) and LIT-150 (the real
ASR/SER/ADD orchestrator): RQ has no built-in chord/fan-in primitive, so
"dispatch N children, wait for all, aggregate" has to be composed from RQ's
own `depends_on` job-dependency mechanism. This module proves the pattern
with three trivial stub jobs. LIT-150 reuses the same pattern with real
inference jobs instead of stubs - see docs/rq_fanout_pattern.md.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from redis import Redis
from rq import Queue, Retry, get_current_job
from rq.job import Dependency, Job

from ..infrastructure.rq_connection import get_redis_connection

CHILD_QUEUE_NAMES = ("fanout_child_a", "fanout_child_b", "fanout_child_c")
AGGREGATOR_QUEUE_NAME = "fanout_aggregator"
PROGRESS_CHANNEL_PREFIX = "fanout:progress:"

# A child job gets one automatic retry. Combined with RQ's own forking
# Worker - which runs each job in a work-horse child process it monitors via
# waitpid() - this is what recovers a killed job with no extra code of ours:
# if the work-horse dies mid-job, the parent worker notices the moment it
# waits on the dead pid, marks the job failed, and (because Retry is set)
# immediately requeues it instead of losing it. See docs/rq_fanout_pattern.md.
CHILD_JOB_RETRY = Retry(max=1)


def _publish_progress(conn: Redis, job_id: str, task_name: str, fraction: float) -> None:
    channel = f"{PROGRESS_CHANNEL_PREFIX}{job_id}"
    conn.publish(channel, json.dumps({"task_name": task_name, "progress": fraction}))


def run_stub_child(
    task_name: str,
    payload: dict[str, Any],
    fail: bool = False,
    work_seconds: float = 0,
) -> dict[str, Any]:
    """
    Stand-in for a real ASR/SER/ADD task: publishes progress, simulates
    `work_seconds` of processing, then succeeds or raises. `work_seconds`
    gives a killed-worker test something to interrupt mid-job.
    """
    conn = get_redis_connection()
    job = get_current_job()
    if job is not None:
        # Which OS process handled this job - useful on its own for
        # debugging a stuck worker, and lets a killed-worker test target
        # the right pid without a special test hook.
        job.meta["pid"] = os.getpid()
        job.save_meta()
        _publish_progress(conn, job.id, task_name, 0.5)

    if work_seconds:
        time.sleep(work_seconds)

    if fail:
        raise RuntimeError(f"stub failure requested for task '{task_name}'")

    result = {"task_name": task_name, "payload": payload, "status": "ok"}
    if job is not None:
        _publish_progress(conn, job.id, task_name, 1.0)
    return result


def aggregate_children(child_job_ids: list[str]) -> dict[str, Any]:
    """
    Runs once, after every child either finished or exhausted its retries
    (see enqueue_fanout's `allow_failure=True` dependency). Never loses a
    succeeded sibling's result just because another child failed.
    """
    conn = get_redis_connection()
    succeeded: dict[str, Any] = {}
    failed: dict[str, str] = {}

    for job_id in child_job_ids:
        job = Job.fetch(job_id, connection=conn)
        if job.is_failed:
            outcome = job.latest_result()
            exc_string = outcome.exc_string if outcome else None
            last_line = exc_string.strip().splitlines()[-1] if exc_string else "unknown error"
            failed[job_id] = last_line
        else:
            succeeded[job_id] = job.return_value()

    return {"succeeded": succeeded, "failed": failed}


def enqueue_fanout(
    payloads: dict[str, dict[str, Any]],
    fail_tasks: set[str] | None = None,
) -> dict[str, Any]:
    """
    Enqueue one stub child per (queue, task_name/payload) pair, then an
    aggregator job depending on all of them. `allow_failure=True` on the
    dependency is the key detail: without it, RQ would never enqueue the
    aggregator at all once one child fails, silently losing the other
    children's results.
    """
    fail_tasks = fail_tasks or set()
    conn = get_redis_connection()

    child_jobs: list[Job] = []
    for queue_name, task_name in zip(CHILD_QUEUE_NAMES, payloads.keys()):
        queue = Queue(queue_name, connection=conn)
        job = queue.enqueue(
            run_stub_child,
            task_name=task_name,
            payload=payloads[task_name],
            fail=task_name in fail_tasks,
            retry=CHILD_JOB_RETRY,
        )
        child_jobs.append(job)

    aggregator_queue = Queue(AGGREGATOR_QUEUE_NAME, connection=conn)
    aggregator_job = aggregator_queue.enqueue(
        aggregate_children,
        child_job_ids=[j.id for j in child_jobs],
        depends_on=Dependency(jobs=child_jobs, allow_failure=True),
    )

    return {
        "child_job_ids": [j.id for j in child_jobs],
        "aggregator_job_id": aggregator_job.id,
    }
