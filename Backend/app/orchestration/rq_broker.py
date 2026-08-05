"""
RQ task broker and per-family worker configuration (LIT-127, FR3).

Deploys the asynchronous task fabric: one RQ queue per model family, backed by
the existing Redis instance (SAD §5.1 orchestration layer; §6.1 "one worker for
each kind of model ... so each worker only ever needs to hold one model in
memory"). GPU-bound families are pinned to a concurrency of 1 to respect the
3-5 GB VRAM budget (SAD constraint C2). Workers run *synchronous* job functions
off the FastAPI request path — nothing is awaited inside a worker.

Scope: this is the broker foundation (queues, worker entrypoint, progress
pub/sub). Two follow-on steps are deliberately **not** here because they either
break the current HTTP contract or depend on work still in review:

  * Rewriting the inference route to enqueue and return a job id + WebSocket URL,
    plus the WebSocket progress relay — changes ``/upload``'s response contract,
    so it must land together with the frontend polling handlers (LIT-157).
  * Removing the old ``queue_service.py`` list-queue — it still backs
    ``session.py``'s ``/queue`` endpoints; it retires with the route rewrite.
  * Wiring the ASR+SER+ADD fan-out/fan-in orchestrator — that is LIT-150
    (in review), which reuses the LIT-225 dependency pattern.

Queue names here are the canonical per-family names; the orchestrator (LIT-150)
will enqueue its child jobs onto these once both land.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional

from redis import Redis
from rq import Queue, Worker

from ..infrastructure.rq_connection import get_redis_connection


class WorkerFamily(str, Enum):
    """One worker family per kind of model, plus a CPU-only mutation worker."""

    ASR = "asr"
    SER = "ser"
    ADD = "add"
    XAI = "xai"
    MUTATION = "mutation"


@dataclass(frozen=True)
class QueueConfig:
    """How one family's queue is provisioned.

    ``concurrency`` is the number of workers to run for the family;
    ``gpu_bound`` families are pinned to 1 so no two model jobs contend for VRAM
    at once (SAD C2). The mutation family is CPU-only and may scale out.
    """

    family: WorkerFamily
    queue_name: str
    gpu_bound: bool
    concurrency: int


# The five families of the fabric. GPU-bound families (the four model families)
# get concurrency 1; the CPU-only mutation worker may run a couple in parallel.
QUEUE_CONFIGS: dict[WorkerFamily, QueueConfig] = {
    WorkerFamily.ASR: QueueConfig(WorkerFamily.ASR, "asr", gpu_bound=True, concurrency=1),
    WorkerFamily.SER: QueueConfig(WorkerFamily.SER, "ser", gpu_bound=True, concurrency=1),
    WorkerFamily.ADD: QueueConfig(WorkerFamily.ADD, "add", gpu_bound=True, concurrency=1),
    WorkerFamily.XAI: QueueConfig(WorkerFamily.XAI, "xai", gpu_bound=True, concurrency=1),
    WorkerFamily.MUTATION: QueueConfig(
        WorkerFamily.MUTATION, "mutation", gpu_bound=False, concurrency=2
    ),
}

# Workers publish progress here, keyed by job id; the gateway relays each
# message to the client's open WebSocket (relay itself is the LIT-157 step).
PROGRESS_CHANNEL_PREFIX = "progress:"


def get_queue_config(family: WorkerFamily | str) -> QueueConfig:
    """Resolve a family's queue config, accepting the enum or its string value."""
    fam = WorkerFamily(family) if not isinstance(family, WorkerFamily) else family
    return QUEUE_CONFIGS[fam]


def get_queue(family: WorkerFamily | str, connection: Optional[Redis] = None) -> Queue:
    """The RQ queue for a model family."""
    conn = connection or get_redis_connection()
    return Queue(get_queue_config(family).queue_name, connection=conn)


def enqueue(
    family: WorkerFamily | str,
    func: Callable[..., Any],
    *args: Any,
    connection: Optional[Redis] = None,
    **kwargs: Any,
):
    """Enqueue ``func`` onto a family's queue and return the RQ ``Job`` handle.

    Non-blocking: the caller (the API gateway) returns the job id immediately and
    never runs the work itself.
    """
    return get_queue(family, connection=connection).enqueue(func, *args, **kwargs)


def progress_channel(job_id: str) -> str:
    """Redis pub/sub channel a job publishes its progress on."""
    return f"{PROGRESS_CHANNEL_PREFIX}{job_id}"


def publish_progress(
    conn: Redis, job_id: str, task_name: str, fraction: float
) -> int:
    """Publish a progress fraction (0..1) for a job; returns subscriber count."""
    payload = json.dumps({"task_name": task_name, "progress": fraction})
    return conn.publish(progress_channel(job_id), payload)


def worker_queue_names(families: Optional[List[WorkerFamily]] = None) -> List[str]:
    """Queue names a worker for ``families`` (default: all) should listen on."""
    fams = families or list(WorkerFamily)
    return [QUEUE_CONFIGS[f].queue_name for f in fams]


def make_worker(
    family: WorkerFamily | str, connection: Optional[Redis] = None
) -> Worker:
    """Build an RQ ``Worker`` bound to one family's queue.

    RQ's forking ``Worker`` (not ``SimpleWorker``) runs each job in a monitored
    work-horse child process, which is what gives us crash recovery for a killed
    job. Call ``.work()`` to start it — done by the worker entrypoint, one
    process per family so each holds only its own model in memory.
    """
    conn = connection or get_redis_connection()
    return Worker([get_queue_config(family).queue_name], connection=conn)
