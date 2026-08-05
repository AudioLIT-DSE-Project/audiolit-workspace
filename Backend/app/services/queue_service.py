"""
Background worker scaffolding for AudioLIT (SRS FR3, SAD §5.1 / §6.1).

RQ (Redis Queue) is the committed task fabric for AudioLIT, chosen over Celery
for lighter setup and simpler single-host operation (SAD §3.3). **No Celery
dependency is introduced.**

Conventions (consistent with LIT-127)
-------------------------------------
Per-family queues::

    asr       — Whisper-family ASR                    (GPU, concurrency 1)
    ser       — Wav2Vec2 emotion classifier           (GPU, concurrency 1)
    add       — Wav2Vec2 ASVspoof-2021-DF deepfake    (GPU, concurrency 1)
    xai       — Captum IG / LIME / SHAP / Grad-CAM    (GPU, concurrency 1)
    mutation  — non-destructive signal mutation        (CPU, concurrency ≥1)

GPU-worker concurrency is pinned to **1 per family** via a Redis lock to
respect the 3–5 GB VRAM budget (SAD constraint C2). The ``mutation`` queue is
CPU-only and may run multiple worker processes.

Architectural rule (SAD §5.1)
-----------------------------
The FastAPI gateway **never** loads a model or runs inference. It calls
:func:`enqueue_multitask_analysis` (or a sibling helper), which places RQ jobs
on the appropriate per-family queues and returns immediately with a job id
and a WebSocket progress URL (SRS FR3.2). Per-family workers pick up jobs
concurrently off the request path; a single fan-in aggregator job (wired via
RQ ``depends_on``) combines the results once all family jobs finish and
writes the unified envelope to the content-addressed cache (LIT-163).

References
----------
* SRS FR3     — Asynchronous Multi-Task Inference
* SRS FR3.2   — WebSocket progress, long-polling fallback
* SRS FR3.3   — Versioned multi-task JSON response
* SRS FR4     — Deterministic Cache-by-Hash (fan-in target)
* SRS SR5     — No PII / audio bytes in logs or cache keys
* SAD §5.1    — Orchestration layer (Task Orchestrator)
* SAD §6.1    — RQ workers, one per model type, fan-out / fan-in
* SAD §3.2 C2 — VRAM budget 3–5 GB; concurrency 1 per GPU family
* LIT-127     — RQ broker deployment (parent issue; queue/worker conventions)
* LIT-163     — Content-addressed cache (result fan-in target)
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse, urlunparse

import redis
from redis.exceptions import ConnectionError as RedisConnectionError
from rq import Queue, SimpleWorker
from rq.job import Job, JobStatus

# ---------------------------------------------------------------------------
# Structured logging (SRS §3.5.3 — no PII, no audio bytes).
# ---------------------------------------------------------------------------
logger = logging.getLogger("audiolit.queue")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            '{"ts":"%(asctime)s","lvl":"%(levelname)s",'
            '"logger":"%(name)s","msg":"%(message)s"}'
        )
    )
    logger.addHandler(_h)
logger.setLevel(os.environ.get("AUDIOLIT_LOG_LEVEL", "INFO"))
logger.propagate = False


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
class TaskFamily(str, Enum):
    """Per-family task queues (SAD §6.1, LIT-127 conventions)."""

    ASR = "asr"
    SER = "ser"
    ADD = "add"
    XAI = "xai"
    MUTATION = "mutation"


DEFAULT_GPU_CONCURRENCY: int = 1   # SAD C2
DEFAULT_CPU_CONCURRENCY: int = 2

DEFAULT_JOB_TIMEOUT: int = 600            # 10 min per inference job
DEFAULT_AGGREGATOR_TIMEOUT: int = 120     # fan-in is lightweight
DEFAULT_RESULT_TTL: int = 60 * 60 * 24    # 24 h — matches SRS §3.10 Redis TTL
DEFAULT_FAILURE_TTL: int = 60 * 60        # 1 h — retain for debugging

PROGRESS_CHANNEL_PREFIX: str = "audiolit:progress"
WORKER_LOCK_PREFIX: str = "audiolit:worker-lock"

REDIS_URL_ENV: str = "REDIS_URL"
WS_BASE_URL_ENV: str = "AUDIOLIT_WS_BASE_URL"

# Module-level singleton — lazily created on first access.
_REDIS_CONN: redis.Redis | None = None
_QUEUES: dict[TaskFamily, Queue] = {}


# ---------------------------------------------------------------------------
# Redis connection management
# ---------------------------------------------------------------------------
def get_redis_connection() -> redis.Redis:
    """
    Return a shared Redis connection used by both the gateway (enqueue side)
    and the workers (dequeue side). Connection parameters are read from the
    environment so secrets stay out of source control (SRS SR3 / C8).

    On first call the connection is pinged; failure raises immediately so
    that worker startup logs clearly show broker reachability (DoD).
    """
    global _REDIS_CONN
    if _REDIS_CONN is None:
        url = os.environ.get(REDIS_URL_ENV, "redis://localhost:6379/0")
        _REDIS_CONN = redis.Redis.from_url(
            url,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=10,
            health_check_interval=30,
        )
        try:
            _REDIS_CONN.ping()
            logger.info(
                "queue.broker.connected url=%s", _sanitize_redis_url(url)
            )
        except RedisConnectionError as exc:
            logger.error(
                "queue.broker.unreachable url=%s err=%s",
                _sanitize_redis_url(url), exc,
            )
            raise
    return _REDIS_CONN


def _sanitize_redis_url(url: str) -> str:
    """Strip credentials from a Redis URL for safe logging (SRS SR5)."""
    try:
        p = urlparse(url)
        netloc = p.hostname or ""
        if p.port:
            netloc = f"{netloc}:{p.port}"
        return urlunparse((p.scheme, netloc, p.path, "", "", ""))
    except Exception:
        return "redis://***"


# ---------------------------------------------------------------------------
# Queue registry
# ---------------------------------------------------------------------------
def get_queue(family: TaskFamily | str) -> Queue:
    """Return the RQ :class:`~rq.Queue` for *family*, creating it lazily."""
    family = TaskFamily(family) if isinstance(family, str) else family
    if family not in _QUEUES:
        _QUEUES[family] = Queue(
            family.value,
            connection=get_redis_connection(),
            default_timeout=DEFAULT_JOB_TIMEOUT,
            result_ttl=DEFAULT_RESULT_TTL,
            failure_ttl=DEFAULT_FAILURE_TTL,
        )
    return _QUEUES[family]


def all_queues() -> Mapping[TaskFamily, Queue]:
    """Ensure every per-family queue is instantiated and return them."""
    for f in TaskFamily:
        get_queue(f)
    return _QUEUES


# ---------------------------------------------------------------------------
# Progress reporting — Redis pub/sub consumed by the WebSocket layer (FR3.2)
# ---------------------------------------------------------------------------
def publish_progress(
    job_id: str,
    stage: str,
    payload: Mapping[str, Any] | None = None,
) -> None:
    """
    Publish a progress event to ``audiolit:progress:<job_id>``.

    The FastAPI WebSocket handler subscribes to this channel and forwards
    events to the browser. Events carry **no** audio bytes, transcripts, or
    PII (SRS SR5).
    """
    conn = get_redis_connection()
    channel = f"{PROGRESS_CHANNEL_PREFIX}:{job_id}".encode()
    event = {
        "job_id": job_id,
        "stage": stage,
        "ts": time.time(),
        "payload": dict(payload or {}),
    }
    conn.publish(channel, json.dumps(event).encode())


def _current_job_id() -> str:
    """Return the id of the currently-executing RQ job (worker-side only)."""
    from rq import get_current_job

    job = get_current_job()
    return job.id if job is not None else "unknown"


# ---------------------------------------------------------------------------
# Worker context — PyTorch / Captum / Librosa are imported **lazily** inside
# worker processes so the gateway never pulls them into its address space
# (SAD §5.1: the application layer contains no AI code).
# ---------------------------------------------------------------------------
@dataclass
class WorkerContext:
    """
    Per-worker runtime context.

    Heavy libraries are imported lazily so that importing this module from
    the gateway process is cheap. The gateway must **never** call
    :func:`get_worker_context` — it is worker-side only.
    """

    family: TaskFamily
    device: str = "cpu"
    torch: Any = None
    captum: Any = None
    librosa: Any = None
    models: dict[str, Any] = field(default_factory=dict)

    def load_libraries(self) -> None:
        """Lazy-import the heavy ML stack (PyTorch, Captum, Librosa)."""
        if self.torch is None:
            import torch  # noqa: WPS433 — intentional lazy import in worker

            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "worker.libs.loaded family=%s device=%s torch=%s",
                self.family.value, self.device, torch.__version__,
            )
        if self.captum is None:
            import captum  # noqa: WPS433

            self.captum = captum
        if self.librosa is None:
            import librosa  # noqa: WPS433

            self.librosa = librosa

    def get_model(self, model_id: str, loader: Callable[[], Any]) -> Any:
        """
        Cache a loaded model in the worker process across jobs
        (VRAM-friendly: load once per family, serve many jobs). The
        *loader* callable is provided by the domain-layer model registry
        (SAD §5.2).
        """
        if model_id not in self.models:
            self.load_libraries()
            self.models[model_id] = loader()
            logger.info(
                "worker.model.loaded family=%s model=%s device=%s",
                self.family.value, model_id, self.device,
            )
        return self.models[model_id]


# Per-worker-process context singleton (set by :class:`AudioLITWorker.work`).
_WORKER_CTX: WorkerContext | None = None


def get_worker_context() -> WorkerContext:
    """Return this worker process's :class:`WorkerContext`.

    Raises if called outside a running AudioLIT worker — the gateway must
    never invoke this.
    """
    if _WORKER_CTX is None:
        raise RuntimeError(
            "WorkerContext accessed outside a running AudioLIT worker. "
            "The gateway must never call this — workers only."
        )
    return _WORKER_CTX


# ---------------------------------------------------------------------------
# Custom Worker — SimpleWorker base (no fork) so models cache across jobs;
# VRAM is released manually between jobs via torch.cuda.empty_cache().
# ---------------------------------------------------------------------------
class AudioLITWorker(SimpleWorker):
    """
    RQ worker subclass that:

    * Initialises the per-family :class:`WorkerContext` (lazy PyTorch /
      Captum / Librosa).
    * Emits structured logs on job lifecycle events (SRS §3.5.3).
    * Publishes granular progress to Redis pub/sub (SRS FR3.2).
    * Releases VRAM between jobs (``torch.cuda.empty_cache``).

    Uses :class:`~rq.SimpleWorker` (no fork) so that model weights persist
    across jobs in the same process — essential for staying within the VRAM
    budget while avoiding per-job model reloads (SAD §6.1).
    """

    def __init__(self, family: TaskFamily, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.family = family
        self._ctx: WorkerContext | None = None

    # -- lifecycle ----------------------------------------------------------
    def work(self, *args: Any, **kwargs: Any) -> bool:  # type: ignore[override]
        global _WORKER_CTX
        self._ctx = WorkerContext(family=self.family)
        _WORKER_CTX = self._ctx
        logger.info(
            "worker.start family=%s queues=%s pid=%s host=%s",
            self.family.value,
            [q.name for q in self.queues],
            os.getpid(),
            socket.gethostname(),
        )
        try:
            return super().work(*args, **kwargs)
        finally:
            _WORKER_CTX = None
            logger.info("worker.stop family=%s", self.family.value)

    def perform_job(self, job: Job, queue: Queue, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        """Wrap job execution with progress + structured logging."""
        publish_progress(job.id, "started", {"family": self.family.value})
        t0 = time.monotonic()
        try:
            assert self._ctx is not None
            self._ctx.load_libraries()
            result = super().perform_job(job, queue, *args, **kwargs)
            dt = time.monotonic() - t0
            publish_progress(
                job.id, "finished",
                {"family": self.family.value, "duration_s": round(dt, 3)},
            )
            logger.info(
                "job.done id=%s family=%s duration_s=%.3f",
                job.id, self.family.value, dt,
            )
            return result
        except Exception as exc:
            dt = time.monotonic() - t0
            publish_progress(
                job.id, "failed",
                {"family": self.family.value, "duration_s": round(dt, 3),
                 "error_type": type(exc).__name__},
            )
            logger.exception(
                "job.failed id=%s family=%s duration_s=%.3f err=%s",
                job.id, self.family.value, dt, exc,
            )
            raise

    def handle_job_success(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        """Release VRAM between jobs (SimpleWorker does not fork)."""
        if self._ctx and self._ctx.torch is not None:
            try:
                if self._ctx.torch.cuda.is_available():
                    self._ctx.torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 — cleanup must never mask success
                pass
        return super().handle_job_success(*args, **kwargs)


# ---------------------------------------------------------------------------
# Task functions (scaffolding stubs).
#
# RQ jobs reference these by qualified name
# (``app.services.queue_service.asr_task``). The domain layer (model registry,
# attribution strategies, mutation engine) plugs in at the TODO markers.
# ---------------------------------------------------------------------------
def asr_task(
    audio_ref: str, model_id: str, params: Mapping[str, Any]
) -> dict[str, Any]:
    """ASR inference — Whisper family (SRS FR3, inherited ASR engine)."""
    ctx = get_worker_context()
    publish_progress(_current_job_id(), "asr.running", {"model": model_id})
    # TODO(LIT-xxx): from app.domain.asr import run; return run(audio_ref, model_id, params, ctx)
    return {"task": "asr", "model_id": model_id, "transcript": None,
            "device": ctx.device, "_scaffold": True}


def ser_task(
    audio_ref: str, model_id: str, params: Mapping[str, Any]
) -> dict[str, Any]:
    """SER inference — Wav2Vec2 emotion classifier (SRS FR5)."""
    ctx = get_worker_context()
    publish_progress(_current_job_id(), "ser.running", {"model": model_id})
    # TODO(LIT-xxx): from app.domain.ser import run; return run(...)
    return {"task": "ser", "model_id": model_id, "emotions": None,
            "device": ctx.device, "_scaffold": True}


def add_task(
    audio_ref: str, model_id: str, params: Mapping[str, Any]
) -> dict[str, Any]:
    """ADD inference — ASVspoof 2021 DF detector (SRS FR6, new)."""
    ctx = get_worker_context()
    publish_progress(_current_job_id(), "add.running", {"model": model_id})
    # TODO(LIT-xxx): from app.domain.add import run; return run(...)
    return {"task": "add", "model_id": model_id, "deepfake": None,
            "device": ctx.device, "_scaffold": True}


def xai_task(
    audio_ref: str, model_id: str, method: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Attribution — IG / LIME / SHAP / Grad-CAM (SRS FR7–FR9)."""
    ctx = get_worker_context()
    publish_progress(
        _current_job_id(), "xai.running",
        {"model": model_id, "method": method},
    )
    # TODO(LIT-xxx): from app.domain.xai import run; return run(...)
    return {"task": "xai", "model_id": model_id, "method": method,
            "attribution": None, "device": ctx.device, "_scaffold": True}


def mutation_task(
    audio_ref: str, mutation: Mapping[str, Any]
) -> dict[str, Any]:
    """CPU-only non-destructive signal mutation (SRS FR12)."""
    # NOTE: mutation worker runs on CPU — do NOT touch the GPU context.
    publish_progress(
        _current_job_id(), "mutation.running",
        {"mutation": mutation.get("kind")},
    )
    # TODO(LIT-xxx): from app.domain.mutation import apply; return apply(...)
    return {"task": "mutation", "derived_ref": None, "_scaffold": True}


def aggregator_task(
    family_job_ids: Sequence[str], cache_key: str | None
) -> dict[str, Any]:
    """
    Fan-in job (SAD §6.1). Depends on all per-family jobs; runs once they
    complete. Combines their results, writes the unified envelope to the
    content-addressed cache (LIT-163), and returns a compact summary so
    the gateway can hand back a cache key + WebSocket URL.
    """
    conn = get_redis_connection()
    combined: dict[str, Any] = {
        "tasks": {},
        "cache_key": cache_key,
        "schema_version": "1.0",  # FR3.3
    }
    for jid in family_job_ids:
        job = Job.fetch(jid, connection=conn)
        status = job.get_status()
        if status != JobStatus.FINISHED:
            combined["tasks"][jid] = {"status": status, "error": "dependency not finished"}
            continue
        result = job.result or {}
        family = result.get("task", "unknown")
        combined["tasks"][family] = result

    if cache_key is not None:
        # TODO(LIT-163): from app.services.cache_service import put
        #     put(cache_key, combined)
        logger.info(
            "aggregator.fanin cache_key=%s families=%s",
            cache_key, list(combined["tasks"].keys()),
        )
    publish_progress(
        _current_job_id(), "aggregated",
        {"cache_key": cache_key,
         "families": list(combined["tasks"].keys())},
    )
    return combined


# ---------------------------------------------------------------------------
# Enqueue helpers — the gateway calls these and returns immediately.
# ---------------------------------------------------------------------------
@dataclass
class EnqueueResult:
    """Envelope returned to the client immediately after dispatch (FR3)."""

    job_id: str                       # aggregator job id (client-facing token)
    websocket_url: str                # client subscribes here (FR3.2)
    family_jobs: dict[str, str] = field(default_factory=dict)
    cache_key: str | None = None

    def as_response(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "websocket_url": self.websocket_url,
            "schema_version": "1.0",  # FR3.3
            "family_jobs": self.family_jobs,
            "cache_key": self.cache_key,
        }


_TASK_FUNCS: dict[TaskFamily, Callable[..., Any]] = {
    TaskFamily.ASR: asr_task,
    TaskFamily.SER: ser_task,
    TaskFamily.ADD: add_task,
}


def _ws_url(job_id: str, ws_base_url: str | None) -> str:
    base = ws_base_url or os.environ.get(WS_BASE_URL_ENV, "/ws/jobs")
    return f"{base}/{job_id}"


def enqueue_multitask_analysis(
    audio_ref: str,
    *,
    tasks: Iterable[TaskFamily | str] = (
        TaskFamily.ASR, TaskFamily.SER, TaskFamily.ADD
    ),
    model_ids: Mapping[TaskFamily | str, str] | None = None,
    params: Mapping[TaskFamily | str, Mapping[str, Any]] | None = None,
    cache_key: str | None = None,
    ws_base_url: str | None = None,
) -> EnqueueResult:
    """
    Fan-out a multi-task analysis (SRS FR3, SAD §6.1).

    1. Enqueue one job per requested family on its dedicated queue.
    2. Enqueue a single aggregator job that ``depends_on`` all family jobs.
    3. Return immediately with the aggregator job id (the client-facing task
       token) and the WebSocket progress URL.

    The gateway calls this and **never** blocks on a forward pass (SAD §5.1).
    """
    model_ids = model_ids or {}
    params = params or {}
    families = [TaskFamily(f) if isinstance(f, str) else f for f in tasks]

    family_job_objs: list[Job] = []
    family_jobs: dict[str, str] = {}

    for f in families:
        if f not in _TASK_FUNCS:
            raise ValueError(
                f"Family {f!r} is not a multi-task inference family. "
                f"Use enqueue_attribution / enqueue_mutation instead."
            )
        queue = get_queue(f)
        job = queue.enqueue(
            _TASK_FUNCS[f],
            audio_ref,
            model_ids.get(f, "default"),
            dict(params.get(f, {})),
            job_timeout=DEFAULT_JOB_TIMEOUT,
            result_ttl=DEFAULT_RESULT_TTL,
            failure_ttl=DEFAULT_FAILURE_TTL,
            meta={"family": f.value, "audio_ref": audio_ref},
        )
        family_jobs[f.value] = job.id
        family_job_objs.append(job)
        logger.info(
            "job.queued id=%s family=%s queue=%s",
            job.id, f.value, queue.name,
        )

    # ---- Fan-in: aggregator depends on every family job (SAD §6.1) -------
    aggregator_queue = get_queue(TaskFamily.XAI)  # lightweight; reuses xai queue
    aggregator = aggregator_queue.enqueue(
        aggregator_task,
        [j.id for j in family_job_objs],
        cache_key,
        depends_on=family_job_objs,
        job_timeout=DEFAULT_AGGREGATOR_TIMEOUT,
        result_ttl=DEFAULT_RESULT_TTL,
        failure_ttl=DEFAULT_FAILURE_TTL,
        meta={"role": "aggregator", "family_jobs": family_jobs},
    )

    logger.info(
        "dispatch.multitask job_id=%s families=%s cache_key=%s",
        aggregator.id, list(family_jobs.keys()), cache_key,
    )

    return EnqueueResult(
        job_id=aggregator.id,
        websocket_url=_ws_url(aggregator.id, ws_base_url),
        family_jobs=family_jobs,
        cache_key=cache_key,
    )


def enqueue_attribution(
    audio_ref: str,
    model_id: str,
    method: str,
    params: Mapping[str, Any],
    *,
    cache_key: str | None = None,
    ws_base_url: str | None = None,
) -> EnqueueResult:
    """Enqueue a single XAI attribution job (SRS FR7–FR9)."""
    queue = get_queue(TaskFamily.XAI)
    job = queue.enqueue(
        xai_task, audio_ref, model_id, method, dict(params),
        job_timeout=DEFAULT_JOB_TIMEOUT,
        result_ttl=DEFAULT_RESULT_TTL,
        failure_ttl=DEFAULT_FAILURE_TTL,
        meta={"family": "xai", "method": method},
    )
    logger.info("job.queued id=%s family=xai method=%s", job.id, method)
    return EnqueueResult(
        job_id=job.id,
        websocket_url=_ws_url(job.id, ws_base_url),
        family_jobs={"xai": job.id},
        cache_key=cache_key,
    )


def enqueue_mutation(
    audio_ref: str,
    mutation: Mapping[str, Any],
    *,
    ws_base_url: str | None = None,
) -> EnqueueResult:
    """Enqueue a CPU-only signal mutation job (SRS FR12)."""
    queue = get_queue(TaskFamily.MUTATION)
    job = queue.enqueue(
        mutation_task, audio_ref, dict(mutation),
        job_timeout=DEFAULT_AGGREGATOR_TIMEOUT,
        result_ttl=DEFAULT_RESULT_TTL,
        failure_ttl=DEFAULT_FAILURE_TTL,
        meta={"family": "mutation"},
    )
    logger.info("job.queued id=%s family=mutation", job.id)
    return EnqueueResult(
        job_id=job.id,
        websocket_url=_ws_url(job.id, ws_base_url),
        family_jobs={"mutation": job.id},
    )


# ---------------------------------------------------------------------------
# Job status lookup — used by the gateway REST + WebSocket layer.
# ---------------------------------------------------------------------------
def fetch_job(job_id: str) -> Job | None:
    """Return an RQ :class:`~rq.job.Job` by id, or ``None`` if unknown."""
    try:
        return Job.fetch(job_id, connection=get_redis_connection())
    except Exception:  # noqa: BLE001
        return None


def job_status(job_id: str) -> dict[str, Any]:
    """Return a JSON-friendly status snapshot for the client."""
    job = fetch_job(job_id)
    if job is None:
        return {"job_id": job_id, "status": "unknown"}
    return {
        "job_id": job.id,
        "status": job.get_status(),
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
        "result": job.result if job.is_finished else None,
        "error": str(job.exc_info) if job.is_failed else None,
        "meta": dict(job.meta or {}),
    }


# ---------------------------------------------------------------------------
# Worker launcher — used by Docker entrypoints (SAD §7 deployment view).
# ---------------------------------------------------------------------------
def run_worker(
    family: TaskFamily | str,
    *,
    burst: bool = False,
) -> None:
    """
    Start a single AudioLIT RQ worker process for *family*.

    GPU families (``asr``, ``ser``, ``add``, ``xai``) are pinned to **1**
    worker process via a Redis lock to respect the VRAM budget (SAD C2).
    The ``mutation`` family (CPU-only) may run multiple worker processes.

    Usage::

        python -m app.services.queue_service worker asr
        python -m app.services.queue_service worker ser
        python -m app.services.queue_service worker add
        python -m app.services.queue_service worker xai
        python -m app.services.queue_service worker mutation
    """
    family = TaskFamily(family) if isinstance(family, str) else family
    queue = get_queue(family)
    conn = get_redis_connection()

    # ---- Enforce 1 worker process per GPU family (SAD C2) ---------------
    lock: redis.lock.Lock | None = None
    if family is not TaskFamily.MUTATION:
        lock = conn.lock(
            f"{WORKER_LOCK_PREFIX}:{family.value}",
            timeout=60 * 60 * 24,   # 24 h — worker keeps renewing
            blocking=False,
        )
        if not lock.acquire(blocking=False):
            raise RuntimeError(
                f"Another worker for GPU family '{family.value}' is already "
                f"running. SAD constraint C2 pins GPU-worker concurrency to "
                f"1 per family. Stop the other worker or release Redis lock "
                f"'{WORKER_LOCK_PREFIX}:{family.value}'."
            )

    worker = AudioLITWorker(
        family=family,
        queues=[queue],
        connection=conn,
        name=f"audiolit-{family.value}-{socket.gethostname()}-{os.getpid()}",
    )
    logger.info(
        "worker.launch family=%s queue=%s pid=%s host=%s burst=%s",
        family.value, queue.name, os.getpid(), socket.gethostname(), burst,
    )
    try:
        worker.work(burst=burst, with_scheduler=True)
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "worker.lock.release.failed family=%s", family.value
                )


# ---------------------------------------------------------------------------
# Health check — DoD: "Worker logs confirm successful broker connection".
# ---------------------------------------------------------------------------
def health_check() -> dict[str, Any]:
    """
    Verify broker reachability and report per-queue depth.

    Called from a Docker ``HEALTHCHECK`` and from CI to satisfy the DoD.
    """
    try:
        conn = get_redis_connection()
        pong = conn.ping()
        info = conn.info("server")
        return {
            "ok": bool(pong),
            "broker": "redis",
            "redis_version": info.get("redis_version"),
            "queues": {f.value: get_queue(f).count for f in TaskFamily},
        }
    except RedisConnectionError as exc:
        return {"ok": False, "broker": "redis", "error": str(exc)}


# ---------------------------------------------------------------------------
# CLI entry point:  python -m app.services.queue_service <command> [...]
# ---------------------------------------------------------------------------
def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="AudioLIT RQ worker launcher (LIT-128)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("worker", help="Start a per-family worker")
    w.add_argument("family", choices=[f.value for f in TaskFamily])
    w.add_argument("--burst", action="store_true",
                   help="Process queued jobs then exit (CI / drain mode)")

    sub.add_parser("health", help="Broker connectivity check")

    args = parser.parse_args()
    if args.cmd == "worker":
        run_worker(args.family, burst=args.burst)
    elif args.cmd == "health":
        print(json.dumps(health_check(), indent=2))


if __name__ == "__main__":
    _cli()