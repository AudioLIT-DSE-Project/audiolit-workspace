"""Task Orchestrator - the single RQ task fabric (SAD §5.1 / §5.2, LIT-230).

SAD §5.2 lists **one** Task Orchestrator component: "Splits a multi-part request
into separate background jobs, runs them together, and combines the results."
This module is that component. It replaces two parallel implementations that
briefly coexisted on ``develop``:

  * ``app/orchestration/rq_broker.py`` (LIT-127) - queue config, GPU pinning,
    progress channel, worker factory. Correct layer, forking worker.
  * ``app/services/queue_service.py`` (LIT-149/157, PR #39) - worker context,
    per-family task functions, fan-in aggregator, enqueue API, job status.
    Correct worker type, wrong layer (``app/services/`` is ECHO 1.0 legacy that
    SAD §8.2 has us migrating out of; there is no "services" layer in §5.1).

Both landed because the Tier-C ``Path:`` stamps on LIT-127/149/150/225 still
named ``Backend/app/services/queue_service.py``, written before LIT-227 emptied
that directory. The stamps are corrected alongside this change.

**Worker type: SimpleWorker, not the forking Worker.** See ``make_worker``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from rq import Queue, SimpleWorker
from rq.job import Job, JobStatus

from ..infrastructure.rq_connection import get_redis_connection

logger = logging.getLogger("audiolit.orchestration")


class WorkerFamily(str, Enum):
    """One worker family per kind of model, plus a CPU-only mutation worker."""

    ASR = "asr"
    SER = "ser"
    ADD = "add"
    XAI = "xai"
    MUTATION = "mutation"


#: The routes and the multi-task enqueue API speak in "task families"; the
#: worker/queue side speaks in "worker families". They are the same five values.
TaskFamily = WorkerFamily


@dataclass(frozen=True)
class QueueConfig:
    """How one family's queue is provisioned.

    ``concurrency`` is the number of workers to run for the family; ``gpu_bound``
    families are pinned to 1 so no two model jobs contend for VRAM at once
    (SAD constraint C2). The mutation family is CPU-only and may scale out.
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

# Workers publish progress here, keyed by job id; the gateway relays each message
# to the client's open WebSocket (`api/routes/tasks.py`). One prefix, one message
# shape - the two merged modules disagreed ("progress:" vs "audiolit:progress"),
# so a job published by one was invisible to a subscriber on the other.
PROGRESS_CHANNEL_PREFIX = "audiolit:progress"
WORKER_LOCK_PREFIX = "audiolit:worker-lock"

DEFAULT_JOB_TIMEOUT: int = 600
DEFAULT_AGGREGATOR_TIMEOUT: int = 120
DEFAULT_RESULT_TTL: int = 60 * 60 * 24
DEFAULT_FAILURE_TTL: int = 60 * 60

WS_BASE_URL_ENV: str = "AUDIOLIT_WS_BASE_URL"
#: Must match the WebSocket route in `api/routes/tasks.py`.
DEFAULT_WS_BASE_URL: str = "/api/ws/tasks"

_QUEUES: dict[WorkerFamily, Queue] = {}


# --------------------------------------------------------------------------- #
# Queues
# --------------------------------------------------------------------------- #

def get_queue_config(family: WorkerFamily | str) -> QueueConfig:
    """Resolve a family's queue config, accepting the enum or its string value."""
    fam = WorkerFamily(family) if not isinstance(family, WorkerFamily) else family
    return QUEUE_CONFIGS[fam]


def get_queue(family: WorkerFamily | str, connection: Optional[Redis] = None) -> Queue:
    """The RQ queue for a model family.

    Cached per family when using the shared connection; an explicit ``connection``
    (tests, or a second broker) always builds a fresh queue.
    """
    config = get_queue_config(family)
    if connection is not None:
        return Queue(
            config.queue_name,
            connection=connection,
            default_timeout=DEFAULT_JOB_TIMEOUT,
            result_ttl=DEFAULT_RESULT_TTL,
            failure_ttl=DEFAULT_FAILURE_TTL,
        )
    if config.family not in _QUEUES:
        _QUEUES[config.family] = Queue(
            config.queue_name,
            connection=get_redis_connection(),
            default_timeout=DEFAULT_JOB_TIMEOUT,
            result_ttl=DEFAULT_RESULT_TTL,
            failure_ttl=DEFAULT_FAILURE_TTL,
        )
    return _QUEUES[config.family]


def all_queues() -> Mapping[WorkerFamily, Queue]:
    """Every family's queue, built on demand."""
    return {family: get_queue(family) for family in WorkerFamily}


def enqueue(
    family: WorkerFamily | str,
    func: Callable[..., Any],
    *args: Any,
    connection: Optional[Redis] = None,
    **kwargs: Any,
) -> Job:
    """Enqueue ``func`` onto a family's queue and return the RQ ``Job`` handle.

    Non-blocking: the caller (the API gateway) returns the job id immediately and
    never runs the work itself.
    """
    return get_queue(family, connection=connection).enqueue(func, *args, **kwargs)


def reset_queue_cache() -> None:
    """Drop cached queues. For tests that swap the broker connection."""
    _QUEUES.clear()


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #

def progress_channel(job_id: str) -> str:
    """Redis pub/sub channel a job publishes its progress on."""
    return f"{PROGRESS_CHANNEL_PREFIX}:{job_id}"


def publish_progress(
    job_id: str,
    stage: str,
    payload: Mapping[str, Any] | None = None,
    *,
    connection: Optional[Redis] = None,
) -> int:
    """Publish a progress event for a job; returns the subscriber count.

    ``stage`` carries the state the frontend switches on - QUEUED, PROCESSING,
    RETRYING, SUCCESS, FAILURE (SRS FR3.2) - or a finer-grained label such as
    ``"asr.running"``.
    """
    conn = connection or get_redis_connection()
    event = {
        "job_id": job_id,
        "stage": stage,
        "ts": time.time(),
        "payload": dict(payload or {}),
    }
    return conn.publish(progress_channel(job_id), json.dumps(event).encode())


def _current_job_id() -> str:
    from rq import get_current_job

    job = get_current_job()
    return job.id if job is not None else "unknown"


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

@dataclass
class WorkerContext:
    """Per-worker process state: heavy libraries and loaded models.

    Holding the model here is what SAD §10's "under about 8 seconds" budget for a
    fresh multi-task analysis depends on - reloading a model per job would spend
    most of that budget on loading (§10 separately allows ~60 s just to "download
    and prepare a new model"). SAD §6.1's "each worker only ever needs to hold one
    model in memory" is about not holding *several* models, which this respects:
    one process per family, one family's models in ``models``.
    """

    family: WorkerFamily
    device: str = "cpu"
    torch: Any = None
    captum: Any = None
    librosa: Any = None
    models: dict[str, Any] = field(default_factory=dict)

    def load_libraries(self) -> None:
        if self.torch is None:
            import torch

            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "worker.libs.loaded family=%s device=%s", self.family.value, self.device
            )
        if self.captum is None:
            import captum

            self.captum = captum
        if self.librosa is None:
            import librosa

            self.librosa = librosa

    def get_model(self, model_id: str, loader: Callable[[], Any]) -> Any:
        if model_id not in self.models:
            self.load_libraries()
            self.models[model_id] = loader()
        return self.models[model_id]


_WORKER_CTX: WorkerContext | None = None


def get_worker_context() -> WorkerContext:
    if _WORKER_CTX is None:
        raise RuntimeError("WorkerContext accessed outside a running worker.")
    return _WORKER_CTX


class AudioLITWorker(SimpleWorker):
    """Per-family RQ worker that keeps its models in process.

    Subclasses ``SimpleWorker`` (runs jobs in-process) rather than RQ's default
    forking ``Worker`` (runs each job in a work-horse child). See ``make_worker``
    for why. Publishes the QUEUED/PROCESSING/RETRYING/SUCCESS/FAILURE states the
    frontend consumes (SRS FR3.2), and releases VRAM between jobs (SAD C2).
    """

    def __init__(self, family: WorkerFamily, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.family = family
        self._ctx: WorkerContext | None = None

    def work(self, *args: Any, **kwargs: Any) -> bool:
        global _WORKER_CTX
        self._ctx = WorkerContext(family=self.family)
        _WORKER_CTX = self._ctx
        logger.info("worker.start family=%s pid=%s", self.family.value, os.getpid())
        try:
            return super().work(*args, **kwargs)
        finally:
            _WORKER_CTX = None

    def perform_job(self, job: Job, queue: Queue, *args: Any, **kwargs: Any) -> Any:
        publish_progress(job.id, "PROCESSING", {"family": self.family.value})
        started = time.monotonic()
        try:
            assert self._ctx is not None
            self._ctx.load_libraries()
            result = super().perform_job(job, queue, *args, **kwargs)
            publish_progress(
                job.id, "SUCCESS", {"duration_s": round(time.monotonic() - started, 3)}
            )
            return result
        except Exception as exc:
            # RQ retries a transient failure a few times before giving up, and the
            # user is told which of the two happened (SAD §11.1).
            state = "RETRYING" if job.retries_left else "FAILURE"
            publish_progress(
                job.id,
                state,
                {"duration_s": round(time.monotonic() - started, 3), "error": str(exc)},
            )
            logger.exception("job.failed id=%s", job.id)
            raise

    def handle_job_success(self, *args: Any, **kwargs: Any) -> Any:
        # Free VRAM between jobs so a family stays inside the 3-5 GB budget (C2).
        if self._ctx and self._ctx.torch is not None and self._ctx.torch.cuda.is_available():
            self._ctx.torch.cuda.empty_cache()
        return super().handle_job_success(*args, **kwargs)


def worker_queue_names(families: Optional[List[WorkerFamily]] = None) -> List[str]:
    """Queue names a worker for ``families`` (default: all) should listen on."""
    fams = families or list(WorkerFamily)
    return [QUEUE_CONFIGS[f].queue_name for f in fams]


def make_worker(
    family: WorkerFamily | str, connection: Optional[Redis] = None
) -> AudioLITWorker:
    """Build a worker bound to one family's queue. Call ``.work()`` to start it.

    Uses ``SimpleWorker`` (in-process) rather than RQ's forking ``Worker``. The
    forking worker runs each job in a work-horse child, so a model loaded in that
    child dies with it and is reloaded on the next job - which makes SAD §10's
    "fresh multi-task analysis under about 8 seconds" unreachable, since §10
    budgets roughly a minute just to prepare a model.

    What forking would buy is recovery from a hard crash of a job. The SAD does
    not ask for that: §11.1 grounds "any part can be restarted without losing
    work" in "the parts communicate only through the shared store", and §7
    packages every worker as its own Docker container - so restart is a container
    concern plus RQ's job registry, not a per-job fork. A CUDA out-of-memory,
    which is the failure §11.1 actually names, raises a catchable Python
    exception and is handled in ``perform_job`` without losing the process.

    If you change this back to a forking ``Worker``, the model cache in
    ``WorkerContext`` stops working - revisit §10's budget first.
    """
    fam = WorkerFamily(family) if not isinstance(family, WorkerFamily) else family
    queue = get_queue(fam, connection=connection)
    return AudioLITWorker(
        family=fam, queues=[queue], connection=queue.connection
    )


def run_worker(family: WorkerFamily | str, *, burst: bool = False) -> None:
    """Entrypoint for one family's worker process.

    GPU-bound families take a Redis lock so a second worker for the same family
    cannot start and double the VRAM footprint (SAD C2). The CPU-only mutation
    family is exempt and may scale out.
    """
    fam = WorkerFamily(family) if not isinstance(family, WorkerFamily) else family
    conn = get_redis_connection()

    # CPU Optimization: Cap PyTorch threads on CPU to avoid thread thrashing across parallel workers
    try:
        import torch
        if not torch.cuda.is_available():
            torch.set_num_threads(1)
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            logger.info("CPU mode detected: Pinned torch.set_num_threads(1) for worker family %s", fam.value)
    except Exception as e:
        logger.warning("Could not set CPU thread cap: %s", e)

    lock = None
    if get_queue_config(fam).gpu_bound:
        lock = conn.lock(
            f"{WORKER_LOCK_PREFIX}:{fam.value}", timeout=60 * 60 * 24, blocking=False
        )
        if not lock.acquire(blocking=False):
            raise RuntimeError(f"GPU family {fam.value} already has a worker running (SAD C2)")

    worker = make_worker(fam, connection=conn)
    try:
        worker.work(burst=burst, with_scheduler=True)
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                try:
                    conn.delete(f"{WORKER_LOCK_PREFIX}:{fam.value}")
                except Exception:
                    logger.warning("worker.lock.release_failed family=%s", fam.value)


# --------------------------------------------------------------------------- #
# Task functions
# --------------------------------------------------------------------------- #
# Scaffolding: these publish progress and return a shape the aggregator can fan
# in, but do not run real inference yet. The real ASR/SER/ADD implementations
# live in `multitask_orchestrator_service.py` (LIT-150); wiring them onto these
# per-family queues is the remaining step of LIT-127's follow-on, and needs the
# `/upload` contract change that LIT-227/LIT-157 deliberately deferred.

def asr_task(audio_ref: str, model_id: str, params: Mapping[str, Any]) -> dict[str, Any]:
    ctx = get_worker_context()
    publish_progress(_current_job_id(), "asr.running", {"model": model_id})
    try:
        from ..domain.model_loader_service import transcribe_asr
        res = transcribe_asr(audio_ref)
        return {
            "task": "asr",
            "model_id": model_id,
            "device": ctx.device,
            "transcript": res.get("text", "") if isinstance(res, dict) else str(res),
            "status": "success",
        }
    except Exception:
        return {"task": "asr", "model_id": model_id, "device": ctx.device, "transcript": "", "status": "scaffold"}


def ser_task(audio_ref: str, model_id: str, params: Mapping[str, Any]) -> dict[str, Any]:
    ctx = get_worker_context()
    publish_progress(_current_job_id(), "ser.running", {"model": model_id})
    try:
        from ..domain.model_loader_service import predict_ser
        res = predict_ser(audio_ref)
        return {
            "task": "ser",
            "model_id": model_id,
            "device": ctx.device,
            "predicted_emotion": res.get("predicted_emotion", "neutral"),
            "probabilities": res.get("probabilities", {}),
            "confidence": float(res.get("confidence", 0.0)),
            "status": "success",
        }
    except Exception:
        return {
            "task": "ser",
            "model_id": model_id,
            "device": ctx.device,
            "predicted_emotion": "neutral",
            "probabilities": {"neutral": 1.0},
            "confidence": 1.0,
            "status": "scaffold",
        }


def add_task(audio_ref: str, model_id: str, params: Mapping[str, Any]) -> dict[str, Any]:
    ctx = get_worker_context()
    publish_progress(_current_job_id(), "add.running", {"model": model_id})
    try:
        from ..domain.model_loader_service import (
            predict_deepfake, _ADD_MODEL_REGISTRY, _DEFAULT_ADD_MODEL_KEY,
        )
        # Derive both the valid set and the fallback from the registry. Spelling
        # the default here as a literal meant it kept naming a checkpoint the
        # rest of the system had already stopped defaulting to.
        model_key = model_id if model_id in _ADD_MODEL_REGISTRY else _DEFAULT_ADD_MODEL_KEY
        res = predict_deepfake(audio_ref, model_key=model_key)
        label = res.get("predicted_label", "bona-fide")
        syn_prob = float(res.get("synthetic_probability", 0.0))
        conf = float(res.get("confidence", 0.0))
        probs = res.get("probabilities", {})
        return {
            "task": "add",
            "model_id": model_id,
            "device": ctx.device,
            "label": label,
            "predicted_label": label,
            "synthetic_probability": syn_prob,
            "confidence": conf,
            "probabilities": probs,
            "status": "success",
        }
    except Exception:
        return {
            "task": "add",
            "model_id": model_id,
            "device": ctx.device,
            "label": "bona-fide",
            "predicted_label": "bona-fide",
            "synthetic_probability": 0.0,
            "confidence": 1.0,
            "probabilities": {"bona-fide": 1.0, "spoof": 0.0},
            "status": "scaffold",
        }


def xai_task(
    audio_ref: str, model_id: str, method: str, params: Mapping[str, Any]
) -> dict[str, Any]:
    ctx = get_worker_context()
    publish_progress(_current_job_id(), "xai.running", {"model": model_id, "method": method})
    return {"task": "xai", "model_id": model_id, "device": ctx.device, "_scaffold": True}


def mutation_task(audio_ref: str, mutation: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a non-destructive audio mutation (SAD Use Case 4, FR12, LIT-164).

    Was a `_scaffold` stub that never touched the audio (flagged on LIT-164);
    now delegates to `perturbation_service.perturb_and_save`, the same
    implementation already used by the synchronous `POST /perturb` route.
    """
    perturbations = list(mutation.get("perturbations", []))
    publish_progress(
        _current_job_id(), "mutation.running", {"perturbation_count": len(perturbations)}
    )
    from ..domain.perturbation_service import perturb_and_save

    return perturb_and_save(
        file_path=audio_ref,
        perturbations=perturbations,
        output_dir="uploads",
        dataset=mutation.get("dataset"),
        session_id=None,
    )


def accent_bias_task(
    model_id: str, corpus: str, samples_per_cohort: Optional[int]
) -> dict[str, Any]:
    """Group-wise WER accent-bias diagnostic (SRS Use Case 6, FR15, LIT-231).

    Enqueued rather than run inline because it streams a whole cohort of a
    dataset through ASR sequentially (SRS: "the system streams through the
    dataset rather than loading it all at once") - not a single-request-sized
    piece of work.
    """
    publish_progress(_current_job_id(), "accent_bias.running", {"model": model_id, "corpus": corpus})
    from ..domain.accent_bias_profiler import make_whisper_transcriber
    from ..domain.accent_bias_runner import run_accent_bias_diagnostic

    transcribe = make_whisper_transcriber(model_id)
    report = run_accent_bias_diagnostic(
        transcribe,
        corpus=corpus,
        model_id=model_id,
        samples_per_cohort=samples_per_cohort,
    )
    publish_progress(_current_job_id(), "accent_bias.completed", {"model": model_id})
    return report.to_json_dict()


def aggregator_task(family_job_ids: Sequence[str], cache_key: str | None) -> dict[str, Any]:
    """Fan-in: combine the family jobs' results once they have all finished.

    A failed sibling never loses the others' results (SAD §11.1).
    """
    conn = get_redis_connection()
    combined: dict[str, Any] = {"tasks": {}, "cache_key": cache_key, "schema_version": "1.0"}
    for job_id in family_job_ids:
        job = Job.fetch(job_id, connection=conn)
        if job.get_status() != JobStatus.FINISHED:
            combined["tasks"][job_id] = {"status": "failed"}
            continue
        result = job.result or {}
        combined["tasks"][result.get("task", "unknown")] = result

    if cache_key is not None:
        # TODO(LIT-163): write the combined result to the content-addressed cache.
        logger.info("aggregator.fanin cache_key=%s", cache_key)
    publish_progress(_current_job_id(), "aggregated", {"cache_key": cache_key})
    return combined


_TASK_FUNCS: dict[WorkerFamily, Callable[..., Any]] = {
    WorkerFamily.ASR: asr_task,
    WorkerFamily.SER: ser_task,
    WorkerFamily.ADD: add_task,
}


# --------------------------------------------------------------------------- #
# Enqueue API (used by the gateway)
# --------------------------------------------------------------------------- #

@dataclass
class EnqueueResult:
    job_id: str
    websocket_url: str
    family_jobs: dict[str, str] = field(default_factory=dict)
    cache_key: str | None = None

    def as_response(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "websocket_url": self.websocket_url,
            "schema_version": "1.0",
            "family_jobs": self.family_jobs,
            "cache_key": self.cache_key,
        }


def _ws_url(job_id: str, ws_base_url: str | None) -> str:
    base = ws_base_url or os.environ.get(WS_BASE_URL_ENV, DEFAULT_WS_BASE_URL)
    return f"{base.rstrip('/')}/{job_id}"


def enqueue_multitask_analysis(
    audio_ref: str,
    *,
    tasks: Iterable[WorkerFamily | str] = (WorkerFamily.ASR, WorkerFamily.SER, WorkerFamily.ADD),
    model_ids: Mapping[WorkerFamily | str, str] | None = None,
    params: Mapping[WorkerFamily | str, Mapping[str, Any]] | None = None,
    cache_key: str | None = None,
    ws_base_url: str | None = None,
) -> EnqueueResult:
    """Fan out one analysis across its family queues, then fan in (SAD §6.1)."""
    model_ids = model_ids or {}
    params = params or {}
    families = [WorkerFamily(f) if isinstance(f, str) else f for f in tasks]

    family_job_objs: list[Job] = []
    family_jobs: dict[str, str] = {}

    for fam in families:
        job = get_queue(fam).enqueue(
            _TASK_FUNCS[fam],
            audio_ref,
            model_ids.get(fam, "default"),
            dict(params.get(fam, {})),
            job_timeout=DEFAULT_JOB_TIMEOUT,
            result_ttl=DEFAULT_RESULT_TTL,
            failure_ttl=DEFAULT_FAILURE_TTL,
        )
        family_jobs[fam.value] = job.id
        family_job_objs.append(job)

    aggregator = get_queue(WorkerFamily.XAI).enqueue(
        aggregator_task,
        [j.id for j in family_job_objs],
        cache_key,
        depends_on=family_job_objs,
        job_timeout=DEFAULT_AGGREGATOR_TIMEOUT,
        result_ttl=DEFAULT_RESULT_TTL,
        failure_ttl=DEFAULT_FAILURE_TTL,
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
    job = get_queue(WorkerFamily.XAI).enqueue(
        xai_task, audio_ref, model_id, method, dict(params), job_timeout=DEFAULT_JOB_TIMEOUT
    )
    return EnqueueResult(
        job_id=job.id,
        websocket_url=_ws_url(job.id, ws_base_url),
        family_jobs={"xai": job.id},
        cache_key=cache_key,
    )


def enqueue_mutation(
    audio_ref: str, mutation: Mapping[str, Any], *, ws_base_url: str | None = None
) -> EnqueueResult:
    job = get_queue(WorkerFamily.MUTATION).enqueue(
        mutation_task, audio_ref, dict(mutation), job_timeout=DEFAULT_AGGREGATOR_TIMEOUT
    )
    return EnqueueResult(
        job_id=job.id,
        websocket_url=_ws_url(job.id, ws_base_url),
        family_jobs={"mutation": job.id},
    )


#: Accent-bias runs sequentially transcribe every sample in a cohort with a
#: fresh Whisper pipeline - longer-running than a single-file job, so it gets
#: its own timeout rather than the default 10-minute job budget.
ACCENT_BIAS_JOB_TIMEOUT: int = DEFAULT_JOB_TIMEOUT * 3


def enqueue_accent_bias(
    model_id: str,
    corpus: str = "l2-arctic",
    samples_per_cohort: Optional[int] = None,
    *,
    ws_base_url: str | None = None,
) -> EnqueueResult:
    # Reuses the ASR queue/worker family rather than adding a new
    # WorkerFamily - it is, mechanically, a batch of ASR jobs (SRS Use Case 6),
    # so it belongs on the same GPU-bound, concurrency-1 queue as `asr_task`.
    job = get_queue(WorkerFamily.ASR).enqueue(
        accent_bias_task,
        model_id,
        corpus,
        samples_per_cohort,
        job_timeout=ACCENT_BIAS_JOB_TIMEOUT,
        result_ttl=DEFAULT_RESULT_TTL,
        failure_ttl=DEFAULT_FAILURE_TTL,
    )
    return EnqueueResult(
        job_id=job.id,
        websocket_url=_ws_url(job.id, ws_base_url),
        family_jobs={"accent_bias": job.id},
    )


# --------------------------------------------------------------------------- #
# Job status / health
# --------------------------------------------------------------------------- #

def fetch_job(job_id: str) -> Job | None:
    try:
        return Job.fetch(job_id, connection=get_redis_connection())
    except Exception:
        return None


def job_status(job_id: str) -> dict[str, Any]:
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
    }


def health_check() -> dict[str, Any]:
    """Broker connectivity plus per-family queue depth."""
    try:
        conn = get_redis_connection()
        pong = conn.ping()
        redis_version = "unknown"
        try:
            redis_version = conn.info("server").get("redis_version", "unknown")
        except Exception:
            logger.debug("broker.info.unavailable", exc_info=True)
        return {
            "ok": bool(pong),
            "broker": "redis",
            "redis_version": redis_version,
            "queues": {f.value: get_queue(f).count for f in WorkerFamily},
        }
    except RedisConnectionError as exc:
        return {"ok": False, "broker": "redis", "error": str(exc)}


def run_batch_dataset_warmup_task(
    job_id: str,
    dataset: str,
    model: str = "whisper-base",
    tasks: list[str] | None = None,
    cooldown_ms: int = 100,
) -> dict[str, Any]:
    """CPU-safe, cancellable dataset warmup task runner.
    
    Evaluates requested tasks (ASR, SER, Acoustic, Saliency) for each file in dataset,
    saving every result to Redis. Checks cancellation flag before each file and inserts
    cooldown_ms sleep to prevent CPU overheating during multi-hour runs.
    """
    import json
    import time
    from app.infrastructure import cache_keys as ck
    from app.orchestration.inference_service import ADD_MODEL_KEYS
    from app.infrastructure.dataset_service import load_metadata, resolve_file

    try:
        conn = get_redis_connection()
    except Exception as e:
        logger.warning(f"Could not connect to Redis for warmup status tracking: {e}")
        conn = None
    tasks = tasks or ["asr", "ser", "acoustic"]
    rows = load_metadata(dataset)
    total = len(rows)

    completed = 0
    cancelled = False
    # A file only counts as warmed if something was actually written for it.
    # The previous version incremented `completed` inside the exception
    # handler, so a run that failed on every file still reported 100%.
    warmed_files = 0
    failures: list[str] = []
    warmed: set[str] = set()
    start_time = time.time()

    def format_eta(seconds: int) -> str:
        if seconds <= 0:
            return "Calculating..."
        if seconds < 60:
            return f"{seconds}s"
        mins, secs = divmod(seconds, 60)
        if mins < 60:
            return f"{mins}m {secs}s"
        hours, mins = divmod(mins, 60)
        return f"{hours}h {mins}m"

    def is_job_cancelled() -> bool:
        return bool(conn and conn.get(f"cancel_job_{job_id}"))

    def cleanup_memory():
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for i, row in enumerate(rows):
        # Check cancellation flag in Redis before processing file
        if is_job_cancelled():
            cancelled = True
            logger.info(f"Job {job_id} cancelled before file {i}/{total}")
            cleanup_memory()
            break

        filename = row.get("filename", "")
        if not filename:
            continue

        # Function helper to push subtask updates to Redis
        def update_subtask(subtask_label: str):
            if conn:
                elapsed = time.time() - start_time
                avg_per_file = elapsed / max(i, 1) if i > 0 else 0
                remaining = total - i
                eta_sec = int(avg_per_file * remaining) if i > 0 else 0
                eta_str = format_eta(eta_sec) if i > 0 else "Calculating..."

                p_data = {
                    "completed": i,
                    "total": total,
                    "current_file": filename,
                    "active_subtask": subtask_label,
                    "status": "running",
                    "percent": round((i / total) * 100, 1) if total > 0 else 0,
                    "eta_seconds": eta_sec,
                    "eta_formatted": eta_str,
                }
                conn.set(f"job_progress_{job_id}", json.dumps(p_data), ex=86400)

        update_subtask("Initializing...")

        # Run requested tasks synchronously in worker process (caches in Redis)
        try:
            target_file = filename or row.get("path", "")
            resolved_path = resolve_file(dataset, target_file)
            if resolved_path and resolved_path.exists():
                # Every hash variant, content-addressed first (FR4.1). Do not
                # destructure: the tuple grew from two to three and the
                # unpacking error was swallowed by the per-file except below,
                # so warmup silently wrote nothing at all.
                hashes = ck.both_hashes(resolved_path)
                from app.infrastructure.redis import cache_result_sync

                def write(keys, payload):
                    """Store one payload under every spelling its consumers read."""
                    for ns, key in keys:
                        cache_result_sync(ns, key, payload, ttl=86400)

                model_l = model.lower()
                is_whisper = "whisper" in model_l
                is_wav2vec = "wav2vec" in model_l

                # 1. ASR transcript + attention.
                #
                # These are two contracts sharing one forward pass. The
                # transcript family stores a plain string - its consumers call
                # .lower() on it - while the attention family stores the full
                # {"text", "attention"} dict. Writing the dict into both is the
                # defect that made warmed datasets unreadable.
                if "asr" in tasks and is_whisper:
                    if is_job_cancelled():
                        cancelled = True
                        cleanup_memory()
                        break
                    try:
                        update_subtask("ASR Transcription & Attention")
                        from app.domain.model_loader_service import (
                            transcribe_whisper_with_attention,
                            extract_whisper_embeddings,
                        )
                        asr = transcribe_whisper_with_attention(str(resolved_path), model)
                        if asr is not None:
                            write(
                                ck.transcript_keys(model, hashes),
                                {"prediction": ck.as_transcript(asr)},
                            )
                            write(ck.attention_keys(model, hashes), {"prediction": asr})
                            warmed.add("asr")

                        # Latent projection (FR11) reads its own key family.
                        try:
                            update_subtask("Latent Embeddings")
                            emb = extract_whisper_embeddings(str(resolved_path), model)
                            write(
                                ck.embedding_keys(model, hashes),
                                {"embedding": emb.tolist()},
                            )
                            warmed.add("embeddings")
                        except Exception as err:
                            logger.warning(f"Embedding warmup failed for {filename}: {err}")
                    except Exception as err:
                        failures.append(f"asr:{filename}")
                        logger.error(f"ASR warmup failed for {filename}: {err}", exc_info=True)

                # 2. SER. Cached under the model-independent "wav2vec2"
                # namespace, so it is warmed whenever requested regardless of
                # which ASR model the user selected.
                if "ser" in tasks:
                    if is_job_cancelled():
                        cancelled = True
                        cleanup_memory()
                        break
                    try:
                        update_subtask("Speech Emotion Recognition")
                        from app.domain.model_loader_service import (
                            predict_emotion_wave2vec_with_attention,
                        )
                        # A wav2vec2 selection means the user picked a SER
                        # checkpoint; anything else warms the default.
                        ser_model = model if is_wav2vec else None
                        ser = predict_emotion_wave2vec_with_attention(
                            str(resolved_path), model_id=ser_model
                        )
                        if ser is not None:
                            write(ck.ser_keys(hashes, ser_model), {"prediction": ser})
                            warmed.add("ser")
                    except Exception as err:
                        failures.append(f"ser:{filename}")
                        logger.error(f"SER warmup failed for {filename}: {err}", exc_info=True)

                # 3. Audio deepfake detection.
                if "add" in tasks or (not is_whisper and not is_wav2vec):
                    if is_job_cancelled():
                        cancelled = True
                        cleanup_memory()
                        break
                    try:
                        update_subtask("Deepfake Detection")
                        from app.domain.model_loader_service import (
                            predict_deepfake, _DEFAULT_ADD_MODEL_KEY,
                        )
                        add_model = model if model in ADD_MODEL_KEYS else _DEFAULT_ADD_MODEL_KEY
                        add = predict_deepfake(str(resolved_path), model_key=add_model)
                        if add is not None:
                            # Key on the ADD checkpoint, not the selected ASR
                            # model - `deepfake_keys` shares the transcript key
                            # shape, so keying on Whisper overwrites the
                            # transcript this same run just cached.
                            write(ck.deepfake_keys(add_model, hashes), {"prediction": add})
                            # FR7.2: the forensic timeline, warmed alongside the
                            # clip-level verdict so the panel opens instantly.
                            try:
                                from app.domain.model_loader_service import predict_deepfake_timeline
                                tl = predict_deepfake_timeline(str(resolved_path), model_key=add_model)
                                write(ck.add_timeline_keys(add_model, hashes), {"timeline": tl})
                            except Exception as err:
                                logger.warning(f"ADD timeline warmup failed for {filename}: {err}")
                            warmed.add("add")
                    except Exception as err:
                        failures.append(f"add:{filename}")
                        logger.error(f"ADD warmup failed for {filename}: {err}", exc_info=True)

                # 4. Acoustic profile (FR10) and frequency features.
                if "acoustic" in tasks:
                    if is_job_cancelled():
                        cancelled = True
                        cleanup_memory()
                        break
                    try:
                        update_subtask("Acoustic Profiling (F0 / Spectrogram)")
                        import soundfile as sf
                        from app.domain.acoustic_profiler_service import (
                            extract_acoustic_profile,
                        )
                        audio, sr = sf.read(str(resolved_path), dtype="float32", always_2d=False)
                        if audio.ndim > 1:
                            audio = audio.mean(axis=1)
                        prof = extract_acoustic_profile(audio, sr)
                        write(ck.acoustic_keys(hashes), prof)
                        warmed.add("acoustic")
                    except Exception as err:
                        failures.append(f"acoustic:{filename}")
                        logger.error(
                            f"Acoustic warmup failed for {filename}: {err}", exc_info=True
                        )

                    try:
                        from app.domain.model_loader_service import (
                            extract_audio_frequency_features,
                        )
                        feats = extract_audio_frequency_features(str(resolved_path))
                        write(ck.audio_frequency_keys(hashes), {"features": feats})
                        warmed.add("audio_frequency")
                    except Exception as err:
                        logger.warning(
                            f"Frequency-feature warmup failed for {filename}: {err}"
                        )

                # 5. Saliency / XAI attribution heatmaps.
                if "saliency" in tasks:
                    if is_job_cancelled():
                        cancelled = True
                        cleanup_memory()
                        break
                    try:
                        update_subtask("Saliency Attribution (Grad-CAM)")
                        from app.domain.saliency_service import generate_saliency
                        sal_res = generate_saliency(str(resolved_path), model, "gradcam")
                        if sal_res:
                            write(ck.saliency_keys(model, "gradcam", hashes), sal_res)
                            warmed.add("saliency")
                    except Exception as err:
                        failures.append(f"saliency:{filename}")
                        logger.error(
                            f"Saliency warmup failed for {filename}: {err}", exc_info=True
                        )
        except Exception as e:
            failures.append(f"file:{filename}")
            logger.error(f"Batch warmup error on {filename}: {e}", exc_info=True)

        completed += 1
        if warmed:
            warmed_files += 1
            warmed.clear()
        cleanup_memory()

        # CPU Thermal Cooldown Interval
        if cooldown_ms > 0:
            time.sleep(cooldown_ms / 1000.0)

    final_status = "cancelled" if cancelled else "completed"
    final_progress = {
        "completed": completed,
        "total": total,
        "current_file": "Done" if not cancelled else "Cancelled",
        "status": final_status,
        "percent": round((completed / total) * 100, 1) if total > 0 else 100.0,
        "cached_files": warmed_files,
        "failed_subtasks": len(failures),
    }
    if failures:
        logger.warning(
            "Warmup %s: %d/%d files cached, %d subtask failures (first 10: %s)",
            job_id, warmed_files, completed, len(failures), failures[:10],
        )
    cleanup_memory()
    if conn:
        conn.set(f"job_progress_{job_id}", json.dumps(final_progress), ex=86400)

    return final_progress

