"""
Background worker scaffolding for AudioLIT (SRS FR3, SAD §5.1 / §6.1).

RQ (Redis Queue) is the committed task fabric for AudioLIT, chosen over Celery
for lighter setup and simpler single-host operation (SAD §3.3). No Celery
dependency is introduced.
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

class TaskFamily(str, Enum):
    ASR = "asr"
    SER = "ser"
    ADD = "add"
    XAI = "xai"
    MUTATION = "mutation"

DEFAULT_GPU_CONCURRENCY: int = 1
DEFAULT_JOB_TIMEOUT: int = 600
DEFAULT_AGGREGATOR_TIMEOUT: int = 120
DEFAULT_RESULT_TTL: int = 60 * 60 * 24
DEFAULT_FAILURE_TTL: int = 60 * 60

PROGRESS_CHANNEL_PREFIX: str = "audiolit:progress"
WORKER_LOCK_PREFIX: str = "audiolit:worker-lock"

REDIS_URL_ENV: str = "REDIS_URL"
WS_BASE_URL_ENV: str = "AUDIOLIT_WS_BASE_URL"

_REDIS_CONN: redis.Redis | None = None
_QUEUES: dict[TaskFamily, Queue] = {}

def get_redis_connection() -> redis.Redis:
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
            logger.info("queue.broker.connected url=%s", _sanitize_redis_url(url))
        except RedisConnectionError as exc:
            logger.error("queue.broker.unreachable url=%s err=%s", _sanitize_redis_url(url), exc)
            raise
    return _REDIS_CONN

def _sanitize_redis_url(url: str) -> str:
    try:
        p = urlparse(url)
        netloc = p.hostname or ""
        if p.port:
            netloc = f"{netloc}:{p.port}"
        return urlunparse((p.scheme, netloc, p.path, "", "", ""))
    except Exception:
        return "redis://***"

def get_queue(family: TaskFamily | str) -> Queue:
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
    for f in TaskFamily:
        get_queue(f)
    return _QUEUES

def publish_progress(job_id: str, stage: str, payload: Mapping[str, Any] | None = None) -> None:
    conn = get_redis_connection()
    channel = f"{PROGRESS_CHANNEL_PREFIX}:{job_id}".encode()
    event = {"job_id": job_id, "stage": stage, "ts": time.time(), "payload": dict(payload or {})}
    conn.publish(channel, json.dumps(event).encode())

def _current_job_id() -> str:
    from rq import get_current_job
    job = get_current_job()
    return job.id if job is not None else "unknown"

@dataclass
class WorkerContext:
    family: TaskFamily
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
            logger.info("worker.libs.loaded family=%s device=%s", self.family.value, self.device)
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
    def __init__(self, family: TaskFamily, *args: Any, **kwargs: Any) -> None:
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
        # Emit PROCESSING state
        publish_progress(job.id, "PROCESSING", {"family": self.family.value})
        t0 = time.monotonic()
        try:
            assert self._ctx is not None
            self._ctx.load_libraries()
            result = super().perform_job(job, queue, *args, **kwargs)
            dt = time.monotonic() - t0
            # Emit SUCCESS state
            publish_progress(job.id, "SUCCESS", {"duration_s": round(dt, 3)})
            return result
        except Exception as exc:
            dt = time.monotonic() - t0
            # Emit RETRYING or FAILURE state based on RQ's retry config
            state = "RETRYING" if job.retries_left > 0 else "FAILURE"
            publish_progress(job.id, state, {"duration_s": round(dt, 3), "error": str(exc)})
            logger.exception("job.failed id=%s", job.id)
            raise

    def handle_job_success(self, *args: Any, **kwargs: Any) -> Any:
        if self._ctx and self._ctx.torch is not None and self._ctx.torch.cuda.is_available():
            self._ctx.torch.cuda.empty_cache()
        return super().handle_job_success(*args, **kwargs)

def asr_task(audio_ref: str, model_id: str, params: Mapping[str, Any]) -> dict[str, Any]:
    ctx = get_worker_context()
    publish_progress(_current_job_id(), "asr.running", {"model": model_id})
    return {"task": "asr", "model_id": model_id, "device": ctx.device, "_scaffold": True}

def ser_task(audio_ref: str, model_id: str, params: Mapping[str, Any]) -> dict[str, Any]:
    ctx = get_worker_context()
    publish_progress(_current_job_id(), "ser.running", {"model": model_id})
    return {"task": "ser", "model_id": model_id, "device": ctx.device, "_scaffold": True}

def add_task(audio_ref: str, model_id: str, params: Mapping[str, Any]) -> dict[str, Any]:
    ctx = get_worker_context()
    publish_progress(_current_job_id(), "add.running", {"model": model_id})
    return {"task": "add", "model_id": model_id, "device": ctx.device, "_scaffold": True}

def xai_task(audio_ref: str, model_id: str, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
    ctx = get_worker_context()
    publish_progress(_current_job_id(), "xai.running", {"model": model_id, "method": method})
    return {"task": "xai", "model_id": model_id, "device": ctx.device, "_scaffold": True}

def mutation_task(audio_ref: str, mutation: Mapping[str, Any]) -> dict[str, Any]:
    publish_progress(_current_job_id(), "mutation.running", {"mutation": mutation.get("kind")})
    return {"task": "mutation", "_scaffold": True}

def aggregator_task(family_job_ids: Sequence[str], cache_key: str | None) -> dict[str, Any]:
    conn = get_redis_connection()
    combined: dict[str, Any] = {"tasks": {}, "cache_key": cache_key, "schema_version": "1.0"}
    for jid in family_job_ids:
        job = Job.fetch(jid, connection=conn)
        if job.get_status() != JobStatus.FINISHED:
            combined["tasks"][jid] = {"status": "failed"}
            continue
        result = job.result or {}
        combined["tasks"][result.get("task", "unknown")] = result

    if cache_key is not None:
        # TODO(LIT-163): Write to content-addressed cache
        logger.info("aggregator.fanin cache_key=%s", cache_key)
    publish_progress(_current_job_id(), "aggregated", {"cache_key": cache_key})
    return combined

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
    tasks: Iterable[TaskFamily | str] = (TaskFamily.ASR, TaskFamily.SER, TaskFamily.ADD),
    model_ids: Mapping[TaskFamily | str, str] | None = None,
    params: Mapping[TaskFamily | str, Mapping[str, Any]] | None = None,
    cache_key: str | None = None,
    ws_base_url: str | None = None,
) -> EnqueueResult:
    model_ids = model_ids or {}
    params = params or {}
    families = [TaskFamily(f) if isinstance(f, str) else f for f in tasks]

    family_job_objs: list[Job] = []
    family_jobs: dict[str, str] = {}

    for f in families:
        queue = get_queue(f)
        job = queue.enqueue(
            _TASK_FUNCS[f], audio_ref, model_ids.get(f, "default"), dict(params.get(f, {})),
            job_timeout=DEFAULT_JOB_TIMEOUT, result_ttl=DEFAULT_RESULT_TTL, failure_ttl=DEFAULT_FAILURE_TTL,
        )
        family_jobs[f.value] = job.id
        family_job_objs.append(job)

    aggregator_queue = get_queue(TaskFamily.XAI)
    aggregator = aggregator_queue.enqueue(
        aggregator_task, [j.id for j in family_job_objs], cache_key,
        depends_on=family_job_objs,
        job_timeout=DEFAULT_AGGREGATOR_TIMEOUT, result_ttl=DEFAULT_RESULT_TTL, failure_ttl=DEFAULT_FAILURE_TTL,
    )

    return EnqueueResult(
        job_id=aggregator.id, websocket_url=_ws_url(aggregator.id, ws_base_url),
        family_jobs=family_jobs, cache_key=cache_key,
    )

def enqueue_attribution(audio_ref: str, model_id: str, method: str, params: Mapping[str, Any], *, cache_key: str | None = None, ws_base_url: str | None = None) -> EnqueueResult:
    queue = get_queue(TaskFamily.XAI)
    job = queue.enqueue(xai_task, audio_ref, model_id, method, dict(params), job_timeout=DEFAULT_JOB_TIMEOUT)
    return EnqueueResult(job_id=job.id, websocket_url=_ws_url(job.id, ws_base_url), family_jobs={"xai": job.id})

def enqueue_mutation(audio_ref: str, mutation: Mapping[str, Any], *, ws_base_url: str | None = None) -> EnqueueResult:
    queue = get_queue(TaskFamily.MUTATION)
    job = queue.enqueue(mutation_task, audio_ref, dict(mutation), job_timeout=DEFAULT_AGGREGATOR_TIMEOUT)
    return EnqueueResult(job_id=job.id, websocket_url=_ws_url(job.id, ws_base_url), family_jobs={"mutation": job.id})

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
        "job_id": job.id, "status": job.get_status(),
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
        "result": job.result if job.is_finished else None,
        "error": str(job.exc_info) if job.is_failed else None,
    }

def run_worker(family: TaskFamily | str, *, burst: bool = False) -> None:
    family = TaskFamily(family) if isinstance(family, str) else family
    queue = get_queue(family)
    conn = get_redis_connection()

    lock: redis.lock.Lock | None = None
    if family is not TaskFamily.MUTATION:
        lock = conn.lock(f"{WORKER_LOCK_PREFIX}:{family.value}", timeout=60*60*24, blocking=False)
        if not lock.acquire(blocking=False):
            raise RuntimeError(f"GPU family {family.value} already running (SAD C2)")

    worker = AudioLITWorker(family=family, queues=[queue], connection=conn)
    try:
        worker.work(burst=burst, with_scheduler=True)
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                try:
                    conn.delete(f"{WORKER_LOCK_PREFIX}:{family.value}")
                except Exception:
                    pass

def health_check() -> dict[str, Any]:
    try:
        conn = get_redis_connection()
        pong = conn.ping()
        redis_version = "unknown"
        try:
            info = conn.info("server")
            redis_version = info.get("redis_version", "unknown")
        except Exception:
            try:
                info = conn.info()
                redis_version = info.get("redis_version", "unknown")
            except Exception:
                pass
        return {"ok": bool(pong), "broker": "redis", "redis_version": redis_version, "queues": {f.value: get_queue(f).count for f in TaskFamily}}
    except RedisConnectionError as exc:
        return {"ok": False, "broker": "redis", "error": str(exc)}

def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="AudioLIT RQ worker launcher")
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker", help="Start a per-family worker")
    w.add_argument("family", choices=[f.value for f in TaskFamily])
    w.add_argument("--burst", action="store_true")
    sub.add_parser("health", help="Broker connectivity check")
    args = parser.parse_args()
    if args.cmd == "worker":
        run_worker(args.family, burst=args.burst)
    elif args.cmd == "health":
        print(json.dumps(health_check(), indent=2))

if __name__ == "__main__":
    _cli()
