from fastapi import APIRouter
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from ...infrastructure import redis as redis_module
from ...infrastructure import metrics as metrics_module

router = APIRouter()

@router.get("/health")
async def health():
    try:
        pong = await redis_module.redis.ping()
        return {"status": "ok", "redis": bool(pong)}
    except (RedisError, Exception) as e:
        # Return 503 if Redis isn’t reachable
        return JSONResponse({"status": "degraded", "redis": False, "detail": str(e)}, status_code=503)


@router.get("/health/workers")
async def health_workers():
    """Detailed live worker and queue status for monitoring execution."""
    from ...orchestration.task_orchestrator import health_check, get_redis_connection
    from rq import Worker

    base = health_check()
    if not base.get("ok"):
        return JSONResponse({"status": "degraded", "detail": base.get("error")}, status_code=503)

    try:
        conn = get_redis_connection()
        workers = Worker.all(connection=conn)
        worker_list = []
        for w in workers:
            current_job = w.get_current_job()
            worker_list.append({
                "name": w.name,
                "queues": w.queue_names(),
                "state": w.get_state(),
                "pid": w.pid,
                "current_job_id": current_job.id if current_job else None,
                "successful_jobs": w.successful_job_count,
                "failed_jobs": w.failed_job_count,
            })

        return {
            "status": "ok",
            "active_worker_count": len(worker_list),
            "queues": base.get("queues", {}),
            "workers": worker_list,
        }
    except Exception as e:
        return JSONResponse({"status": "degraded", "detail": str(e)}, status_code=500)


def _cuda_state() -> dict:
    """Lazily imported so a CPU-only box (or when torch is not installed) never
    blocks the request path."""
    try:
        import torch

        if torch.cuda.is_available():
            return {"cuda_available": True, "device": torch.cuda.get_device_name(0)}
        return {"cuda_available": False, "device": "cpu"}
    except Exception:
        return {"cuda_available": False, "device": "cpu"}


@router.get("/metrics")
async def operational_metrics():
    """Operational metrics (LIT-259): task counters, duration aggregates, queue
    depths, cache hit ratio and GPU state - the five keys an ops card needs.

    Counter recovery is best-effort (``audiolit:metrics:*`` under
    ``app/infrastructure/metrics.py``); a broker that is down degrades the
    counters and the queue depths, never the response's shape.
    """
    from ...orchestration.task_orchestrator import (
        QUEUE_CONFIGS, WORKER_LOCK_PREFIX, WorkerFamily, health_check,
    )

    try:
        client = redis_module.redis
        tasks_raw = await client.hgetall(metrics_module.TASKS_HASH)
        dur_sum = await client.hgetall(metrics_module.DUR_SUM_MS_HASH)
        dur_count = await client.hgetall(metrics_module.DUR_COUNT_HASH)
        cache_raw = await client.hgetall(metrics_module.CACHE_HASH)
        tasks = metrics_module.tasks_summary(tasks_raw)
        durations = metrics_module.durations_summary(dur_sum, dur_count)
        cache = metrics_module.cache_summary(cache_raw)
    except Exception as e:
        return JSONResponse({"status": "degraded", "detail": str(e)}, status_code=503)

    # Queue depths come from the RQ side, exactly like /health/workers.
    base = health_check()
    queues = (
        base.get("queues")
        if base.get("ok")
        else {f.value: 0 for f in WorkerFamily}
    )

    # GPU-bound families whose worker lock is held currently own the GPU.
    locked: list[str] = []
    try:
        for fam, config in QUEUE_CONFIGS.items():
            if config.gpu_bound and await redis_module.redis.exists(
                f"{WORKER_LOCK_PREFIX}:{fam.value}"
            ):
                locked.append(fam.value)
    except Exception:
        locked = []

    gpu = {**_cuda_state(), "families_locked": locked}

    return {
        "tasks": tasks,
        "durations": durations,
        "queues": queues,
        "cache": cache,
        "gpu": gpu,
    }

