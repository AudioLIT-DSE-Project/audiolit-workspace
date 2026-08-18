from fastapi import APIRouter
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from ...infrastructure.redis import redis

router = APIRouter()

@router.get("/health")
async def health():
    try:
        pong = await redis.ping()
        return {"status": "ok", "redis": bool(pong)}
    except RedisError as e:
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

