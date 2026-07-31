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
