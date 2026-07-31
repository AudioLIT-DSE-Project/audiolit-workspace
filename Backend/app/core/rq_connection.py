from redis import Redis

from ..infrastructure.settings import settings


def get_redis_connection() -> Redis:
    """
    Sync Redis connection for RQ.

    RQ (unlike the rest of the app) needs a synchronous redis-py client -
    the async client in `app.infrastructure.redis` is for the FastAPI request path,
    not for workers. Same `REDIS_URL`, separate connection.
    """
    return Redis.from_url(settings.REDIS_URL)
