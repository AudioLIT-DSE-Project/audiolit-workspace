"""Synchronous Redis connection for the RQ fabric (SAD §5.1 infrastructure layer).

RQ (unlike the rest of the app) needs a synchronous redis-py client - the async
client in `app.infrastructure.redis` is for the FastAPI request path, not for
workers. Same `REDIS_URL`, separate connection.

The connection lives here rather than in `app/orchestration/` because SAD §5.1
has each layer relying only on the layer below it: orchestration uses
infrastructure, so infrastructure is what owns the broker connection. The
orchestrator previously read `os.environ["REDIS_URL"]` directly, bypassing both
this module and `settings` (LIT-230).
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse, urlunparse

from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from .settings import settings

logger = logging.getLogger("audiolit.infrastructure.rq")

_CONNECTION: Redis | None = None


def sanitize_redis_url(url: str) -> str:
    """Strip credentials from a Redis URL so it is safe to log (SAD §11.3)."""
    try:
        parsed = urlparse(url)
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))
    except Exception:
        return "redis://***"


def get_redis_connection() -> Redis:
    """The process-wide synchronous Redis connection used by RQ.

    Cached after the first call: RQ queues, workers and the progress pub/sub all
    share one client. Pings on first connect so an unreachable broker fails here
    with a clear message rather than at the first enqueue.
    """
    global _CONNECTION
    if _CONNECTION is None:
        url = settings.REDIS_URL
        connection = Redis.from_url(
            url,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=10,
            health_check_interval=30,
        )
        try:
            connection.ping()
        except RedisConnectionError as exc:
            logger.error("broker.unreachable url=%s err=%s", sanitize_redis_url(url), exc)
            raise
        logger.info("broker.connected url=%s", sanitize_redis_url(url))
        _CONNECTION = connection
    return _CONNECTION


def reset_connection() -> None:
    """Drop the cached connection. For tests that swap in a fake Redis."""
    global _CONNECTION
    _CONNECTION = None
