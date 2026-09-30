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


_WORKER_CONNECTION: Redis | None = None


def get_worker_redis_connection() -> Redis:
    """A separate connection for RQ workers, without the read timeout.

    A worker waiting for a job sits in a blocking ``BLPOP`` for
    ``worker_ttl - 15`` seconds - 405 s with RQ 2.10's defaults. The shared
    connection above sets ``socket_timeout=10`` so that a hung broker fails a
    *request* quickly, and redis-py applies that same timeout to the blocking
    read: the socket times out 10 s into a 405 s wait, RQ sees a Redis
    connection timeout, and the worker quits.

    That is why workers here died only while idle and survived under load -
    when jobs are queued, ``BLPOP`` returns long before 10 s. Observed twice in
    one session, both times after a quiet period, both logged as
    "Redis connection timeout, quitting..." while Redis itself was healthy and
    answering PING. Nothing restarts them, so asynchronous work silently stops
    being processed while the API still looks fine.

    Raising ``socket_timeout`` on the shared connection would fix the worker and
    break the request path, which uses the same client (acoustic, health and
    inference routes all call ``get_redis_connection``); a stalled broker would
    hang a request for seven minutes instead of failing in ten seconds. So the
    worker gets its own client, with no read deadline, and the request path
    keeps its fail-fast one.

    ``socket_keepalive`` asks the OS to notice a genuinely dead peer, which is
    the failure the removed timeout would otherwise have masked.
    """
    global _WORKER_CONNECTION
    if _WORKER_CONNECTION is None:
        url = settings.REDIS_URL
        connection = Redis.from_url(
            url,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=None,      # must outlast RQ's blocking dequeue
            socket_keepalive=True,
            health_check_interval=30,
        )
        try:
            connection.ping()
        except RedisConnectionError as exc:
            logger.error("broker.unreachable url=%s err=%s", sanitize_redis_url(url), exc)
            raise
        logger.info("broker.connected.worker url=%s", sanitize_redis_url(url))
        _WORKER_CONNECTION = connection
    return _WORKER_CONNECTION


def reset_connection() -> None:
    """Drop the cached connections. For tests that swap in a fake Redis."""
    global _CONNECTION, _WORKER_CONNECTION
    _CONNECTION = None
    _WORKER_CONNECTION = None
