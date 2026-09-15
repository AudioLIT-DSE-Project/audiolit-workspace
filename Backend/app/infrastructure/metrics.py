"""Operational metrics counters and readers (LIT-259).

Workers and the cache chokepoints record lightweight counters straight into
Redis; ``GET /metrics`` (``app/api/routes/health.py``) aggregates them. Every
write here is best-effort - a metrics counter must never be able to take the
request or worker path down - so recording swallows Redis errors and logs at
DEBUG.

Counter keys (all under the ``audiolit:metrics:`` prefix):

  ``tasks``                       hash, field ``<family>:<status>`` -> count
                                  (e.g. ``asr:success``). The ``status`` values
                                  are the task-event tails: ``processing``,
                                  ``success``, ``failed``, ``retrying``.
  ``durations:sum_ms`` / count    hashes, field ``<family>``. Durations are
                                  accumulated in whole milliseconds via HINCRBY
                                  so the aggregate is exact and never has to
                                  round-trip a float.
  ``cache``                       hash, fields ``hits`` / ``misses``.

The counters stay where the facts live: the task events are recorded in
``AudioLITWorker.perform_job`` (which alone knows the outcome + duration), and
cache hits/misses in the two read chokepoints every route's lookup passes
through (``get_result`` / ``get_result_sync`` in ``app/infrastructure/redis.py``
and ``RedisCacheManager.get`` in ``app/core/redis.py``).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

logger = logging.getLogger("audiolit.metrics")

TASKS_HASH = "audiolit:metrics:tasks"
DUR_SUM_MS_HASH = "audiolit:metrics:durations:sum_ms"
DUR_COUNT_HASH = "audiolit:metrics:durations:count"
CACHE_HASH = "audiolit:metrics:cache"


def _as_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #

def record_task(
    conn: Any,
    family: str,
    status: str,
    duration_s: float | None = None,
) -> None:
    """Count one finished task and (optionally) its elapsed duration.

    ``conn`` is the synchronous broker connection (workers are sync). Never
    raises; a down broker degrades metrics, not work.
    """
    try:
        conn.hincrby(TASKS_HASH, f"{family}:{status}", 1)
        if duration_s is not None:
            conn.hincrby(DUR_SUM_MS_HASH, family, int(round(duration_s * 1000)))
            conn.hincrby(DUR_COUNT_HASH, family, 1)
    except Exception:
        logger.debug(
            "metrics.task.record_failed family=%s status=%s", family, status, exc_info=True
        )


def record_cache(conn: Any, hit: bool) -> None:
    """Count a cache hit/miss on a synchronous lookup."""
    try:
        conn.hincrby(CACHE_HASH, "hits" if hit else "misses", 1)
    except Exception:
        logger.debug("metrics.cache.record_failed hit=%s", hit, exc_info=True)


def record_cache_via(conn_getter: Callable[[], Any], hit: bool) -> None:
    """``record_cache`` behind a lazy connection getter.

    For a caller that does not already hold a connection (e.g. the FR4 tensor
    cache manager): the getter is invoked inside the guard, so a broker that
    raises at connect time still cannot leak an exception onto the lookup path.
    """
    try:
        record_cache(conn_getter(), hit)
    except Exception:
        logger.debug("metrics.cache.record_failed_via hit=%s", hit, exc_info=True)


async def arecord_cache(client: Any, hit: bool) -> None:
    """Count a cache hit/miss on an asynchronous lookup."""
    try:
        await client.hincrby(CACHE_HASH, "hits" if hit else "misses", 1)
    except Exception:
        logger.debug("metrics.cache.arecord_failed hit=%s", hit, exc_info=True)


# --------------------------------------------------------------------------- #
# Reading / shaping for GET /metrics
# --------------------------------------------------------------------------- #

def hit_ratio(hits: int, misses: int) -> float:
    """Cache hit ratio, 0.0 when there has never been a lookup (no div-by-zero)."""
    denominator = hits + misses
    return round(hits / denominator, 4) if denominator else 0.0


def tasks_summary(groups: Mapping[str, Any]) -> dict[str, Any]:
    """``{<family>:<status>: count, ..., total: <sum>}`` from a ``hgetall``."""
    out: dict[str, Any] = {_as_str(k): _as_int(v) for k, v in (groups or {}).items()}
    out["total"] = sum(out.values())
    return out


def durations_summary(
    sum_ms: Mapping[str, Any], counts: Mapping[str, Any]
) -> dict[str, Any]:
    """Per-family ``{count, sum_ms, avg_ms}`` from the two duration hashes."""
    sums = {_as_str(k): _as_int(v) for k, v in (sum_ms or {}).items()}
    counts_by_family = {_as_str(k): _as_int(v) for k, v in (counts or {}).items()}
    result: dict[str, Any] = {}
    for family in sorted(set(sums) | set(counts_by_family)):
        count = counts_by_family.get(family, 0)
        total_ms = sums.get(family, 0)
        avg_ms = round(total_ms / count, 1) if count else 0.0
        result[family] = {"count": count, "sum_ms": total_ms, "avg_ms": avg_ms}
    return result


def cache_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    """``{hits, misses, hit_ratio}`` from the cache counter hash."""
    hits = _as_int((raw or {}).get("hits", 0))
    misses = _as_int((raw or {}).get("misses", 0))
    return {"hits": hits, "misses": misses, "hit_ratio": hit_ratio(hits, misses)}