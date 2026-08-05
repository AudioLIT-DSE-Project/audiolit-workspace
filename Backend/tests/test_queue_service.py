"""
Unit tests for queue_service (SRS FR3, SAD §5.1 / §6.1).

Covers: no-Celery DoD, per-family queue conventions, fan-out/fan-in via RQ
``depends_on``, GPU concurrency lock (SAD C2), progress pub/sub, credential
sanitisation, and health-check structure.
"""
from __future__ import annotations

import sys
from unittest import mock

import fakeredis
import pytest

from app.services import queue_service
from app.services.queue_service import (
    DEFAULT_GPU_CONCURRENCY,
    PROGRESS_CHANNEL_PREFIX,
    TaskFamily,
    _sanitize_redis_url,
    enqueue_attribution,
    enqueue_multitask_analysis,
    enqueue_mutation,
    get_queue,
    health_check,
    publish_progress,
    run_worker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_redis(monkeypatch):
    """Replace the shared Redis connection with a fakeredis instance."""
    fake = fakeredis.FakeStrictRedis()
    monkeypatch.setattr(queue_service, "_REDIS_CONN", fake)
    monkeypatch.setattr(queue_service, "_QUEUES", {})
    return fake


@pytest.fixture
def stub_worker(monkeypatch):
    """Replace AudioLITWorker with a no-op stub that captures constructor args."""

    captured: dict = {}

    class _StubWorker:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def work(self, burst=False, with_scheduler=False):
            captured["burst"] = burst
            captured["with_scheduler"] = with_scheduler
            return True

    monkeypatch.setattr(queue_service, "AudioLITWorker", _StubWorker)
    return captured


# ---------------------------------------------------------------------------
# DoD: no Celery dependency
# ---------------------------------------------------------------------------
def test_no_celery_imported():
    """DoD: RQ is the committed task fabric; Celery must not be imported."""
    assert "celery" not in sys.modules, "Celery must not be a dependency"


def test_rq_is_importable():
    import rq

    assert rq.__name__ == "rq"


# ---------------------------------------------------------------------------
# Queue conventions (LIT-127)
# ---------------------------------------------------------------------------
def test_all_five_families_defined():
    assert {f.value for f in TaskFamily} == {
        "asr", "ser", "add", "xai", "mutation",
    }


def test_get_queue_returns_named_queue(fake_redis):
    q = get_queue(TaskFamily.ASR)
    assert q.name == "asr"
    assert q.connection is fake_redis


def test_all_queues_instantiates_every_family(fake_redis):
    qs = queue_service.all_queues()
    assert set(qs.keys()) == set(TaskFamily)


# ---------------------------------------------------------------------------
# Fan-out + fan-in (SAD §6.1)
# ---------------------------------------------------------------------------
def test_enqueue_multitask_creates_per_family_jobs_and_aggregator(fake_redis):
    from rq.job import Job

    result = enqueue_multitask_analysis(
        "audio://sha256/abc",
        tasks=[TaskFamily.ASR, TaskFamily.SER, TaskFamily.ADD],
        model_ids={TaskFamily.ASR: "openai/whisper-base"},
        cache_key="sha256:deadbeef",
    )

    # Three family jobs + one aggregator.
    assert set(result.family_jobs.keys()) == {"asr", "ser", "add"}
    assert result.job_id  # aggregator id
    assert result.websocket_url.endswith(result.job_id)
    assert result.cache_key == "sha256:deadbeef"

    resp = result.as_response()
    assert resp["schema_version"] == "1.0"  # FR3.3

    # Aggregator depends on every family job.
    agg = Job.fetch(result.job_id, connection=fake_redis)
    assert set(agg.dependency_ids) == set(result.family_jobs.values())


def test_enqueue_multitask_rejects_non_inference_family(fake_redis):
    with pytest.raises(ValueError):
        enqueue_multitask_analysis(
            "audio://sha256/abc",
            tasks=[TaskFamily.XAI],
        )


def test_enqueue_attribution_uses_xai_queue(fake_redis):
    r = enqueue_attribution("audio://sha256/abc", "openai/whisper-base", "ig", {})
    assert "xai" in r.family_jobs


def test_enqueue_mutation_uses_mutation_queue(fake_redis):
    r = enqueue_mutation("audio://sha256/abc", {"kind": "silence"})
    assert "mutation" in r.family_jobs


# ---------------------------------------------------------------------------
# GPU concurrency lock (SAD C2)
# ---------------------------------------------------------------------------
def test_gpu_concurrency_constant_is_one():
    assert DEFAULT_GPU_CONCURRENCY == 1


def test_run_worker_acquires_lock_for_gpu_family(fake_redis, stub_worker):
    """Starting a GPU-family worker acquires a Redis lock (SAD C2)."""
    run_worker(TaskFamily.ASR)
    # Lock key should exist in Redis.
    assert fake_redis.exists("audiolit:worker-lock:asr")


def test_run_worker_raises_if_gpu_lock_already_held(fake_redis, stub_worker):
    """A second GPU-family worker is rejected (concurrency pinned to 1)."""
    run_worker(TaskFamily.ASR)
    with pytest.raises(RuntimeError, match="already running"):
        run_worker(TaskFamily.ASR)


def test_run_worker_no_lock_for_mutation_family(fake_redis, stub_worker):
    """The CPU-only mutation queue does not acquire a GPU lock."""
    run_worker(TaskFamily.MUTATION)
    assert not fake_redis.exists("audiolit:worker-lock:mutation")


def test_run_worker_releases_lock_on_exit(fake_redis, stub_worker):
    run_worker(TaskFamily.SER)
    # After work() returns (stubbed), the lock should be released.
    assert not fake_redis.exists("audiolit:worker-lock:ser")


# ---------------------------------------------------------------------------
# Progress reporting (FR3.2)
# ---------------------------------------------------------------------------
def test_publish_progress_emits_to_pubsub(fake_redis):
    publish_progress("job-123", "started", {"family": "asr"})
    ps = fake_redis.pubsub()
    ps.subscribe(f"{PROGRESS_CHANNEL_PREFIX}:job-123")
    ps.get_message()  # discard subscription ack
    msg = ps.get_message()
    assert msg is not None
    assert b"job-123" in msg["data"]
    assert b"started" in msg["data"]


# ---------------------------------------------------------------------------
# Health check (DoD)
# ---------------------------------------------------------------------------
def test_health_check_ok(fake_redis):
    h = health_check()
    assert h["ok"] is True
    assert h["broker"] == "redis"
    assert "redis_version" in h
    assert set(h["queues"].keys()) == {
        "asr", "ser", "add", "xai", "mutation",
    }


def test_health_check_reports_broker_failure(monkeypatch):
    """If Redis is unreachable, health_check returns ok=False."""
    monkeypatch.setattr(queue_service, "_REDIS_CONN", None)

    def _fail():
        from redis.exceptions import ConnectionError
        raise ConnectionError("boom")

    monkeypatch.setattr(queue_service, "get_redis_connection", _fail)
    h = health_check()
    assert h["ok"] is False
    assert "boom" in h["error"]


# ---------------------------------------------------------------------------
# Credential sanitisation (SRS SR5)
# ---------------------------------------------------------------------------
def test_sanitize_redis_url_strips_password():
    safe = _sanitize_redis_url("redis://:s3cret@host:6379/0")
    assert "s3cret" not in safe
    assert "host" in safe


def test_sanitize_redis_url_no_credentials():
    safe = _sanitize_redis_url("redis://localhost:6379/0")
    assert safe == "redis://localhost:6379/0"


# ---------------------------------------------------------------------------
# Job status lookup
# ---------------------------------------------------------------------------
def test_job_status_unknown(fake_redis):
    s = queue_service.job_status("nonexistent")
    assert s["status"] == "unknown"


def test_job_status_after_enqueue(fake_redis):
    r = enqueue_attribution("audio://sha256/abc", "m", "ig", {})
    s = queue_service.job_status(r.job_id)
    assert s["job_id"] == r.job_id
    assert s["status"] in ("queued", "finished", "failed", "started", "deferred")
