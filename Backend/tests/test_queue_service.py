"""Unit tests for queue_service (SRS FR3, SAD §5.1 / §6.1)."""
from __future__ import annotations

import sys
import pytest
import fakeredis

from app.services import queue_service
from app.services.queue_service import (
    DEFAULT_GPU_CONCURRENCY, PROGRESS_CHANNEL_PREFIX, TaskFamily,
    _sanitize_redis_url, enqueue_attribution, enqueue_multitask_analysis,
    enqueue_mutation, get_queue, health_check, publish_progress, run_worker,
)

@pytest.fixture
def fake_redis(monkeypatch):
    fake = fakeredis.FakeStrictRedis()
    monkeypatch.setattr(queue_service, "_REDIS_CONN", fake)
    monkeypatch.setattr(queue_service, "_QUEUES", {})
    return fake

@pytest.fixture
def stub_worker(monkeypatch):
    captured: dict = {}
    class _StubWorker:
        def __init__(self, **kwargs): captured.update(kwargs)
        def work(self, burst=False, with_scheduler=False): return True
    monkeypatch.setattr(queue_service, "AudioLITWorker", _StubWorker)
    return captured

def test_no_celery_imported():
    assert "celery" not in sys.modules

def test_all_five_families_defined():
    assert {f.value for f in TaskFamily} == {"asr", "ser", "add", "xai", "mutation"}

def test_get_queue_returns_named_queue(fake_redis):
    q = get_queue(TaskFamily.ASR)
    assert q.name == "asr"

def test_enqueue_multitask_creates_per_family_jobs_and_aggregator(fake_redis):
    from rq.job import Job
    result = enqueue_multitask_analysis(
        "audio://sha256/abc",
        tasks=[TaskFamily.ASR, TaskFamily.SER, TaskFamily.ADD],
        cache_key="sha256:deadbeef",
    )
    assert set(result.family_jobs.keys()) == {"asr", "ser", "add"}
    assert result.job_id
    agg = Job.fetch(result.job_id, connection=fake_redis)
    assert set(agg.dependency_ids) == set(result.family_jobs.values())

def test_run_worker_acquires_lock_for_gpu_family(fake_redis, stub_worker):
    run_worker(TaskFamily.ASR)
    assert fake_redis.exists("audiolit:worker-lock:asr")

def test_run_worker_raises_if_gpu_lock_already_held(fake_redis, stub_worker):
    run_worker(TaskFamily.ASR)
    with pytest.raises(RuntimeError):
        run_worker(TaskFamily.ASR)

def test_run_worker_no_lock_for_mutation_family(fake_redis, stub_worker):
    run_worker(TaskFamily.MUTATION)
    assert not fake_redis.exists("audiolit:worker-lock:mutation")

def test_run_worker_releases_lock_on_exit(fake_redis, stub_worker):
    run_worker(TaskFamily.SER)
    assert not fake_redis.exists("audiolit:worker-lock:ser")

def test_publish_progress_emits_to_pubsub(fake_redis):
    # Subscribe FIRST
    ps = fake_redis.pubsub()
    ps.subscribe(f"{PROGRESS_CHANNEL_PREFIX}:job-123")
    ps.get_message() # discard ack
    
    # Publish message
    publish_progress("job-123", "started", {"family": "asr"})
    
    # Retrieve
    msg = ps.get_message(timeout=1.0)
    assert msg is not None
    assert msg.get("type") == "message"
    assert b"job-123" in msg["data"]

def test_health_check_ok(fake_redis):
    h = health_check()
    assert h["ok"] is True
    assert h["broker"] == "redis"
    assert set(h["queues"].keys()) == {"asr", "ser", "add", "xai", "mutation"}

def test_health_check_reports_broker_failure(monkeypatch):
    monkeypatch.setattr(queue_service, "_REDIS_CONN", None)
    def _fail():
        from redis.exceptions import ConnectionError
        raise ConnectionError("boom")
    monkeypatch.setattr(queue_service, "get_redis_connection", _fail)
    h = health_check()
    assert h["ok"] is False
