"""LIT-259: structured JSON task logs and operational metrics.

Covers the three acceptance tests - JsonFormatter round-trip, the worker's
success path (counter + caplog capture + no audio identity in the logs), and
``GET /metrics`` with all five keys and a zero hit ratio - plus the cache
counters and the model-id slot extraction.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import pytest
from fakeredis import FakeServer, FakeStrictRedis

from app.infrastructure import metrics as metrics_module
from app.infrastructure.logging_config import JsonFormatter


def _noop_task(audio_ref: str, model_id: str) -> str:
    """Module-level so RQ can reference it by import path."""
    return "ok"


def _boom_task(audio_ref: str, model_id: str) -> str:
    """Module-level so RQ can reference it by import path."""
    raise RuntimeError("inference exploded")


@pytest.fixture
def broker(monkeypatch):
    """Install a fake broker as *the* shared sync connection (RQ side)."""
    from app.infrastructure import rq_connection

    from app.orchestration import task_orchestrator as to

    fake = FakeStrictRedis(server=FakeServer())
    monkeypatch.setattr(rq_connection, "_CONNECTION", fake)
    to.reset_queue_cache()
    yield fake
    to.reset_queue_cache()
    fake.flushall()


def _decoded(hash_map: dict) -> dict:
    """Normalise a sync ``hgetall`` (bytes keys/values) to int values."""
    return {k.decode() if isinstance(k, bytes) else k: int(v) for k, v in (hash_map or {}).items()}


class TestJsonFormatter:
    def _record(self, **extra) -> logging.LogRecord:
        record = logging.LogRecord(
            "audiolit.metrics.test", logging.INFO, __file__, 1, "task.success", (), None
        )
        for key, value in extra.items():
            setattr(record, key, value)
        return record

    def test_emits_one_json_object_with_expected_fields(self):
        parsed = json.loads(JsonFormatter().format(
            self._record(job_id="job-1", family="asr", duration_s=0.123)
        ))
        assert {"ts", "level", "logger", "event"} <= set(parsed)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "audiolit.metrics.test"
        assert parsed["event"] == "task.success"
        assert parsed["job_id"] == "job-1"
        assert parsed["duration_s"] == 0.123

    def test_round_trip_preserves_extra_values(self):
        parsed = json.loads(JsonFormatter().format(
            self._record(job_id="job-9", family="asr", queue="asr", model_id="openai/whisper-base")
        ))
        assert parsed["model_id"] == "openai/whisper-base"
        assert parsed["queue"] == "asr"

    def test_reserved_attributes_not_dumped_as_extra(self):
        parsed = json.loads(JsonFormatter().format(
            self._record(pathname="/x", lineno=42, funcName="f", msg="task.success")
        ))
        for key in ("pathname", "lineno", "funcName", "msg", "args", "created"):
            assert key not in parsed

    def test_exc_info_rendered_under_exc(self):
        try:
            raise ValueError("boom")
        except ValueError:
            etype, evalue, tb = sys.exc_info()
        record = logging.LogRecord(
            "audiolit.metrics.test", logging.ERROR, __file__, 1, "task.failure", (), None
        )
        record.exc_info = (etype, evalue, tb)
        parsed = json.loads(JsonFormatter().format(record))
        assert parsed["level"] == "ERROR"
        assert "ValueError: boom" in parsed["exc"]


class TestWorkerMetrics:
    def test_success_records_counter_duration_and_safe_structured_log(
        self, broker, caplog
    ):
        from app.orchestration.task_orchestrator import (
            WorkerContext, WorkerFamily, get_queue, make_worker,
        )

        q = get_queue(WorkerFamily.ASR)
        # Real task slottage: position 0 is the audio ref, position 1 the model.
        job = q.enqueue(_noop_task, "uploads/secret-audio.wav", "whisper-base")

        w = make_worker(WorkerFamily.ASR, connection=broker)
        w._ctx = WorkerContext(family=WorkerFamily.ASR)
        w._ctx.load_libraries = lambda: None  # keep the test off torch/captum

        with caplog.at_level(logging.INFO, logger="audiolit.orchestration"):
            w.perform_job(job, q)

        # RQ's perform_job stores the func's return on the Job, not on the
        # worker's return value (which is merely True/False).
        assert job.result == "ok"

        tasks = _decoded(broker.hgetall(metrics_module.TASKS_HASH))
        assert tasks["asr:processing"] == 1
        assert tasks["asr:success"] == 1
        assert metrics_module.tasks_summary(tasks)["total"] == 2
        assert _decoded(broker.hgetall(metrics_module.DUR_COUNT_HASH)) == {"asr": 1}
        assert _decoded(broker.hgetall(metrics_module.DUR_SUM_MS_HASH))["asr"] >= 0

        events = [r for r in caplog.records if r.getMessage() == "task.success"]
        assert events, "expected a task.success structured event"
        rec = events[0]
        assert rec.family == "asr"
        assert rec.queue == "asr"
        assert rec.job_id == job.id
        assert rec.worker == os.getpid()
        assert "duration_s" in rec.__dict__
        assert "task.processing" in [r.getMessage() for r in caplog.records]

        # SR6: no audio ref / filename ever enters the structured logs.
        joined = " ".join(
            str(getattr(r, key, ""))
            for r in caplog.records
            for key in ("msg", "job_id", "family", "queue", "model_id")
        )
        assert "uploads/secret-audio.wav" not in joined
        assert "audio_ref" not in joined

    def test_failure_records_failed_counter(self, broker, caplog):
        from app.orchestration.task_orchestrator import (
            WorkerContext, WorkerFamily, get_queue, make_worker,
        )

        q = get_queue(WorkerFamily.SER)
        job = q.enqueue(_boom_task, "uploads/other.wav", "some-model")

        w = make_worker(WorkerFamily.SER, connection=broker)
        w._ctx = WorkerContext(family=WorkerFamily.SER)
        w._ctx.load_libraries = lambda: None

        with caplog.at_level(logging.INFO, logger="audiolit.orchestration"):
            # RQ's perform_job swallows the job-func exception and returns False.
            assert w.perform_job(job, q) is False

        tasks = _decoded(broker.hgetall(metrics_module.TASKS_HASH))
        assert tasks["ser:processing"] == 1
        assert tasks["ser:failed"] == 1

        failures = [r for r in caplog.records if r.getMessage() == "task.failure"]
        assert failures and "error" in failures[0].__dict__
        assert "uploads/other.wav" not in failures[0].error

    def test_model_id_reads_only_the_mapped_slot(self):
        from app.orchestration.task_orchestrator import _task_model_id

        class _Job:
            func_name = ""
            args: tuple = ()

        job = _Job()
        job.func_name = "app.orchestration.task_orchestrator.asr_task"
        job.args = ("uploads/x.wav", "openai/whisper-base", {})
        assert _task_model_id(job) == "openai/whisper-base"

        job = _Job()
        job.func_name = "app.orchestration.task_orchestrator.mutation_task"
        job.args = ("uploads/x.wav", {"perturbations": []})
        assert _task_model_id(job) is None

        job = _Job()
        job.func_name = "app.orchestration.task_orchestrator.asr_task"
        job.args = ("uploads/x.wav",)  # malformed: no model slot present
        assert _task_model_id(job) is None


class TestCacheCounters:
    async def test_async_lookup_counts_hits_and_misses(self):
        from app.infrastructure import redis as redis_module
        from app.infrastructure.redis import get_result

        assert await get_result("whisper-base", "k1") is None
        await redis_module.redis.set("result:whisper-base:k1", '{"x": 1}')
        assert await get_result("whisper-base", "k1") == {"x": 1}

        counts = await redis_module.redis.hgetall(metrics_module.CACHE_HASH)
        assert counts.get("misses") == "1"
        assert counts.get("hits") == "1"

    def test_sync_lookup_counts_hits_and_misses(self, broker):
        from app.infrastructure.redis import cache_result_sync, get_result_sync

        assert get_result_sync("asr", "k2") is None
        cache_result_sync("asr", "k2", {"transcript": "hello"})
        assert get_result_sync("asr", "k2") == {"transcript": "hello"}

        counts = _decoded(broker.hgetall(metrics_module.CACHE_HASH))
        assert counts == {"hits": 1, "misses": 1}
        assert metrics_module.hit_ratio(1, 1) == 0.5


class TestMetricsEndpoint:
    async def test_returns_five_keys_with_zero_hit_ratio(self, client, monkeypatch):
        from app.infrastructure import rq_connection

        from app.orchestration import task_orchestrator as to

        # Give the RQ side a fake broker too, so health_check()/queue counts do
        # not try to reach a real Redis during the request.
        sync_fake = FakeStrictRedis(server=FakeServer())
        monkeypatch.setattr(rq_connection, "_CONNECTION", sync_fake)
        to.reset_queue_cache()

        resp = await client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {"tasks", "durations", "queues", "cache", "gpu"}
        assert body["cache"] == {"hits": 0, "misses": 0, "hit_ratio": 0.0}
        assert body["tasks"] == {"total": 0}
        assert body["durations"] == {}
        assert set(body["queues"]) == {"asr", "ser", "add", "xai", "mutation"}
        assert isinstance(body["gpu"]["cuda_available"], bool)

    async def test_metrics_reflects_a_recorded_counter(self, client, monkeypatch):
        from app.infrastructure import rq_connection

        from app.orchestration import task_orchestrator as to

        from app.infrastructure import redis as redis_module

        sync_fake = FakeStrictRedis(server=FakeServer())
        monkeypatch.setattr(rq_connection, "_CONNECTION", sync_fake)
        to.reset_queue_cache()

        await redis_module.redis.hincrby(metrics_module.TASKS_HASH, "asr:success", 1)
        await redis_module.redis.hincrby(metrics_module.CACHE_HASH, "misses", 2)

        resp = await client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tasks"]["asr:success"] == 1
        assert body["cache"] == {"hits": 0, "misses": 2, "hit_ratio": 0.0}