"""Warmup liveness, reconciliation and cancellation.

Regression cover for a warmup whose worker died under it (containers
recreated mid-run): its progress record stayed "running" for the full 24h
TTL, the Cancel button set a flag nothing was left to read while the API
answered "cancelled" anyway, and the UI reattached to the ghost on every
reload.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest
from fakeredis import FakeServer, FakeStrictRedis
from rq import Queue
from rq.job import Job, JobStatus

from app.api.routes import inference as inference_routes
from app.infrastructure import rq_connection
from app.orchestration import task_orchestrator
from app.orchestration.task_orchestrator import (
    WARMUP_HEARTBEAT_TTL,
    WARMUP_PROGRESS_TTL,
    WARMUP_STALE_SECONDS,
    _WarmupHeartbeat,
    cancel_warmup,
    reconcile_warmup_progress,
    warmup_heartbeat_key,
    warmup_liveness,
    warmup_progress_key,
)


@pytest.fixture
def broker(monkeypatch):
    """Install a fake broker as *the* shared connection (see test_task_orchestrator)."""
    fake = FakeStrictRedis(server=FakeServer())
    monkeypatch.setattr(rq_connection, "_CONNECTION", fake)
    task_orchestrator.reset_queue_cache()
    yield fake
    task_orchestrator.reset_queue_cache()
    fake.flushall()


def _noop() -> None:
    return None


def _progress(conn, job_id: str, *, ttl: int = WARMUP_PROGRESS_TTL, **fields) -> dict:
    data = {"completed": 4, "total": 100, "status": "running", "percent": 4.0, **fields}
    conn.set(warmup_progress_key(job_id), json.dumps(data), ex=ttl)
    return data


def _stored(conn, job_id: str) -> dict:
    return json.loads(conn.get(warmup_progress_key(job_id)))


def _started_job(conn, job_id: str, *, started_ago: float) -> Job:
    job = Queue("asr", connection=conn).enqueue(_noop, job_id=job_id)
    job.started_at = datetime.now(timezone.utc) - timedelta(seconds=started_ago)
    job.set_status(JobStatus.STARTED)
    job.save()
    return job


def _beat(conn, job_id: str) -> None:
    conn.set(warmup_heartbeat_key(job_id), "1", ex=WARMUP_HEARTBEAT_TTL)


class TestLiveness:
    def test_queued_rq_job(self, broker):
        Queue("asr", connection=broker).enqueue(_noop, job_id="warmup_q")
        assert warmup_liveness(broker, "warmup_q", {}) == "queued"

    def test_queued_job_missing_from_its_queue_is_dead(self, broker):
        # What allkeys-lru eviction left behind: status says QUEUED, but the
        # queue list that would deliver it to a worker is gone.
        Queue("asr", connection=broker).enqueue(_noop, job_id="warmup_lost")
        broker.delete("rq:queue:asr")
        assert warmup_liveness(broker, "warmup_lost", {}) == "dead"

    def test_heartbeating_run_is_alive(self, broker):
        _started_job(broker, "warmup_live", started_ago=3600)
        _beat(broker, "warmup_live")
        assert warmup_liveness(broker, "warmup_live", {}) == "alive"

    def test_started_job_without_heartbeat_is_dead(self, broker):
        # The OOM-kill case: RQ still says "started" (its registry entry lives
        # for the 24h job timeout), but the runner's heartbeat has expired.
        _started_job(broker, "warmup_ghost", started_ago=3600)
        assert warmup_liveness(broker, "warmup_ghost", {}) == "dead"

    def test_just_started_job_gets_a_grace_period_before_its_first_beat(self, broker):
        _started_job(broker, "warmup_new", started_ago=1)
        assert warmup_liveness(broker, "warmup_new", {}) == "alive"

    def test_no_rq_job_uses_the_heartbeat(self, broker):
        assert warmup_liveness(broker, "w", {"updated_at": time.time()}) == "alive"
        stale = time.time() - WARMUP_STALE_SECONDS - 1
        assert warmup_liveness(broker, "w", {"updated_at": stale}) == "dead"

    def test_legacy_record_without_heartbeat_is_aged_from_its_ttl(self, broker):
        # Written ~10h ago by a runner that predates `updated_at`.
        _progress(broker, "warmup_old", ttl=WARMUP_PROGRESS_TTL - 10 * 3600)
        assert warmup_liveness(broker, "warmup_old", _stored(broker, "warmup_old")) == "dead"


class TestReconcile:
    def test_orphaned_run_becomes_interrupted(self, broker):
        data = _progress(broker, "warmup_old", ttl=WARMUP_PROGRESS_TTL - 10 * 3600)

        result = reconcile_warmup_progress(broker, "warmup_old", data)

        assert result["status"] == "interrupted"
        assert result["completed"] == 4
        assert _stored(broker, "warmup_old")["status"] == "interrupted"

    def test_orphaned_run_with_a_cancel_request_becomes_cancelled(self, broker):
        data = _progress(broker, "warmup_old", ttl=WARMUP_PROGRESS_TTL - 10 * 3600)
        broker.set("cancel_job_warmup_old", "1")

        assert reconcile_warmup_progress(broker, "warmup_old", data)["status"] == "cancelled"

    def test_live_and_terminal_runs_are_left_alone(self, broker):
        live = _progress(broker, "warmup_live", updated_at=time.time())
        assert reconcile_warmup_progress(broker, "warmup_live", live) == live
        done = {"status": "completed", "completed": 100, "total": 100}
        assert reconcile_warmup_progress(broker, "warmup_done", done) == done


class TestCancel:
    def test_orphaned_run_is_cancelled_immediately(self, broker):
        _progress(broker, "warmup_old", ttl=WARMUP_PROGRESS_TTL - 10 * 3600)

        result = cancel_warmup(broker, "warmup_old")

        assert result["status"] == "cancelled"
        assert _stored(broker, "warmup_old")["status"] == "cancelled"

    def test_queued_run_is_cancelled_in_rq_too(self, broker):
        Queue("asr", connection=broker).enqueue(_noop, job_id="warmup_q")
        _progress(broker, "warmup_q", completed=0, updated_at=time.time())

        assert cancel_warmup(broker, "warmup_q")["status"] == "cancelled"
        assert Job.fetch("warmup_q", connection=broker).get_status() == JobStatus.CANCELED

    def test_live_run_moves_to_cancelling_and_is_flagged(self, broker):
        _started_job(broker, "warmup_live", started_ago=600)
        _beat(broker, "warmup_live")
        _progress(broker, "warmup_live", updated_at=time.time())

        result = cancel_warmup(broker, "warmup_live")

        assert result["status"] == "cancelling"
        assert _stored(broker, "warmup_live")["status"] == "cancelling"
        assert broker.get("cancel_job_warmup_live") == b"1"

    def test_unknown_job(self, broker):
        assert cancel_warmup(broker, "warmup_nope")["status"] == "not_found"


class TestHeartbeat:
    def test_beats_while_running_and_clears_on_exit(self, broker):
        with _WarmupHeartbeat(broker, "warmup_hb"):
            assert broker.exists(warmup_heartbeat_key("warmup_hb"))
            assert 0 < broker.ttl(warmup_heartbeat_key("warmup_hb")) <= WARMUP_HEARTBEAT_TTL
        assert not broker.exists(warmup_heartbeat_key("warmup_hb"))

    def test_no_broker_is_a_no_op(self):
        with _WarmupHeartbeat(None, "warmup_hb"):
            pass


class TestRoutes:
    async def test_active_list_drops_orphaned_runs(self, broker):
        _progress(broker, "warmup_old", ttl=WARMUP_PROGRESS_TTL - 10 * 3600)
        _progress(broker, "warmup_live", updated_at=time.time())

        result = await inference_routes.list_active_warmups()

        assert [j["job_id"] for j in result["jobs"]] == ["warmup_live"]
        assert _stored(broker, "warmup_old")["status"] == "interrupted"

    async def test_progress_reports_orphaned_run_as_interrupted(self, broker):
        _progress(broker, "warmup_old", ttl=WARMUP_PROGRESS_TTL - 10 * 3600)
        assert (await inference_routes.get_job_progress("warmup_old"))["status"] == "interrupted"

    async def test_cancel_route_reports_the_real_state(self, broker):
        _progress(broker, "warmup_old", ttl=WARMUP_PROGRESS_TTL - 10 * 3600)
        assert (await inference_routes.cancel_batch_job("warmup_old"))["status"] == "cancelled"
