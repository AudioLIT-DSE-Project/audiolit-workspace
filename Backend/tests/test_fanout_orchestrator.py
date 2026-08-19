"""
Tests for the LIT-225 RQ fan-out/fan-in prototype
(app.orchestration.fanout_orchestrator_service).

Success/failure/progress cases run against fakeredis - fast, no external
service. The killed-worker recovery case needs a *real* Redis instance,
because it forks an actual OS process to run the job and kills it mid-job;
it skips cleanly if Redis isn't reachable, which is the expected outcome in
CI (no Redis service is provisioned there - see the note in the LIT-225 PR
about why one wasn't added). Run `docker-compose up -d redis` from
`Backend/` to exercise it locally.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import time

import pytest
from fakeredis import FakeServer, FakeStrictRedis
from redis import Redis
from redis.exceptions import RedisError
from rq import Queue, SimpleWorker, Worker
from rq.job import Job, JobStatus

from app.orchestration import fanout_orchestrator_service as fanout

# DB 15, not DB 0. This test calls flushdb(), and DB 0 is where the running
# application keeps its dataset-warmup cache - pointing them at the same place
# meant a plain `pytest` run silently destroyed hours of warmed inference.
# Override with TEST_REDIS_URL, but never at the app's own REDIS_URL.
REAL_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")


def _drain(conn, *queue_names) -> None:
    queues = [Queue(name, connection=conn) for name in queue_names]
    SimpleWorker(queues, connection=conn).work(burst=True)


@pytest.fixture
def fake_conn(monkeypatch):
    conn = FakeStrictRedis(server=FakeServer())
    monkeypatch.setattr(fanout, "get_redis_connection", lambda: conn)
    yield conn
    conn.flushall()


class TestFanoutSuccess:
    def test_aggregator_deferred_until_children_finish(self, fake_conn):
        result = fanout.enqueue_fanout({"asr": {"a": 1}, "ser": {"b": 2}, "add": {"c": 3}})
        agg_job = Job.fetch(result["aggregator_job_id"], connection=fake_conn)
        assert agg_job.get_status() == JobStatus.DEFERRED

    def test_aggregates_three_children_once_on_success(self, fake_conn):
        result = fanout.enqueue_fanout({"asr": {"a": 1}, "ser": {"b": 2}, "add": {"c": 3}})

        _drain(fake_conn, *fanout.CHILD_QUEUE_NAMES)

        # Aggregate directly over the finished child jobs. Running the aggregator
        # as a *dependent* RQ job under SimpleWorker(burst) races RQ's
        # dependency-completion callback on fakeredis and can intermittently hang
        # the burst worker (this is what stalled CI for ~40 min; same root cause
        # as the LIT-150 fix). The fan-in logic is what we're asserting, so call
        # it on the child ids rather than through the flaky aggregator drain.
        agg_result = fanout.aggregate_children(result["child_job_ids"])

        assert set(agg_result["succeeded"]) == set(result["child_job_ids"])
        assert agg_result["failed"] == {}

        task_names = {r["task_name"] for r in agg_result["succeeded"].values()}
        assert task_names == {"asr", "ser", "add"}


class TestFanoutFailure:
    def test_degrades_gracefully_on_one_child_failure(self, fake_conn, monkeypatch):
        # No retry here: CHILD_JOB_RETRY exists for real forking-Worker crash
        # recovery (see test_recovers_a_killed_job); under SimpleWorker+fakeredis
        # a retried failure can hang the burst drain, and this test is about fan-in
        # degradation (allow_failure), not retry. Patch it off so 'ser' fails once.
        monkeypatch.setattr(fanout, "CHILD_JOB_RETRY", None)

        result = fanout.enqueue_fanout(
            {"asr": {}, "ser": {}, "add": {}},
            fail_tasks={"ser"},
        )

        _drain(fake_conn, *fanout.CHILD_QUEUE_NAMES)

        # Aggregate directly (see the success test) — the aggregator still
        # combines the two survivors even though one child failed, which is what
        # `allow_failure=True` on the Dependency buys us.
        agg_result = fanout.aggregate_children(result["child_job_ids"])
        assert len(agg_result["succeeded"]) == 2
        assert len(agg_result["failed"]) == 1

        (typed_error,) = agg_result["failed"].values()
        assert "stub failure requested for task 'ser'" in typed_error


class TestFanoutProgress:
    def test_progress_relayed_over_pubsub(self, fake_conn):
        result = fanout.enqueue_fanout({"asr": {}, "ser": {}, "add": {}})
        child_id = result["child_job_ids"][0]

        pubsub = fake_conn.pubsub()
        pubsub.subscribe(f"{fanout.PROGRESS_CHANNEL_PREFIX}{child_id}")
        pubsub.get_message(timeout=1)  # subscribe confirmation

        _drain(fake_conn, *fanout.CHILD_QUEUE_NAMES)

        fractions = []
        for _ in range(5):
            msg = pubsub.get_message(timeout=1)
            if msg and msg["type"] == "message":
                fractions.append(json.loads(msg["data"])["progress"])

        assert 0.5 in fractions
        assert 1.0 in fractions


def _assert_not_the_application_database(conn: Redis) -> None:
    """Refuse to flush the database the application itself is using.

    A guard, not a formality: this test used to default to DB 0 - the app's
    own cache - so running the suite wiped every warmed dataset entry.
    """
    from app.infrastructure.settings import settings

    app_conn = Redis.from_url(settings.REDIS_URL)
    app_db = app_conn.connection_pool.connection_kwargs.get("db", 0)
    test_db = conn.connection_pool.connection_kwargs.get("db", 0)
    app_host = app_conn.connection_pool.connection_kwargs.get("host")
    test_host = conn.connection_pool.connection_kwargs.get("host")
    if (app_host, app_db) == (test_host, test_db):
        pytest.skip(
            f"refusing to flushdb() on the application's own Redis database "
            f"({settings.REDIS_URL}) - that would destroy the dataset-warmup "
            f"cache. Set TEST_REDIS_URL to a different db index."
        )


def _real_redis_or_skip() -> Redis:
    conn = Redis.from_url(REAL_REDIS_URL)
    try:
        conn.ping()
    except RedisError:
        pytest.skip(
            f"no reachable Redis at {REAL_REDIS_URL} - start it with "
            "`docker-compose up -d redis` (Backend/docker-compose.yml) to "
            "run the killed-worker recovery test locally"
        )
    _assert_not_the_application_database(conn)
    conn.flushdb()
    return conn


def _run_burst_worker(redis_url: str, queue_name: str) -> None:
    conn = Redis.from_url(redis_url)
    fanout.get_redis_connection = lambda: conn
    queue = Queue(queue_name, connection=conn)
    Worker([queue], connection=conn).work(burst=True)


class TestFanoutRecovery:
    def test_recovers_a_killed_job(self, monkeypatch):
        """
        Kills the *work-horse* (the OS process RQ forks to run one job)
        mid-execution, while its parent Worker process is still alive and
        watching it - the realistic "a worker died mid-job" case. No custom
        recovery code is involved: RQ's own Worker notices the work-horse
        died (waitpid returns a signal), marks the job failed, and - because
        it was enqueued with `retry=CHILD_JOB_RETRY` - immediately requeues
        it, so the *same* burst run picks it back up and finishes it.
        """
        conn = _real_redis_or_skip()
        monkeypatch.setattr(fanout, "get_redis_connection", lambda: conn)

        queue_name = fanout.CHILD_QUEUE_NAMES[0]
        queue = Queue(queue_name, connection=conn)
        job = queue.enqueue(
            fanout.run_stub_child,
            task_name="asr",
            payload={},
            work_seconds=2,
            retry=fanout.CHILD_JOB_RETRY,
        )

        if "fork" not in multiprocessing.get_all_start_methods():
            pytest.skip(
                "needs a forking start method - RQ's Worker forks a work-horse "
                "process that this test kills mid-job; unavailable on Windows"
            )
        ctx = multiprocessing.get_context("fork")
        proc = ctx.Process(target=_run_burst_worker, args=(REAL_REDIS_URL, queue_name))
        proc.start()
        try:
            horse_pid = None
            for _ in range(100):
                job.refresh()
                horse_pid = job.meta.get("pid")
                if horse_pid:
                    break
                time.sleep(0.1)
            assert horse_pid, "work-horse never reported its pid - job never started"

            os.kill(horse_pid, signal.SIGKILL)
            proc.join(timeout=20)
            assert not proc.is_alive(), "worker process did not finish within timeout"
        finally:
            if proc.is_alive():
                proc.terminate()

        job.refresh()
        assert job.get_status() == JobStatus.FINISHED
        assert job.retries_left == 0  # the one retry was consumed by the kill
        assert job.return_value() == {"task_name": "asr", "payload": {}, "status": "ok"}
