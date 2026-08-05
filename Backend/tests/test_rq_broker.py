"""Tests for the RQ task broker (LIT-127, FR3).

Sync fakeredis, no worker execution: these assert queue/config wiring and the
progress channel directly, so there is no SimpleWorker/burst draining (and thus
none of the fan-in flakiness those can cause).
"""

from __future__ import annotations

import json

import pytest
from fakeredis import FakeServer, FakeStrictRedis
from rq import Queue, Worker

from app.orchestration import rq_broker
from app.orchestration import worker as worker_entry
from app.orchestration.rq_broker import (
    QUEUE_CONFIGS,
    WorkerFamily,
    enqueue,
    get_queue,
    get_queue_config,
    make_worker,
    progress_channel,
    publish_progress,
    worker_queue_names,
)


def _noop_job() -> str:  # module-level so RQ can reference it by import path
    return "ok"


@pytest.fixture
def conn():
    c = FakeStrictRedis(server=FakeServer())
    yield c
    c.flushall()


class TestQueueConfig:
    def test_five_families_registered(self):
        assert set(QUEUE_CONFIGS) == {
            WorkerFamily.ASR,
            WorkerFamily.SER,
            WorkerFamily.ADD,
            WorkerFamily.XAI,
            WorkerFamily.MUTATION,
        }

    def test_gpu_families_pinned_to_concurrency_one(self):
        # SAD C2: no two model jobs may contend for VRAM at once.
        for fam in (WorkerFamily.ASR, WorkerFamily.SER, WorkerFamily.ADD, WorkerFamily.XAI):
            cfg = QUEUE_CONFIGS[fam]
            assert cfg.gpu_bound is True
            assert cfg.concurrency == 1

    def test_mutation_is_cpu_only_and_may_scale(self):
        cfg = QUEUE_CONFIGS[WorkerFamily.MUTATION]
        assert cfg.gpu_bound is False
        assert cfg.concurrency >= 1

    def test_get_config_accepts_enum_and_string(self):
        assert get_queue_config("asr") is get_queue_config(WorkerFamily.ASR)

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError):
            get_queue_config("gpu-mining")


class TestQueuesAndEnqueue:
    def test_get_queue_uses_family_queue_name(self, conn):
        q = get_queue(WorkerFamily.SER, connection=conn)
        assert isinstance(q, Queue)
        assert q.name == "ser"

    def test_enqueue_places_job_on_the_right_queue(self, conn):
        job = enqueue(WorkerFamily.ASR, _noop_job, connection=conn)
        q = get_queue(WorkerFamily.ASR, connection=conn)
        assert len(q) == 1
        assert q.job_ids == [job.id]
        assert job.func_name.endswith("_noop_job")

    def test_enqueue_does_not_leak_across_families(self, conn):
        enqueue(WorkerFamily.ASR, _noop_job, connection=conn)
        assert len(get_queue(WorkerFamily.SER, connection=conn)) == 0


class TestProgressChannel:
    def test_channel_is_keyed_by_job_id(self):
        assert progress_channel("job-123") == "progress:job-123"

    def test_publish_progress_delivers_payload(self, conn):
        pubsub = conn.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(progress_channel("job-1"))

        subscribers = publish_progress(conn, "job-1", "asr", 0.5)
        assert subscribers == 1

        msg = None
        for _ in range(10):
            m = pubsub.get_message(timeout=0.5)
            if m and m.get("type") == "message":
                msg = m
                break
        assert msg is not None
        assert json.loads(msg["data"]) == {"task_name": "asr", "progress": 0.5}


class TestWorkers:
    def test_worker_queue_names_defaults_to_all_families(self):
        assert worker_queue_names() == ["asr", "ser", "add", "xai", "mutation"]

    def test_worker_queue_names_for_subset(self):
        assert worker_queue_names([WorkerFamily.ADD]) == ["add"]

    def test_make_worker_binds_to_family_queue(self, conn):
        w = make_worker(WorkerFamily.XAI, connection=conn)
        assert isinstance(w, Worker)
        assert w.queue_names() == ["xai"]


class TestWorkerEntrypoint:
    def test_no_argument_is_usage_error(self):
        assert worker_entry.main([]) == 2

    def test_unknown_family_is_usage_error(self):
        assert worker_entry.main(["gpu-mining"]) == 2

    def test_valid_family_starts_worker(self, monkeypatch):
        started = {}

        class _FakeWorker:
            def work(self):
                started["ran"] = True

        monkeypatch.setattr(worker_entry, "make_worker", lambda fam: _FakeWorker())
        assert worker_entry.main(["asr"]) == 0
        assert started["ran"] is True
