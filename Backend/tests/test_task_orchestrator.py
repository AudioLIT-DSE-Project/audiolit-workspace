"""Tests for the consolidated Task Orchestrator (LIT-230; FR3, SAD §5.1/§5.2/§6.1).

Merges the two suites that shipped with the two duplicate implementations -
`test_rq_broker.py` (LIT-127) and `test_queue_service.py` (LIT-149/157) - now
that there is a single module.

Sync fakeredis throughout. Queue/config wiring and the progress channel are
asserted directly rather than by draining a worker, so there is no burst/fan-in
flakiness.
"""

from __future__ import annotations

import json
import sys

import pytest
from fakeredis import FakeServer, FakeStrictRedis
from rq import Queue, SimpleWorker, Worker
from rq.job import Job

from app.infrastructure import rq_connection
from app.orchestration import task_orchestrator
from app.orchestration import worker as worker_entry
from app.orchestration.task_orchestrator import (
    QUEUE_CONFIGS,
    AudioLITWorker,
    WorkerFamily,
    accent_bias_task,
    enqueue,
    enqueue_accent_bias,
    enqueue_multitask_analysis,
    get_queue,
    get_queue_config,
    health_check,
    make_worker,
    mutation_task,
    progress_channel,
    publish_progress,
    run_worker,
    worker_queue_names,
)


def _noop_job() -> str:  # module-level so RQ can reference it by import path
    return "ok"


@pytest.fixture
def conn():
    """A bare fake broker, passed explicitly where the API takes a connection."""
    c = FakeStrictRedis(server=FakeServer())
    yield c
    c.flushall()


@pytest.fixture
def broker(monkeypatch):
    """Install a fake broker as *the* shared connection.

    Patches the module the connection actually lives in (infrastructure) rather
    than binding a name into the orchestrator - see LIT-229 on why importing the
    module beats importing the name.
    """
    fake = FakeStrictRedis(server=FakeServer())
    monkeypatch.setattr(rq_connection, "_CONNECTION", fake)
    task_orchestrator.reset_queue_cache()
    yield fake
    task_orchestrator.reset_queue_cache()
    fake.flushall()


@pytest.fixture
def stub_worker(monkeypatch):
    """Replace the worker so `run_worker` exercises locking without consuming jobs."""
    captured: dict = {}

    class _StubWorker:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def work(self, burst=False, with_scheduler=False):
            return True

    monkeypatch.setattr(task_orchestrator, "AudioLITWorker", _StubWorker)
    return captured


class TestProjectRules:
    def test_no_celery_imported(self):
        # RQ + Redis only; Celery is removed project-wide.
        assert "celery" not in sys.modules


class TestQueueConfig:
    def test_five_families_registered(self):
        assert set(QUEUE_CONFIGS) == {
            WorkerFamily.ASR,
            WorkerFamily.SER,
            WorkerFamily.ADD,
            WorkerFamily.XAI,
            WorkerFamily.MUTATION,
        }

    def test_all_five_families_defined(self):
        assert {f.value for f in WorkerFamily} == {"asr", "ser", "add", "xai", "mutation"}

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

    def test_task_family_is_the_same_enum_as_worker_family(self):
        # The routes speak "task family", the workers "worker family" - one enum,
        # so a route and a queue can never disagree about what "asr" means.
        assert task_orchestrator.TaskFamily is WorkerFamily


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

    def test_shared_queue_is_cached_per_family(self, broker):
        assert get_queue(WorkerFamily.ASR) is get_queue(WorkerFamily.ASR)

    def test_explicit_connection_bypasses_the_cache(self, broker, conn):
        assert get_queue(WorkerFamily.ASR, connection=conn) is not get_queue(WorkerFamily.ASR)


class TestMultiTaskFanOut:
    def test_creates_per_family_jobs_and_a_dependent_aggregator(self, broker):
        result = enqueue_multitask_analysis(
            "audio://sha256/abc",
            tasks=[WorkerFamily.ASR, WorkerFamily.SER, WorkerFamily.ADD],
            cache_key="sha256:deadbeef",
        )
        assert set(result.family_jobs) == {"asr", "ser", "add"}
        assert result.job_id

        aggregator = Job.fetch(result.job_id, connection=broker)
        assert set(aggregator.dependency_ids) == set(result.family_jobs.values())

    def test_websocket_url_matches_the_tasks_route(self, broker):
        # The URL handed to the client must be the route tasks.py actually serves;
        # they were previously "/ws/jobs" vs "/api/ws/tasks".
        result = enqueue_multitask_analysis("audio://sha256/abc", tasks=[WorkerFamily.ASR])
        assert result.websocket_url == f"/api/ws/tasks/{result.job_id}"


class TestProgressChannel:
    def test_channel_is_keyed_by_job_id(self):
        assert progress_channel("job-123") == "audiolit:progress:job-123"

    def test_publish_progress_delivers_payload(self, conn):
        pubsub = conn.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(progress_channel("job-1"))

        subscribers = publish_progress(
            "job-1", "PROCESSING", {"family": "asr"}, connection=conn
        )
        assert subscribers == 1

        msg = None
        for _ in range(10):
            m = pubsub.get_message(timeout=0.5)
            if m and m.get("type") == "message":
                msg = m
                break
        assert msg is not None

        event = json.loads(msg["data"])
        assert event["job_id"] == "job-1"
        assert event["stage"] == "PROCESSING"
        assert event["payload"] == {"family": "asr"}

    def test_one_channel_format_for_publisher_and_subscriber(self, conn):
        # The two merged modules disagreed ("progress:" vs "audiolit:progress"), so
        # a job published by one was invisible to a subscriber on the other.
        pubsub = conn.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(progress_channel("job-x"))
        assert publish_progress("job-x", "SUCCESS", connection=conn) == 1


class TestWorkers:
    def test_worker_queue_names_defaults_to_all_families(self):
        assert worker_queue_names() == ["asr", "ser", "add", "xai", "mutation"]

    def test_worker_queue_names_for_subset(self):
        assert worker_queue_names([WorkerFamily.ADD]) == ["add"]

    def test_make_worker_binds_to_family_queue(self, conn):
        w = make_worker(WorkerFamily.XAI, connection=conn)
        assert isinstance(w, AudioLITWorker)
        assert w.queue_names() == ["xai"]

    def test_worker_is_in_process_not_forking(self, conn):
        """SAD §10 budgets a fresh multi-task analysis at ~8s, and ~60s just to
        prepare a model - so the model cache in WorkerContext has to survive
        between jobs. RQ's forking Worker discards it with the work-horse child.

        If this fails because someone switched to the forking Worker, re-read
        §10 before changing the assertion.
        """
        w = make_worker(WorkerFamily.ASR, connection=conn)
        assert isinstance(w, SimpleWorker)
        assert not isinstance(w, Worker)


class TestGpuWorkerLock:
    def test_lock_is_released_after_the_worker_exits(self, broker, stub_worker):
        assert not broker.exists("audiolit:worker-lock:asr")
        run_worker(WorkerFamily.ASR)
        assert not broker.exists("audiolit:worker-lock:asr")

    def test_second_worker_for_a_gpu_family_is_refused(self, broker, stub_worker):
        # SAD C2: a second ASR worker would double the VRAM footprint.
        broker.set("audiolit:worker-lock:asr", "locked")
        with pytest.raises(RuntimeError):
            run_worker(WorkerFamily.ASR)

    def test_cpu_only_mutation_family_takes_no_lock(self, broker, stub_worker):
        run_worker(WorkerFamily.MUTATION)
        assert not broker.exists("audiolit:worker-lock:mutation")


class TestHealthCheck:
    def test_reports_ok_and_every_queue(self, broker):
        health = health_check()
        assert health["ok"] is True
        assert health["broker"] == "redis"
        assert set(health["queues"]) == {"asr", "ser", "add", "xai", "mutation"}

    def test_reports_broker_failure(self, monkeypatch):
        from redis.exceptions import ConnectionError as RedisConnectionError

        def _fail():
            raise RedisConnectionError("boom")

        monkeypatch.setattr(task_orchestrator, "get_redis_connection", _fail)
        assert health_check()["ok"] is False


class TestMutationTask:
    """LIT-164/LIT-231: `mutation_task` was a `_scaffold` stub that never
    touched the audio - it now delegates to `perturbation_service.perturb_and_save`,
    the same function the synchronous `POST /perturb` route already uses.
    """

    def test_delegates_to_perturb_and_save(self, monkeypatch):
        from app.domain import perturbation_service

        captured = {}

        def _fake_perturb_and_save(**kwargs):
            captured.update(kwargs)
            return {"success": True, "perturbed_file": "uploads/x_perturbed.wav"}

        monkeypatch.setattr(perturbation_service, "perturb_and_save", _fake_perturb_and_save)

        result = mutation_task(
            "uploads/original.wav",
            {"perturbations": [{"type": "noise", "params": {"noise_level": 0.1}}], "dataset": None},
        )

        assert result == {"success": True, "perturbed_file": "uploads/x_perturbed.wav"}
        assert captured["file_path"] == "uploads/original.wav"
        assert captured["perturbations"] == [{"type": "noise", "params": {"noise_level": 0.1}}]
        assert captured["dataset"] is None

    def test_no_longer_returns_scaffold_flag(self, monkeypatch):
        from app.domain import perturbation_service

        monkeypatch.setattr(
            perturbation_service, "perturb_and_save", lambda **kwargs: {"success": True}
        )
        result = mutation_task("uploads/original.wav", {"perturbations": []})
        assert "_scaffold" not in result


class TestAccentBiasTask:
    def test_runs_diagnostic_and_returns_json_dict(self, monkeypatch):
        from app.domain import accent_bias_profiler, accent_bias_runner

        class _FakeReport:
            def to_json_dict(self):
                return {"corpus": "l2-arctic", "model_id": "openai/whisper-base", "cohorts": []}

        captured = {}

        def _fake_transcriber(model_id):
            captured["model_id"] = model_id
            return lambda path: "hello world"

        def _fake_run_diagnostic(transcribe, corpus, model_id, samples_per_cohort):
            captured["corpus"] = corpus
            captured["samples_per_cohort"] = samples_per_cohort
            return _FakeReport()

        monkeypatch.setattr(accent_bias_profiler, "make_whisper_transcriber", _fake_transcriber)
        monkeypatch.setattr(accent_bias_runner, "run_accent_bias_diagnostic", _fake_run_diagnostic)

        result = accent_bias_task("openai/whisper-base", "l2-arctic", 5)

        assert result == {"corpus": "l2-arctic", "model_id": "openai/whisper-base", "cohorts": []}
        assert captured["model_id"] == "openai/whisper-base"
        assert captured["samples_per_cohort"] == 5


class TestEnqueueAccentBias:
    def test_places_job_on_asr_queue(self, broker):
        result = enqueue_accent_bias("openai/whisper-base", corpus="l2-arctic", samples_per_cohort=3)

        asr_queue = get_queue(WorkerFamily.ASR)
        assert result.job_id in asr_queue.job_ids
        assert result.family_jobs == {"accent_bias": result.job_id}

    def test_websocket_url_matches_the_tasks_route(self, broker):
        result = enqueue_accent_bias("openai/whisper-base")
        assert result.websocket_url == f"/api/ws/tasks/{result.job_id}"


class TestWorkerEntrypoint:
    def test_no_argument_is_usage_error(self):
        assert worker_entry.main([]) == 2

    def test_unknown_family_is_usage_error(self):
        assert worker_entry.main(["gpu-mining"]) == 2

    def test_valid_family_starts_worker(self, monkeypatch):
        started = {}
        monkeypatch.setattr(
            worker_entry, "run_worker", lambda fam: started.update(family=fam)
        )
        assert worker_entry.main(["asr"]) == 0
        assert started["family"] is WorkerFamily.ASR
