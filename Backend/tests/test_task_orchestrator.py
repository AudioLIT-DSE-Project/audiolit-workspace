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
from rq.job import Job, JobStatus

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

    def test_aggregator_dependency_receives_audio_ref(self, broker):
        # LIT-257: the fan-in needs the path ref to write the sample + analysis
        # records, so enqueue_multitask_analysis must thread it into the
        # aggregator job (family_job_ids, cache_key, audio_ref).
        result = enqueue_multitask_analysis(
            "uploads/test_sample.wav",
            tasks=[WorkerFamily.ASR],
            cache_key="sha256:deadbeef",
        )
        aggregator = Job.fetch(result.job_id, connection=broker)
        assert aggregator.args[1] == "sha256:deadbeef"
        assert aggregator.args[2] == "uploads/test_sample.wav"


class _StubFinishedJob:
    """Enough of an RQ Job for the aggregator's fan-in: finished with a result."""

    def __init__(self, result):
        self.result = result

    def get_status(self):
        return JobStatus.FINISHED


class _StubFailedJob:
    """The failed-sibling shape the aggregator's fan-in tolerates."""

    def get_status(self):
        return JobStatus.FAILED


def _mongomock_store():
    """A MetadataStore bound to an in-memory mongomock db."""
    import mongomock

    from app.infrastructure import metadata_store as ms

    mock_db = mongomock.MongoClient().db
    store = ms.MetadataStore(client=mock_db.client, db=mock_db)
    store.ensure_schema()
    return store


class TestAggregatorMetadataWriteThrough:
    """LIT-257: the fan-in aggregator writes each task's result and the audio
    sample through to the durable MongoDB metadata tier, without ever letting
    a metadata failure fail the analysis."""

    @staticmethod
    def _stub_outputs():
        # Second result carries a tensor-like array that must be stripped before
        # it reaches the durable tier (C4/SR4).
        return {
            "j-asr": {
                "task": "asr",
                "model_id": "openai/whisper-base",
                "transcript": "hello",
            },
            "j-ser": {
                "task": "ser",
                "model_id": "wav2vec2-base",
                "predicted_emotion": "neutral",
                "probabilities": {"neutral": 0.9, "happy": 0.1},
                "attention": [0.5] * 500,
            },
        }

    @staticmethod
    def _install_stub_jobs(monkeypatch, outputs):
        class _StubJobRegistry:
            @staticmethod
            def fetch(job_id, connection=None):
                return _StubFinishedJob(outputs[job_id])

        monkeypatch.setattr(task_orchestrator, "Job", _StubJobRegistry)

    def test_writes_one_analysis_doc_per_task_and_one_sample_doc(
        self, broker, monkeypatch, sample_audio_file
    ):
        from app.infrastructure import metadata_store as ms

        outputs = self._stub_outputs()
        self._install_stub_jobs(monkeypatch, outputs)
        store = _mongomock_store()
        monkeypatch.setattr(ms, "get_metadata_store", lambda: store)

        combined = task_orchestrator.aggregator_task(
            list(outputs), "sha256:deadbeef", str(sample_audio_file)
        )

        assert set(combined["tasks"]) == {"asr", "ser"}

        analyses = list(store._collection("analysis_results").find())
        assert len(analyses) == 2
        by_task = {a["task"]: a for a in analyses}
        assert by_task["asr"]["model_id"] == "openai/whisper-base"
        assert by_task["asr"]["redis_tensor_key"] == "sha256:deadbeef"
        assert by_task["ser"]["redis_tensor_key"] == "sha256:deadbeef"

        samples = list(store._collection("audio_samples").find())
        assert len(samples) == 1
        assert samples[0]["file_path_reference"] == str(sample_audio_file)
        assert samples[0]["sample_rate"] == 16000
        assert samples[0]["duration"] == pytest.approx(5.0, abs=0.2)
        # Only file-path metadata is stored, never audio bytes (C4/SR4).
        assert "audio_bytes" not in samples[0]

    def test_no_document_contains_a_list_longer_than_the_bound(
        self, broker, monkeypatch, sample_audio_file
    ):
        from app.infrastructure import metadata_store as ms

        outputs = self._stub_outputs()
        self._install_stub_jobs(monkeypatch, outputs)
        store = _mongomock_store()
        monkeypatch.setattr(ms, "get_metadata_store", lambda: store)

        task_orchestrator.aggregator_task(
            list(outputs), "sha256:deadbeef", str(sample_audio_file)
        )

        docs = list(store._collection("analysis_results").find())
        assert len(docs) == 2

        def _any_long_list(value):
            if isinstance(value, list):
                if len(value) > task_orchestrator._MAX_PERSISTED_LIST_LEN:
                    return True
                return any(_any_long_list(v) for v in value)
            if isinstance(value, dict):
                return any(_any_long_list(v) for v in value.values())
            return False

        for doc in docs:
            assert not _any_long_list(doc)

    def test_a_failing_metadata_tier_never_fails_the_fan_in(
        self, broker, monkeypatch, sample_audio_file
    ):
        from app.infrastructure import metadata_store as ms

        outputs = self._stub_outputs()
        self._install_stub_jobs(monkeypatch, outputs)

        class _FailingStore:
            def upsert_audio_sample(self, *args, **kwargs):
                raise RuntimeError("mongo down")

            def insert_analysis(self, *args, **kwargs):
                raise RuntimeError("mongo down")

        monkeypatch.setattr(ms, "get_metadata_store", lambda: _FailingStore())

        combined = task_orchestrator.aggregator_task(
            list(outputs), "sha256:deadbeef", str(sample_audio_file)
        )

        assert set(combined["tasks"]) == {"asr", "ser"}
        assert combined["tasks"]["asr"]["transcript"] == "hello"

    def test_skips_writes_when_metadata_tier_is_configured_off(
        self, broker, monkeypatch, sample_audio_file
    ):
        from app.infrastructure import metadata_store as ms

        outputs = self._stub_outputs()
        self._install_stub_jobs(monkeypatch, outputs)
        monkeypatch.setattr(ms, "get_metadata_store", lambda: None)

        combined = task_orchestrator.aggregator_task(
            list(outputs), "sha256:deadbeef", str(sample_audio_file)
        )

        assert combined["tasks"]["ser"]["predicted_emotion"] == "neutral"

    def test_failed_sibling_records_no_analysis_doc(
        self, broker, monkeypatch, sample_audio_file
    ):
        from app.infrastructure import metadata_store as ms

        class _StubJobRegistry:
            @staticmethod
            def fetch(job_id, connection=None):
                jobs = {
                    "j-asr": _StubFinishedJob(
                        {"task": "asr", "model_id": "openai/whisper-base", "transcript": "hi"}
                    ),
                    "j-failed": _StubFailedJob(),
                }
                return jobs[job_id]

        monkeypatch.setattr(task_orchestrator, "Job", _StubJobRegistry)
        store = _mongomock_store()
        monkeypatch.setattr(ms, "get_metadata_store", lambda: store)

        combined = task_orchestrator.aggregator_task(
            ["j-asr", "j-failed"], "sha256:deadbeef", str(sample_audio_file)
        )

        assert combined["tasks"]["j-failed"] == {"status": "failed"}
        # The finished task is recorded; the failed sibling is not - it has no
        # reproducible (task, model) identity to persist.
        analyses = list(store._collection("analysis_results").find())
        assert len(analyses) == 1
        assert analyses[0]["task"] == "asr"
        assert list(store._collection("audio_samples").find())[0]["sample_rate"] == 16000


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

    def test_worker_read_timeout_outlasts_the_blocking_dequeue(self, conn, monkeypatch):
        """A worker's socket must not time out while it waits for a job.

        An idle RQ worker sits in a blocking BLPOP for ``worker_ttl - 15`` -
        405 s on RQ 2.10's defaults. The request-path connection sets
        ``socket_timeout=10`` so a hung broker fails a request fast, and
        redis-py applies that same deadline to the blocking read: the socket
        times out 10 s into a 405 s wait, RQ reports "Redis connection timeout,
        quitting...", and the worker exits. Observed twice in one session, both
        times after a quiet period, while Redis itself answered PING.

        Nothing restarts a worker that quits, so async work stops being
        processed while the API still reports healthy - which is why this is
        pinned rather than left to a comment.

        The assertion is about configuration, not connectivity, so we capture the
        kwargs passed to ``Redis.from_url`` without making a real connection.
        """
        import unittest.mock as mock
        from app.infrastructure import rq_connection

        captured_kwargs: dict = {}

        def _fake_from_url(url, **kwargs):
            captured_kwargs.update(kwargs)
            fake = mock.MagicMock()
            fake.connection_pool.connection_kwargs = kwargs
            return fake

        rq_connection.reset_connection()
        monkeypatch.setattr("app.infrastructure.rq_connection.Redis.from_url", _fake_from_url)

        worker_conn = rq_connection.get_worker_redis_connection()
        read_timeout = worker_conn.connection_pool.connection_kwargs.get("socket_timeout")

        w = make_worker(WorkerFamily.ASR, connection=conn)
        assert read_timeout is None or read_timeout > w.dequeue_timeout, (
            f"worker socket_timeout={read_timeout}s would fire during a "
            f"{w.dequeue_timeout}s blocking dequeue and kill an idle worker"
        )

        # Restore the global so the MagicMock doesn't leak into other tests.
        rq_connection.reset_connection()

    def test_request_path_connection_keeps_its_fail_fast_timeout(self):
        """The other half: raising the worker's timeout must not raise the API's.

        `get_redis_connection` is shared by the acoustic, health and inference
        routes. Without a read deadline there, a stalled broker would hang a
        request for minutes instead of failing in seconds.
        """
        from app.infrastructure.rq_connection import (
            get_redis_connection,
            reset_connection,
        )

        reset_connection()
        try:
            conn = get_redis_connection()
        except Exception:
            pytest.skip("no broker reachable to inspect the request-path client")
        timeout = conn.connection_pool.connection_kwargs.get("socket_timeout")
        assert timeout is not None and timeout <= 30, (
            f"request-path socket_timeout={timeout} - the API should fail fast, "
            "only the worker connection may block indefinitely"
        )


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

    def test_delegates_to_perturb_and_save(self, broker, monkeypatch):
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

    def test_no_longer_returns_scaffold_flag(self, broker, monkeypatch):
        from app.domain import perturbation_service

        monkeypatch.setattr(
            perturbation_service, "perturb_and_save", lambda **kwargs: {"success": True}
        )
        result = mutation_task("uploads/original.wav", {"perturbations": []})
        assert "_scaffold" not in result


class TestAccentBiasTask:
    def test_runs_diagnostic_and_returns_json_dict(self, broker, monkeypatch):
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

    def test_writes_bias_report_after_a_run(self, broker, monkeypatch):
        # LIT-257: an accent-bias run writes one `bias_reports` document per
        # cohort (retained permanently, SAD §9).
        from app.domain import accent_bias_profiler, accent_bias_runner
        from app.infrastructure import metadata_store as ms

        store = _mongomock_store()
        monkeypatch.setattr(ms, "get_metadata_store", lambda: store)

        class _Cohort:
            accent = "Arabic"
            sample_count = 10
            scored_count = 8
            mean_wer = 0.142
            median_wer = 0.13
            stdev_wer = 0.03
            min_wer = 0.05
            max_wer = 0.2

        class _FakeReport:
            corpus = "l2-arctic"
            model_id = "openai/whisper-base"
            cohorts = [_Cohort()]

            def to_json_dict(self):
                return {"corpus": "l2-arctic", "model_id": "openai/whisper-base", "cohorts": []}

        monkeypatch.setattr(accent_bias_profiler, "make_whisper_transcriber", lambda m: None)
        monkeypatch.setattr(
            accent_bias_runner, "run_accent_bias_diagnostic", lambda *a, **k: _FakeReport()
        )

        accent_bias_task("openai/whisper-base", "l2-arctic", 5)

        docs = list(store._collection("bias_reports").find())
        assert len(docs) == 1
        assert docs[0]["model_id"] == "openai/whisper-base"
        assert docs[0]["cohort"] == "Arabic"
        assert docs[0]["WER"] == 0.142
        assert docs[0]["disparity_metrics"]["sample_count"] == 10
        assert docs[0]["disparity_metrics"]["median_wer"] == 0.13
        assert "created_at" in docs[0]


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
