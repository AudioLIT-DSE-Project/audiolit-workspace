"""A finished job's outcome has to reach the client (SRS FR3.2).

The worker, the gateway relay and the frontend hook were each tested alone, with
the neighbour mocked, so nothing covered the path between them - and that path
was broken in three places:

* the worker's SUCCESS event carries only ``{"duration_s"}``, the gateway
  relayed it unchanged, and the client read that as the job's result;
* a task that raised published no terminal event at all, because RQ catches the
  exception and the worker's own ``except`` never ran;
* a result containing bytes made both the socket and the polling route raise.

These tests run a real job through a real ``AudioLITWorker`` against a fake
broker and assert on what a client would actually receive.
"""

from __future__ import annotations

import json

import pytest
from fakeredis import FakeServer, FakeStrictRedis
from starlette.testclient import TestClient

from app.api.routes.tasks import terminal_event
from app.infrastructure import rq_connection
from app.main import app
from app.orchestration import task_orchestrator
from app.orchestration.task_orchestrator import (
    WorkerFamily,
    enqueue,
    fetch_job,
    make_worker,
    progress_channel,
)


def _report_job() -> dict:  # module-level so RQ can reference it by import path
    return {"cohorts": [{"accent": "Hindi", "mean_wer": 0.21}]}


def _bytes_job() -> dict:
    return {"success": True, "perturbed_file": "uploads/x.wav", "preview_bytes": b"RIFF\xfa\x00"}


def _failing_job() -> None:
    raise ValueError("Unknown corpus 'custom'")


@pytest.fixture
def broker(monkeypatch):
    """One fake broker for both the request path and the worker."""
    fake = FakeStrictRedis(server=FakeServer())
    monkeypatch.setattr(rq_connection, "_CONNECTION", fake)
    monkeypatch.setattr(rq_connection, "_WORKER_CONNECTION", fake)
    task_orchestrator.reset_queue_cache()
    yield fake
    task_orchestrator.reset_queue_cache()
    fake.flushall()


def _run(broker, func) -> tuple[str, list[dict]]:
    """Enqueue ``func``, drain it, and return the job id and its progress events."""
    job = enqueue(WorkerFamily.MUTATION, func)
    pubsub = broker.pubsub()
    pubsub.subscribe(progress_channel(job.id))
    make_worker(WorkerFamily.MUTATION, connection=broker).work(burst=True)

    events = []
    while (message := pubsub.get_message(timeout=0.2)) is not None:
        if message["type"] == "message":
            events.append(json.loads(message["data"]))
    pubsub.close()
    return job.id, events


class TestWorkerPublishesATerminalEvent:
    def test_success_event_alone_does_not_carry_the_result(self, broker):
        """Pins the worker's side of the contract: the result is not on the
        channel, so the gateway has to add it."""
        _, events = _run(broker, _report_job)
        assert events[-1]["stage"] == "SUCCESS"
        assert "result" not in events[-1]["payload"]

    def test_a_raising_task_publishes_failure(self, broker):
        job_id, events = _run(broker, _failing_job)
        assert fetch_job(job_id).is_failed
        assert events[-1]["stage"] == "FAILURE"
        assert events[-1]["payload"]["error"] == "ValueError: Unknown corpus 'custom'"


class TestTerminalEvent:
    def test_success_gains_the_job_result(self, broker):
        job_id, events = _run(broker, _report_job)
        event = terminal_event(events[-1], fetch_job(job_id))
        assert event["state"] == "SUCCESS"
        assert event["payload"]["result"] == _report_job()
        assert "duration_s" in event["payload"]

    def test_bytes_in_a_result_do_not_break_encoding(self, broker):
        job_id, events = _run(broker, _bytes_job)
        event = terminal_event(events[-1], fetch_job(job_id))
        json.dumps(event)
        assert event["payload"]["result"]["perturbed_file"] == "uploads/x.wav"
        assert event["payload"]["result"]["preview_bytes"] is None

    def test_failure_keeps_the_workers_error(self, broker):
        job_id, events = _run(broker, _failing_job)
        event = terminal_event(events[-1], fetch_job(job_id))
        assert event["state"] == "FAILURE"
        assert event["payload"]["error"] == "ValueError: Unknown corpus 'custom'"

    def test_failure_without_an_error_falls_back_to_the_traceback(self, broker):
        job_id, _ = _run(broker, _failing_job)
        event = terminal_event({"stage": "FAILURE", "payload": {}}, fetch_job(job_id))
        assert event["payload"]["error"] == "ValueError: Unknown corpus 'custom'"


class TestStatusRoute:
    @pytest.mark.asyncio
    async def test_finished_job_returns_its_result(self, client, broker):
        job_id, _ = _run(broker, _report_job)
        body = (await client.get(f"/api/tasks/{job_id}/status")).json()
        assert body["state"] == "SUCCESS"
        assert body["result"] == _report_job()
        assert body["error"] is None

    @pytest.mark.asyncio
    async def test_result_with_bytes_is_still_a_200(self, client, broker):
        job_id, _ = _run(broker, _bytes_job)
        r = await client.get(f"/api/tasks/{job_id}/status")
        assert r.status_code == 200
        assert r.json()["result"]["success"] is True

    @pytest.mark.asyncio
    async def test_failed_job_returns_the_exception_not_the_stack(self, client, broker):
        job_id, _ = _run(broker, _failing_job)
        body = (await client.get(f"/api/tasks/{job_id}/status")).json()
        assert body["state"] == "FAILURE"
        assert body["error"] == "ValueError: Unknown corpus 'custom'"


class TestWebSocket:
    def test_socket_opened_after_the_job_finished_gets_the_result(self, broker):
        job_id, _ = _run(broker, _report_job)
        with TestClient(app).websocket_connect(f"/api/ws/tasks/{job_id}") as ws:
            message = ws.receive_json()
        assert message["state"] == "SUCCESS"
        assert message["payload"]["result"] == _report_job()

    def test_socket_open_during_the_job_gets_the_result_with_success(self, broker):
        """The case every slow job hits, and the one that was broken: the socket
        is already listening when the worker publishes SUCCESS."""
        job = enqueue(WorkerFamily.MUTATION, _report_job)
        with TestClient(app).websocket_connect(f"/api/ws/tasks/{job.id}") as ws:
            assert ws.receive_json()["state"] == "QUEUED"
            make_worker(WorkerFamily.MUTATION, connection=broker).work(burst=True)
            while True:
                message = ws.receive_json()
                if message.get("stage") == "SUCCESS":
                    break
        assert message["state"] == "SUCCESS"
        assert message["payload"]["result"] == _report_job()


class TestMutationRoute:
    """`POST /api/inference/mutation` resolves the audio before it enqueues."""

    @pytest.fixture
    def enqueued(self, monkeypatch):
        from app.api.routes import inference as inference_routes
        from app.orchestration.task_orchestrator import EnqueueResult

        calls: list[dict] = []

        def _fake_enqueue(audio_ref, mutation):
            calls.append({"audio_ref": audio_ref, "mutation": mutation})
            return EnqueueResult(job_id="job-1", websocket_url="/api/ws/tasks/job-1")

        monkeypatch.setattr(inference_routes, "enqueue_mutation", _fake_enqueue)
        return calls

    @pytest.mark.asyncio
    async def test_upload_is_not_looked_up_inside_the_active_corpus(
        self, client, enqueued, sample_audio_file
    ):
        """The UI sends the active corpus with every request. For an upload it
        used to win, and the clip was reported missing from that corpus."""
        perturbations = [{"type": "noise", "params": {"noise_level": 0.1}}]
        r = await client.post(
            "/api/inference/mutation",
            json={
                "audio_ref": str(sample_audio_file),
                "mutation": {
                    "perturbations": perturbations,
                    "is_uploaded": True,
                    "dataset": "common-voice",
                },
            },
        )
        assert r.status_code == 200
        assert r.json()["job_id"] == "job-1"
        assert enqueued == [
            {
                "audio_ref": str(sample_audio_file.resolve()),
                "mutation": {"perturbations": perturbations},
            }
        ]

    @pytest.mark.asyncio
    async def test_dataset_row_is_resolved_through_its_corpus(
        self, client, enqueued, sample_audio_file, monkeypatch
    ):
        from app.api.routes import inference as inference_routes

        seen = {}

        def _fake_resolve_file(dataset, file_path, session_id=None):
            seen.update(dataset=dataset, file_path=file_path)
            return sample_audio_file

        monkeypatch.setattr(inference_routes, "resolve_file", _fake_resolve_file)
        r = await client.post(
            "/api/inference/mutation",
            json={
                "audio_ref": "sample-000001.mp3",
                "mutation": {"perturbations": [], "is_uploaded": False, "dataset": "common-voice"},
            },
        )
        assert r.status_code == 200
        assert seen == {"dataset": "common-voice", "file_path": "sample-000001.mp3"}
        assert enqueued[0]["audio_ref"] == str(sample_audio_file.resolve())

    @pytest.mark.asyncio
    async def test_missing_audio_is_a_404_and_nothing_is_enqueued(self, client, enqueued):
        r = await client.post(
            "/api/inference/mutation",
            json={"audio_ref": "uploads/does-not-exist.wav", "mutation": {"perturbations": []}},
        )
        assert r.status_code == 404
        assert "does-not-exist.wav" in r.json()["detail"]
        assert enqueued == []

    @pytest.mark.asyncio
    async def test_unknown_corpus_is_a_400(self, client, enqueued):
        r = await client.post(
            "/api/inference/mutation",
            json={
                "audio_ref": "clip.wav",
                "mutation": {"perturbations": [], "dataset": "not-a-corpus"},
            },
        )
        assert r.status_code == 400
        assert "not-a-corpus" in r.json()["detail"]
        assert enqueued == []


class TestAsrTaskTranscribes:
    """`asr_task` imported a function that did not exist and reported an empty
    transcript as a success. Run through a real worker, since the task needs
    its worker context."""

    def _run_asr(self, broker, model_id):
        from app.orchestration.task_orchestrator import asr_task

        job = enqueue(WorkerFamily.ASR, asr_task, "uploads/clip.wav", model_id, {})
        make_worker(WorkerFamily.ASR, connection=broker).work(burst=True)
        return fetch_job(job.id).result

    def test_returns_the_models_transcript(self, broker, monkeypatch):
        from app.domain import model_loader_service

        calls = []

        def _fake_transcribe(audio_file_path, model=None):
            calls.append((audio_file_path, model))
            return " It must have fallen."

        # raising=True (the default): fails if the name the task imports is gone.
        monkeypatch.setattr(model_loader_service, "transcribe_whisper_base", _fake_transcribe)

        result = self._run_asr(broker, "whisper-base")
        assert result["status"] == "success"
        assert result["transcript"] == " It must have fallen."
        assert calls == [("uploads/clip.wav", "whisper-base")]

    def test_default_model_placeholder_is_not_passed_as_a_model_name(self, broker, monkeypatch):
        from app.domain import model_loader_service

        calls = []
        monkeypatch.setattr(
            model_loader_service,
            "transcribe_whisper_base",
            lambda audio_file_path, model=None: calls.append(model) or "ok",
        )
        assert self._run_asr(broker, "default")["transcript"] == "ok"
        assert calls == [None]
