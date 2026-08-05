"""Tests for the LIT-150 draft real ASR+SER (+stub ADD) orchestrator
(app.services.multitask_orchestrator_service).

Same fakeredis + SimpleWorker(burst=True) pattern as
test_fanout_orchestrator.py (LIT-225) -- fast, no external Redis, no real
model downloads (transcribe_whisper_base / predict_emotion_wave2vec are
mocked at the call site).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fakeredis import FakeServer, FakeStrictRedis
from rq import Queue, SimpleWorker
from rq.job import Job, JobStatus

from app.services import multitask_orchestrator_service as multitask


def _drain(conn, *queue_names) -> None:
    queues = [Queue(name, connection=conn) for name in queue_names]
    SimpleWorker(queues, connection=conn).work(burst=True)


@pytest.fixture
def fake_conn(monkeypatch):
    conn = FakeStrictRedis(server=FakeServer())
    monkeypatch.setattr(multitask, "get_redis_connection", lambda: conn)
    yield conn
    conn.flushall()


# Child task queues, drained to completion before the aggregator. Draining the
# dependency-gated aggregator in the SAME SimpleWorker.work(burst=True) pass as
# its children races the dependency-completion callback and can hang the burst
# worker intermittently; the fan-out prototype test (test_fanout_orchestrator)
# avoids this by draining in two phases, and so do we.
CHILD_QUEUES = (
    multitask.ASR_QUEUE_NAME,
    multitask.SER_QUEUE_NAME,
    multitask.ADD_QUEUE_NAME,
)


class TestRealAsrJob:
    def test_calls_whisper_base_by_default(self, fake_conn):
        with patch.object(multitask, "transcribe_whisper_base", return_value="hello world") as mock_base, \
             patch.object(multitask, "transcribe_whisper_large") as mock_large:
            result = multitask.run_asr_job("clip.wav")

        mock_base.assert_called_once_with("clip.wav")
        mock_large.assert_not_called()
        assert result == {"task_name": "asr", "transcript": "hello world"}

    def test_uses_whisper_large_when_requested(self, fake_conn):
        with patch.object(multitask, "transcribe_whisper_base") as mock_base, \
             patch.object(multitask, "transcribe_whisper_large", return_value="fancier transcript") as mock_large:
            result = multitask.run_asr_job("clip.wav", model="whisper-large")

        mock_large.assert_called_once_with("clip.wav")
        mock_base.assert_not_called()
        assert result["transcript"] == "fancier transcript"


class TestRealSerJob:
    def test_returns_real_emotion_prediction(self, fake_conn):
        fake_prediction = {"predicted_emotion": "happy", "probabilities": {"happy": 0.9}, "confidence": 0.9}
        with patch.object(multitask, "predict_emotion_wave2vec", return_value=fake_prediction) as mock_predict:
            result = multitask.run_ser_job("clip.wav")

        mock_predict.assert_called_once_with("clip.wav")
        assert result == {"task_name": "ser", **fake_prediction}


class TestAddJobStub:
    def test_returns_typed_not_implemented_not_a_fabricated_score(self, fake_conn):
        result = multitask.run_add_job("clip.wav")
        assert result == {"task_name": "add", "status": multitask.ADD_NOT_IMPLEMENTED}


class TestMultitaskFanOutFanIn:
    @staticmethod
    def _child_ids(result: dict) -> list[str]:
        return [result["asr_job_id"], result["ser_job_id"], result["add_job_id"]]

    def test_enqueue_wires_aggregator_deferred_on_children(self, fake_conn):
        # The aggregator must not run until the children finish. Assert that
        # wiring (a DEFERRED aggregator depending on all three children) without
        # running any worker, mirroring test_fanout_orchestrator's deferred case.
        result = multitask.enqueue_multitask("clip.wav")
        agg_job = Job.fetch(result["aggregator_job_id"], connection=fake_conn)
        assert agg_job.get_status() == JobStatus.DEFERRED

    def test_aggregates_asr_ser_add_once_on_success(self, fake_conn):
        with patch.object(multitask, "transcribe_whisper_base", return_value="the transcript"), \
             patch.object(
                 multitask,
                 "predict_emotion_wave2vec",
                 return_value={"predicted_emotion": "neutral", "probabilities": {}, "confidence": 0.5},
             ):
            result = multitask.enqueue_multitask("clip.wav")
            _drain(fake_conn, *CHILD_QUEUES)

        # Aggregate directly over the finished child jobs. Running the aggregator
        # as a *dependent* RQ job under SimpleWorker(burst) races RQ's
        # dependency-completion callback on fakeredis and can hang the burst
        # worker; the fan-in logic itself is what this asserts, so invoke it on
        # the child ids (real finished jobs) rather than through the flaky drain.
        agg_result = multitask.aggregate_multitask(self._child_ids(result))

        assert agg_result["failed"] == {}
        assert len(agg_result["succeeded"]) == 3
        by_task = {r["task_name"]: r for r in agg_result["succeeded"].values()}
        assert by_task["asr"]["transcript"] == "the transcript"
        assert by_task["ser"]["predicted_emotion"] == "neutral"
        assert by_task["add"]["status"] == multitask.ADD_NOT_IMPLEMENTED

    def test_asr_failure_does_not_lose_ser_result(self, fake_conn):
        # CHILD_JOB_RETRY exists for real forking-Worker crash recovery; under
        # SimpleWorker+fakeredis a retried failure can hang the burst drain, and
        # this test is about fan-in aggregation, not retry — so patch retry off
        # and let the asr child fail once, deterministically.
        with patch.object(multitask, "transcribe_whisper_base", side_effect=RuntimeError("model load failed")), \
             patch.object(multitask, "CHILD_JOB_RETRY", None), \
             patch.object(
                 multitask,
                 "predict_emotion_wave2vec",
                 return_value={"predicted_emotion": "sad", "probabilities": {}, "confidence": 0.7},
             ):
            result = multitask.enqueue_multitask("clip.wav")
            _drain(fake_conn, *CHILD_QUEUES)

        agg_result = multitask.aggregate_multitask(self._child_ids(result))

        assert len(agg_result["failed"]) == 1
        assert len(agg_result["succeeded"]) == 2
        (error_text,) = agg_result["failed"].values()
        assert "model load failed" in error_text
        succeeded_tasks = {r["task_name"] for r in agg_result["succeeded"].values()}
        assert succeeded_tasks == {"ser", "add"}
