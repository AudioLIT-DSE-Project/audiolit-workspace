"""Tests for real multi-task inference through the orchestrator (LIT-151, FR7/FR3).

Until now `asr_task` / `ser_task` / `add_task` were scaffolding that returned
`{"_scaffold": True}`, so `/api/inference/multitask` enqueued jobs producing
nothing. These cover the real delegation, the binary deepfake head's output
shape, the unified response, and the fan-out guard.

No model is downloaded — the domain functions are patched, matching how
`test_deepfake_classifier.py` handles the same problem.
"""

from __future__ import annotations

import pytest
from fakeredis import FakeServer, FakeStrictRedis
from rq.job import Job

from app.infrastructure import rq_connection
from app.orchestration import task_orchestrator as t


@pytest.fixture
def broker(monkeypatch):
    fake = FakeStrictRedis(server=FakeServer())
    monkeypatch.setattr(rq_connection, "_CONNECTION", fake)
    t.reset_queue_cache()
    yield fake
    t.reset_queue_cache()
    fake.flushall()


@pytest.fixture
def quiet_progress(monkeypatch):
    """Task functions publish progress; without a broker that would connect."""
    monkeypatch.setattr(t, "publish_progress", lambda *a, **k: 0)
    monkeypatch.setattr(t, "_current_job_id", lambda: "job-test")


class TestSerTask:
    def test_returns_label_confidence_and_full_distribution(self, monkeypatch, quiet_progress):
        """FR6.2 wants the distribution, top-1 label and confidence - not just argmax."""
        import app.domain.model_loader_service as ml

        monkeypatch.setattr(
            ml,
            "predict_emotion_wave2vec",
            lambda path: {
                "predicted_emotion": "happy",
                "confidence": 0.87,
                "probabilities": {"happy": 0.87, "sad": 0.08, "neutral": 0.05},
            },
        )
        out = t.ser_task("uploads/clip.wav", "default", {})
        assert out["task"] == "ser"
        # Key name matters: the merged frontend reads `predicted_emotion`.
        assert out["predicted_emotion"] == "happy"
        assert out["confidence"] == pytest.approx(0.87)
        assert out["probabilities"]["sad"] == pytest.approx(0.08)
        assert "_scaffold" not in out

    def test_passes_the_audio_reference_through(self, monkeypatch, quiet_progress):
        import app.domain.model_loader_service as ml

        seen = {}

        def _fake(path):
            seen["path"] = path
            return {"predicted_emotion": "sad", "confidence": 0.5, "probabilities": {"sad": 0.5}}

        monkeypatch.setattr(ml, "predict_emotion_wave2vec", _fake)
        t.ser_task("uploads/abc.wav", "default", {})
        assert seen["path"] == "uploads/abc.wav"


class TestAsrAndAddTasks:
    def test_asr_returns_a_transcript(self, monkeypatch, quiet_progress):
        import app.domain.model_loader_service as ml

        monkeypatch.setattr(ml, "transcribe_whisper_base", lambda p: "hello world")
        out = t.asr_task("uploads/clip.wav", "whisper-base", {})
        assert out == {"task": "asr", "model_id": "whisper-base", "transcript": "hello world"}

    def test_asr_honours_the_large_model_selection(self, monkeypatch, quiet_progress):
        import app.domain.model_loader_service as ml

        monkeypatch.setattr(ml, "transcribe_whisper_base", lambda p: "base")
        monkeypatch.setattr(ml, "transcribe_whisper_large", lambda p: "large")
        assert t.asr_task("c.wav", "whisper-large", {})["transcript"] == "large"
        assert t.asr_task("c.wav", "whisper-base", {})["transcript"] == "base"

    def test_add_returns_label_and_synthetic_probability(self, monkeypatch, quiet_progress):
        import app.domain.model_loader_service as ml
        from app.domain.model_loader_service import DEEPFAKE_BONA_FIDE, DEEPFAKE_SPOOF

        monkeypatch.setattr(
            ml,
            "predict_deepfake",
            lambda p: {
                "predicted_label": DEEPFAKE_SPOOF,
                "synthetic_probability": 0.91,
                "confidence": 0.91,
                "probabilities": {DEEPFAKE_SPOOF: 0.91, DEEPFAKE_BONA_FIDE: 0.09},
            },
        )
        out = t.add_task("uploads/clip.wav", "default", {})
        assert out["task"] == "add"
        assert out["label"] == DEEPFAKE_SPOOF
        assert out["synthetic_probability"] == pytest.approx(0.91)
        assert out["probabilities"][DEEPFAKE_BONA_FIDE] == pytest.approx(0.09)

    def test_add_keeps_fraud_probability_distinct_from_confidence(
        self, monkeypatch, quiet_progress
    ):
        """FR7 asks for a fraud probability. On a bona-fide clip that is the
        complement of `confidence`, so collapsing the two would report a clip as
        91% synthetic when the model said the opposite."""
        import app.domain.model_loader_service as ml
        from app.domain.model_loader_service import DEEPFAKE_BONA_FIDE, DEEPFAKE_SPOOF

        monkeypatch.setattr(
            ml,
            "predict_deepfake",
            lambda p: {
                "predicted_label": DEEPFAKE_BONA_FIDE,
                "synthetic_probability": 0.09,
                "confidence": 0.91,
                "probabilities": {DEEPFAKE_SPOOF: 0.09, DEEPFAKE_BONA_FIDE: 0.91},
            },
        )
        out = t.add_task("uploads/clip.wav", "default", {})
        assert out["label"] == DEEPFAKE_BONA_FIDE
        assert out["confidence"] == pytest.approx(0.91)
        assert out["synthetic_probability"] == pytest.approx(0.09)


class TestFanOutGuard:
    def test_xai_is_rejected_with_a_clear_error(self, broker):
        """Regression: the frontend sent tasks=[...,"xai"], which hit
        _TASK_FUNCS[XAI] and raised KeyError deep in the enqueue loop - a 500 on
        every real multi-task call."""
        with pytest.raises(ValueError, match="Unsupported multi-task families: xai"):
            t.enqueue_multitask_analysis("clip.wav", tasks=["asr", "ser", "add", "xai"])

    def test_mutation_is_rejected_too(self, broker):
        with pytest.raises(ValueError, match="mutation"):
            t.enqueue_multitask_analysis("clip.wav", tasks=["asr", "mutation"])

    def test_empty_task_list_is_rejected(self, broker):
        with pytest.raises(ValueError, match="At least one task family"):
            t.enqueue_multitask_analysis("clip.wav", tasks=[])

    def test_the_three_supported_families_fan_out(self, broker):
        result = t.enqueue_multitask_analysis("clip.wav", tasks=["asr", "ser", "add"])
        assert set(result.family_jobs) == {"asr", "ser", "add"}

    def test_multitask_families_excludes_xai_and_mutation(self):
        assert set(t.MULTITASK_FAMILIES) == {
            t.WorkerFamily.ASR, t.WorkerFamily.SER, t.WorkerFamily.ADD
        }


class TestFrontendContract:
    """Pin the exact key names `UnifiedTaskResult` in PredictionPanel.tsx reads.

    A mismatch here does not fail anywhere - the panel just renders nothing, and
    the job still reports SUCCESS. Both of these were wrong when the real task
    functions first landed: the backend emitted `label` where the UI reads
    `predicted_emotion`, and the UI's type said the deepfake label could be
    "synthetic", which the backend never emits.
    """

    def test_ser_emits_the_keys_the_panel_reads(self, monkeypatch, quiet_progress):
        import app.domain.model_loader_service as ml

        monkeypatch.setattr(
            ml,
            "predict_emotion_wave2vec",
            lambda p: {"predicted_emotion": "sad", "confidence": 0.7,
                       "probabilities": {"sad": 0.7}},
        )
        out = t.ser_task("c.wav", "default", {})
        assert {"predicted_emotion", "probabilities"} <= set(out)

    def test_asr_emits_the_keys_the_panel_reads(self, monkeypatch, quiet_progress):
        import app.domain.model_loader_service as ml

        monkeypatch.setattr(ml, "transcribe_whisper_base", lambda p: "hi")
        assert "transcript" in t.asr_task("c.wav", "whisper-base", {})

    def test_add_label_is_one_of_the_backend_constants(self, monkeypatch, quiet_progress):
        import app.domain.model_loader_service as ml
        from app.domain.model_loader_service import DEEPFAKE_BONA_FIDE, DEEPFAKE_SPOOF

        monkeypatch.setattr(
            ml,
            "predict_deepfake",
            lambda p: {"predicted_label": DEEPFAKE_SPOOF, "synthetic_probability": 0.9,
                       "confidence": 0.9, "probabilities": {}},
        )
        out = t.add_task("c.wav", "default", {})
        assert {"label", "confidence"} <= set(out)
        assert out["label"] in {DEEPFAKE_BONA_FIDE, DEEPFAKE_SPOOF}
        assert out["label"] != "synthetic", "the UI's type union once claimed this value"


class TestUnifiedResponse:
    def test_aggregator_keys_results_by_task_name(self, broker, monkeypatch):
        """DoD: one response carrying transcript, emotion distribution and
        deepfake score together."""
        monkeypatch.setattr(t, "publish_progress", lambda *a, **k: 0)
        monkeypatch.setattr(t, "_current_job_id", lambda: "agg")

        finished = {
            "asr": {"task": "asr", "transcript": "hello"},
            "ser": {"task": "ser", "predicted_emotion": "happy", "confidence": 0.9,
                    "probabilities": {"happy": 0.9}},
            "add": {"task": "add", "label": "bona-fide", "confidence": 0.8,
                    "synthetic_probability": 0.2, "probabilities": {}},
        }

        class _Job:
            def __init__(self, payload):
                self.result = payload

            def get_status(self):
                from rq.job import JobStatus
                return JobStatus.FINISHED

        monkeypatch.setattr(
            Job, "fetch", staticmethod(lambda jid, connection=None: _Job(finished[jid]))
        )
        combined = t.aggregator_task(["asr", "ser", "add"], cache_key="sha256:abc")

        assert set(combined["tasks"]) == {"asr", "ser", "add"}
        assert combined["tasks"]["asr"]["transcript"] == "hello"
        assert combined["tasks"]["ser"]["probabilities"]["happy"] == pytest.approx(0.9)
        assert combined["tasks"]["add"]["label"] == "bona-fide"
        assert combined["cache_key"] == "sha256:abc"
        assert combined["schema_version"] == "1.0"

    def test_a_failed_sibling_does_not_lose_the_others(self, broker, monkeypatch):
        """SAD §11.1: a failed job must not take the successful results with it."""
        monkeypatch.setattr(t, "publish_progress", lambda *a, **k: 0)
        monkeypatch.setattr(t, "_current_job_id", lambda: "agg")
        from rq.job import JobStatus

        class _Job:
            def __init__(self, jid):
                self.jid = jid
                self.result = {"task": "asr", "transcript": "hello"} if jid == "ok" else None

            def get_status(self):
                return JobStatus.FINISHED if self.jid == "ok" else JobStatus.FAILED

        monkeypatch.setattr(Job, "fetch", staticmethod(lambda jid, connection=None: _Job(jid)))
        combined = t.aggregator_task(["ok", "boom"], cache_key=None)

        assert combined["tasks"]["asr"]["transcript"] == "hello"
        assert combined["tasks"]["boom"] == {"status": "failed"}
