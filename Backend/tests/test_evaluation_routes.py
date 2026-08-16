"""Endpoint tests for evaluation.py (LIT-231, FR15/FR16).

`generate_saliency`/`evaluate_downstream_degradation`/`enqueue_accent_bias`
are already covered at the unit level elsewhere (test_saliency_service.py,
test_degradation_scoring.py, test_task_orchestrator.py) -- these exercise the
route layer (request validation, wiring, HTTP status mapping) with those
calls mocked so no real model inference or RQ connection is needed in CI.
"""
from __future__ import annotations

import pytest

from app.api.routes import evaluation as evaluation_routes


class TestEvaluationFaithfulness:
    @pytest.mark.asyncio
    async def test_success_returns_degradation_result(self, client, monkeypatch, sample_audio_file):
        monkeypatch.setattr(
            evaluation_routes,
            "generate_saliency",
            lambda audio_path, model, method: {"series": [0.1, 0.5, 0.9, 0.2]},
        )
        monkeypatch.setattr(
            evaluation_routes,
            "evaluate_downstream_degradation",
            lambda **kwargs: {
                "success": True,
                "audc": 0.42,
                "degradation_curve": [{"k_percent": 10.0, "degradation_ratio": 0.1}],
                "audit_verdict": "faithful",
            },
        )

        r = await client.post(
            "/evaluation/faithfulness",
            json={"model": "wav2vec2", "method": "gradcam", "file_path": str(sample_audio_file)},
        )

        assert r.status_code == 200
        body = r.json()
        assert body["audc"] == 0.42
        assert body["audit_verdict"] == "faithful"

    @pytest.mark.asyncio
    async def test_empty_saliency_series_returns_422(self, client, monkeypatch, sample_audio_file):
        monkeypatch.setattr(
            evaluation_routes, "generate_saliency", lambda audio_path, model, method: {"series": []}
        )

        r = await client.post(
            "/evaluation/faithfulness",
            json={"model": "wav2vec2", "file_path": str(sample_audio_file)},
        )

        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_degradation_failure_returns_500(self, client, monkeypatch, sample_audio_file):
        monkeypatch.setattr(
            evaluation_routes, "generate_saliency", lambda audio_path, model, method: {"series": [0.1, 0.2]}
        )
        monkeypatch.setattr(
            evaluation_routes,
            "evaluate_downstream_degradation",
            lambda **kwargs: {"success": False, "error": "model inference failed"},
        )

        r = await client.post(
            "/evaluation/faithfulness",
            json={"model": "wav2vec2", "file_path": str(sample_audio_file)},
        )

        assert r.status_code == 500

    @pytest.mark.asyncio
    async def test_missing_file_returns_404(self, client):
        r = await client.post(
            "/evaluation/faithfulness", json={"model": "wav2vec2", "file_path": "/no/such/file.wav"}
        )
        assert r.status_code == 404


class TestEvaluationAccentBias:
    @pytest.mark.asyncio
    async def test_enqueues_and_returns_job_response(self, client, monkeypatch):
        class _FakeEnqueueResult:
            def as_response(self):
                return {
                    "job_id": "job-123",
                    "websocket_url": "/api/ws/tasks/job-123",
                    "schema_version": "1.0",
                    "family_jobs": {"accent_bias": "job-123"},
                    "cache_key": None,
                }

        captured = {}

        def _fake_enqueue(model_id, corpus="l2-arctic", samples_per_cohort=None):
            captured["model_id"] = model_id
            captured["corpus"] = corpus
            captured["samples_per_cohort"] = samples_per_cohort
            return _FakeEnqueueResult()

        monkeypatch.setattr(evaluation_routes, "enqueue_accent_bias", _fake_enqueue)

        r = await client.post(
            "/evaluation/accent-bias", json={"model_id": "openai/whisper-base", "corpus": "l2-arctic"}
        )

        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == "job-123"
        assert body["family_jobs"] == {"accent_bias": "job-123"}
        assert captured["model_id"] == "openai/whisper-base"
        assert captured["samples_per_cohort"] == 10  # route default
