"""Endpoint test for POST /inferences/run's force_refresh passthrough (LIT-248).

The regenerate-button behaviour itself (cache bypass, recompute, cache
overwrite) is covered at the inference_service.run_inference level in
test_custom_model_fidelity.py; this only checks the route reads
"force_refresh" off the request body and forwards it, defaulting to False
when the field is absent (every existing caller - the batch queue, /upload,
warmup - omits it and must keep the current cached-first behaviour).
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.api.routes import inferences as inferences_routes


class TestRunInferenceForceRefreshPassthrough:
    @pytest.mark.asyncio
    async def test_force_refresh_true_is_forwarded(self, client, monkeypatch):
        mock_run = AsyncMock(return_value="prediction")
        monkeypatch.setattr(inferences_routes, "run_inference", mock_run)

        r = await client.post(
            "/inferences/run",
            json={"model": "whisper-base", "file_path": "a.wav", "force_refresh": True},
        )

        assert r.status_code == 200
        mock_run.assert_awaited_once()
        assert mock_run.call_args.kwargs.get("force_refresh") is True

    @pytest.mark.asyncio
    async def test_force_refresh_defaults_to_false_when_omitted(self, client, monkeypatch):
        mock_run = AsyncMock(return_value="prediction")
        monkeypatch.setattr(inferences_routes, "run_inference", mock_run)

        r = await client.post("/inferences/run", json={"model": "whisper-base", "file_path": "a.wav"})

        assert r.status_code == 200
        mock_run.assert_awaited_once()
        assert mock_run.call_args.kwargs.get("force_refresh") is False
