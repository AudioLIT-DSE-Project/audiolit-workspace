"""Endpoint tests for models.py (LIT-231, FR1).

The registry itself (Hub resolution, safetensors validation, hook wiring) is
already covered by test_model_registry_service.py -- these exercise the route
layer: request validation and the typed-error-to-HTTP-status mapping SRS Use
Case 5 asks for, via a mocked `registry.get` so no real Hugging Face model is
ever downloaded in CI.
"""
from __future__ import annotations

from unittest.mock import Mock

import pytest

from app.api.routes import models as models_routes
from app.domain.model_registry_service import LoadedModel, ModelRegistryError


def _fake_loaded_model() -> LoadedModel:
    return LoadedModel(
        model_id="openai/whisper-base",
        revision="abc123",
        family="whisper",
        weights_sha256="deadbeef",
        model=Mock(),
        available_layers=["encoder.layers.0", "encoder.layers.1"],
    )


class TestResolveModel:
    @pytest.mark.asyncio
    async def test_success_returns_family_and_layers(self, client, monkeypatch):
        monkeypatch.setattr(models_routes.registry, "get", lambda model_id, revision="main": _fake_loaded_model())

        r = await client.post("/models/resolve", json={"model_id": "openai/whisper-base"})

        assert r.status_code == 200
        body = r.json()
        assert body["family"] == "whisper"
        assert body["weights_sha256"] == "deadbeef"
        assert body["available_layers"] == ["encoder.layers.0", "encoder.layers.1"]

    @pytest.mark.asyncio
    async def test_unsupported_architecture_returns_422_with_code(self, client, monkeypatch):
        def _raise(model_id, revision="main"):
            raise ModelRegistryError("UNSUPPORTED_ARCHITECTURE", "model_type 'bert' is not supported")

        monkeypatch.setattr(models_routes.registry, "get", _raise)

        r = await client.post("/models/resolve", json={"model_id": "bert-base-uncased"})

        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "UNSUPPORTED_ARCHITECTURE"

    @pytest.mark.asyncio
    async def test_unsafe_artifact_returns_422_with_code(self, client, monkeypatch):
        def _raise(model_id, revision="main"):
            raise ModelRegistryError("UNSAFE_ARTIFACT", "no safetensors weights found")

        monkeypatch.setattr(models_routes.registry, "get", _raise)

        r = await client.post("/models/resolve", json={"model_id": "some/unsafe-model"})

        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "UNSAFE_ARTIFACT"

    @pytest.mark.asyncio
    async def test_hub_unavailable_returns_502_with_code(self, client, monkeypatch):
        def _raise(model_id, revision="main"):
            raise ModelRegistryError("HUB_UNAVAILABLE", "Hugging Face Hub is unreachable")

        monkeypatch.setattr(models_routes.registry, "get", _raise)

        r = await client.post("/models/resolve", json={"model_id": "openai/whisper-base"})

        assert r.status_code == 502
        assert r.json()["detail"]["code"] == "HUB_UNAVAILABLE"


class TestCancelAndActiveModels:
    @pytest.mark.asyncio
    async def test_cancel_model_resolution(self, client):
        r = await client.post("/models/cancel", json={"model_id": "openai/whisper-base"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "openai/whisper-base" in body["message"]

    @pytest.mark.asyncio
    async def test_get_active_downloads(self, client):
        r = await client.get("/models/active")
        assert r.status_code == 200
        assert "active_downloads" in r.json()

