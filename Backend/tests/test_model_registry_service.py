"""ModelRegistry — resolver, safetensors enforcement, circuit breaker, LRU cache (LIT-207).

Everything here mocks the Hugging Face Hub / transformers loading calls —
no real network access or model downloads, matching how test_hook_manager_service.py
avoids downloading real weights for LIT-211.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from huggingface_hub.utils import HfHubHTTPError

from app.domain.model_registry_service import (
    HUB_UNAVAILABLE,
    UNSAFE_ARTIFACT,
    UNSUPPORTED_ARCHITECTURE,
    LoadedModel,
    ModelRegistry,
    ModelRegistryError,
    ResolvedModel,
    _CircuitBreaker,
    download_and_load,
    resolve_model_id,
)


def _sibling(name):
    return SimpleNamespace(rfilename=name)


def _hub_error(message="boom"):
    return HfHubHTTPError(message, response=MagicMock())


class TestCircuitBreaker:
    def test_closed_below_threshold(self):
        breaker = _CircuitBreaker(failure_threshold=5, recovery_seconds=60)
        for _ in range(4):
            breaker.record_failure(now=0.0)
        assert breaker.is_open(now=0.0) is False

    def test_opens_at_threshold_within_window(self):
        breaker = _CircuitBreaker(failure_threshold=5, recovery_seconds=60)
        for i in range(5):
            breaker.record_failure(now=float(i))
        assert breaker.is_open(now=5.0) is True

    def test_recovers_after_window_elapses(self):
        breaker = _CircuitBreaker(failure_threshold=5, recovery_seconds=60)
        for i in range(5):
            breaker.record_failure(now=float(i))
        assert breaker.is_open(now=5.0) is True
        assert breaker.is_open(now=65.0) is False

    def test_success_clears_failures(self):
        breaker = _CircuitBreaker(failure_threshold=5, recovery_seconds=60)
        for i in range(4):
            breaker.record_failure(now=float(i))
        breaker.record_success()
        assert breaker.is_open(now=4.0) is False

    def test_guard_fast_fails_without_calling_func_when_open(self):
        breaker = _CircuitBreaker(failure_threshold=1, recovery_seconds=60)
        breaker.record_failure()
        func = MagicMock()
        with pytest.raises(ModelRegistryError) as exc_info:
            breaker.guard(func)
        assert exc_info.value.code == HUB_UNAVAILABLE
        func.assert_not_called()

    def test_guard_records_failure_on_hub_error(self):
        breaker = _CircuitBreaker(failure_threshold=5, recovery_seconds=60)
        func = MagicMock(side_effect=_hub_error())
        with pytest.raises(HfHubHTTPError):
            breaker.guard(func)
        assert len(breaker._failures) == 1


class TestResolveModelId:
    def _mock_api(self, siblings, sha="abc123"):
        api = MagicMock()
        api.model_info.return_value = SimpleNamespace(sha=sha, siblings=siblings)
        return api

    def test_rejects_non_safetensors_before_config_lookup(self):
        api = self._mock_api([_sibling("pytorch_model.bin")])
        with patch("app.domain.model_registry_service.hf_hub_download") as mock_download:
            with pytest.raises(ModelRegistryError) as exc_info:
                resolve_model_id("some/model", api=api)
        assert exc_info.value.code == UNSAFE_ARTIFACT
        mock_download.assert_not_called()

    def test_rejects_unsupported_model_type(self, tmp_path):
        api = self._mock_api([_sibling("model.safetensors")])
        config_path = tmp_path / "config.json"
        config_path.write_text('{"model_type": "bert"}')
        with patch("app.domain.model_registry_service.hf_hub_download", return_value=str(config_path)):
            with pytest.raises(ModelRegistryError) as exc_info:
                resolve_model_id("some/bert-model", api=api)
        assert exc_info.value.code == UNSUPPORTED_ARCHITECTURE

    def test_resolves_supported_whisper_family(self, tmp_path):
        api = self._mock_api([_sibling("model.safetensors")], sha="deadbeef")
        config_path = tmp_path / "config.json"
        config_path.write_text('{"model_type": "whisper"}')
        with patch("app.domain.model_registry_service.hf_hub_download", return_value=str(config_path)):
            resolved = resolve_model_id("openai/whisper-base", revision="main", api=api)
        assert resolved == ResolvedModel(model_id="openai/whisper-base", revision="deadbeef", family="whisper")

    def test_resolves_supported_wav2vec2_family(self, tmp_path):
        api = self._mock_api([_sibling("model.safetensors")], sha="cafef00d")
        config_path = tmp_path / "config.json"
        config_path.write_text('{"model_type": "wav2vec2"}')
        with patch("app.domain.model_registry_service.hf_hub_download", return_value=str(config_path)):
            resolved = resolve_model_id("facebook/wav2vec2-base-960h", api=api)
        assert resolved.family == "wav2vec2"

    def test_hub_error_on_model_info_raises_hub_unavailable(self):
        api = MagicMock()
        api.model_info.side_effect = _hub_error("down")
        with pytest.raises(ModelRegistryError) as exc_info:
            resolve_model_id("some/model", api=api)
        assert exc_info.value.code == HUB_UNAVAILABLE


class TestModelRegistryLRU:
    def _fake_loaded(self, model_id, revision="rev"):
        return LoadedModel(
            model_id=model_id,
            revision=revision,
            family="whisper",
            weights_sha256="deadbeef",
            model=MagicMock(),
        )

    def test_cache_hit_by_resolved_revision_skips_download(self):
        """A moving ref ('main') must be checked against the cache by its
        resolved commit sha, not the literal string -- otherwise repeated
        calls with revision='main' would never hit the cache. resolve_model_id
        (a cheap metadata call) still runs every time; download_and_load
        (the expensive part) must not.
        """
        reg = ModelRegistry(max_cache_size=4)
        loaded = self._fake_loaded("openai/whisper-base", revision="deadbeef")
        reg._cache["openai/whisper-base@deadbeef"] = loaded

        with patch("app.domain.model_registry_service.resolve_model_id") as mock_resolve, \
             patch("app.domain.model_registry_service.download_and_load") as mock_download:
            mock_resolve.return_value = ResolvedModel(
                model_id="openai/whisper-base", revision="deadbeef", family="whisper"
            )
            result = reg.get("openai/whisper-base", revision="main")

        assert result is loaded
        mock_resolve.assert_called_once()
        mock_download.assert_not_called()

    def test_evicts_oldest_beyond_max_cache_size(self):
        reg = ModelRegistry(max_cache_size=2)
        with patch("app.domain.model_registry_service.resolve_model_id") as mock_resolve, \
             patch("app.domain.model_registry_service.download_and_load") as mock_download:
            for name in ["model-a", "model-b", "model-c"]:
                mock_resolve.return_value = ResolvedModel(model_id=name, revision="main", family="whisper")
                mock_download.return_value = self._fake_loaded(name, revision="main")
                reg.get(name)

        assert len(reg) == 2
        assert "model-a@main" not in reg._cache
        assert "model-b@main" in reg._cache
        assert "model-c@main" in reg._cache

    def test_eviction_releases_the_model_reference(self):
        reg = ModelRegistry(max_cache_size=1)
        evicted_model = MagicMock()
        with patch("app.domain.model_registry_service.resolve_model_id") as mock_resolve, \
             patch("app.domain.model_registry_service.download_and_load") as mock_download:
            mock_resolve.return_value = ResolvedModel(model_id="model-a", revision="main", family="whisper")
            mock_download.return_value = LoadedModel(
                model_id="model-a", revision="main", family="whisper",
                weights_sha256="x", model=evicted_model,
            )
            reg.get("model-a")

            mock_resolve.return_value = ResolvedModel(model_id="model-b", revision="main", family="whisper")
            mock_download.return_value = self._fake_loaded("model-b", revision="main")
            reg.get("model-b")

        evicted_model.to.assert_called_with("cpu")
        assert "model-a@main" not in reg._cache

    def test_recently_used_entry_is_not_the_eviction_target(self):
        reg = ModelRegistry(max_cache_size=2)
        with patch("app.domain.model_registry_service.resolve_model_id") as mock_resolve, \
             patch("app.domain.model_registry_service.download_and_load") as mock_download:
            for name in ["model-a", "model-b"]:
                mock_resolve.return_value = ResolvedModel(model_id=name, revision="main", family="whisper")
                mock_download.return_value = self._fake_loaded(name, revision="main")
                reg.get(name)

            # Touch model-a again so model-b becomes the least-recently-used one.
            # get() always re-resolves, so the mock must be pointed back at model-a
            # (its return_value otherwise still holds the last loop iteration's model-b).
            mock_resolve.return_value = ResolvedModel(model_id="model-a", revision="main", family="whisper")
            reg.get("model-a")

            mock_resolve.return_value = ResolvedModel(model_id="model-c", revision="main", family="whisper")
            mock_download.return_value = self._fake_loaded("model-c", revision="main")
            reg.get("model-c")

        assert "model-a@main" in reg._cache
        assert "model-b@main" not in reg._cache
        assert "model-c@main" in reg._cache


class _FakeAttention(nn.Module):
    def forward(self, x):
        return x, torch.softmax(x, dim=-1)


class _FakeWhisperEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttention()

    def forward(self, x):
        out, _ = self.self_attn(x)
        return out


class _FakeWhisperModel(nn.Module):
    """Shaped like a real WhisperModel closely enough for HookManager to
    resolve encoder/attention layers, without downloading real weights."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([_FakeWhisperEncoderLayer()])
        self.config = SimpleNamespace(output_attentions=False)

    def forward(self, x):
        for layer in self.encoder.layers:
            x = layer(x)
        return x

    @classmethod
    def from_pretrained(cls, path, attn_implementation="eager"):
        return cls()


class _FakeUnresolvableModel(nn.Module):
    """No encoder/layers HookManager can find -- simulates an architecture
    whose config.json claims a supported model_type but whose module tree
    doesn't actually match (download_and_load must degrade, not crash)."""

    @classmethod
    def from_pretrained(cls, path, attn_implementation="eager"):
        return cls()


class TestDownloadAndLoadHookWiring:
    """LIT-207's DoD requires a loaded model to expose selectable layers via
    hooks -- these exercise that wiring (ModelRegistry -> HookManager)."""

    def test_populates_available_layers_for_a_resolvable_architecture(self, tmp_path):
        resolved = ResolvedModel(model_id="fake/whisper", revision="abc123", family="whisper")
        with patch("app.domain.model_registry_service.snapshot_download", return_value=str(tmp_path)):
            loaded = download_and_load(resolved, model_class=_FakeWhisperModel)

        assert loaded.available_layers == ["encoder.layers.0", "encoder.layers.0.self_attn"]

    def test_degrades_to_empty_layers_without_failing_the_load(self, tmp_path):
        resolved = ResolvedModel(model_id="fake/mystery", revision="abc123", family="whisper")
        with patch("app.domain.model_registry_service.snapshot_download", return_value=str(tmp_path)):
            loaded = download_and_load(resolved, model_class=_FakeUnresolvableModel)

        # The model still loads (usable for inference) even though hook
        # layer resolution failed -- LIT-207 doesn't gate loading on it.
        assert loaded.available_layers == []
        assert loaded.model is not None

    def test_attach_hooks_returns_a_working_hook_manager(self, tmp_path):
        resolved = ResolvedModel(model_id="fake/whisper", revision="abc123", family="whisper")
        with patch("app.domain.model_registry_service.snapshot_download", return_value=str(tmp_path)):
            loaded = download_and_load(resolved, model_class=_FakeWhisperModel)

        with loaded.attach_hooks() as hooks:
            loaded.model(torch.randn(1, 4, 8))
            assert set(hooks.captured) == set(loaded.available_layers)
