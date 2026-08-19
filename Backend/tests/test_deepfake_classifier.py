"""Tests for the binary audio-deepfake classifier (LIT-128, FR7).

The real Wav2Vec2 model is never downloaded — the model + feature extractor are
mocked, and predict_deepfake is exercised over synthetic audio. Label
normalization is unit-tested directly.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from transformers import BatchFeature

import app.domain.model_loader_service as ml
from app.domain.model_loader_service import (
    DEEPFAKE_BONA_FIDE,
    DEEPFAKE_SPOOF,
    _normalize_deepfake_label,
    predict_deepfake,
)


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    def __init__(self, id2label, logits):
        self.config = types.SimpleNamespace(id2label=id2label)
        self._logits = logits

    def __call__(self, input_values=None, attention_mask=None):
        return _FakeOutput(torch.tensor([self._logits], dtype=torch.float32))


def _fake_feature_extractor(audio, sampling_rate, return_tensors, padding):
    return BatchFeature({"input_values": torch.zeros(1, len(audio))})


def _install_fake(monkeypatch, id2label, logits):
    # add_model is set (not None), so ensure_add_model_loaded() skips real loading.
    monkeypatch.setattr(ml, "add_feature_extractor", _fake_feature_extractor)
    monkeypatch.setattr(ml, "add_model", _FakeModel(id2label, logits))


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    p = tmp_path / "clip.wav"
    t = np.linspace(0, 0.2, 3200, endpoint=False)
    sf.write(p, (0.1 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), 16_000)
    return p


class TestNormalizeLabel:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("fake", DEEPFAKE_SPOOF),
            ("spoof", DEEPFAKE_SPOOF),
            ("synthetic", DEEPFAKE_SPOOF),
            ("deepfake", DEEPFAKE_SPOOF),
            ("real", DEEPFAKE_BONA_FIDE),
            ("bonafide", DEEPFAKE_BONA_FIDE),
            ("genuine", DEEPFAKE_BONA_FIDE),
            ("LABEL_0", DEEPFAKE_BONA_FIDE),
        ],
    )
    def test_maps_varied_label_strings(self, raw, expected):
        assert _normalize_deepfake_label(raw) == expected


class TestPredictDeepfake:
    def test_spoof_when_fake_logit_dominates(self, monkeypatch, clip):
        _install_fake(monkeypatch, {0: "real", 1: "fake"}, [0.1, 3.0])
        out = predict_deepfake(str(clip))

        assert out["predicted_label"] == DEEPFAKE_SPOOF
        assert out["synthetic_probability"] > 0.5
        assert set(out["probabilities"]) == {DEEPFAKE_BONA_FIDE, DEEPFAKE_SPOOF}
        assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-5
        assert out["confidence"] == max(out["probabilities"].values())

    def test_bona_fide_when_real_logit_dominates(self, monkeypatch, clip):
        _install_fake(monkeypatch, {0: "bonafide", 1: "spoof"}, [3.0, 0.1])
        out = predict_deepfake(str(clip))

        assert out["predicted_label"] == DEEPFAKE_BONA_FIDE
        assert out["synthetic_probability"] < 0.5

    def test_handles_string_keyed_id2label(self, monkeypatch, clip):
        # Some checkpoints key id2label by strings ("0"/"1") rather than ints.
        _install_fake(monkeypatch, {"0": "real", "1": "fake"}, [0.1, 3.0])
        out = predict_deepfake(str(clip))
        assert out["predicted_label"] == DEEPFAKE_SPOOF


class TestSecondAddCheckpoint:
    """model_key="wav2vec2-add" must use its own cache entry, independent of
    the default (MelodyMachine) globals mocked via _install_fake above."""

    def test_uses_extra_cache_not_default_globals(self, monkeypatch, clip):
        # Default-model globals stay unset/mocked to something that would fail
        # loudly if wrongly consulted for the second checkpoint.
        monkeypatch.setattr(ml, "add_feature_extractor", None)
        monkeypatch.setattr(ml, "add_model", None)

        gustking_id = ml._ADD_MODEL_REGISTRY["wav2vec2-add"]
        fake_model = _FakeModel({0: "real", 1: "fake"}, [0.1, 3.0])
        monkeypatch.setitem(
            ml._add_model_extra_cache, gustking_id, (_fake_feature_extractor, fake_model)
        )

        out = predict_deepfake(str(clip), model_key="wav2vec2-add")

        assert out["predicted_label"] == DEEPFAKE_SPOOF


class TestDeepfakeTimeline:
    """FR7.2 — windowed confidence across the clip."""

    def _stub(self, monkeypatch, n_classes=2):
        import app.domain.model_loader_service as ml
        import numpy as np, torch

        loads = []

        class _FE:
            def __call__(self, audio, sampling_rate=16000, **kw):
                return type("O", (), {
                    "input_values": torch.zeros(1, len(audio)),
                    "__contains__": lambda self, k: False,
                })()

        class _M:
            config = type("C", (), {"id2label": {0: "bonafide", 1: "spoof"}})()

            def __call__(self, input_values=None, attention_mask=None):
                return type("O", (), {"logits": torch.tensor([[0.2, 0.8]])})()

        def fake_ensure(model_key="melody-machine"):
            loads.append(model_key)
            return _FE(), _M(), "cpu"

        monkeypatch.setattr(ml, "ensure_add_model_loaded", fake_ensure)
        monkeypatch.setattr(ml.librosa, "load",
                            lambda p, sr=16000: (np.zeros(5 * sr, dtype="float32"), sr))
        return loads

    def test_window_count_and_ordering(self, monkeypatch):
        from app.domain.model_loader_service import predict_deepfake_timeline
        self._stub(monkeypatch)
        tl = predict_deepfake_timeline("x.wav", window_s=1.0, overlap=0.5)
        assert len(tl) == 9  # 5 s at 1 s / 50 % overlap
        assert all(a["start_s"] <= b["start_s"] for a, b in zip(tl, tl[1:]))
        assert all(0.0 <= w["synthetic_probability"] <= 1.0 for w in tl)
        assert tl[-1]["end_s"] == 5.0

    def test_model_is_loaded_once_not_per_window(self, monkeypatch):
        from app.domain.model_loader_service import predict_deepfake_timeline
        loads = self._stub(monkeypatch)
        predict_deepfake_timeline("x.wav", window_s=1.0, overlap=0.5)
        assert len(loads) == 1, f"model loaded {len(loads)} times; reload per window is the bug"

    def test_rejects_impossible_parameters(self, monkeypatch):
        from app.domain.model_loader_service import predict_deepfake_timeline
        self._stub(monkeypatch)
        with pytest.raises(ValueError):
            predict_deepfake_timeline("x.wav", overlap=1.0)
        with pytest.raises(ValueError):
            predict_deepfake_timeline("x.wav", window_s=0)
