"""Tests for the SER (Speech Emotion Recognition) inference entry point (LIT-206, FR6).

The emotion model + feature extractor are mocked (no HF download); predict_ser
is exercised end-to-end from a synthetic clip through to the emotion/probability/
confidence schema the multi-task response needs.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import BatchFeature

import app.domain.model_loader_service as ml
from app.domain.model_loader_service import predict_ser


class _Output:
    def __init__(self, logits):
        self.logits = logits


class _FakeEmoModel:
    def __init__(self, id2label, logits):
        self.config = types.SimpleNamespace(id2label=id2label)
        self._logits = logits

    def __call__(self, input_values, attention_mask=None):
        return _Output(self._logits)


def _fake_feature_extractor(audio, sampling_rate, return_tensors, padding):
    return BatchFeature({"input_values": torch.zeros(1, 3200)})


@pytest.fixture
def clip(tmp_path: Path):
    import soundfile as sf
    p = tmp_path / "c.wav"
    t = np.linspace(0, 0.2, 3200, endpoint=False)
    sf.write(p, (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), 16000)
    return str(p)


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch):
    # A non-None emo_model makes ensure_emo_model_loaded a no-op; patch it too for safety.
    monkeypatch.setattr(ml, "feature_extractor", _fake_feature_extractor)
    monkeypatch.setattr(ml, "ensure_emo_model_loaded", lambda: None)


class TestPredictSer:
    def test_returns_emotion_probabilities_and_confidence(self, monkeypatch, clip):
        # logits favor "happy" (index 1)
        model = _FakeEmoModel({0: "sad", 1: "happy", 2: "neutral"}, torch.tensor([[0.1, 3.0, 0.2]]))
        monkeypatch.setattr(ml, "emo_model", model)

        r = predict_ser(clip)
        assert r["predicted_emotion"] == "happy"
        assert set(r["probabilities"]) == {"sad", "happy", "neutral"}
        assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-5
        assert r["probabilities"]["happy"] > r["probabilities"]["sad"]

    def test_confidence_is_top_class_probability(self, monkeypatch, clip):
        model = _FakeEmoModel({0: "sad", 1: "happy"}, torch.tensor([[0.5, 2.5]]))
        monkeypatch.setattr(ml, "emo_model", model)

        r = predict_ser(clip)
        assert r["confidence"] == pytest.approx(r["probabilities"][r["predicted_emotion"]])
        assert r["confidence"] == pytest.approx(max(r["probabilities"].values()))

    def test_handles_string_keyed_id2label(self, monkeypatch, clip):
        # some checkpoints key id2label by string ("0", "1", ...)
        model = _FakeEmoModel({"0": "angry", "1": "calm"}, torch.tensor([[2.0, 0.1]]))
        monkeypatch.setattr(ml, "emo_model", model)

        r = predict_ser(clip)
        assert r["predicted_emotion"] == "angry"
        assert set(r["probabilities"]) == {"angry", "calm"}
