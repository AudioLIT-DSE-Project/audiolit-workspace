import pytest
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from app.domain.perturbation_service import (
    evaluate_downstream_degradation,
)

@pytest.fixture
def dummy_audio_file(tmp_path):
    """Generates a 1-second synthetic 16kHz audio file for testing degradation scoring."""
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    file_path = tmp_path / "test_degradation_audio.wav"
    sf.write(str(file_path), data, sample_rate)
    return str(file_path)

def test_evaluate_downstream_degradation_ser(dummy_audio_file):
    """Verify evaluate_downstream_degradation evaluates SER degradation curves."""
    attributions = np.random.randn(16000).astype(np.float32)
    k_percentages = [10.0, 20.0, 50.0]

    res = evaluate_downstream_degradation(
        audio_path=dummy_audio_file,
        attributions=attributions,
        model_type="ser",
        model_id="speech_emotion_eval",
        k_percentages=k_percentages,
    )

    assert res["success"] is True
    assert res["model_type"] == "ser"
    assert "baseline_confidence" in res
    assert "target_class" in res
    assert len(res["degradation_curve"]) == 3
    assert "audc" in res
    assert "mean_degradation_score" in res
    assert res["degradation_trend"] in ("monotonic_decline", "non_monotonic")
    assert res["audit_verdict"] in ("faithful", "unfaithful")

def test_evaluate_downstream_degradation_add(dummy_audio_file):
    """Verify evaluate_downstream_degradation evaluates Deepfake ADD degradation curves."""
    attributions = np.random.randn(16000).astype(np.float32)
    k_percentages = [5.0, 15.0, 30.0]

    res = evaluate_downstream_degradation(
        audio_path=dummy_audio_file,
        attributions=attributions,
        model_type="add",
        model_id="aasist_deepfake",
        k_percentages=k_percentages,
    )

    assert res["success"] is True
    assert res["model_type"] == "add"
    assert "baseline_confidence" in res
    assert len(res["degradation_curve"]) == 3
    assert "audc" in res
    assert "mean_degradation_score" in res

def test_evaluate_downstream_degradation_missing_file():
    """Verify evaluate_downstream_degradation returns structured error on missing audio file."""
    res = evaluate_downstream_degradation(
        audio_path="non_existent_clip.wav",
        attributions=[0.1, 0.5, 0.2],
        model_type="ser",
    )

    assert res["success"] is False
    assert "error" in res
