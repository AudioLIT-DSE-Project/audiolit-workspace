import pytest
import torch
import numpy as np
import os
import soundfile as sf
from pathlib import Path
from Backend.app.domain.perturbation_service import (
    mask_top_k_features,
    compute_deletion_score,
)

@pytest.fixture
def dummy_audio_file(tmp_path):
    """Generates a 1-second synthetic 16kHz audio file for testing."""
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    file_path = tmp_path / "test_faithfulness_audio.wav"
    sf.write(str(file_path), data, sample_rate)
    return str(file_path)

def test_mask_top_k_features_zeros_salient_timesteps():
    """Verify mask_top_k_features correctly zeroes out the top-K% salient timesteps."""
    waveform = torch.ones((1, 1000), dtype=torch.float32)
    attributions = np.zeros(1000, dtype=np.float32)
    attributions[100:200] = 5.0  # 100 timesteps have high saliency (10% of total)

    masked = mask_top_k_features(waveform, attributions, k_percent=10.0)

    assert masked.shape == (1, 1000)
    # The 100 highest saliency timesteps should be set to 0.0
    assert torch.sum(masked[:, 100:200]) == 0.0
    # Remaining timesteps should stay untouched (1.0)
    assert torch.sum(masked[:, :100]) == 100.0
    assert torch.sum(masked[:, 200:]) == 800.0

def test_mask_top_k_features_resampling():
    """Verify mask_top_k_features handles attribution maps with different frame lengths."""
    waveform = torch.ones((1, 16000), dtype=torch.float32)
    attributions = np.array([0.1, 0.9, 0.2, 0.8, 0.1], dtype=np.float32)  # 5 frames

    masked = mask_top_k_features(waveform, attributions, k_percent=20.0)
    assert masked.shape == (1, 16000)
    assert torch.sum(masked == 0.0) > 0

def test_compute_deletion_score_ser(dummy_audio_file):
    """Verify compute_deletion_score returns valid deletion metrics for SER models."""
    attributions = np.random.randn(16000).astype(np.float32)
    res = compute_deletion_score(
        audio_path=dummy_audio_file,
        attributions=attributions,
        model_type="ser",
        model_id="speech_emotion_eval",
        k_percent=15.0,
    )

    assert res["success"] is True
    assert res["model_type"] == "ser"
    assert "initial_confidence" in res
    assert "masked_confidence" in res
    assert "confidence_drop" in res
    assert "deletion_score" in res
    assert res["faithfulness_verdict"] in ("faithful", "unfaithful")

def test_compute_deletion_score_add(dummy_audio_file):
    """Verify compute_deletion_score returns valid deletion metrics for Deepfake ADD models."""
    attributions = np.random.randn(16000).astype(np.float32)
    res = compute_deletion_score(
        audio_path=dummy_audio_file,
        attributions=attributions,
        model_type="add",
        model_id="aasist_deepfake",
        k_percent=10.0,
    )

    assert res["success"] is True
    assert res["model_type"] == "add"
    assert "initial_confidence" in res
    assert "masked_confidence" in res
    assert "confidence_drop" in res
    assert "deletion_score" in res
    assert res["faithfulness_verdict"] in ("faithful", "unfaithful")
