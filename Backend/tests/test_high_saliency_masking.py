import pytest
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from app.domain.perturbation_service import (
    HighSaliencyMaskingEngine,
    apply_high_saliency_mask,
)

@pytest.fixture
def dummy_audio_file(tmp_path):
    """Generates a 1-second synthetic 16kHz audio file for testing."""
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    data = 0.8 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    file_path = tmp_path / "test_saliency_audio.wav"
    sf.write(str(file_path), data, sample_rate)
    return str(file_path)

def test_extract_top_k_ms_intervals():
    """Verify extract_top_k_ms_intervals extracts correct millisecond intervals."""
    attributions = np.zeros(100, dtype=np.float32)
    attributions[20:30] = 10.0  # 10% top saliency peak

    intervals = HighSaliencyMaskingEngine.extract_top_k_ms_intervals(
        attributions=attributions,
        sample_rate=16000,
        k_percent=10.0,
        total_samples=16000,
    )

    assert len(intervals) > 0
    # 20 to 30 out of 100 on a 1000ms clip -> 200ms to 300ms
    t_start, t_end = intervals[0]
    assert t_start == 200.0
    assert t_end == 300.0

def test_mask_audio_by_intervals_zero_mode():
    """Verify mask_audio_by_intervals zeroes out specified intervals."""
    waveform = torch.ones((1, 16000), dtype=torch.float32)
    intervals = [(100.0, 200.0)]  # 100ms to 200ms

    masked = HighSaliencyMaskingEngine.mask_audio_by_intervals(
        waveform=waveform,
        sample_rate=16000,
        intervals=intervals,
        mode="zero",
    )

    start_sample = int(0.1 * 16000)
    end_sample = int(0.2 * 16000)
    assert torch.sum(masked[:, start_sample:end_sample]) == 0.0
    assert torch.sum(masked[:, :start_sample]) == start_sample
    assert torch.sum(masked[:, end_sample:]) == (16000 - end_sample)

def test_mask_audio_by_intervals_modes():
    """Verify noise and mean-blur masking modes alter target regions."""
    waveform = torch.ones((1, 16000), dtype=torch.float32)
    intervals = [(200.0, 400.0)]

    masked_noise = HighSaliencyMaskingEngine.mask_audio_by_intervals(
        waveform, 16000, intervals, mode="noise"
    )
    masked_mean = HighSaliencyMaskingEngine.mask_audio_by_intervals(
        waveform, 16000, intervals, mode="mean"
    )

    start_sample = int(0.2 * 16000)
    end_sample = int(0.4 * 16000)
    assert not torch.allclose(masked_noise[:, start_sample:end_sample], waveform[:, start_sample:end_sample])
    assert masked_mean.shape == waveform.shape

def test_apply_high_saliency_mask_service(dummy_audio_file):
    """Verify high-level apply_high_saliency_mask creates file output and metadata."""
    attributions = np.random.randn(100).astype(np.float32)
    res = apply_high_saliency_mask(
        audio_path=dummy_audio_file,
        attributions=attributions,
        k_percent=20.0,
        mode="zero",
    )

    assert res["success"] is True
    assert "masked_file" in res
    assert Path(res["masked_file"]).exists()
    assert res["k_percent"] == 20.0
    assert res["mode"] == "zero"
    assert "masked_intervals_ms" in res
    assert len(res["preview_bytes"]) > 0
