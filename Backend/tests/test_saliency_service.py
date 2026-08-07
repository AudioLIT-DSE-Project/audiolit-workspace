"""Unit tests for SaliencyService (SRS FR9, SAD §5.2)."""
import pytest
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

# Import pure helper functions
from app.domain.saliency_service import (
    audio_to_spectrogram,
    spectrogram_patch_bounds,
    occlusion_attribution,
    attribution_timeline
)

# Mock the model_loader_service to avoid downloading HuggingFace models in CI
import app.domain.model_loader_service as model_loader_service
from app.domain.saliency_service import generate_wav2vec2_saliency

class MockFeatureExtractor:
    def __call__(self, audio, sampling_rate, return_tensors="pt", padding=True):
        class Inputs:
            def __init__(self, audio):
                self.input_values = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                self.attention_mask = torch.ones_like(self.input_values)
        return Inputs(audio)

class MockEmoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 16, 3, padding=1)
        self.linear = torch.nn.Linear(16000, 6)  # 6 emotion classes
        self.config = type('obj', (object,), {'id2label': {0: 'neutral', 1: 'happy', 2: 'sad'}})()

    def forward(self, input_values, attention_mask=None):
        x = input_values[:, :16000]
        if x.shape[-1] < 16000:
            x = torch.nn.functional.pad(x, (0, 16000 - x.shape[-1]))
        x = x.unsqueeze(1)  # [B, 1, T]
        x = self.conv(x)
        x = x.squeeze(1)     # [B, T]
        logits = self.linear(x)
        
        class MockOutput:
            def __init__(self, logits):
                self.logits = logits
        return MockOutput(logits)

    def wav2vec2(self, input_values, attention_mask=None):
        # Mock internal wav2vec2 for the fallback energy map
        x = input_values[:, :16000]
        if x.shape[-1] < 16000:
            x = torch.nn.functional.pad(x, (0, 16000 - x.shape[-1]))
        class MockOutput:
            def __init__(self, x):
                self.last_hidden_state = x.unsqueeze(-1).repeat(1, 1, 32)
        return MockOutput(x)

@pytest.fixture
def mock_model_loader(monkeypatch):
    monkeypatch.setattr(model_loader_service, "feature_extractor", MockFeatureExtractor())
    monkeypatch.setattr(model_loader_service, "emo_model", MockEmoModel())
    monkeypatch.setattr(model_loader_service, "emo_device", torch.device("cpu"))
    monkeypatch.setattr(model_loader_service, "ensure_emo_model_loaded", lambda: None)

@pytest.fixture
def dummy_audio_file(tmp_path: Path) -> Path:
    """Creates a temporary dummy WAV file."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    file_path = tmp_path / "dummy_audio.wav"
    sf.write(file_path, y, sr)
    return file_path


class TestPureHelperFunctions:
    """Test the mathematical helpers used for spectrogram mapping and occlusion."""
    
    def test_audio_to_spectrogram(self, dummy_audio_file):
        import librosa
        audio, sr = librosa.load(str(dummy_audio_file), sr=16000)
        spect = audio_to_spectrogram(audio)
        
        assert spect.ndim == 2
        assert spect.shape[0] == 257  # n_fft/2 + 1
        assert spect.shape[1] > 0

    def test_spectrogram_patch_bounds_exact(self):
        bounds = spectrogram_patch_bounds(100, 10)
        assert len(bounds) == 10
        assert bounds[0] == (0, 10)
        assert bounds[-1] == (90, 100)

    def test_spectrogram_patch_bounds_uneven(self):
        bounds = spectrogram_patch_bounds(105, 10)
        assert len(bounds) == 10
        assert bounds[0] == (0, 10)
        assert bounds[-1] == (94, 105)  # Last patch takes the remainder

    def test_occlusion_attribution(self):
        # Create a dummy 2D spectrogram
        spect = np.random.rand(32, 32).astype(np.float32)
        
        # Mock score function: returns the mean of the spectrogram
        def mock_score_fn(s):
            return float(np.mean(s))
        
        importance = occlusion_attribution(mock_score_fn, spect, n_freq_patches=4, n_time_patches=4)
        
        assert importance.shape == (4, 4)
        # Occluding a patch should reduce the score, so importance (base - occluded) should be positive
        assert np.all(importance >= 0)

    def test_attribution_timeline(self):
        # Create dummy attributions: [batch, channels, time]
        attributions = torch.randn(1, 16, 100)
        timeline = attribution_timeline(attributions, sample_rate=16000, hop_length=512)
        
        assert isinstance(timeline, list)
        assert len(timeline) == 100
        assert "t_ms" in timeline[0]
        assert "weight" in timeline[0]
        assert 0.0 <= timeline[0]["weight"] <= 1.0
        assert timeline[0]["t_ms"] == 0.0


class TestWav2Vec2Saliency:
    """Test the main generate_wav2vec2_saliency pipeline with mocks."""
    
    def test_generates_2d_aligned_saliency(self, mock_model_loader, dummy_audio_file):
        """Verify DoD: emotion prediction inspected with spectrogram-aligned saliency overlay."""
        result = generate_wav2vec2_saliency(str(dummy_audio_file), method="gradcam")
        
        assert result["model"] == "wav2vec2"
        assert result["method"] == "gradcam"
        assert "emotion" in result
        
        # Verify 1D series exists
        assert "series" in result
        assert len(result["series"]) > 0
        
        # Verify 2D matrices exist
        assert "base_spectrogram" in result
        assert "saliency_matrix" in result
        
        base_spect = np.array(result["base_spectrogram"])
        saliency_matrix = np.array(result["saliency_matrix"])
        
        # Both must be 2D
        assert base_spect.ndim == 2
        assert saliency_matrix.ndim == 2
        
        # The shapes MUST match for UI canvas overlay (DoD requirement)
        assert base_spect.shape == saliency_matrix.shape
        
        # Values should be normalized
        assert np.max(saliency_matrix) <= 1.0
        assert np.min(saliency_matrix) >= 0.0
