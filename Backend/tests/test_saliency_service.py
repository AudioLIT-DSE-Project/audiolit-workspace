"""Unit tests for SaliencyService & Core (SRS FR9, SAD §5.2)."""
import pytest
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import librosa
from pathlib import Path

# Import pure helper functions and core IG/Grad-CAM
from app.domain.saliency_service import (
    audio_to_spectrogram,
    spectrogram_patch_bounds,
    occlusion_attribution,
    attribution_timeline,
    integrated_gradients,
    compute_grad_cam,
    find_last_conv_layer,
    generate_wav2vec2_saliency
)

# Mock the model_loader_service to avoid downloading HuggingFace models in CI
import app.domain.model_loader_service as model_loader_service

class MockFeatureExtractor:
    def __call__(self, audio, sampling_rate, return_tensors="pt", padding=True):
        class Inputs:
            def __init__(self, audio):
                self.input_values = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                self.attention_mask = torch.ones_like(self.input_values)
            
            def __contains__(self, key):
                return key in ["input_values", "attention_mask"]
                
        return Inputs(audio)

class MockEmoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16000, 6)
        self.config = type('obj', (object,), {'id2label': {0: 'neutral', 1: 'happy', 2: 'sad'}})()

    def forward(self, input_values, attention_mask=None):
        x = input_values[:, :16000]
        if x.shape[-1] < 16000:
            x = torch.nn.functional.pad(x, (0, 16000 - x.shape[-1]))
        logits = self.linear(x)
        
        class MockOutput:
            def __init__(self, logits):
                self.logits = logits
        return MockOutput(logits)

    def wav2vec2(self, input_values, attention_mask=None):
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

class DummyAudioModel(nn.Module):
    """Dummy model mimicking an audio classifier for testing hooks."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 16, 3, padding=1)
        self.linear = nn.Linear(16, 6)  # 6 output classes
        
    def forward(self, x):
        # x shape: [1, time] -> [1, 1, time] for Conv1d
        x = x.unsqueeze(1)
        x = self.conv(x)        # [1, 16, time]
        x = x.mean(dim=-1)     # Global average pool -> [1, 16]
        logits = self.linear(x)# [1, 6]
        return logits


class TestPureHelperFunctions:
    """Test the mathematical helpers used for spectrogram mapping and occlusion."""
    
    def test_audio_to_spectrogram(self, dummy_audio_file):
        audio, sr = librosa.load(str(dummy_audio_file), sr=16000)
        spect = audio_to_spectrogram(audio)
        assert spect.ndim == 2
        assert spect.shape[0] == 257
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
        assert bounds[-1] == (94, 105)

    def test_occlusion_attribution(self):
        spect = np.random.rand(32, 32).astype(np.float32)
        def mock_score_fn(s):
            return float(np.sum(s))
        importance = occlusion_attribution(mock_score_fn, spect, n_freq_patches=4, n_time_patches=4)
        assert importance.shape == (4, 4)
        assert importance.dtype == np.float32


class TestSaliencyCore:
    """Test IntegratedGradients, Grad-CAM, and Timeline mapping (SRS FR9)."""
    
    @pytest.fixture
    def dummy_inputs(self):
        return torch.randn(1, 16000, requires_grad=True)

    def test_integrated_gradients(self, dummy_inputs):
        """Test IG maps output scores back to input dimensions."""
        model = DummyAudioModel()
        # Return the score for class 0 directly to satisfy Captum's scalar requirement
        def forward_fn(x):
            return model(x)[:, 0]
            
        attributions = integrated_gradients(
            forward_fn=forward_fn,
            inputs=dummy_inputs,
            n_steps=10
        )
        assert isinstance(attributions, torch.Tensor)
        assert attributions.shape == dummy_inputs.shape

    def test_compute_grad_cam(self, dummy_inputs):
        """Test Grad-CAM isolates spatial activation shifts."""
        model = DummyAudioModel()
        target_layer = model.conv
        
        cam_map = compute_grad_cam(
            model=model,
            inputs=dummy_inputs,
            target_layer=target_layer,
            target_index=0
        )
        assert isinstance(cam_map, np.ndarray)
        # Conv1d output spatial dim should be 16000
        assert cam_map.ndim == 1
        assert len(cam_map) == 16000
        assert np.max(cam_map) <= 1.0
        assert np.min(cam_map) >= 0.0

    def test_find_last_conv_layer(self):
        model = DummyAudioModel()
        layer = find_last_conv_layer(model)
        assert isinstance(layer, nn.Conv1d)

    def test_attribution_timeline(self):
        """Test gradient arrays are wrapped into ms-aligned JSON streams."""
        attributions = torch.randn(1, 16, 100)
        timeline = attribution_timeline(attributions, sample_rate=16000, hop_length=512)
        
        assert isinstance(timeline, list)
        assert len(timeline) == 100
        assert "t_ms" in timeline[0]
        assert "weight" in timeline[0]
        assert 0.0 <= timeline[0]["weight"] <= 1.0
        assert timeline[0]["t_ms"] == 0.0
        assert timeline[1]["t_ms"] == round((512 / 16000) * 1000, 3)


class TestWav2Vec2Saliency:
    """Test the main generate_wav2vec2_saliency pipeline with mocks."""
    
    def test_generates_2d_aligned_saliency(self, mock_model_loader, dummy_audio_file):
        """Verify DoD: emotion prediction inspected with spectrogram-aligned saliency overlay."""
        result = generate_wav2vec2_saliency(str(dummy_audio_file), method="gradcam")
        
        assert result["model"] == "wav2vec2"
        assert result["method"] == "gradcam"
        assert "emotion" in result
        
        assert "series" in result
        assert len(result["series"]) > 0
        
        assert "base_spectrogram" in result
        assert "saliency_matrix" in result
        
        base_spect = np.array(result["base_spectrogram"])
        saliency_matrix = np.array(result["saliency_matrix"])
        
        assert base_spect.ndim == 2
        assert saliency_matrix.ndim == 2
        assert base_spect.shape == saliency_matrix.shape
        
        assert np.max(saliency_matrix) <= 1.0
        assert np.min(saliency_matrix) >= 0.0
