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
import time
from app.domain.saliency_service import generate_perturbation_matrix, audio_to_spectrogram

class TestPerturbationMatrixEngine:
    """Test 2D Spectrogram Patch Segmentation & Perturbation Matrix Engine (SRS FR8)."""

    def test_generate_500_variants_within_duration(self, dummy_audio_file):
        """Verify DoD: Perturbation routine generates 500 masked variants successfully."""
        import librosa
        audio, sr = librosa.load(str(dummy_audio_file), sr=16000)
        spectrogram = audio_to_spectrogram(audio)
        
        start_time = time.time()
        
        # Generate 500 variants using zero masking
        variants = generate_perturbation_matrix(
            spectrogram=spectrogram,
            n_patches_freq=8,
            n_patches_time=8,
            n_variants=500,
            perturbation_type="zero",
            random_state=42
        )
        
        duration = time.time() - start_time
        
        # Should return a 3D array: (500, freq_bins, time_bins)
        assert variants.ndim == 3
        assert variants.shape[0] == 500
        assert variants.shape[1] == spectrogram.shape[0]
        assert variants.shape[2] == spectrogram.shape[1]
        
        # The base spectrogram was modified (contains zeros)
        assert not np.array_equal(variants[0], spectrogram)
        
        # DoD: "within expected duration thresholds". 500 variants of a 2s audio 
        # spectrogram should take less than 2 seconds even on CPU.
        assert duration < 5.0, f"Perturbation generation took too long: {duration:.2f}s"
        
    def test_noise_perturbation_type(self, dummy_audio_file):
        """Verify noise perturbation applies random values instead of zeros."""
        import librosa
        audio, sr = librosa.load(str(dummy_audio_file), sr=16000)
        spectrogram = audio_to_spectrogram(audio)
        
        variants = generate_perturbation_matrix(
            spectrogram=spectrogram,
            n_variants=10,
            perturbation_type="noise",
            noise_level=0.5,
            random_state=42
        )
        
        # Check that the variants contain noise (values not strictly zero or original)
        assert np.any(variants != 0.0)
        assert not np.array_equal(variants[0], spectrogram)
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
        self.conv = torch.nn.Conv1d(1, 4, 3, padding=1)
        self.linear = torch.nn.Linear(16000 * 4, 6)
        self.config = type('obj', (object,), {'id2label': {0: 'neutral', 1: 'happy', 2: 'sad'}})()

    def forward(self, input_values, attention_mask=None):
        inp = input_values.unsqueeze(1) if input_values.dim() == 2 else input_values
        conv_out = self.conv(inp)
        x = conv_out.view(conv_out.shape[0], -1)
        if x.shape[-1] < 16000 * 4:
            x = torch.nn.functional.pad(x, (0, 16000 * 4 - x.shape[-1]))
        else:
            x = x[:, :16000 * 4]
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


class DummyAddModel(nn.Module):
    """Mimics the real HF Wav2Vec2 feature-encoder conv stack: the *first*
    Conv1d has in_channels=1, every later one has in_channels == a wider
    channel count (like config.conv_dim). find_last_conv_layer() returns the
    LAST one, so a test whose model has only a single Conv1d layer can't
    catch the "target_layer.in_channels == 1" bug in
    generate_add_gradcam_saliency - this shape is what actually exposed it.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv1d(8, 8, 3, padding=1)
        self.linear = nn.Linear(8, 2)
        self.config = type("obj", (object,), {"id2label": {0: "bonafide", 1: "spoof"}})()

    def forward(self, input_values, attention_mask=None):
        x = input_values.unsqueeze(1) if input_values.dim() == 2 else input_values
        x = self.conv1(x)
        x = self.conv2(x)
        logits = self.linear(x.mean(dim=-1))

        class MockOutput:
            def __init__(self, logits):
                self.logits = logits

        return MockOutput(logits)

    def wav2vec2(self, input_values, attention_mask=None):
        x = input_values if input_values.dim() == 2 else input_values.squeeze(1)
        x = x[:, :16000]
        if x.shape[-1] < 16000:
            x = torch.nn.functional.pad(x, (0, 16000 - x.shape[-1]))

        class MockOutput:
            def __init__(self, x):
                self.last_hidden_state = x.unsqueeze(-1).repeat(1, 1, 8)

        return MockOutput(x)


class TestAddAndGradCamSaliency:
    """Test ADD model saliency across all XAI methods (FR8.1, FR8.2, FR9)."""

    def test_generate_saliency_add_model_gradcam_success(self, monkeypatch, dummy_audio_file):
        from app.domain.saliency_service import generate_saliency

        dummy_model = DummyAddModel()
        monkeypatch.setattr(
            "app.domain.model_loader_service.ensure_add_model_loaded",
            lambda key: (MockFeatureExtractor(), dummy_model, "cpu"),
        )

        res = generate_saliency(str(dummy_audio_file), model="melody-machine", method="gradcam")
        assert res["model"] == "melody-machine"
        assert res["method"] == "gradcam"
        assert res["provenance"] == "measured"
        assert res["provenance_reason"] is None
        # Every other saliency path keys segments as start_time/end_time, and
        # the frontend (SaliencyVisualization.tsx) only reads those two names.
        # This function alone used to emit "start"/"end", which stayed hidden
        # as long as Grad-CAM never actually succeeded for a real ADD
        # checkpoint - once fixed, the mismatch surfaced as a page-crashing
        # `undefined.toFixed()` with no error boundary to catch it.
        assert len(res["segments"]) > 0
        for segment in res["segments"]:
            assert isinstance(segment["start_time"], (int, float))
            assert isinstance(segment["end_time"], (int, float))
        assert "saliency_matrix" in res
        assert "base_spectrogram" in res

    @pytest.mark.parametrize("method", ["integrated_gradients", "ig", "lime", "shap"])
    def test_generate_saliency_add_model_non_gradcam_now_works(self, monkeypatch, dummy_audio_file, method):
        """FR9 commits Integrated Gradients "for ASR, SER, and ADD"; FR8.1 commits
        LIME/SHAP generally. These used to unconditionally 400 for every ADD
        model regardless of method - see LIT-* fix wiring generate_add_saliency.
        """
        from app.domain.saliency_service import generate_saliency

        dummy_model = DummyAddModel()
        monkeypatch.setattr(
            "app.domain.model_loader_service.ensure_add_model_loaded",
            lambda key: (MockFeatureExtractor(), dummy_model, "cpu"),
        )

        res = generate_saliency(str(dummy_audio_file), model="wav2vec2-add", method=method)
        assert res["model"] == "wav2vec2-add"
        assert res["method"] == method
        assert res["predicted_label"] in ("bona-fide", "spoof")
        assert len(res["series"]) > 0
        assert len(res["segments"]) > 0

    def test_generate_saliency_add_model_unsupported_method_raises(self, monkeypatch, dummy_audio_file):
        from app.domain.saliency_service import generate_saliency

        dummy_model = DummyAddModel()
        monkeypatch.setattr(
            "app.domain.model_loader_service.ensure_add_model_loaded",
            lambda key: (MockFeatureExtractor(), dummy_model, "cpu"),
        )

        with pytest.raises(ValueError, match="Unsupported saliency method"):
            generate_saliency(str(dummy_audio_file), model="melody-machine", method="not-a-real-method")

    def test_generate_saliency_no_conv_layer_returns_unavailable(self, monkeypatch, dummy_audio_file):
        from app.domain.saliency_service import generate_add_gradcam_saliency

        class LinearOnlyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16000, 2)

            def forward(self, input_values):
                return self.fc(input_values)

        class DummyFeatureExtractor:
            def __call__(self, audio, sampling_rate, return_tensors="pt", padding=True):
                class Inputs:
                    def __init__(self, audio):
                        self.input_values = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

                return Inputs(audio)

        monkeypatch.setattr(
            "app.domain.model_loader_service.ensure_add_model_loaded",
            lambda key: (DummyFeatureExtractor(), LinearOnlyModule(), "cpu"),
        )

        res = generate_add_gradcam_saliency(str(dummy_audio_file), model_key="melody-machine")
        assert res["provenance"] == "unavailable"
        assert "no Conv1d/Conv2d layer" in res["provenance_reason"]

    def test_gradcam_not_equal_to_ig_for_wav2vec2(self, mock_model_loader, dummy_audio_file):
        from app.domain.saliency_service import generate_saliency

        res_gradcam = generate_saliency(str(dummy_audio_file), model="wav2vec2", method="gradcam")
        res_ig = generate_saliency(str(dummy_audio_file), model="wav2vec2", method="ig")

        cam_arr = np.array(res_gradcam["series"])
        ig_arr = np.array(res_ig["series"])
        assert not np.allclose(cam_arr, ig_arr)

    def test_generate_whisper_saliency_gradcam_end_to_end(self, dummy_audio_file):
        """End-to-end test for generate_whisper_saliency with method='gradcam'.

        Asserts a non-empty saliency_matrix is returned with MEASURED provenance.
        """
        from app.domain.saliency_service import generate_whisper_saliency

        res = generate_whisper_saliency(str(dummy_audio_file), model_size="whisper-base", method="gradcam")
        assert res["model"] == "openai/whisper-base"
        assert res["method"] == "gradcam"
        assert res["provenance"] == "measured"
        assert res["provenance_reason"] is None
        assert "saliency_matrix" in res
        matrix = np.array(res["saliency_matrix"])
        assert matrix.size > 0
        assert matrix.shape[0] == 128
        assert "base_spectrogram" in res
        base_spect = np.array(res["base_spectrogram"])
        assert base_spect.size > 0
        assert base_spect.shape[0] == 128
        assert res["total_duration"] > 0



