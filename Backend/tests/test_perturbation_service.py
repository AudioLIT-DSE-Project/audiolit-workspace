"""
Perturbation service — soundfile I/O & NumPy-Driven Masking (LIT-226 / LIT-175)

Verifies the torchaudio -> soundfile swap preserves the (channels, time)
float32 tensor contract the perturbation functions rely on, and that
perturb_and_save still produces a valid, playable output file.
Also verifies the new 2D time-freq slice masking, band-pass filter, 
Web Audio preview bytes, and the 16kHz mono orientation adapter.
"""

import numpy as np
import soundfile as sf
import torch
from pathlib import Path
import pytest

from app.domain.perturbation_service import (
    _load_waveform,
    _save_waveform,
    export_to_wav_bytes,
    add_gaussian_noise,
    apply_time_masking,
    apply_frequency_masking,
    apply_2d_time_freq_mask,
    apply_band_pass_filter,
    apply_pitch_shift,
    apply_time_stretch,
    apply_perturbations,
    perturb_and_save,
)


@pytest.fixture
def sample_audio_file(tmp_path: Path) -> Path:
    """Creates a temporary dummy WAV file for testing (Stereo, 22050 Hz)."""
    sr = 22050  # Intentionally not 16kHz to test adapter resampling
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Generate a stereo sine wave to test channel downmixing
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 660 * t)
    stereo_audio = np.stack([left, right], axis=1)  # Shape: (samples, channels)
    
    file_path = tmp_path / "dummy_stereo.wav"
    sf.write(file_path, stereo_audio, sr)
    return file_path


class TestWaveformIO:
    """_load_waveform / _save_waveform match torchaudio.load/save's old contract."""

    def test_orientation_adapter_downmixes_and_resamples(self, sample_audio_file: Path):
        """Verify soundfile reads (samples, channels) and adapter transposes to (channels, samples), 
        downmixes to mono, and resamples to 16kHz."""
        waveform, sample_rate = _load_waveform(str(sample_audio_file))

        assert isinstance(waveform, torch.Tensor)
        assert waveform.dtype == torch.float32
        assert waveform.dim() == 2  # (channels, time)
        assert waveform.shape[0] == 1  # Downmixed to mono
        assert sample_rate == 16000  # Resampled to 16kHz

    def test_save_waveform_round_trip(self, tmp_path: Path):
        # Two-channel synthetic waveform, matches the (channels, time) shape
        t = np.linspace(0, 1.0, 16000, False)
        left = 0.3 * np.sin(2 * np.pi * 440 * t)
        right = 0.3 * np.sin(2 * np.pi * 660 * t)
        waveform = torch.from_numpy(np.stack([left, right]).astype("float32"))

        out_path = tmp_path / "roundtrip.wav"
        _save_waveform(str(out_path), waveform, 16000)

        assert out_path.exists()
        reloaded, sr = _load_waveform(str(out_path))
        assert sr == 16000
        # Reloaded will be downmixed to mono by _load_waveform, so shape changes to (1, 16000)
        assert reloaded.shape[0] == 1
        assert reloaded.shape[1] == 16000

    def test_save_waveform_mono(self, tmp_path: Path):
        # apply_pitch_shift/apply_time_stretch always return a single-channel
        # (1, time) tensor regardless of input channel count — confirm that
        # still saves and reloads correctly.
        t = np.linspace(0, 0.5, 8000, False)
        mono = torch.from_numpy((0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32")).unsqueeze(0)

        out_path = tmp_path / "mono.wav"
        _save_waveform(str(out_path), mono, 16000)

        reloaded, sr = _load_waveform(str(out_path))
        assert sr == 16000
        assert reloaded.shape == mono.shape
        np.testing.assert_allclose(reloaded.numpy(), mono.numpy(), atol=1e-4)


class TestPerturbAndSave:
    """End-to-end perturb_and_save, unchanged behaviour via the new I/O path."""

    def test_noise_perturbation_succeeds(self, sample_audio_file: Path, tmp_path: Path):
        result = perturb_and_save(
            file_path=str(sample_audio_file),
            perturbations=[{"type": "noise", "params": {"noise_level": 0.01}}],
            output_dir=str(tmp_path),
        )

        assert result["success"] is True
        assert result["sample_rate"] == 16000
        assert result["duration_ms"] > 0
        assert result["applied_perturbations"][0]["status"] == "applied"

        output_path = Path(result["perturbed_file"])
        assert output_path.exists()

        # Output must be loadable and differ from the original (noise applied).
        perturbed, sr = _load_waveform(str(output_path))
        original, _ = _load_waveform(str(sample_audio_file))
        assert sr == 16000
        assert perturbed.shape == original.shape
        assert not torch.allclose(perturbed, original)

    def test_missing_file_reports_failure_not_exception(self, tmp_path: Path):
        result = perturb_and_save(
            file_path=str(tmp_path / "does-not-exist.wav"),
            perturbations=[{"type": "noise", "params": {}}],
            output_dir=str(tmp_path),
        )

        assert result["success"] is False
        assert "error" in result


class TestMutationEngines:
    """Tests individual mutation algorithms, 2D masking, and band-pass filter."""

    @pytest.fixture
    def dummy_waveform(self):
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        waveform = torch.from_numpy(y).unsqueeze(0)
        return waveform, sr

    def test_export_to_wav_bytes(self, dummy_waveform):
        """Test WAV byte export for Web Audio preview (SRS FR12.2)."""
        waveform, sr = dummy_waveform
        wav_bytes = export_to_wav_bytes(waveform, sr)
        
        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > 0
        # Check WAV header magic bytes
        assert wav_bytes[:4] == b'RIFF'

    def test_apply_2d_time_freq_mask(self, dummy_waveform):
        """Test NumPy-Driven 2D Spectrogram Slice Masking (mute regions)."""
        waveform, sr = dummy_waveform
        params = {
            "t_start_ms": 500,
            "t_end_ms": 1000,
            "f_low_hz": 1000,
            "f_high_hz": 4000
        }
        masked_waveform = apply_2d_time_freq_mask(waveform, sr, params)
        
        assert masked_waveform.shape == waveform.shape
        # The masked region should be different
        assert not torch.equal(masked_waveform, waveform)

    def test_apply_band_pass_filter(self, dummy_waveform):
        """Test signal modification routine for band-pass filtering."""
        waveform, sr = dummy_waveform
        params = {"f_low_hz": 1000, "f_high_hz": 3000}
        filtered_waveform = apply_band_pass_filter(waveform, sr, params)
        
        assert filtered_waveform.shape == waveform.shape
        assert not torch.equal(filtered_waveform, waveform)

    def test_apply_pitch_shift(self, dummy_waveform):
        """Test pitch shifting via librosa."""
        waveform, sr = dummy_waveform
        shifted_waveform = apply_pitch_shift(waveform, sr, pitch_shift_semitones=2)
        
        assert shifted_waveform.shape[0] == waveform.shape[0]
        # Length might vary slightly due to algorithm, but should be very close
        assert abs(shifted_waveform.shape[-1] - waveform.shape[-1]) < 100

    def test_apply_time_stretch(self, dummy_waveform):
        """Test time stretching via librosa."""
        waveform, sr = dummy_waveform
        stretched_waveform = apply_time_stretch(waveform, stretch_factor=1.5)
        
        # If stretched by 1.5 (faster), it should be shorter
        assert stretched_waveform.shape[-1] < waveform.shape[-1]

    def test_perturb_and_save_end_to_end(self, sample_audio_file: Path, tmp_path: Path):
        """Test the full end-to-end perturbation and save flow (FR12.1)."""
        perturbations = [
            {"type": "time_freq_mask", "params": {"t_start_ms": 100, "t_end_ms": 500, "f_low_hz": 500, "f_high_hz": 2000}},
            {"type": "pitch_shift", "params": {"pitch_shift_semitones": 1}}
        ]
        
        result = perturb_and_save(
            file_path=str(sample_audio_file),
            perturbations=perturbations,
            output_dir=str(tmp_path)
        )
        
        assert result["success"] is True
        assert Path(result["perturbed_file"]).exists()
        assert result["duration_ms"] > 0
        assert result["sample_rate"] == 16000
        assert len(result["applied_perturbations"]) == 2
        assert all(p["status"] == "applied" for p in result["applied_perturbations"])
        
        # Check Web Audio preview bytes (FR12.2)
        assert isinstance(result["preview_bytes"], bytes)
        assert result["preview_bytes"][:4] == b'RIFF'
        
        # Check non-destructive: original file should still exist
        assert sample_audio_file.exists()
