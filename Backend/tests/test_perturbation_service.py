"""
Perturbation service — soundfile I/O (LIT-226)

Verifies the torchaudio -> soundfile swap preserves the (channels, time)
float32 tensor contract the perturbation functions rely on, and that
perturb_and_save still produces a valid, playable output file.
"""

import numpy as np
import soundfile as sf
import torch
from pathlib import Path

from app.services.perturbation_service import (
    _load_waveform,
    _save_waveform,
    perturb_and_save,
)


class TestWaveformIO:
    """_load_waveform / _save_waveform match torchaudio.load/save's old contract."""

    def test_load_waveform_shape_and_dtype(self, sample_audio_file: Path):
        waveform, sample_rate = _load_waveform(str(sample_audio_file))

        assert isinstance(waveform, torch.Tensor)
        assert waveform.dtype == torch.float32
        assert waveform.dim() == 2  # (channels, time), same as torchaudio.load
        assert waveform.shape[0] == 1  # sample_audio_file fixture writes mono
        assert sample_rate == 16000

    def test_load_waveform_values_match_soundfile(self, sample_audio_file: Path):
        waveform, sample_rate = _load_waveform(str(sample_audio_file))
        raw, sr = sf.read(str(sample_audio_file), dtype="float32")

        assert sample_rate == sr
        np.testing.assert_allclose(waveform[0].numpy(), raw, atol=1e-6)

    def test_save_waveform_round_trip(self, temp_dir: Path):
        # Two-channel synthetic waveform, matches the (channels, time) shape
        # apply_perturbations()/torchaudio.load() used to hand back.
        t = np.linspace(0, 1.0, 16000, False)
        left = 0.3 * np.sin(2 * np.pi * 440 * t)
        right = 0.3 * np.sin(2 * np.pi * 660 * t)
        waveform = torch.from_numpy(np.stack([left, right]).astype("float32"))

        out_path = temp_dir / "roundtrip.wav"
        _save_waveform(str(out_path), waveform, 16000)

        assert out_path.exists()
        reloaded, sr = _load_waveform(str(out_path))
        assert sr == 16000
        assert reloaded.shape == waveform.shape
        np.testing.assert_allclose(reloaded.numpy(), waveform.numpy(), atol=1e-4)

    def test_save_waveform_mono(self, temp_dir: Path):
        # apply_pitch_shift/apply_time_stretch always return a single-channel
        # (1, time) tensor regardless of input channel count — confirm that
        # still saves and reloads correctly.
        t = np.linspace(0, 0.5, 8000, False)
        mono = torch.from_numpy((0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32")).unsqueeze(0)

        out_path = temp_dir / "mono.wav"
        _save_waveform(str(out_path), mono, 16000)

        reloaded, sr = _load_waveform(str(out_path))
        assert sr == 16000
        assert reloaded.shape == mono.shape
        np.testing.assert_allclose(reloaded.numpy(), mono.numpy(), atol=1e-4)


class TestPerturbAndSave:
    """End-to-end perturb_and_save, unchanged behaviour via the new I/O path."""

    def test_noise_perturbation_succeeds(self, sample_audio_file: Path, temp_dir: Path):
        result = perturb_and_save(
            file_path=str(sample_audio_file),
            perturbations=[{"type": "noise", "params": {"noise_level": 0.01}}],
            output_dir=str(temp_dir),
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

    def test_missing_file_reports_failure_not_exception(self, temp_dir: Path):
        result = perturb_and_save(
            file_path=str(temp_dir / "does-not-exist.wav"),
            perturbations=[{"type": "noise", "params": {}}],
            output_dir=str(temp_dir),
        )

        assert result["success"] is False
        assert "error" in result
