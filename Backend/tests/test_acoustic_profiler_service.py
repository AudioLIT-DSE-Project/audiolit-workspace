"""Tests for the Acoustic Profiler's pitch tracking (LIT-145, FR10).

A pure sine wave has a known, constant fundamental frequency, so it gives a
ground truth to assert pYIN's output against directly — no recorded speech or
external fixtures needed.
"""

from __future__ import annotations

import numpy as np

from app.domain.acoustic_profiler_service import track_pitch_contour

SR = 16000


def _sine(freq_hz: float, duration_s: float = 1.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


class TestTrackPitchContour:
    def test_recovers_known_frequency_of_a_pure_tone(self):
        f0 = track_pitch_contour(_sine(220.0), sr=SR)
        voiced = f0[~np.isnan(f0)]
        assert voiced.size > 0
        # pYIN is a probabilistic estimator, not exact — a steady tone should
        # still land within a few Hz of the true 220 Hz.
        assert np.allclose(voiced, 220.0, atol=5.0)

    def test_output_is_frame_aligned_to_hop_length(self):
        hop_length = 512
        frame_length = 2048
        audio = _sine(220.0, duration_s=1.0)
        f0 = track_pitch_contour(audio, sr=SR, hop_length=hop_length, frame_length=frame_length)
        expected_frames = 1 + len(audio) // hop_length
        assert f0.shape == (expected_frames,)

    def test_silence_is_entirely_unvoiced(self):
        silence = np.zeros(SR, dtype=np.float32)
        f0 = track_pitch_contour(silence, sr=SR)
        assert np.all(np.isnan(f0))

    def test_stricter_voiced_threshold_yields_no_more_voiced_frames(self):
        audio = _sine(220.0)
        lenient = track_pitch_contour(audio, sr=SR, voiced_prob_threshold=0.1)
        strict = track_pitch_contour(audio, sr=SR, voiced_prob_threshold=0.9)
        assert np.count_nonzero(~np.isnan(strict)) <= np.count_nonzero(~np.isnan(lenient))
