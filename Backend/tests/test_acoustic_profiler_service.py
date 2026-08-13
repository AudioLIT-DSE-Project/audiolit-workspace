"""Tests for the Acoustic Profiler's pitch tracking (LIT-145, FR10).

A pure sine wave has a known, constant fundamental frequency, so it gives a
ground truth to assert pYIN's output against directly — no recorded speech or
external fixtures needed.
"""

from __future__ import annotations

import numpy as np

from app.domain.acoustic_profiler_service import estimate_rms_contour, track_pitch_contour

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


class TestEstimateRmsContour:
    def test_matches_known_amplitude_of_a_pure_tone(self):
        # RMS of a sine wave of amplitude A is A / sqrt(2), independent of frequency.
        amplitude = 0.8
        audio = amplitude * np.sin(2 * np.pi * 220.0 * np.linspace(0, 1.0, SR, endpoint=False)).astype(np.float32)
        rms = estimate_rms_contour(audio)
        # Edge frames (STFT padding) are less reliable; the interior should be accurate.
        interior = rms[2:-2]
        assert np.allclose(interior, amplitude / np.sqrt(2), atol=0.02)

    def test_silence_is_near_zero(self):
        silence = np.zeros(SR, dtype=np.float32)
        rms = estimate_rms_contour(silence)
        assert np.allclose(rms, 0.0, atol=1e-6)

    def test_output_is_frame_aligned_to_hop_length(self):
        hop_length = 512
        frame_length = 2048
        audio = _sine(220.0, duration_s=1.0)
        rms = estimate_rms_contour(audio, hop_length=hop_length, frame_length=frame_length)
        expected_frames = 1 + len(audio) // hop_length
        assert rms.shape == (expected_frames,)

    def test_aligned_with_pitch_contour_frame_for_frame(self):
        # The whole point of sharing defaults with track_pitch_contour: same
        # audio in -> same-length contours out, so callers can zip them.
        audio = _sine(220.0, duration_s=1.0)
        f0 = track_pitch_contour(audio, sr=SR)
        rms = estimate_rms_contour(audio)
        assert f0.shape == rms.shape
