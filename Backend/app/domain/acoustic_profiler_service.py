"""Acoustic Profiler — DSP feature extraction (LIT-125, FR10).

Time-aligned pitch (LIT-145) and energy (LIT-146) contours computed directly
from raw audio via librosa — no model involved. Both share one
frame_length/hop_length so the two contours line up on the same time axis,
which is what LIT-125's combined pipeline needs in order to overlay them.
"""

from __future__ import annotations

import librosa
import numpy as np

DEFAULT_FRAME_LENGTH = 2048
DEFAULT_HOP_LENGTH = 512
# Typical human voice fundamental-frequency range, matches librosa's own
# pYIN example range for speech (C2-C7).
DEFAULT_FMIN = librosa.note_to_hz("C2")
DEFAULT_FMAX = librosa.note_to_hz("C7")


def track_pitch_contour(
    audio: np.ndarray,
    sr: int,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    voiced_prob_threshold: float = 0.5,
) -> np.ndarray:
    """Frame-wise fundamental-frequency (F0) trajectory via pYIN (LIT-145, FR10).

    Returns a 1-D array, one value per frame at ``hop_length`` spacing.
    Unvoiced frames — silence, noise, or frames pYIN itself isn't confident
    about (``voiced_prob`` below ``voiced_prob_threshold``) — are ``np.nan``
    rather than 0.0, so they read as "no pitch" instead of a spurious 0 Hz
    value, matching how desktop tools like Praat render pitch gaps.
    """
    audio = np.asarray(audio, dtype=np.float32)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    f0 = np.asarray(f0, dtype=np.float64)
    unvoiced = ~np.asarray(voiced_flag, dtype=bool) | (np.asarray(voiced_prob) < voiced_prob_threshold)
    f0[unvoiced] = np.nan
    return f0


def estimate_rms_contour(
    audio: np.ndarray,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """Frame-wise RMS energy / localized amplitude contour (LIT-146, FR10).

    Returns a 1-D array, one value per frame, using the same
    ``frame_length``/``hop_length`` defaults (and the same STFT-style
    ``center=True`` framing) as `track_pitch_contour`, so the two contours
    come out the same length and line up frame-for-frame — the "aligned
    contours" LIT-125's combined pipeline needs to overlay pitch and
    intensity on one timeline.
    """
    audio = np.asarray(audio, dtype=np.float32)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    return rms.astype(np.float64)
