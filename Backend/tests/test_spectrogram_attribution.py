"""Tests for spectrogram-patch LIME/SHAP-style attribution (LIT-130, FR8).

Model-agnostic: attribution is driven by a plain ``score_fn`` callable, so no
model download or audio is needed. A deterministic score function that depends
on a known region lets us assert the attribution lands there.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.domain.saliency_service import (
    audio_to_spectrogram,
    occlusion_attribution,
    spectrogram_patch_bounds,
)


class TestPatchBounds:
    def test_tiles_range_exactly(self):
        bounds = spectrogram_patch_bounds(16, 4)
        assert bounds == [(0, 4), (4, 8), (8, 12), (12, 16)]

    def test_uneven_split_tiles_without_gaps(self):
        bounds = spectrogram_patch_bounds(10, 3)
        # contiguous, covers [0, 10), sizes differ by at most 1
        assert bounds[0][0] == 0 and bounds[-1][1] == 10
        for (a0, a1), (b0, b1) in zip(bounds, bounds[1:]):
            assert a1 == b0
        sizes = [b - a for a, b in bounds]
        assert max(sizes) - min(sizes) <= 1

    def test_caps_patches_at_length(self):
        assert len(spectrogram_patch_bounds(3, 8)) == 3

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            spectrogram_patch_bounds(16, 0)


class TestAudioToSpectrogram:
    def test_returns_2d_freq_time(self):
        audio = np.zeros(4096, dtype=np.float32)
        spec = audio_to_spectrogram(audio, n_fft=512, hop_length=256)
        assert spec.ndim == 2
        assert spec.shape[0] == 512 // 2 + 1        # freq bins


class TestOcclusionAttribution:
    def test_highlights_the_responsible_patch(self):
        # Score depends only on the top-left freq/time corner.
        spec = np.zeros((16, 16), dtype=np.float32)
        spec[:4, :4] = 1.0

        def score_fn(s):
            return float(s[:4, :4].mean())

        imp = occlusion_attribution(score_fn, spec, n_freq_patches=4, n_time_patches=4)
        assert imp.shape == (4, 4)
        # patch (0,0) exactly covers freq[0:4], time[0:4] -> biggest score drop
        assert np.unravel_index(int(np.argmax(imp)), imp.shape) == (0, 0)
        # a patch that doesn't touch the corner has ~no effect
        assert abs(imp[2, 2]) < 1e-6

    def test_shape_matches_patch_grid(self):
        spec = np.random.default_rng(0).random((24, 32)).astype(np.float32)
        imp = occlusion_attribution(lambda s: float(s.mean()), spec, n_freq_patches=6, n_time_patches=8)
        assert imp.shape == (6, 8)

    def test_fixed_baseline_value(self):
        spec = np.ones((8, 8), dtype=np.float32)
        # every patch occluded to 0.0; mean-based score drops by the occluded fraction
        imp = occlusion_attribution(lambda s: float(s.mean()), spec, 4, 4, baseline=0.0)
        assert np.all(imp >= 0)          # occluding-to-0 can only lower a mean of ones
        assert imp.sum() > 0

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            occlusion_attribution(lambda s: 0.0, np.zeros(16, dtype=np.float32))
