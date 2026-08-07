"""Tests for the Integrated Gradients core (LIT-126, FR9).

Integrated Gradients is *exact* for a linear model (constant gradient along the
path), so a linear forward function gives closed-form attributions to assert
against — no model download needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.domain.saliency_service import attribution_timeline, integrated_gradients


class TestIntegratedGradients:
    def test_exact_for_linear_scalar_output(self):
        # f(x) = w . x  ->  IG_i (baseline 0) = w_i * x_i, exactly.
        w = torch.tensor([[2.0, -3.0, 0.5, 1.0]])

        def forward_fn(x):
            return (x * w).sum(dim=1)

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        attr = integrated_gradients(forward_fn, x, n_steps=64)
        assert torch.allclose(attr, w * x, atol=1e-3)

    def test_target_selects_the_right_output(self):
        # outputs = [x0, 2*x1]; IG for each target isolates that output's inputs.
        W = torch.tensor([[1.0, 0.0], [0.0, 2.0]])

        def forward_fn(x):
            return x @ W

        x = torch.tensor([[3.0, 5.0]])
        assert torch.allclose(integrated_gradients(forward_fn, x, target=0),
                              torch.tensor([[3.0, 0.0]]), atol=1e-3)
        assert torch.allclose(integrated_gradients(forward_fn, x, target=1),
                              torch.tensor([[0.0, 10.0]]), atol=1e-3)

    def test_default_baseline_is_zeros(self):
        def forward_fn(x):
            return x.sum(dim=1)

        x = torch.tensor([[1.0, 2.0, 3.0]])
        # f = sum(x), so IG_i = x_i with a zero baseline.
        assert torch.allclose(integrated_gradients(forward_fn, x), x, atol=1e-3)


class TestAttributionTimeline:
    def test_normalized_and_millisecond_aligned(self):
        tl = attribution_timeline(torch.tensor([[0.0, 4.0, 2.0, 0.0]]), sample_rate=16000)
        assert len(tl) == 4
        assert tl[0]["t_ms"] == 0.0
        assert tl[1]["t_ms"] == round(1 / 16000 * 1000, 3)
        assert tl[1]["weight"] == 1.0                       # 4.0 is the peak -> 1.0
        assert all(0.0 <= e["weight"] <= 1.0 for e in tl)

    def test_hop_length_steps_by_frame(self):
        tl = attribution_timeline(torch.tensor([[1.0, 1.0, 1.0]]), sample_rate=16000, hop_length=256)
        assert tl[1]["t_ms"] == round(256 / 16000 * 1000, 3)   # 16.0 ms/frame

    def test_collapses_non_time_dims(self):
        # [batch=1, channels=2, time=2] -> collapse channels -> length-2 timeline
        attr = torch.tensor([[[1.0, 0.0], [3.0, 0.0]]])
        tl = attribution_timeline(attr)
        assert len(tl) == 2
        assert tl[0]["weight"] == 1.0 and tl[1]["weight"] == 0.0

    def test_accepts_numpy_input(self):
        tl = attribution_timeline(np.array([0.0, 2.0]), sample_rate=8000)
        assert [round(e["weight"], 3) for e in tl] == [0.0, 1.0]
