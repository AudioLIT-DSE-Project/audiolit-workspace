"""Tests for the Grad-CAM attribution utility (LIT-148, FR8).

Exercises the real hook/backward machinery on tiny conv nets — no model
download, no audio needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.domain.saliency_service import compute_grad_cam, find_last_conv_layer


class _ConvNet2d(torch.nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 8, 3, padding=1)  # the last conv
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(8, n_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.fc(self.pool(x).flatten(1))


class _ConvNet1d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 4, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        return self.fc(self.pool(x).flatten(1))


class _Output:
    def __init__(self, logits):
        self.logits = logits


class _ConvNetLogitsObj(_ConvNet2d):
    def forward(self, x):
        return _Output(super().forward(x))


class _LinearOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def _is_unit_normalized(cam: np.ndarray) -> bool:
    return cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6 and not np.isnan(cam).any()


class TestFindLastConv:
    def test_returns_the_last_conv(self):
        m = _ConvNet2d()
        assert find_last_conv_layer(m) is m.conv2

    def test_raises_without_conv(self):
        with pytest.raises(ValueError):
            find_last_conv_layer(_LinearOnly())


class TestGradCam:
    def test_2d_shape_and_normalized_range(self):
        torch.manual_seed(0)
        cam = compute_grad_cam(_ConvNet2d(), torch.randn(1, 1, 8, 8))
        assert cam.shape == (8, 8)            # matches the conv's spatial dims
        assert _is_unit_normalized(cam)

    def test_1d_attribution_over_time(self):
        cam = compute_grad_cam(_ConvNet1d(), torch.randn(1, 1, 16))
        assert cam.shape == (16,)
        assert _is_unit_normalized(cam)

    def test_explicit_target_index(self):
        cam = compute_grad_cam(_ConvNet2d(n_classes=3), torch.randn(1, 1, 8, 8), target_index=2)
        assert cam.shape == (8, 8)
        assert _is_unit_normalized(cam)

    def test_accepts_output_object_with_logits(self):
        cam = compute_grad_cam(_ConvNetLogitsObj(), torch.randn(1, 1, 8, 8))
        assert cam.shape == (8, 8)

    def test_explicit_target_layer(self):
        m = _ConvNet2d()
        cam = compute_grad_cam(m, torch.randn(1, 1, 8, 8), target_layer=m.conv1)
        assert cam.shape == (8, 8)

    def test_hooks_are_removed(self):
        m = _ConvNet2d()
        compute_grad_cam(m, torch.randn(1, 1, 8, 8))
        assert not m.conv2._forward_hooks
        assert not (m.conv2._backward_hooks or {})
        assert not (getattr(m.conv2, "_full_backward_hooks", None) or {})

    def test_repeatable_without_hook_leak(self):
        m = _ConvNet2d()
        for _ in range(3):
            compute_grad_cam(m, torch.randn(1, 1, 8, 8))
        assert not m.conv2._forward_hooks
