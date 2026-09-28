"""SER attention is pooled at the source so long clips cannot exhaust memory.

wav2vec2 attention is [heads, T, T] per layer at ~50 frames/s. Unbounded, a
23 s clip OOM-killed the warmup worker (>12 GB). transformers 5 collects the
weights with its own hooks on each attention module, so pooling must run
before those hooks - these tests go through a real (tiny, random) model to
hold that ordering in place.
"""

from __future__ import annotations

import pytest
import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForSequenceClassification

from app.domain.model_loader_service import _pool_attention, _pooled_attention


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    cfg = Wav2Vec2Config(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=37,
        conv_dim=(8, 8), conv_stride=(5, 4), conv_kernel=(10, 8), num_labels=3,
        do_stable_layer_norm=True, feat_extract_norm="layer",
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
        attn_implementation="eager",
    )
    return Wav2Vec2ForSequenceClassification(cfg).eval()


def _hooks(model) -> int:
    return sum(len(m._forward_hooks) for m in model.modules())


def test_pool_bounds_frames_and_keeps_rows_normalised():
    attn = torch.softmax(torch.randn(1, 2, 700, 700), dim=-1)
    pooled = _pool_attention(attn, max_frames=100)
    assert pooled.shape == (1, 2, 100, 100)
    assert torch.allclose(pooled.sum(-1), torch.ones(1, 2, 100), atol=1e-5)


def test_short_attention_is_untouched():
    attn = torch.softmax(torch.randn(1, 2, 40, 40), dim=-1)
    assert _pool_attention(attn, max_frames=100) is attn


def test_model_attentions_are_pooled_and_predictions_unchanged(tiny_model):
    audio = torch.randn(1, 16000)  # ~800 frames in this toy config
    with torch.no_grad():
        # Warm call first: this is what installs transformers' own capture
        # hooks, which our pooling has to precede.
        full = tiny_model(audio, output_attentions=True)
        with _pooled_attention(tiny_model, max_frames=50):
            pooled = tiny_model(audio, output_attentions=True)

    assert full.attentions[0].shape[-1] > 50
    assert all(a.shape[-2:] == (50, 50) for a in pooled.attentions)
    assert torch.allclose(full.logits, pooled.logits)


def test_hooks_are_removed_afterwards(tiny_model):
    with torch.no_grad():
        tiny_model(torch.randn(1, 16000), output_attentions=True)
    before = _hooks(tiny_model)
    with _pooled_attention(tiny_model):
        assert _hooks(tiny_model) > before
    assert _hooks(tiny_model) == before
