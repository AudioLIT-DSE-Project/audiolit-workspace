"""HookManager — forward/attention hook registration (LIT-211).

Uses small fake nn.Module trees shaped like the real Whisper/Wav2Vec2
encoder layouts (rather than downloading actual HF weights) so hook
attach/detach/capture behaviour can be verified in CI without network
access or GPU.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from app.domain.hook_manager_service import (
    UNSUPPORTED_ARCHITECTURE,
    HookManager,
    HookRegistrationError,
)


class FakeAttention(nn.Module):
    def forward(self, x):
        weights = torch.softmax(x, dim=-1)
        return x, weights


class FakeWhisperEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()

    def forward(self, x):
        out, _ = self.self_attn(x)
        return out


class FakeWhisperEncoder(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([FakeWhisperEncoderLayer() for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FakeWhisperModel(nn.Module):
    """Class name deliberately matches detect_model_family's 'whisper' check."""

    def __init__(self, n_layers=2):
        super().__init__()
        self.encoder = FakeWhisperEncoder(n_layers)
        self.config = SimpleNamespace(output_attentions=False)

    def forward(self, x):
        return self.encoder(x)


class FakeWhisperModelNoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(output_attentions=False)

    def forward(self, x):
        return x


class FakeWav2Vec2EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = FakeAttention()

    def forward(self, x):
        out, _ = self.attention(x)
        return out


class FakeWav2Vec2Encoder(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([FakeWav2Vec2EncoderLayer() for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FakeWav2Vec2Base(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.encoder = FakeWav2Vec2Encoder(n_layers)

    def forward(self, x):
        return self.encoder(x)


class FakeWav2Vec2ForSequenceClassification(nn.Module):
    """Mirrors the real fine-tuned model's `.wav2vec2` base attribute."""

    def __init__(self, n_layers=2):
        super().__init__()
        self.wav2vec2 = FakeWav2Vec2Base(n_layers)
        self.config = SimpleNamespace(output_attentions=False)

    def forward(self, x):
        return self.wav2vec2(x)


@pytest.fixture
def sample_input():
    return torch.randn(1, 5, 8)


class TestUnsupportedArchitecture:
    def test_unrecognized_class_raises_typed_error(self):
        with pytest.raises(HookRegistrationError) as exc_info:
            HookManager(nn.Linear(4, 4))
        assert exc_info.value.code == UNSUPPORTED_ARCHITECTURE

    def test_recognized_family_without_resolvable_layers_raises(self):
        with pytest.raises(HookRegistrationError) as exc_info:
            HookManager(FakeWhisperModelNoEncoder())
        assert exc_info.value.code == UNSUPPORTED_ARCHITECTURE

    def test_does_not_attach_partial_hooks_on_failure(self):
        model = FakeWhisperModelNoEncoder()
        with pytest.raises(HookRegistrationError):
            HookManager(model)
        # No hook should have been left on any submodule.
        for module in model.modules():
            assert len(module._forward_hooks) == 0


class TestWhisperHooks:
    def test_available_layers_lists_encoder_and_attention(self):
        manager = HookManager(FakeWhisperModel(n_layers=2))
        assert manager.available_layers() == [
            "encoder.layers.0",
            "encoder.layers.0.self_attn",
            "encoder.layers.1",
            "encoder.layers.1.self_attn",
        ]

    def test_forward_pass_captures_activations(self, sample_input):
        model = FakeWhisperModel(n_layers=2)
        manager = HookManager(model)
        with manager:
            model(sample_input)
        assert set(manager.captured) == set(manager.available_layers())
        for tensor in manager.captured.values():
            assert torch.is_tensor(tensor)

    def test_attention_layer_captures_weights_not_hidden_state(self, sample_input):
        model = FakeWhisperModel(n_layers=1)
        manager = HookManager(model)
        with manager:
            model(sample_input)
        attn_tensor = manager.captured["encoder.layers.0.self_attn"]
        # FakeAttention returns (hidden_state, softmax_weights); the hook
        # must capture the attention weights (output[1]), not the passthrough.
        expected = torch.softmax(sample_input, dim=-1)
        assert torch.allclose(attn_tensor, expected)


class TestWav2Vec2Hooks:
    def test_available_layers_lists_encoder_and_attention(self):
        manager = HookManager(FakeWav2Vec2ForSequenceClassification(n_layers=1))
        assert manager.available_layers() == [
            "encoder.layers.0",
            "encoder.layers.0.attention",
        ]

    def test_forward_pass_captures_activations(self, sample_input):
        model = FakeWav2Vec2ForSequenceClassification(n_layers=1)
        manager = HookManager(model)
        with manager:
            model(sample_input)
        assert set(manager.captured) == set(manager.available_layers())


class TestHookLifecycle:
    def test_hooks_removed_after_context_exit(self, sample_input):
        model = FakeWhisperModel(n_layers=2)
        manager = HookManager(model)
        with manager:
            model(sample_input)
        for module in model.modules():
            assert len(module._forward_hooks) == 0

    def test_hooks_removed_even_when_block_raises(self):
        model = FakeWhisperModel(n_layers=2)
        manager = HookManager(model)
        with pytest.raises(RuntimeError):
            with manager:
                raise RuntimeError("boom")
        for module in model.modules():
            assert len(module._forward_hooks) == 0

    def test_output_attentions_toggled_then_restored(self, sample_input):
        model = FakeWhisperModel(n_layers=1)
        assert model.config.output_attentions is False
        manager = HookManager(model)
        with manager:
            assert model.config.output_attentions is True
            model(sample_input)
        assert model.config.output_attentions is False

    def test_reentering_context_manager_reattaches_hooks(self, sample_input):
        model = FakeWhisperModel(n_layers=1)
        manager = HookManager(model)
        with manager:
            model(sample_input)
        first_capture = dict(manager.captured)

        manager.captured.clear()
        with manager:
            model(sample_input)
        assert set(manager.captured) == set(first_capture)
