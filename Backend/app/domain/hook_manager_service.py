import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

UNSUPPORTED_ARCHITECTURE = "UNSUPPORTED_ARCHITECTURE"


class HookRegistrationError(Exception):
    """Raised when hooks cannot be attached to a model's encoder/attention layers.

    Carries a typed `code` (currently only UNSUPPORTED_ARCHITECTURE) so callers
    can distinguish this from an unrelated runtime error and surface it as a
    clear API response instead of a stack trace.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class HookableLayer:
    name: str  # dotted path, e.g. "encoder.layers.3.self_attn"
    module: nn.Module
    kind: str  # "encoder" | "attention"


def detect_model_family(model: nn.Module) -> Optional[str]:
    """Best-effort family detection from the model instance's class/attributes.

    Mirrors saliency_service.detect_model_type's naming ("whisper"/"wav2vec2"),
    but works off the loaded object rather than a model-id string, since a
    HookManager only ever receives an already-instantiated model.
    """
    class_name = type(model).__name__.lower()
    if "whisper" in class_name:
        return "whisper"
    if "wav2vec2" in class_name:
        return "wav2vec2"
    if hasattr(model, "wav2vec2"):
        return "wav2vec2"
    return None


def _resolve_whisper_layers(model: nn.Module) -> List[HookableLayer]:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        encoder = getattr(getattr(model, "model", None), "encoder", None)
    if encoder is None or not hasattr(encoder, "layers"):
        return []

    layers: List[HookableLayer] = []
    for idx, layer in enumerate(encoder.layers):
        layers.append(HookableLayer(f"encoder.layers.{idx}", layer, "encoder"))
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is not None:
            layers.append(HookableLayer(f"encoder.layers.{idx}.self_attn", self_attn, "attention"))
    return layers


def _resolve_wav2vec2_layers(model: nn.Module) -> List[HookableLayer]:
    base = getattr(model, "wav2vec2", model)
    encoder = getattr(base, "encoder", None)
    if encoder is None or not hasattr(encoder, "layers"):
        return []

    layers: List[HookableLayer] = []
    for idx, layer in enumerate(encoder.layers):
        layers.append(HookableLayer(f"encoder.layers.{idx}", layer, "encoder"))
        attention = getattr(layer, "attention", None)
        if attention is not None:
            layers.append(HookableLayer(f"encoder.layers.{idx}.attention", attention, "attention"))
    return layers


_FAMILY_RESOLVERS = {
    "whisper": _resolve_whisper_layers,
    "wav2vec2": _resolve_wav2vec2_layers,
}


class HookManager:
    """Context manager registering forward/attention hooks on a supported model.

    Committed support: Whisper and Wav2Vec2 families (SRS FR1). For any other
    architecture, or one whose encoder/attention layers can't be resolved,
    construction raises HookRegistrationError(UNSUPPORTED_ARCHITECTURE) rather
    than attaching partial or guessed hooks.

    Hooks are only live inside the `with` block; __exit__ always removes them
    and releases CUDA cache, even if the block raises, so hooks never leak
    across cached-model inferences.
    """

    def __init__(self, model: nn.Module, family: Optional[str] = None):
        self.model = model
        self.family = family or detect_model_family(model)

        if self.family is None or self.family not in _FAMILY_RESOLVERS:
            raise HookRegistrationError(
                UNSUPPORTED_ARCHITECTURE,
                f"No hook support for architecture '{type(model).__name__}'; "
                f"supported families: {', '.join(_FAMILY_RESOLVERS)}.",
            )

        self.layers: List[HookableLayer] = _FAMILY_RESOLVERS[self.family](model)
        if not self.layers:
            raise HookRegistrationError(
                UNSUPPORTED_ARCHITECTURE,
                f"Could not resolve encoder/attention layers for '{type(model).__name__}'.",
            )

        self.captured: Dict[str, Any] = {}
        self._handles: List[Any] = []
        self._prev_output_attentions: Optional[bool] = None

    def available_layers(self) -> List[str]:
        """Layer names selectable for attribution, in registration order."""
        return [layer.name for layer in self.layers]

    def __enter__(self) -> "HookManager":
        config = getattr(self.model, "config", None)
        if config is not None and hasattr(config, "output_attentions"):
            self._prev_output_attentions = config.output_attentions
            config.output_attentions = True

        for layer in self.layers:
            handle = layer.module.register_forward_hook(self._make_hook(layer.name, layer.kind))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

        config = getattr(self.model, "config", None)
        if config is not None and self._prev_output_attentions is not None:
            config.output_attentions = self._prev_output_attentions

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _make_hook(self, name: str, kind: str):
        def hook(module, inputs, output):
            tensor = self._extract_tensor(output, kind)
            if tensor is not None:
                self.captured[name] = tensor.detach()

        return hook

    @staticmethod
    def _extract_tensor(output: Any, kind: str) -> Optional[torch.Tensor]:
        if torch.is_tensor(output):
            return output
        if isinstance(output, tuple):
            if kind == "attention" and len(output) > 1 and torch.is_tensor(output[1]):
                return output[1]
            if torch.is_tensor(output[0]):
                return output[0]
        return None
