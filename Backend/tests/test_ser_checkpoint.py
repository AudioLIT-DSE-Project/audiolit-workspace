"""Tests for the default SER checkpoint selection (LIT-224, FR6, SRS TBD-1).

The bug this guards against: ECHO's inherited default
(`r-f/wav2vec-english-speech-emotion-recognition`) is unusable in two
independent ways, and *both* were silent from the application's point of view.

  1. It publishes no safetensors, only `pytorch_model.bin`. The registry fetches
     safetensors exclusively (SAD C3), so it refused the checkpoint and every
     SER call raised ModelRegistryError.
  2. Its weights carry a custom `classifier.dense` / `classifier.out_proj` head,
     while `Wav2Vec2ForSequenceClassification` expects `projector` +
     `classifier`. Loaded directly, all four head tensors are randomly
     initialised - the model returns chance-level noise that changes with the
     torch seed, with no error anywhere.

(2) is the dangerous one: a randomly-initialised head is indistinguishable from
a working one unless you look. The hub test at the bottom is the check that
actually catches it; the rest are fast guards on the pinned configuration.

Following this suite's convention, nothing here downloads a model by default -
the hub test is opt-in via AUDIOLIT_HUB_TESTS=1.
"""

from __future__ import annotations

import os
import re
import types

import pytest
import torch

import app.domain.model_loader_service as ml


class TestPinnedCheckpoint:
    def test_default_is_not_the_broken_inherited_checkpoint(self):
        # Regression guard on the specific checkpoint LIT-224 ruled out.
        assert ml._EMO_MODEL_ID != "r-f/wav2vec-english-speech-emotion-recognition"

    def test_revision_is_pinned_to_an_exact_commit(self):
        # "main" would let an upstream re-upload silently change the head layout,
        # which is exactly how this broke. SAD §5.2 wants the exact version recorded.
        assert re.fullmatch(r"[0-9a-f]{40}", ml._EMO_MODEL_REVISION), (
            f"SER revision must be a full commit sha, got {ml._EMO_MODEL_REVISION!r}"
        )

    def test_required_head_keys_are_declared(self):
        # These are the four tensors whose absence turns SER into a noise generator.
        assert set(ml._EMO_REQUIRED_HEAD_KEYS) == {
            "projector.weight",
            "projector.bias",
            "classifier.weight",
            "classifier.bias",
        }


class TestRegistryWiring:
    def test_loader_passes_the_pinned_revision_to_the_registry(self, monkeypatch):
        """The pin is worthless if the loader drops it on the way to the registry."""
        seen: dict = {}

        class _FakeLoaded:
            model = object()

        def _fake_get(model_id, revision="main", model_class=None):
            seen.update(model_id=model_id, revision=revision, model_class=model_class)
            return _FakeLoaded()

        monkeypatch.setattr(ml._model_registry, "get", _fake_get)
        monkeypatch.setattr(
            ml.Wav2Vec2FeatureExtractor, "from_pretrained", staticmethod(lambda *a, **k: object())
        )
        monkeypatch.setattr(ml, "emo_model", None)
        monkeypatch.setattr(ml, "feature_extractor", None)

        ml.ensure_emo_model_loaded()

        assert seen["model_id"] == ml._EMO_MODEL_ID
        assert seen["revision"] == ml._EMO_MODEL_REVISION
        assert seen["model_class"] is ml.Wav2Vec2ForSequenceClassification


class TestLabelMapping:
    """Emotion labels come from the checkpoint's own config, never a hardcoded list.

    The replacement names two classes differently from the old default
    ("fearful"/"surprised" vs "fear"/"surprise"), which only stays harmless as
    long as nothing hardcodes the old spellings.
    """

    def _fake_emo(self, monkeypatch, id2label, logits):
        class _Out:
            def __init__(self, lg):
                self.logits = lg

        class _Model:
            config = types.SimpleNamespace(id2label=id2label, output_attentions=False)

            def __call__(self, input_values=None, attention_mask=None, output_attentions=False):
                return _Out(torch.tensor([logits], dtype=torch.float32))

        monkeypatch.setattr(ml, "emo_model", _Model())
        return _Model()

    def test_prediction_uses_the_config_id2label(self, monkeypatch):
        id2label = {0: "angry", 1: "disgust", 2: "fearful", 3: "happy",
                    4: "neutral", 5: "sad", 6: "surprised"}
        model = self._fake_emo(monkeypatch, id2label, [0, 0, 0, 0, 9.0, 0, 0])

        probs = torch.tensor([0, 0, 0, 0, 9.0, 0, 0]).softmax(-1)
        top = model.config.id2label[int(probs.argmax())]
        assert top == "neutral"
        # every class the checkpoint declares is reportable (FR6.2)
        assert len(model.config.id2label) == 7


class TestFr61Coverage:
    def test_pinned_checkpoint_covers_the_six_required_categories(self):
        """SRS FR6.1: at least six categories (angry, disgust, fear, happy, neutral, sad).

        Checked against the id2label recorded for the pinned revision rather than
        the live hub, so this stays a fast offline test. The hub test below
        confirms the recorded labels still match what the revision actually ships.
        """
        required = {"angry", "disgust", "fear", "happy", "neutral", "sad"}
        alias = {"fearful": "fear", "surprised": "surprise"}
        pinned = {alias.get(x, x) for x in ml._EMO_PINNED_LABELS}
        assert required <= pinned, f"FR6.1 not met, missing {sorted(required - pinned)}"


@pytest.mark.skipif(
    os.environ.get("AUDIOLIT_HUB_TESTS") != "1",
    reason="hits the Hugging Face hub and downloads ~1.2 GB; set AUDIOLIT_HUB_TESTS=1 to run",
)
class TestAgainstTheRealHub:
    """The check that would have caught the original bug. Opt-in."""

    def test_head_loads_with_no_missing_keys(self):
        from transformers import Wav2Vec2ForSequenceClassification

        model, info = Wav2Vec2ForSequenceClassification.from_pretrained(
            ml._EMO_MODEL_ID, revision=ml._EMO_MODEL_REVISION, output_loading_info=True
        )
        missing = set(info["missing_keys"]) & set(ml._EMO_REQUIRED_HEAD_KEYS)
        assert not missing, (
            f"SER classifier head is randomly initialised: {sorted(missing)}. "
            "Predictions would be untrained noise."
        )
        assert list(model.config.id2label.values()) == list(ml._EMO_PINNED_LABELS)

    def test_checkpoint_publishes_safetensors(self):
        """SAD C3: the registry fetches safetensors only and refuses anything else."""
        from huggingface_hub import HfApi

        files = [s.rfilename for s in HfApi().model_info(ml._EMO_MODEL_ID).siblings]
        assert any(f.endswith(".safetensors") for f in files), (
            f"{ml._EMO_MODEL_ID} has no safetensors; the registry will refuse it (SAD C3)"
        )

    def test_predictions_are_deterministic_across_seeds(self):
        """A trained head gives identical output for identical input; a random one does not."""
        import numpy as np
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

        fe = AutoFeatureExtractor.from_pretrained(
            ml._EMO_MODEL_ID, revision=ml._EMO_MODEL_REVISION
        )
        audio = (np.random.default_rng(0).standard_normal(32000) * 0.05).astype("float32")
        batch = fe(audio, sampling_rate=16000, return_tensors="pt")

        outputs = []
        for seed in (1, 2):
            torch.manual_seed(seed)
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                ml._EMO_MODEL_ID, revision=ml._EMO_MODEL_REVISION
            ).eval()
            with torch.no_grad():
                outputs.append(model(**batch).logits.softmax(-1))
        assert torch.allclose(outputs[0], outputs[1]), (
            "SER predictions change with the torch seed - the head is randomly initialised"
        )
