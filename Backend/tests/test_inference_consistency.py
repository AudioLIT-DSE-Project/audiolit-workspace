"""Same audio, same model, twice -> the same answer, and the answer the model gave.

The unit suite was fully green while the running application returned
attribution maps no model produced and word labels no transcript contained. All
of it passed because every test called a function directly, and the defects
lived in what happened *between* the functions: a cache round-trip, a decode
that was configured differently from its neighbour, a result assembled without
the field it had just computed.

These are the checks that would have caught them. Nothing here downloads a
model; the model boundary is stubbed and the wiring around it is what is
asserted.
"""

from unittest.mock import patch

import numpy as np
import pytest
import torch

import app.domain.model_loader_service as ml
import app.domain.saliency_service as sal
import app.orchestration.inference_service as isvc


# --------------------------------------------------------------------------- #
# Determinism through the cache
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
class TestSameInputSameOutput:
    """A warm read must equal the cold computation it claims to be."""

    async def _run_twice(self, model, prediction, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\0")
        calls = []

        def fake(path, *a, **k):
            calls.append(model)
            return prediction

        with patch.object(isvc, "_resolve_audio_path", return_value=audio), \
             patch.dict(isvc.MODEL_FUNCTIONS, {model: fake}, clear=False):
            first = await isvc.run_inference(model, file_path=str(audio))
            second = await isvc.run_inference(model, file_path=str(audio))
        return first, second, calls

    async def test_asr_transcript_is_identical_on_the_second_call(self, tmp_path):
        first, second, calls = await self._run_twice(
            "whisper-base", " Mines in the door.", tmp_path
        )
        assert first == second == " Mines in the door."
        assert len(calls) == 1, "the second call recomputed instead of using the cache"

    async def test_ser_distribution_is_identical_on_the_second_call(self, tmp_path):
        payload = {
            "predicted_emotion": "surprised",
            "probabilities": {"angry": 0.30898746, "surprised": 0.50981038},
            "confidence": 0.50981038,
        }
        first, second, _ = await self._run_twice("wav2vec2", payload, tmp_path)
        assert first == second == payload, "cache round-trip altered the distribution"

    async def test_deepfake_verdict_is_identical_on_the_second_call(self, tmp_path):
        payload = {
            "predicted_label": "spoof",
            "synthetic_probability": 0.999981164932251,
            "confidence": 0.999981164932251,
            "probabilities": {"bona-fide": 1.8783854e-05, "spoof": 0.999981164932251},
        }
        first, second, _ = await self._run_twice("melody-machine", payload, tmp_path)
        assert first == second == payload
        assert first["synthetic_probability"] == payload["synthetic_probability"], \
            "float precision lost through the JSON cache round-trip"

    async def test_a_second_model_does_not_overwrite_the_first(self, tmp_path):
        """One clip, two models: each must keep its own answer."""
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\0")

        with patch.object(isvc, "_resolve_audio_path", return_value=audio), \
             patch.dict(isvc.MODEL_FUNCTIONS,
                        {"whisper-base": lambda p, *a, **k: " Mines in the door.",
                         "melody-machine": lambda p, *a, **k: {"predicted_label": "spoof"}},
                        clear=False):
            asr_before = await isvc.run_inference("whisper-base", file_path=str(audio))
            await isvc.run_inference("melody-machine", file_path=str(audio))
            asr_after = await isvc.run_inference("whisper-base", file_path=str(audio))

        assert asr_after == asr_before, (
            "the deepfake result landed on the transcript's cache entry"
        )
        assert isinstance(asr_after, str)


# --------------------------------------------------------------------------- #
# The model that ran is the model that was asked for
# --------------------------------------------------------------------------- #

class TestEmbeddingExtractorRouting:
    """`extract_single_embedding` picked its extractor by substring."""

    @pytest.mark.asyncio
    async def test_add_model_does_not_get_the_ser_extractor(self, tmp_path):
        """The key 'wav2vec2-add' contains 'wav2vec', so it matched the SER branch."""
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\0")
        used = []

        with patch.object(isvc, "_resolve_audio_path", return_value=audio), \
             patch.object(isvc, "extract_add_embeddings",
                          side_effect=lambda p, m: used.append(("add", m)) or np.zeros(4)), \
             patch.object(isvc, "extract_wav2vec2_embeddings",
                          side_effect=lambda p, m=None: used.append(("ser", m)) or np.zeros(4)):
            await isvc.extract_single_embedding("wav2vec2-add", file_path=str(audio))

        assert used == [("add", "wav2vec2-add")], (
            "deepfake embeddings were produced by the emotion model - same 1024 dims, "
            "no error, wrong latent space"
        )

    @pytest.mark.asyncio
    async def test_custom_ser_checkpoint_reaches_the_ser_extractor(self, tmp_path):
        """A name with neither "whisper" nor "wav2vec" in it went to Whisper.

        `myorg/custom-ser` is a perfectly ordinary emotion checkpoint name and
        the substring chain had no branch for it, so a custom SER model was
        plotted in Whisper's latent space. The registry knows the family.
        """
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\0")
        used = []

        class Registry:
            def get(self, mid, **kw):
                return type("L", (), {"family": "wav2vec2"})()

        with patch.object(isvc, "_resolve_audio_path", return_value=audio), \
             patch.dict("sys.modules", {"app.domain.model_registry_service":
                                        type("M", (), {"registry": Registry()})}), \
             patch.object(isvc, "extract_wav2vec2_embeddings",
                          side_effect=lambda p, m=None: used.append(m) or np.zeros(4)), \
             patch.object(isvc, "extract_whisper_embeddings",
                          side_effect=lambda p, m=None: used.append(("whisper", m)) or np.zeros(4)):
            await isvc.extract_single_embedding("myorg/custom-ser", file_path=str(audio))
        assert used == ["myorg/custom-ser"], (
            "a custom emotion checkpoint was sent to the Whisper extractor"
        )

    @pytest.mark.asyncio
    async def test_named_wav2vec_checkpoint_still_takes_the_fast_path(self, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\0")
        used = []

        with patch.object(isvc, "_resolve_audio_path", return_value=audio), \
             patch.object(isvc, "extract_wav2vec2_embeddings",
                          side_effect=lambda p, m=None: used.append(m) or np.zeros(4)):
            await isvc.extract_single_embedding("myorg/wav2vec2-emotion", file_path=str(audio))
        assert used == ["myorg/wav2vec2-emotion"]

    @pytest.mark.asyncio
    async def test_family_alias_is_not_passed_as_a_hub_id(self, tmp_path):
        """'wav2vec2' names the family, not a checkpoint; it resolves to the default."""
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\0")
        used = []

        with patch.object(isvc, "_resolve_audio_path", return_value=audio), \
             patch.object(isvc, "extract_wav2vec2_embeddings",
                          side_effect=lambda p, m=None: used.append(m) or np.zeros(4)):
            await isvc.extract_single_embedding("wav2vec2", file_path=str(audio))
        assert used == [None]


# --------------------------------------------------------------------------- #
# The transcript and the XAI segments describe one prediction
# --------------------------------------------------------------------------- #

class TestTranscriptReconciliation:
    """Word timestamps need a constrained decode; the panels must still agree."""

    def _timestamped(self, text, words):
        return {
            "text": text,
            "chunks": [{"text": " " + w, "timestamp": (i, i + 1)}
                       for i, w in enumerate(words)],
            "audio": np.zeros(16000, dtype=np.float32),
            "sample_rate": 16000,
        }

    def test_segments_are_relabelled_from_the_canonical_transcript(self):
        """The exact divergence seen on common-voice sample-000037."""
        def fake(model_id, path, **kw):
            if kw.get("return_timestamps"):
                return self._timestamped(" Minds in the door.",
                                         ["Minds", "in", "the", "door."])
            return " Mines in the door."

        with patch.object(ml, "transcribe_whisper", side_effect=fake):
            out = ml.transcribe_whisper_with_timestamps("a.wav", "whisper-base")

        assert out["text"] == " Mines in the door."
        assert [c["text"].strip() for c in out["chunks"]] == \
            ["Mines", "in", "the", "door."], \
            "the XAI panel would label a segment with a word the transcript lacks"
        assert out["word_labels_diverged"] is False

    def test_mismatched_word_counts_are_flagged_not_forced(self):
        """Relabelling a different-length sequence puts right words on wrong times."""
        def fake(model_id, path, **kw):
            if kw.get("return_timestamps"):
                return self._timestamped(" a b c", ["a", "b", "c"])
            return " completely different text here please"

        with patch.object(ml, "transcribe_whisper", side_effect=fake):
            out = ml.transcribe_whisper_with_timestamps("a.wav", "whisper-base")

        assert out["word_labels_diverged"] is True
        assert [c["text"].strip() for c in out["chunks"]] == ["a", "b", "c"]

    def test_agreeing_decodes_are_left_alone(self):
        def fake(model_id, path, **kw):
            if kw.get("return_timestamps"):
                return self._timestamped(" hello world", ["hello", "world"])
            return " hello world"

        with patch.object(ml, "transcribe_whisper", side_effect=fake):
            out = ml.transcribe_whisper_with_timestamps("a.wav", "whisper-base")
        assert out["word_labels_diverged"] is False
        assert out["text"] == " hello world"


# --------------------------------------------------------------------------- #
# Attribution must be attribution
# --------------------------------------------------------------------------- #

class TestGradCamProducesAMap:
    """Grad-CAM returned an all-zero map on Whisper, so every heatmap was the
    energy fallback wearing a Grad-CAM label."""

    def test_the_whisper_target_is_the_transcript_not_encoder_energy(self):
        """The regression itself: which scalar Grad-CAM differentiates.

        Encoder energy produced a pre-ReLU map that was negative at all 1500
        positions on whisper-base, so ReLU flattened it to zero and the caller
        silently served the energy fallback instead.
        """
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(8, 8, kernel_size=3, padding=1)

            def forward(self, x):
                return type("O", (), {"last_hidden_state": self.conv(x).transpose(1, 2)})()

        class CondGen(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = type("M", (), {"encoder": Encoder()})()
                self.proj = torch.nn.Linear(8, 5)

            def generate(self, feats, **kw):
                return torch.tensor([[0, 1, 2]])

            def forward(self, input_features=None, decoder_input_ids=None):
                pooled = input_features.mean(dim=2)
                steps = decoder_input_ids.shape[1]
                return type("O", (), {
                    "logits": self.proj(pooled).unsqueeze(1).repeat(1, steps, 1)
                })()

        cond = CondGen()
        with patch.object(sal.model_loader_service, "_get_whisper_cond_gen",
                          return_value=(None, cond)):
            wrapper, layer = sal._build_whisper_gradcam_target(
                "openai/whisper-base", None, torch.randn(1, 8, 16)
            )
        assert isinstance(wrapper, sal._WhisperTranscriptScore), (
            "fell back to the encoder-energy target, which is the broken one"
        )
        assert layer is cond.model.encoder.conv, (
            "hooks must sit on the encoder that actually runs the forward pass"
        )

    def test_the_energy_target_is_still_available_when_generation_fails(self):
        class Broken(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Sequential(torch.nn.Conv1d(4, 4, 3, padding=1))

        base = Broken()
        with patch.object(sal.model_loader_service, "_get_whisper_cond_gen",
                          side_effect=RuntimeError("no such repo")):
            wrapper, layer = sal._build_whisper_gradcam_target(
                "openai/whisper-base", base, torch.randn(1, 4, 16)
            )
        assert isinstance(wrapper, sal._WhisperEncoderEnergy)

    def test_a_class_style_target_yields_a_varying_map(self):
        conv = torch.nn.Conv1d(4, 4, kernel_size=3, padding=1)

        class ClassScore(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = conv
                self.head = torch.nn.Linear(4, 2)

            def forward(self, x):
                return self.head(self.conv(x).mean(dim=2))

        torch.manual_seed(0)
        cam = sal.compute_grad_cam(ClassScore(), torch.randn(1, 4, 32), target_layer=conv)
        assert float(np.ptp(cam)) > 0.0

    def test_whisper_gradcam_target_scores_the_transcript(self):
        """Not encoder energy: the score is the log-probability of the tokens."""
        class TinyDecoderModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(8, 5)

            def forward(self, input_features=None, decoder_input_ids=None):
                pooled = input_features.mean(dim=2)                  # [B, 8]
                steps = decoder_input_ids.shape[1]
                logits = self.proj(pooled).unsqueeze(1).repeat(1, steps, 1)
                return type("O", (), {"logits": logits})()

        wrapper = sal._WhisperTranscriptScore(
            TinyDecoderModel(), torch.tensor([[0, 1]]), torch.tensor([[1, 2]])
        )
        score = wrapper(torch.randn(1, 8, 16))
        assert tuple(score.shape) == (1, 1)
        assert float(score) < 0.0, "a summed log-probability is negative"


class TestLimeSurrogateIsNotDegenerate:
    """Captum's default Lasso(alpha=0.01) zeroed every coefficient, so LIME
    silently became the energy fallback."""

    def test_surrogate_is_not_the_default_lasso(self):
        from captum._utils.models.linear_model import SkLearnLasso
        assert not isinstance(sal._lime_surrogate(), SkLearnLasso)

    def test_surrogate_recovers_a_known_linear_signal(self):
        """A surrogate that shrinks a real effect to zero is the whole bug."""
        from captum.attr import Lime

        weights = torch.tensor([3.0, -2.0, 0.5, 4.0])

        def f(x):
            return (x.squeeze(1) * weights).sum(dim=-1)

        inp = torch.ones(1, 1, 4)
        mask = torch.tensor([[[0, 1, 2, 3]]])
        attr = Lime(f, interpretable_model=sal._lime_surrogate()).attribute(
            inp, feature_mask=mask, n_samples=200
        )
        assert float(attr.abs().max()) > 1e-6, "surrogate collapsed to zero"

    def test_time_band_mask_groups_frames_not_cells(self):
        mask = sal._time_band_feature_mask(torch.zeros(1, 80, 3000), n_bands=32)
        assert tuple(mask.shape) == (1, 80, 3000)
        assert int(mask.max()) + 1 == 32, "one group per time band, not per mel cell"
        assert torch.equal(mask[0, 0], mask[0, 79]), "a band must span all mel bins"

    def test_time_band_mask_handles_a_2d_waveform_input(self):
        mask = sal._time_band_feature_mask(torch.zeros(1, 8000), n_bands=16)
        assert tuple(mask.shape) == (1, 8000)
        assert int(mask.max()) + 1 == 16

    def test_bands_are_capped_by_available_frames(self):
        mask = sal._time_band_feature_mask(torch.zeros(1, 4, 5), n_bands=32)
        assert int(mask.max()) + 1 <= 5


# --------------------------------------------------------------------------- #
# Computed and then thrown away
# --------------------------------------------------------------------------- #

class TestSerAttentionIsReturned:
    """~300 lines extracted 24 layers of attention and the result dict dropped it."""

    def _fake_model(self, attentions):
        class Out:
            def __init__(self, logits, attns):
                self.logits = logits
                self.attentions = attns

            def keys(self):
                return ["logits", "attentions"]

        class Base:
            def __init__(self, attns):
                self.attns = attns
                self.config = type("C", (), {"output_attentions": False})()

            def __call__(self, **kw):
                return Out(None, self.attns)

        class Model:
            def __init__(self):
                self.config = type("C", (), {"output_attentions": False,
                                             "id2label": {0: "happy", 1: "sad"}})()
                self.wav2vec2 = Base(attentions)

            def __call__(self, **kw):
                return Out(torch.tensor([[2.0, 0.5]]), attentions)

        return Model()

    def _patch(self, monkeypatch, model):
        class Inputs(dict):
            def __init__(self, n):
                super().__init__(attention_mask=torch.ones(1, n))
                self.input_values = torch.zeros(1, n)
                self.attention_mask = torch.ones(1, n)

        class FE:
            def __call__(self, audio, **kw):
                return Inputs(len(audio))

        monkeypatch.setattr(ml, "ensure_emo_model_loaded",
                            lambda *a, **k: (FE(), model, "cpu"))
        monkeypatch.setattr(ml.librosa, "load",
                            lambda *a, **k: (np.zeros(16000, dtype=np.float32), 16000))

    def test_real_attention_reaches_the_caller(self, monkeypatch):
        attns = tuple(torch.rand(1, 2, 8, 8) for _ in range(3))
        self._patch(monkeypatch, self._fake_model(attns))
        out = ml.predict_emotion_wave2vec("a.wav", return_attention=True)
        assert out.get("attention"), "attention was extracted and then discarded"
        assert len(out["attention"]) == 3
        assert out["attention_is_fallback"] is False
        assert out["provenance"] == "measured"

    def test_a_synthesised_pattern_is_never_reported_as_measured(self, monkeypatch):
        """When the checkpoint yields nothing, the stand-in must say so.

        It used to be worse than unflagged: the fallback loaded
        facebook/wav2vec2-base-960h and returned *that* model's real attention
        as the emotion model's, which no badge would have caught.
        """
        self._patch(monkeypatch, self._fake_model(None))
        out = ml.predict_emotion_wave2vec("a.wav", return_attention=True)
        assert out["provenance"] != "measured"
        if out["attention"] is not None:
            assert out["attention_is_fallback"] is True
            assert out["provenance"] == "fallback"
            assert "NOT real attention" in out["provenance_reason"]
        else:
            assert out["provenance"] == "unavailable"
            assert out["provenance_reason"]

    def test_no_other_checkpoint_is_loaded_to_fill_the_gap(self, monkeypatch):
        """A different network's attention is not an explanation of this one."""
        self._patch(monkeypatch, self._fake_model(None))
        loaded = []
        for name in ("Wav2Vec2Model", "Wav2Vec2Processor"):
            target = getattr(ml, name, None)
            if target is not None:
                monkeypatch.setattr(
                    target, "from_pretrained",
                    classmethod(lambda cls, *a, **k: loaded.append(a[0] if a else "?")),
                )
        ml.predict_emotion_wave2vec("a.wav", return_attention=True)
        assert loaded == [], f"a substitute model was loaded: {loaded}"

    def test_prediction_only_calls_carry_no_provenance_noise(self, monkeypatch):
        attns = tuple(torch.rand(1, 2, 8, 8) for _ in range(3))
        self._patch(monkeypatch, self._fake_model(attns))
        out = ml.predict_emotion_wave2vec("a.wav", return_attention=False)
        assert "attention" not in out
        assert set(out) == {"predicted_emotion", "probabilities", "confidence"}
