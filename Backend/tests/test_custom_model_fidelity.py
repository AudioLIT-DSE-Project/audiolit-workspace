"""The model that runs must be the model that was asked for (FR1, FR4).

Several entry points used to hardcode ``openai/whisper-base``: the saliency
generator, the attention route, ``transcribe_whisper_base``, and the
embedding extractor's failure fallback. Because the *result* was still
cached under the requested model's key, selecting a custom checkpoint gave
you whisper-base's output filed under the custom model's name - wrong
output, and a cache entry that stays wrong until it expires.

These tests assert the requested id reaches the loader untouched.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

import app.domain.model_loader_service as ml
from app.domain.model_loader_service import resolve_whisper_model_id


class TestResolveWhisperModelId:
    @pytest.mark.parametrize(
        "given,expected",
        [
            ("base", "openai/whisper-base"),
            ("whisper-base", "openai/whisper-base"),
            ("tiny", "openai/whisper-tiny"),
            ("whisper-tiny", "openai/whisper-tiny"),
            ("small", "openai/whisper-small"),
            (None, "openai/whisper-base"),
            ("", "openai/whisper-base"),
        ],
    )
    def test_aliases_expand(self, given, expected):
        assert resolve_whisper_model_id(given) == expected

    @pytest.mark.parametrize(
        "repo",
        [
            "openai/whisper-tiny",
            "openai/whisper-large-v3",
            "distil-whisper/distil-small.en",
            "myorg/my-finetuned-whisper",
            "some-user/whisper-base",  # must NOT collapse to openai/whisper-base
        ],
    )
    def test_custom_repo_ids_pass_through_untouched(self, repo):
        assert resolve_whisper_model_id(repo) == repo


@pytest.fixture
def loaded():
    """Record the model id each entry point actually loads."""
    seen: list[str] = []
    original = ml.transcribe_whisper

    def spy(model_id, audio_file, **kw):
        seen.append(model_id)
        if kw.get("return_attention"):
            return {"text": "stub", "attention": None, "attention_is_fallback": False}
        if kw.get("return_timestamps"):
            return {"text": "stub", "chunks": [], "audio": [], "sample_rate": 16000}
        return "stub"

    ml.transcribe_whisper = spy
    yield seen
    ml.transcribe_whisper = original


CUSTOM = "myorg/my-finetuned-whisper"


class TestEntryPointsHonourTheSelection:
    def test_transcribe_whisper_base_uses_the_given_model(self, loaded):
        ml.transcribe_whisper_base("a.wav", model=CUSTOM)
        assert loaded == [CUSTOM]

    def test_transcribe_whisper_base_defaults_when_unspecified(self, loaded):
        ml.transcribe_whisper_base("a.wav")
        assert loaded == ["openai/whisper-base"]

    def test_attention_entry_point(self, loaded):
        ml.transcribe_whisper_with_attention("a.wav", CUSTOM)
        assert loaded == [CUSTOM]

    def test_timestamps_entry_point(self, loaded):
        ml.transcribe_whisper_with_timestamps("a.wav", CUSTOM)
        assert loaded == [CUSTOM]

    def test_attention_pairs_entry_point(self, loaded):
        ml.extract_whisper_attention_pairs("a.wav", CUSTOM)
        assert CUSTOM in loaded


class TestRunInferenceBindsTheSelection:
    """`/inferences/run` resolves by family; the id must survive that hop."""

    @pytest.mark.parametrize(
        "model", ["whisper-base", "openai/whisper-tiny", "distil-whisper/distil-small.en"]
    )
    async def test_selected_model_reaches_the_loader(self, loaded, model):
        import app.orchestration.inference_service as isvc

        async def no_cache(*a, **k):
            return None

        async def noop(*a, **k):
            return None

        with patch.object(isvc, "_resolve_audio_path", return_value=Path("a.wav")), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(isvc, "get_result", new=no_cache), \
             patch.object(isvc, "cache_result", new=noop):
            await isvc.run_inference(model, file_path="a.wav")

        assert loaded == [resolve_whisper_model_id(model)]


class TestEmbeddingsDoNotSilentlySubstitute:
    def test_a_broken_custom_model_raises_rather_than_using_whisper_base(self, tmp_path):
        """Falling back would file whisper-base vectors under the custom key."""
        import soundfile as sf
        import numpy as np

        wav = tmp_path / "a.wav"
        sf.write(wav, np.zeros(16000, dtype="float32"), 16000)

        broken = patch(
            "transformers.WhisperProcessor.from_pretrained",
            side_effect=OSError("no such repo"),
        )
        with patch.object(ml, "get_whisper_base_models") as base_loader, broken:
            with pytest.raises(OSError):
                ml.extract_whisper_embeddings(str(wav), "myorg/does-not-exist")
            base_loader.assert_not_called()

    def test_the_default_model_still_uses_the_shared_loader(self, tmp_path):
        import soundfile as sf
        import numpy as np

        wav = tmp_path / "a.wav"
        sf.write(wav, np.zeros(16000, dtype="float32"), 16000)

        with patch.object(ml, "get_whisper_base_models",
                          side_effect=RuntimeError("reached")) as base_loader:
            with pytest.raises(RuntimeError):
                ml.extract_whisper_embeddings(str(wav), "whisper-base")
            base_loader.assert_called_once_with("openai/whisper-base")


class TestFabricatedAttentionIsFlagged:
    """FR17.1: a synthesised pattern must never pass for real attention."""

    def test_result_carries_the_flag(self, loaded):
        result = ml.transcribe_whisper_with_attention("a.wav", CUSTOM)
        assert "attention_is_fallback" in result
        assert result["attention_is_fallback"] is False
