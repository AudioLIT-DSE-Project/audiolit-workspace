"""The model that runs must be the model that was asked for (FR1, FR4).

Several entry points used to hardcode ``openai/whisper-base``: the saliency
generator, the attention route, ``transcribe_whisper_base``, and the
embedding extractor's failure fallback. Because the *result* was still
cached under the requested model's key, selecting a custom checkpoint gave
you whisper-base's output filed under the custom model's name - wrong
output, and a cache entry that stays wrong until it expires.

These tests assert the requested id reaches the loader untouched.
"""

import asyncio
import threading
import time
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


class TestRunInferenceSerializesAgainstSaliencyLock:
    """A plain transcription forward pass racing a saliency Grad-CAM call on
    the same shared model corrupts both. Live-reproduced against real
    L2-ARCTIC audio + whisper-base: concurrently firing /saliency/generate
    and /inferences/run gave one request a Grad-CAM "size of tensor a (2)
    must match the size of tensor b (0)" crash and the other a garbage,
    non-matching transcript - register_forward_hook fires on ANY forward
    pass through the hooked layer, not just the one that registered it.
    run_inference must serialize its model calls through the exact same
    per-model lock saliency_service.generate_saliency uses (model_lock_key /
    lock_for_model), not just saliency-vs-saliency.
    """

    async def test_concurrent_calls_for_the_same_model_do_not_overlap(self, loaded):
        import app.orchestration.inference_service as isvc

        concurrent = 0
        max_concurrent = 0
        guard = threading.Lock()

        def slow_transcribe(model_id, audio_file, **kw):
            nonlocal concurrent, max_concurrent
            with guard:
                concurrent += 1
                max_concurrent = max(max_concurrent, concurrent)
            time.sleep(0.2)
            with guard:
                concurrent -= 1
            return "stub"

        ml.transcribe_whisper = slow_transcribe

        async def no_cache(*a, **k):
            return None

        async def noop(*a, **k):
            return None

        with patch.object(isvc, "_resolve_audio_path", return_value=Path("a.wav")), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(isvc, "get_result", new=no_cache), \
             patch.object(isvc, "cache_result", new=noop):
            await asyncio.gather(
                isvc.run_inference("whisper-base", file_path="a.wav"),
                isvc.run_inference("whisper-base", file_path="b.wav"),
            )

        assert max_concurrent == 1


class TestForceRefreshBypassesCache:
    """LIT-248: the per-row Regenerate button needs a real recompute, not
    the same cached prediction handed back unchanged."""

    async def test_default_returns_cached_result_without_recomputing(self, loaded):
        import app.orchestration.inference_service as isvc

        async def cached(*a, **k):
            return {"prediction": "cached-value"}

        with patch.object(isvc, "_resolve_audio_path", return_value=Path("a.wav")), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(isvc, "get_result", new=cached):
            result = await isvc.run_inference("whisper-base", file_path="a.wav")

        assert result == "cached-value"
        assert loaded == []  # never recomputed

    async def test_force_refresh_skips_cache_and_recomputes(self, loaded):
        import app.orchestration.inference_service as isvc

        async def cached(*a, **k):
            return {"prediction": "stale-cached-value"}

        cache_writes = []

        async def record_write(model, key, payload, ttl=None):
            cache_writes.append((model, key, payload))

        with patch.object(isvc, "_resolve_audio_path", return_value=Path("a.wav")), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(isvc, "get_result", new=cached), \
             patch.object(isvc, "cache_result", new=record_write):
            result = await isvc.run_inference("whisper-base", file_path="a.wav", force_refresh=True)

        assert result == "stub"  # the freshly computed value, not the stale cache
        assert loaded == ["openai/whisper-base"]  # the model actually ran
        assert len(cache_writes) == 1  # fresh result overwrites the cache entry
        assert cache_writes[0][2] == {"prediction": "stub"}

    async def test_force_refresh_false_is_the_default(self, loaded):
        import app.orchestration.inference_service as isvc
        import inspect as _inspect

        sig = _inspect.signature(isvc.run_inference)
        assert sig.parameters["force_refresh"].default is False


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


class TestCustomSerModelIsNotSubstituted:
    """A selected SER checkpoint must run, and must not share another's cache.

    Both halves were broken. `predict_emotion_wave2vec` took no model at all and
    `ensure_emo_model_loaded` cached one model in a module global, so the first
    checkpoint loaded answered for every later selection. And `ser_keys` carried
    no model, so even a correct prediction was written where a different model
    would read it. Selecting a custom emotion model returned the default
    model's answer, which is what "custom models give wrong predictions" looks
    like from the UI.
    """

    CUSTOM = "myorg/custom-ser"

    def test_ser_cache_keys_differ_per_model(self):
        from app.infrastructure import cache_keys as ck
        h = ("0" * 32,)
        default = set(ck.ser_keys(h))
        custom = set(ck.ser_keys(h, self.CUSTOM))
        assert not (default & custom), (
            "a custom SER model shares cache keys with the default; "
            "one model's prediction would be served for the other"
        )

    def test_default_keeps_its_historical_spelling(self):
        from app.infrastructure import cache_keys as ck
        h = ("0" * 32,)
        assert ck.ser_keys(h) == ck.ser_keys(h, None)
        assert ck.ser_keys(h) == ck.ser_keys(h, ck.DEFAULT_SER_MODEL)
        assert ("wav2vec2", f"wav2vec2_detailed_{h[0]}") in ck.ser_keys(h)

    def test_selected_checkpoint_reaches_the_loader(self, monkeypatch):
        import app.domain.model_loader_service as ml

        asked = []

        def fake_loader(model_id=None, revision=None):
            asked.append(model_id)
            raise RuntimeError("stop here — the model id is what we are asserting")

        monkeypatch.setattr(ml, "ensure_emo_model_loaded", fake_loader)
        for call in (
            lambda: ml.predict_emotion_wave2vec("a.wav", model_id=self.CUSTOM),
            lambda: ml.predict_emotion_wave2vec_with_attention("a.wav", model_id=self.CUSTOM),
        ):
            with pytest.raises(RuntimeError):
                call()
        assert asked == [self.CUSTOM, self.CUSTOM]

    def test_a_second_model_is_not_answered_by_the_first(self, monkeypatch):
        """The module-global cache held exactly one model."""
        import app.domain.model_loader_service as ml

        built = []

        class _FE:
            @classmethod
            def from_pretrained(cls, mid, **kw):
                built.append(("fe", mid))
                return cls()

        class _Loaded:
            model = object()

        monkeypatch.setattr(ml, "Wav2Vec2FeatureExtractor", _FE)
        monkeypatch.setattr(ml._model_registry, "get",
                            lambda mid, **kw: built.append(("model", mid)) or _Loaded())
        ml._emo_model_cache.clear()

        a = ml.ensure_emo_model_loaded("org/first")
        b = ml.ensure_emo_model_loaded("org/second")
        assert ("model", "org/first") in built and ("model", "org/second") in built
        assert a[1] is not None and b[1] is not None
