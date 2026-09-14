"""Dataset warmup must write the payload shape each key family promises (FR4).

The defect these tests lock out: warmup ran
``transcribe_whisper_with_attention`` and stored its ``{"text", "attention"}``
dict under *every* key it wrote, including the transcript family whose
consumers all treat ``prediction`` as a plain string. The keys matched, so no
consumer fell back to recomputation - they read the value and died on
``.lower()``, and the UI showed a spinner that never resolved.

A key-alignment test alone would have passed. These assert the shapes.
"""

import hashlib
from unittest.mock import patch

import pytest

from app.infrastructure import cache_keys as ck


ASR_WITH_ATTENTION = {"text": " hello world", "attention": [[[0.1, 0.2]]]}
SER_RESULT = {"predicted_emotion": "happy", "confidence": 0.91, "probabilities": {}}


class TestTranscriptNormalisation:
    def test_flattens_asr_dict_to_string(self):
        assert ck.as_transcript(ASR_WITH_ATTENTION) == " hello world"

    def test_passes_string_through(self):
        assert ck.as_transcript(" already a string") == " already a string"

    def test_none_becomes_empty_not_none_string(self):
        assert ck.as_transcript(None) == ""

    def test_nested_prediction_wrapper(self):
        assert ck.as_transcript({"prediction": {"text": "x"}}) == "x"

    @pytest.mark.parametrize("value", [ASR_WITH_ATTENTION, " plain", None, {"text": "t"}])
    def test_result_always_supports_string_ops(self, value):
        """`/inferences/whisper-accuracy` calls .lower() on this. Always must work."""
        assert ck.as_transcript(value).lower() is not None

    def test_unwrap_leaves_non_asr_dicts_intact(self):
        """SER/ADD payloads are dicts by contract and must not be flattened."""
        assert ck.unwrap_prediction({"prediction": SER_RESULT}) == SER_RESULT


class TestKeyFamiliesMatchRouteSpelling:
    """Keys the routes compute, reproduced literally from the route source."""

    MODEL = "openai/whisper-tiny"
    H = "0" * 32

    def test_transcript_family(self):
        keys = dict.fromkeys(ck.transcript_keys(self.MODEL, (self.H,)))
        # inferences.py `/whisper-accuracy` and inference_service.run_inference
        assert (self.MODEL, f"v2_{self.MODEL}_{self.H}") in keys
        # inferences.py `/batch-check` and `/whisper-batch`
        assert (self.MODEL, f"{self.MODEL}_{self.H}") in keys
        # inferences.py `/whisper-accuracy` final fallback
        assert ("predictions", f"v2_{self.MODEL}_{self.H}") in keys

    def test_attention_family(self):
        assert (self.MODEL, f"{self.MODEL}_attention_v2_{self.H}") in ck.attention_keys(
            self.MODEL, (self.H,)
        )

    def test_ser_family_covers_both_detail_endpoints(self):
        keys = ck.ser_keys((self.H,))
        assert ("wav2vec2", f"wav2vec2_detailed_{self.H}") in keys
        assert ("wav2vec2", f"wav2vec2_detailed_attention_v3_{self.H}") in keys

    def test_acoustic_and_saliency_and_embedding_families(self):
        # Versioned: LIT-248 added `spectrogram` to the payload, and an
        # unversioned key kept serving pre-LIT-248 entries without it.
        assert (
            "acoustic",
            f"acoustic_profile_{ck.ACOUSTIC_SCHEMA_VERSION}_{self.H}",
        ) in ck.acoustic_keys((self.H,))
        assert (
            "saliency",
            f"saliency_v2_{self.MODEL}_gradcam_{self.H}",
        ) in ck.saliency_keys(self.MODEL, "gradcam", (self.H,))
        assert (self.MODEL, f"{self.MODEL}_embeddings_{self.H}") in ck.embedding_keys(
            self.MODEL, (self.H,)
        )
        assert ("audio_frequency", f"audio_frequency_{self.H}") in ck.audio_frequency_keys(
            (self.H,)
        )

    def test_path_hash_matches_route_formula(self, sample_audio_file):
        expected = hashlib.md5(str(sample_audio_file).encode()).hexdigest()
        assert ck.path_hash(sample_audio_file) == expected

    def test_content_hash_differs_from_path_hash(self, sample_audio_file):
        hashes = ck.both_hashes(sample_audio_file)
        assert len(set(hashes)) == len(hashes), "hash variants must be distinct"
        # content first: readers should prefer the content-addressed key (FR4.1)
        assert hashes[0] == ck.content_sha256(sample_audio_file)


class TestWarmupWritesCorrectShapes:
    """Drive the real warmup task with models stubbed; inspect what it stored."""

    def _run(self, sample_audio_file, tasks):
        from app.orchestration import task_orchestrator as to

        writes: dict[tuple[str, str], object] = {}

        def fake_cache(ns, key, payload, ttl=None):
            writes[(ns, key)] = payload

        with patch.object(to, "get_redis_connection", side_effect=Exception("no redis")), \
             patch("app.infrastructure.dataset_service.load_metadata",
                   return_value=[{"filename": sample_audio_file.name}]), \
             patch("app.infrastructure.dataset_service.resolve_file",
                   return_value=sample_audio_file), \
             patch("app.infrastructure.redis.cache_result_sync", side_effect=fake_cache), \
             patch("app.domain.model_loader_service.transcribe_whisper_with_attention",
                   return_value=ASR_WITH_ATTENTION), \
             patch("app.domain.model_loader_service.predict_emotion_wave2vec_with_attention",
                   return_value=SER_RESULT), \
             patch("app.domain.model_loader_service.extract_whisper_embeddings",
                   side_effect=Exception("skip")), \
             patch("app.domain.model_loader_service.extract_audio_frequency_features",
                   side_effect=Exception("skip")):
            to.run_batch_dataset_warmup_task(
                "job-test", "common-voice", "openai/whisper-tiny", tasks, cooldown_ms=0
            )
        return writes

    def test_transcript_keys_get_a_string_not_the_attention_dict(self, sample_audio_file):
        """The regression. Every transcript-family key must hold a string."""
        writes = self._run(sample_audio_file, ["asr"])
        h = ck.path_hash(sample_audio_file)
        transcript_written = False
        for ns, key in ck.transcript_keys("openai/whisper-tiny", (h,)):
            if (ns, key) in writes:
                transcript_written = True
                payload = writes[(ns, key)]
                assert isinstance(payload["prediction"], str), (
                    f"{ns}:{key} holds {type(payload['prediction']).__name__}; "
                    "consumers call .lower() on it"
                )
        assert transcript_written, "warmup wrote no transcript keys at all"

    def test_attention_keys_keep_the_full_dict(self, sample_audio_file):
        writes = self._run(sample_audio_file, ["asr"])
        h = ck.path_hash(sample_audio_file)
        for ns, key in ck.attention_keys("openai/whisper-tiny", (h,)):
            assert (ns, key) in writes, f"missing attention key {ns}:{key}"
            assert writes[(ns, key)]["prediction"] == ASR_WITH_ATTENTION

    def test_ser_is_warmed_even_when_the_selected_model_is_whisper(self, sample_audio_file):
        """SER lives in a model-independent namespace; asking for it must run it.

        Warmup used to dispatch purely on the model name, so `tasks=["ser"]`
        with a Whisper model silently warmed nothing.
        """
        writes = self._run(sample_audio_file, ["ser"])
        h = ck.path_hash(sample_audio_file)
        for ns, key in ck.ser_keys((h,)):
            assert (ns, key) in writes, f"missing SER key {ns}:{key}"
            assert writes[(ns, key)]["prediction"] == SER_RESULT

    def test_unrequested_tasks_are_not_warmed(self, sample_audio_file):
        writes = self._run(sample_audio_file, ["asr"])
        h = ck.path_hash(sample_audio_file)
        for ns, key in ck.ser_keys((h,)):
            assert (ns, key) not in writes

    def test_add_does_not_overwrite_the_asr_transcript(self, sample_audio_file):
        """ADD and the transcript family share a key shape (`v2_{model}_{h}`).

        Warmup keyed ADD on the *selected* model, so running asr+add together
        with a Whisper model wrote the deepfake dict straight over the
        transcript it had just cached - and every transcript consumer then read
        a dict and failed on .lower(). ADD must key on its own checkpoint.
        """
        with patch("app.domain.model_loader_service.predict_deepfake",
                   return_value={"predicted_label": "bona-fide",
                                 "synthetic_probability": 0.02, "confidence": 0.98}):
            writes = self._run(sample_audio_file, ["asr", "add"])

        h = ck.path_hash(sample_audio_file)
        for ns, key in ck.transcript_keys("openai/whisper-tiny", (h,)):
            if (ns, key) in writes:
                assert isinstance(writes[(ns, key)]["prediction"], str), (
                    f"{ns}:{key} holds a dict - ADD overwrote the transcript"
                )
        # and the ADD result still landed, under its own checkpoint
        from app.domain.model_loader_service import _DEFAULT_ADD_MODEL_KEY
        add_written = any(
            k in writes for k in ck.deepfake_keys(_DEFAULT_ADD_MODEL_KEY, (h,))
        )
        assert add_written, "ADD result was not cached under its own model key"

    def test_both_hash_spellings_are_written(self, sample_audio_file):
        """Consumers disagree on which hash to use; warm both or they miss."""
        writes = self._run(sample_audio_file, ["asr"])
        for h in ck.both_hashes(sample_audio_file):
            assert ("openai/whisper-tiny", f"v2_openai/whisper-tiny_{h}") in writes


class TestContentAddressing:
    """FR4.1 — cache identity follows the audio, not the path."""

    def test_same_audio_at_two_paths_shares_a_key(self, tmp_path, sample_audio_file):
        import shutil
        copy = tmp_path / "renamed_copy.wav"
        shutil.copyfile(sample_audio_file, copy)

        assert ck.content_sha256(sample_audio_file) == ck.content_sha256(copy)
        # ...while the location-derived hashes necessarily differ, which is
        # exactly why FR4.1 asks for content addressing.
        assert ck.path_hash(sample_audio_file) != ck.path_hash(copy)

        a = ck.transcript_keys("m", (ck.content_sha256(sample_audio_file),))
        b = ck.transcript_keys("m", (ck.content_sha256(copy),))
        assert a == b

    def test_editing_a_file_in_place_changes_its_key(self, tmp_path):
        import numpy as np, soundfile as sf
        p = tmp_path / "clip.wav"
        sf.write(p, np.zeros(16000, dtype="float32"), 16000)
        before = ck.content_sha256(p)
        sf.write(p, np.ones(16000, dtype="float32") * 0.5, 16000)
        assert ck.content_sha256(p) != before

    def test_content_hash_is_tried_first(self, sample_audio_file):
        """Readers must prefer the content key over the path keys."""
        assert ck.both_hashes(sample_audio_file)[0] == ck.content_sha256(sample_audio_file)

    def test_edited_file_rehashes_despite_the_memo(self, tmp_path):
        """The memo is keyed on size and mtime, so an edit must not be cached."""
        import numpy as np, soundfile as sf, os, time
        p = tmp_path / "clip.wav"
        sf.write(p, np.zeros(16000, dtype="float32"), 16000)
        first = ck.content_sha256(p)
        time.sleep(0.01)
        sf.write(p, np.ones(8000, dtype="float32") * 0.5, 16000)
        os.utime(p, None)
        assert ck.content_sha256(p) != first
