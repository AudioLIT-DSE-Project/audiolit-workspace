"""The ASR pipeline must be built once per model, not once per request.

`_get_whisper_pipeline` and its `_pipeline_cache` existed for a long time
without a single caller: `transcribe_whisper` constructed the pipeline inline
instead, so the cache stayed empty and every transcription rebuilt the model.
Measured on whisper-base, construction was 5.22 s against 1.89 s of real
inference, so roughly three quarters of each call was rebuilding a model the
process already had.

Nothing failed while that was true, which is why it survived: the output was
correct, only slow. These tests make the reuse itself observable, so the caller
cannot be quietly disconnected again.
"""

from __future__ import annotations

import torch

import app.domain.model_loader_service as ml


class _FakePipe:
    """Stands in for an HF pipeline; records how many were constructed."""

    builds = 0

    def __init__(self, model_id: str):
        self.model_id = model_id
        type(self).builds += 1

    def __call__(self, audio, **kwargs):
        return {"text": f"transcript from {self.model_id}"}


def _install_counting_pipeline(monkeypatch):
    """Replace transformers.pipeline with a counter, and clear the cache."""
    _FakePipe.builds = 0
    monkeypatch.setattr(ml, "_pipeline_cache", {})

    def fake_pipeline(task, model=None, **kwargs):
        return _FakePipe(model)

    monkeypatch.setattr(ml, "pipeline", fake_pipeline)


class TestPipelineIsCached:
    def test_repeated_calls_build_the_model_once(self, monkeypatch):
        _install_counting_pipeline(monkeypatch)

        for _ in range(4):
            ml._get_whisper_pipeline("openai/whisper-base", -1, torch.float32)

        assert _FakePipe.builds == 1, (
            f"built the pipeline {_FakePipe.builds} times for one model; the "
            "cache is not being consulted"
        )

    def test_a_second_model_gets_its_own_entry(self, monkeypatch):
        _install_counting_pipeline(monkeypatch)

        a = ml._get_whisper_pipeline("openai/whisper-base", -1, torch.float32)
        b = ml._get_whisper_pipeline("openai/whisper-small", -1, torch.float32)

        assert _FakePipe.builds == 2
        # Two checkpoints must not collide on one cache entry, or selecting a
        # different model would silently return the first model's pipeline.
        assert a is not b
        assert a.model_id != b.model_id

    def test_the_cache_returns_the_same_object(self, monkeypatch):
        _install_counting_pipeline(monkeypatch)

        first = ml._get_whisper_pipeline("openai/whisper-base", -1, torch.float32)
        second = ml._get_whisper_pipeline("openai/whisper-base", -1, torch.float32)

        assert first is second


class TestTranscribeUsesTheCache:
    """The regression that actually happened: the helper existed but the
    caller bypassed it, so the cache was never populated."""

    def test_transcribe_whisper_populates_the_pipeline_cache(self, monkeypatch, tmp_path):
        _install_counting_pipeline(monkeypatch)

        # transcribe_whisper loads audio before inference; stub that out so the
        # test does not need a real file or librosa's decode path.
        monkeypatch.setattr(
            ml.librosa, "load", lambda p, sr=16000: (__import__("numpy").zeros(sr, dtype="float32"), sr)
        )

        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"")

        ml.transcribe_whisper("openai/whisper-base", str(clip))

        assert ml._pipeline_cache, (
            "transcribe_whisper left the pipeline cache empty, so it built its "
            "own pipeline instead of using the cached one"
        )

    def test_two_transcriptions_share_one_pipeline(self, monkeypatch, tmp_path):
        _install_counting_pipeline(monkeypatch)
        monkeypatch.setattr(
            ml.librosa, "load", lambda p, sr=16000: (__import__("numpy").zeros(sr, dtype="float32"), sr)
        )

        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"")

        ml.transcribe_whisper("openai/whisper-base", str(clip))
        ml.transcribe_whisper("openai/whisper-base", str(clip))

        assert _FakePipe.builds == 1, (
            f"two transcriptions built {_FakePipe.builds} pipelines; the model "
            "is being reloaded per request"
        )
