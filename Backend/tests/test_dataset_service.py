"""Tests for dataset_service.py's fallback to the real dataset_ingestion
registry (LIT-235).

Before this, dataset_service.py hardcoded only two corpora (common-voice,
ravdess -- and ravdess's path was stale, pointing at a directory that no
longer exists) despite dataset_ingestion.py having working, tested loaders
for six. These use a fake loader via monkeypatch rather than real local data
under Backend/data/, so they run the same in CI as locally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
import soundfile as sf

from app.infrastructure import dataset_ingestion, dataset_service
from app.infrastructure.dataset_ingestion import DatasetLoader, SampleMetadata, TaskFamily


class _FakeLoader(DatasetLoader):
    """A minimal DatasetLoader standing in for a real corpus loader."""

    def __init__(self, samples):
        super().__init__(name="fake-corpus", task_family=TaskFamily.ASR)
        self._samples = samples

    def iter_metadata(self) -> Iterator[SampleMetadata]:
        yield from self._samples


def _sample(tmp_path: Path, filename: str, label: str = "hello", **kwargs) -> SampleMetadata:
    audio_path = tmp_path / filename
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"RIFF....WAVEfmt ")  # not real audio, just needs to exist
    return SampleMetadata(
        dataset="fake-corpus",
        sample_id=filename,
        audio_path=audio_path,
        task_family=TaskFamily.ASR,
        label=label,
        **kwargs,
    )


class TestLoadMetadataFallsBackToRegistry:
    def test_unknown_legacy_dataset_uses_registry_loader(self, tmp_path, monkeypatch):
        samples = [_sample(tmp_path, "a.wav", label="hi"), _sample(tmp_path, "b.wav", label="there")]
        monkeypatch.setattr(dataset_ingestion, "get_loader", lambda name, **kw: _FakeLoader(samples))

        rows = dataset_service.load_metadata("some-new-corpus")

        assert [r["filename"] for r in rows] == ["a.wav", "b.wav"]
        assert [r["label"] for r in rows] == ["hi", "there"]

    def test_speaker_and_accent_included_when_present(self, tmp_path, monkeypatch):
        samples = [_sample(tmp_path, "a.wav", speaker_id="spk1", accent="l1-hindi")]
        monkeypatch.setattr(dataset_ingestion, "get_loader", lambda name, **kw: _FakeLoader(samples))

        rows = dataset_service.load_metadata("l2-arctic")

        assert rows[0]["speaker_id"] == "spk1"
        assert rows[0]["accent"] == "l1-hindi"

    def test_unregistered_corpus_raises_value_error(self, monkeypatch):
        with pytest.raises(ValueError):
            dataset_service.load_metadata("not-a-real-corpus")

    def test_unloaded_corpus_raises_value_error_not_crash(self, monkeypatch):
        # e.g. "esd" - registered but has no loader_factory yet (LIT-208).
        def _raise(name, **kw):
            raise NotImplementedError(f"No loader registered for corpus '{name}' yet")

        monkeypatch.setattr(dataset_ingestion, "get_loader", _raise)
        with pytest.raises(ValueError):
            dataset_service.load_metadata("esd")

    def test_common_voice_still_uses_the_legacy_csv_path(self, monkeypatch):
        # Explicitly confirm the known-working path wasn't touched - it must
        # never reach the registry fallback.
        def _should_not_be_called(name, **kw):
            raise AssertionError("common-voice must not fall back to the registry")

        monkeypatch.setattr(dataset_ingestion, "get_loader", _should_not_be_called)
        # common-voice's real CSV may or may not exist in this environment;
        # either a successful read or a FileNotFoundError from the legacy
        # path both prove the registry fallback was never reached.
        try:
            dataset_service.load_metadata("common-voice")
        except FileNotFoundError:
            pass


class TestResolveFileFallsBackToRegistry:
    def test_unknown_legacy_dataset_resolves_via_registry_loader(self, tmp_path, monkeypatch):
        samples = [_sample(tmp_path, "clip.wav")]
        monkeypatch.setattr(dataset_ingestion, "get_loader", lambda name, **kw: _FakeLoader(samples))

        resolved = dataset_service.resolve_file("some-new-corpus", "clip.wav")

        assert resolved == samples[0].audio_path

    def test_missing_file_raises_file_not_found(self, tmp_path, monkeypatch):
        samples = [_sample(tmp_path, "clip.wav")]
        monkeypatch.setattr(dataset_ingestion, "get_loader", lambda name, **kw: _FakeLoader(samples))

        with pytest.raises(FileNotFoundError):
            dataset_service.resolve_file("some-new-corpus", "does-not-exist.wav")

    def test_ravdess_no_longer_points_at_the_stale_subset_path(self):
        # The old hardcoded "ravdess_subset" directory doesn't exist on disk
        # anymore - ravdess must not be in the legacy dicts, so it always
        # goes through the registry (which points at the real "ravdess" dir).
        assert "ravdess" not in dataset_service.DATASET_PATHS
        assert "ravdess" not in dataset_service.DATASET_BASE_DIRS


class TestRegistryMetadataRowsFR2Fixes:
    """LIT-237: FR2.1 integrity + FR2.2 cap + FR2.3 richer rows."""

    def test_cap_limits_rows_returned(self, tmp_path, monkeypatch):
        samples = [_sample(tmp_path, f"{i}.wav") for i in range(5)]
        monkeypatch.setattr(dataset_ingestion, "get_loader", lambda name, **kw: _FakeLoader(samples))

        rows = dataset_service.load_metadata("some-new-corpus", limit=2)

        assert len(rows) == 2
        assert [r["filename"] for r in rows] == ["0.wav", "1.wav"]

    def test_default_cap_from_settings(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dataset_service.settings, "DATASET_METADATA_ROW_CAP", 3)
        samples = [_sample(tmp_path, f"{i}.wav") for i in range(10)]
        monkeypatch.setattr(dataset_ingestion, "get_loader", lambda name, **kw: _FakeLoader(samples))

        rows = dataset_service.load_metadata("some-new-corpus")

        assert len(rows) == 3

    def test_offset_skips_rows(self, tmp_path, monkeypatch):
        samples = [_sample(tmp_path, f"{i}.wav") for i in range(5)]
        monkeypatch.setattr(dataset_ingestion, "get_loader", lambda name, **kw: _FakeLoader(samples))

        rows = dataset_service.load_metadata("some-new-corpus", limit=2, offset=2)

        assert [r["filename"] for r in rows] == ["2.wav", "3.wav"]

    def test_row_for_missing_audio_file_is_excluded(self, tmp_path, monkeypatch):
        present = _sample(tmp_path, "present.wav")
        ghost = SampleMetadata(
            dataset="fake-corpus",
            sample_id="ghost.wav",
            audio_path=tmp_path / "ghost.wav",  # never written
            task_family=TaskFamily.ASR,
            label="unreachable",
        )
        monkeypatch.setattr(
            dataset_ingestion, "get_loader", lambda name, **kw: _FakeLoader([ghost, present])
        )

        rows = dataset_service.load_metadata("some-new-corpus")

        assert [r["filename"] for r in rows] == ["present.wav"]

    def test_license_and_language_included_when_present(self, tmp_path, monkeypatch):
        samples = [_sample(tmp_path, "a.wav", license="CC0-1.0", language="en")]
        monkeypatch.setattr(dataset_ingestion, "get_loader", lambda name, **kw: _FakeLoader(samples))

        rows = dataset_service.load_metadata("some-new-corpus")

        assert rows[0]["license"] == "CC0-1.0"
        assert rows[0]["language"] == "en"

    def test_demographic_fields_are_flattened_into_the_row(self, tmp_path, monkeypatch):
        samples = [_sample(tmp_path, "a.wav", demographic={"age": "30", "sex": "female"})]
        monkeypatch.setattr(dataset_ingestion, "get_loader", lambda name, **kw: _FakeLoader(samples))

        rows = dataset_service.load_metadata("some-new-corpus")

        assert rows[0]["age"] == "30"
        assert rows[0]["sex"] == "female"

    def test_duration_included_for_real_decodable_audio(self, tmp_path, monkeypatch):
        audio_path = tmp_path / "real.wav"
        sf.write(str(audio_path), (0.1 * np.ones(16_000)).astype(np.float32), 16_000)
        sample = SampleMetadata(
            dataset="fake-corpus",
            sample_id="real.wav",
            audio_path=audio_path,
            task_family=TaskFamily.ASR,
            label="hi",
        )
        monkeypatch.setattr(dataset_ingestion, "get_loader", lambda name, **kw: _FakeLoader([sample]))

        rows = dataset_service.load_metadata("some-new-corpus")

        assert rows[0]["duration"] == "1.0"


class TestResolveAudioReference:
    def test_resolves_dataset_file_with_relative_directory_prefix(self):
        resolved = dataset_service.resolve_audio_reference(file_path="cv-valid-dev/sample-000775.mp3")
        assert resolved.exists()
        assert resolved.name == "sample-000775.mp3"

    def test_resolves_dataset_and_dataset_file(self):
        resolved = dataset_service.resolve_audio_reference(dataset="cv-valid-dev", dataset_file="sample-000775.mp3")
        assert resolved.exists()
        assert resolved.name == "sample-000775.mp3"

