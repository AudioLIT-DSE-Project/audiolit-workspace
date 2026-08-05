"""Tests for the multi-task dataset ingestion core (LIT-123, FR2).

Uses synthetic audio written with soundfile and temp CSV catalogs — no real
corpora are downloaded, so these run fast and offline.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.infrastructure.dataset_ingestion import (
    CORPUS_REGISTRY,
    ColumnMap,
    CommonVoiceLoader,
    CORPUS_REGISTRY,
    CsvCatalogLoader,
    TARGET_SAMPLE_RATE,
    TaskFamily,
    get_corpus_spec,
    get_loader,
    list_supported_corpora,
    load_standardized_audio,
)


def _write_tone(path: Path, *, sr: int, seconds: float = 0.5, channels: int = 1) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    mono = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    data = mono if channels == 1 else np.stack([mono] * channels, axis=1)
    sf.write(str(path), data, sr)


class TestStandardization:
    def test_stereo_highrate_becomes_mono_16k(self, tmp_path: Path):
        src = tmp_path / "stereo_44k.wav"
        _write_tone(src, sr=44_100, seconds=1.0, channels=2)

        audio, sr = load_standardized_audio(src)

        assert sr == TARGET_SAMPLE_RATE
        assert audio.ndim == 1                      # down-mixed to mono
        assert audio.dtype == np.float32
        # ~1 s resampled to 16 kHz — allow the resampler's small edge tolerance.
        assert abs(len(audio) - TARGET_SAMPLE_RATE) < 50

    def test_already_16k_mono_is_passed_through(self, tmp_path: Path):
        src = tmp_path / "mono_16k.wav"
        _write_tone(src, sr=TARGET_SAMPLE_RATE, seconds=0.5, channels=1)

        audio, sr = load_standardized_audio(src)

        assert sr == TARGET_SAMPLE_RATE
        assert audio.ndim == 1
        assert len(audio) == TARGET_SAMPLE_RATE // 2


@pytest.fixture
def catalog_corpus(tmp_path: Path):
    """A tiny catalog-driven corpus: 5 clips + a CSV with accent/demographic."""
    audio_dir = tmp_path / "clips"
    audio_dir.mkdir()
    rows = []
    for i in range(5):
        fname = f"clip_{i}.wav"
        _write_tone(audio_dir / fname, sr=22_050, seconds=0.3)
        rows.append(
            {
                "filename": fname,
                "transcript": f"utterance number {i}",
                "accent": "us" if i % 2 == 0 else "indian",
                "gender": "female" if i % 2 == 0 else "male",
                "client_id": f"spk{i}",
            }
        )

    catalog = tmp_path / "catalog.csv"
    with catalog.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    loader = CsvCatalogLoader(
        name="fixture-asr",
        task_family=TaskFamily.ASR,
        catalog_path=catalog,
        audio_base_dir=audio_dir,
        column_map=ColumnMap(
            filename="filename",
            label="transcript",
            speaker_id="client_id",
            accent="accent",
            demographic=("gender",),
        ),
        license="CC0-1.0",
    )
    return loader


class TestCsvCatalogLoader:
    def test_parses_labels_accent_and_demographic(self, catalog_corpus):
        samples = list(catalog_corpus.iter_metadata())

        assert len(samples) == 5
        first = samples[0]
        assert first.dataset == "fixture-asr"
        assert first.task_family is TaskFamily.ASR
        assert first.label == "utterance number 0"
        assert first.speaker_id == "spk0"
        assert first.accent == "us"
        assert first.demographic == {"gender": "female"}
        assert first.license == "CC0-1.0"
        assert first.audio_path.name == "clip_0.wav"

    def test_stream_limit_is_lazy(self, catalog_corpus):
        assert len(list(catalog_corpus.stream(limit=2))) == 2

    def test_missing_catalog_raises(self, tmp_path: Path):
        loader = CsvCatalogLoader(
            name="missing",
            task_family=TaskFamily.ASR,
            catalog_path=tmp_path / "nope.csv",
            audio_base_dir=tmp_path,
            column_map=ColumnMap(),
        )
        with pytest.raises(FileNotFoundError):
            list(loader.iter_metadata())

    def test_load_sample_audio_is_standardized(self, catalog_corpus):
        sample = next(iter(catalog_corpus))
        audio, sr = catalog_corpus.load_sample_audio(sample)
        assert sr == TARGET_SAMPLE_RATE
        assert audio.ndim == 1
        assert audio.dtype == np.float32


class TestSubsample:
    def test_deterministic_and_bounded(self, catalog_corpus):
        a = catalog_corpus.subsample(3, seed=42)
        b = catalog_corpus.subsample(3, seed=42)
        assert len(a) == 3
        assert [s.sample_id for s in a] == [s.sample_id for s in b]

    def test_n_larger_than_corpus_returns_all(self, catalog_corpus):
        assert len(catalog_corpus.subsample(100)) == 5

    def test_negative_n_rejected(self, catalog_corpus):
        with pytest.raises(ValueError):
            catalog_corpus.subsample(-1)


class TestCorpusRegistry:
    def test_seven_approved_corpora_registered(self):
        names = list_supported_corpora()
        assert len(names) == 7
        assert set(names) == {
            "common-voice",
            "librispeech",
            "crema-d",
            "ravdess",
            "esd",
            "l2-arctic",
            "asvspoof-2021",
        }

    def test_every_task_family_is_covered(self):
        families = {spec.task_family for spec in CORPUS_REGISTRY.values()}
        assert families == {TaskFamily.ASR, TaskFamily.SER, TaskFamily.DEEPFAKE}

    def test_lookup_is_case_insensitive(self):
        assert get_corpus_spec("Common-Voice").task_family is TaskFamily.ASR

    def test_unknown_corpus_raises_value_error(self):
        with pytest.raises(ValueError):
            get_corpus_spec("not-a-corpus")

    def test_pending_loader_raises_with_owner_issue(self):
        # Concrete loaders not yet contributed must signal that honestly.
        # Pick a still-pending corpus dynamically so this test survives each new
        # loader landing (rather than hardcoding a name every PR has to update).
        pending = [n for n, s in CORPUS_REGISTRY.items() if s.loader_factory is None]
        assert pending, "expected at least one not-yet-implemented loader"
        name = pending[0]
        with pytest.raises(NotImplementedError) as exc:
            get_loader(name)
        assert CORPUS_REGISTRY[name].owner_issue in str(exc.value)


def _write_cv_catalog(tmp_path: Path, *, columns: str = "processed") -> tuple[Path, Path]:
    """Write a tiny Common Voice-shaped catalog + clips. ``columns`` selects the
    processed export schema (filename/accent) or the raw CV schema (path/accents).
    """
    audio_dir = tmp_path / "common_voice_valid_dev"
    audio_dir.mkdir()
    file_col, accent_col = ("filename", "accent") if columns == "processed" else ("path", "accents")
    rows = []
    for i in range(4):
        fname = f"cv_{i}.wav"
        _write_tone(audio_dir / fname, sr=48_000, seconds=0.25)  # CV clips are 48 kHz
        rows.append(
            {
                file_col: fname if columns == "processed" else f"clips/{fname}",
                "sentence": f"the quick brown fox {i}",
                accent_col: "us" if i % 2 == 0 else "england",
                "age": "twenties",
                "gender": "female" if i % 2 == 0 else "male",
                "client_id": f"cvspk{i}",
                "locale": "en",
            }
        )
    catalog = audio_dir / "common_voice_valid_data_metadata.csv"
    with catalog.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return catalog, audio_dir


class TestCommonVoiceLoader:
    def test_maps_processed_catalog_schema(self, tmp_path: Path):
        catalog, audio_dir = _write_cv_catalog(tmp_path, columns="processed")
        loader = CommonVoiceLoader(catalog_path=catalog, audio_base_dir=audio_dir)

        samples = list(loader)
        assert loader.task_family is TaskFamily.ASR
        assert len(samples) == 4
        s0 = samples[0]
        assert s0.label == "the quick brown fox 0"          # 'sentence' column
        assert s0.accent == "us"
        assert s0.speaker_id == "cvspk0"
        assert s0.language == "en"
        assert s0.demographic == {"age": "twenties", "gender": "female"}
        assert s0.license == "CC0-1.0"
        assert s0.audio_path.name == "cv_0.wav"

    def test_falls_back_to_raw_cv_schema(self, tmp_path: Path):
        # Raw Common Voice uses 'path' and 'accents' instead of 'filename'/'accent'.
        catalog, audio_dir = _write_cv_catalog(tmp_path, columns="raw")
        loader = CommonVoiceLoader(catalog_path=catalog, audio_base_dir=audio_dir)

        s0 = next(iter(loader))
        assert s0.audio_path.name == "cv_0.wav"             # basename from 'clips/cv_0.wav'
        assert s0.accent == "us"                            # resolved from 'accents'

    def test_load_audio_standardizes_48k_to_16k_mono(self, tmp_path: Path):
        catalog, audio_dir = _write_cv_catalog(tmp_path)
        loader = CommonVoiceLoader(catalog_path=catalog, audio_base_dir=audio_dir)
        audio, sr = loader.load_sample_audio(next(iter(loader)))
        assert sr == TARGET_SAMPLE_RATE
        assert audio.ndim == 1 and audio.dtype == np.float32

    def test_registry_get_loader_returns_common_voice(self, tmp_path: Path):
        catalog, audio_dir = _write_cv_catalog(tmp_path)
        loader = get_loader("common-voice", catalog_path=catalog, audio_base_dir=audio_dir)
        assert isinstance(loader, CommonVoiceLoader)
        assert len(list(loader)) == 4

    @pytest.mark.skipif(
        not CommonVoiceLoader.DEFAULT_CATALOG.exists(),
        reason="real Common Voice data not provisioned (Backend/data/ is gitignored)",
    )
    def test_real_catalog_loads_when_present(self):
        loader = CommonVoiceLoader()
        first = next(iter(loader.stream(limit=1)))
        assert first.dataset == "common-voice"
        assert first.label  # every validated CV clip has a transcript
        audio, sr = loader.load_sample_audio(first)
        assert sr == TARGET_SAMPLE_RATE and audio.ndim == 1
