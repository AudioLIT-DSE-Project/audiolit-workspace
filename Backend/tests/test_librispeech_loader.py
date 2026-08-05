"""Tests for the LibriSpeech loader + silence validation (LIT-141, FR2).

Builds a synthetic LibriSpeech tree (speaker/chapter/*.trans.txt + .flac and a
SPEAKERS.TXT) and synthetic audio — no real corpus downloaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.infrastructure.dataset_ingestion import (
    LibriSpeechLoader,
    SILENCE_RMS_FLOOR,
    TARGET_SAMPLE_RATE,
    TaskFamily,
    get_loader,
    is_silent,
)


def _flac(path: Path, *, silent: bool = False, sr: int = 16_000) -> None:
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    data = np.zeros_like(t, dtype=np.float32) if silent else (0.1 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    sf.write(path, data, sr, format="FLAC")


@pytest.fixture
def libri_root(tmp_path: Path):
    """A 2-speaker / 2-chapter synthetic LibriSpeech subset with SPEAKERS.TXT."""
    root = tmp_path / "librispeech"
    specs = {
        ("103", "1240"): [("103-1240-0000", "HELLO WORLD"), ("103-1240-0001", "GOODBYE NOW")],
        ("1034", "121119"): [("1034-121119-0000", "THE QUICK BROWN FOX")],
    }
    for (spk, chap), utts in specs.items():
        d = root / spk / chap
        d.mkdir(parents=True)
        trans = d / f"{spk}-{chap}.trans.txt"
        trans.write_text("\n".join(f"{uid} {text}" for uid, text in utts) + "\n", encoding="utf-8")
        for uid, _ in utts:
            _flac(d / f"{uid}.flac")

    (root / "SPEAKERS.TXT").write_text(
        ";ID  | SEX | SUBSET | MINUTES | NAME\n"
        "103  | F   | train-clean-100 | 25.0 | Alice\n"
        "1034 | M   | train-clean-100 | 25.0 | Bob\n",
        encoding="utf-8",
    )
    return root


class TestLibriSpeechLoader:
    def test_walks_tree_and_parses_transcripts(self, libri_root):
        loader = LibriSpeechLoader(libri_root)
        samples = {s.sample_id: s for s in loader}

        assert loader.task_family is TaskFamily.ASR
        assert len(samples) == 3
        s = samples["103-1240-0000"]
        assert s.label == "HELLO WORLD"
        assert s.speaker_id == "103"
        assert s.audio_path.name == "103-1240-0000.flac"
        assert s.audio_path.exists()

    def test_gender_parsed_from_speakers_txt(self, libri_root):
        by_id = {s.sample_id: s for s in LibriSpeechLoader(libri_root)}
        assert by_id["103-1240-0000"].demographic == {"gender": "female"}
        assert by_id["1034-121119-0000"].demographic == {"gender": "male"}

    def test_load_audio_standardized(self, libri_root):
        loader = LibriSpeechLoader(libri_root)
        audio, sr = loader.load_sample_audio(next(iter(loader)))
        assert sr == TARGET_SAMPLE_RATE
        assert audio.ndim == 1 and audio.dtype == np.float32

    def test_missing_root_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            list(LibriSpeechLoader(tmp_path / "nope"))

    def test_registry_get_loader_returns_librispeech(self, libri_root):
        loader = get_loader("librispeech", root_dir=libri_root)
        assert isinstance(loader, LibriSpeechLoader)
        assert len(list(loader)) == 3

    def test_subsample_is_deterministic(self, libri_root):
        loader = LibriSpeechLoader(libri_root)
        a = [s.sample_id for s in loader.subsample(2, seed=1)]
        b = [s.sample_id for s in loader.subsample(2, seed=1)]
        assert len(a) == 2 and a == b


class TestSilenceValidation:
    def test_empty_is_silent(self):
        assert is_silent(np.array([], dtype=np.float32)) is True

    def test_all_zero_is_silent(self):
        assert is_silent(np.zeros(16_000, dtype=np.float32)) is True

    def test_real_tone_is_not_silent(self):
        t = np.linspace(0, 1, 16_000, endpoint=False)
        assert is_silent((0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)) is False

    def test_below_floor_is_silent(self):
        faint = np.full(16_000, SILENCE_RMS_FLOOR / 10, dtype=np.float32)
        assert is_silent(faint) is True

    def test_loaded_silent_flac_is_flagged(self, tmp_path: Path):
        p = tmp_path / "silent.flac"
        _flac(p, silent=True)
        from app.infrastructure.dataset_ingestion import load_standardized_audio
        audio, _ = load_standardized_audio(p)
        assert is_silent(audio) is True
