"""Tests for the ASVspoof 2021 DF deepfake loader (LIT-142, FR2/FR7).

Uses synthetic whitespace-delimited protocol files (both the 2021 DF
trial_metadata.txt shape and the 2019 LA CM protocol shape) plus synthetic
audio — no real corpus is downloaded (Backend/data/ is gitignored).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.infrastructure.dataset_ingestion import (
    ASVspoofLoader,
    BONA_FIDE,
    SPOOF,
    TARGET_SAMPLE_RATE,
    TaskFamily,
    get_loader,
)


def _clip(path: Path, sr: int = 16_000) -> None:
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    sf.write(path, (0.1 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), sr)


@pytest.fixture
def df_corpus(tmp_path: Path):
    """A 2021 DF-shaped protocol: SPEAKER FILE codec source attack LABEL trim subset."""
    audio = tmp_path / "asvspoof2021_df"
    audio.mkdir()
    rows = [
        "DF_0001 DF_E_2000001 nocodec asvspoof -   bonafide notrim eval",
        "DF_0002 DF_E_2000002 low_mp3 vcc2020  A14 spoof    notrim eval",
        "DF_0003 DF_E_2000003 nocodec asvspoof -   bonafide notrim eval",
        "DF_0004 DF_E_2000004 low_ogg vcc2018  A07 spoof    notrim eval",
    ]
    for r in rows:
        _clip(audio / f"{r.split()[1]}.wav")
    protocol = audio / "trial_metadata.txt"
    protocol.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return protocol, audio


class TestAsvspoofLabels:
    def test_parses_bonafide_and_spoof(self, df_corpus):
        protocol, audio = df_corpus
        loader = ASVspoofLoader(protocol, audio, audio_ext=".wav")

        samples = list(loader)
        assert loader.task_family is TaskFamily.DEEPFAKE
        assert [s.label for s in samples] == [BONA_FIDE, SPOOF, BONA_FIDE, SPOOF]
        first = samples[0]
        assert first.sample_id == "DF_E_2000001"
        assert first.speaker_id == "DF_0001"
        assert first.audio_path.name == "DF_E_2000001.wav"
        assert "research use only" in first.license.lower()

    def test_handles_2019_la_protocol_layout(self, tmp_path: Path):
        # 2019 LA CM protocol: SPEAKER FILE - SYSTEM KEY  (label token still scanned)
        audio = tmp_path / "la"; audio.mkdir()
        protocol = audio / "cm.trl.txt"
        protocol.write_text(
            "LA_0069 LA_E_9332430 - - bonafide\nLA_0070 LA_E_9332431 - A01 spoof\n",
            encoding="utf-8",
        )
        loader = ASVspoofLoader(protocol, audio, audio_ext=".flac")
        samples = list(loader)
        assert [s.label for s in samples] == [BONA_FIDE, SPOOF]
        assert samples[0].sample_id == "LA_E_9332430"

    def test_skips_lines_without_a_label_token(self, tmp_path: Path):
        p = tmp_path / "p.txt"
        p.write_text("DF_0001 DF_E_1 nocodec asvspoof - bonafide\ngarbage line no label\n", encoding="utf-8")
        loader = ASVspoofLoader(p, tmp_path)
        assert len(list(loader)) == 1

    def test_missing_protocol_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            list(ASVspoofLoader(tmp_path / "nope.txt", tmp_path))


class TestAsvspoofAudioAndRegistry:
    def test_load_audio_is_standardized(self, df_corpus):
        protocol, audio = df_corpus
        loader = ASVspoofLoader(protocol, audio, audio_ext=".wav")
        audio_arr, sr = loader.load_sample_audio(next(iter(loader)))
        assert sr == TARGET_SAMPLE_RATE
        assert audio_arr.ndim == 1 and audio_arr.dtype == np.float32

    def test_default_audio_ext_is_flac(self):
        assert ASVspoofLoader().audio_ext == ".flac"

    def test_subsample_is_bounded_and_deterministic(self, df_corpus):
        protocol, audio = df_corpus
        loader = ASVspoofLoader(protocol, audio, audio_ext=".wav")
        a = [s.sample_id for s in loader.subsample(2, seed=3)]
        b = [s.sample_id for s in loader.subsample(2, seed=3)]
        assert len(a) == 2 and a == b

    def test_registry_get_loader_returns_asvspoof(self, df_corpus):
        protocol, audio = df_corpus
        loader = get_loader("asvspoof-2021", protocol_path=protocol, audio_base_dir=audio, audio_ext=".wav")
        assert isinstance(loader, ASVspoofLoader)
        assert len(list(loader)) == 4
