"""Tests for the L2-ARCTIC non-native reading corpus loader (LIT-181, FR2/FR15).

Synthetic per-speaker tree + audio; no real corpus downloaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.infrastructure.dataset_ingestion import (
    L2ArcticLoader,
    TARGET_SAMPLE_RATE,
    TaskFamily,
    get_loader,
)


def _wav(path: Path, sr: int = 16_000) -> None:
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    sf.write(path, (0.1 * np.sin(2 * np.pi * 190 * t)).astype(np.float32), sr)


@pytest.fixture
def arctic_root(tmp_path: Path):
    """Two speakers with different L1s: ABA (Arabic), HJK (Korean).

    HJK's second clip has no transcript file, to exercise the missing case.
    """
    root = tmp_path / "l2_arctic"
    layout = {
        "ABA": {"arctic_a0001": "authors of the danger trail", "arctic_a0002": "the boy was there"},
        "HJK": {"arctic_a0001": "she had your dark suit", "arctic_a0002": None},
    }
    for speaker, utts in layout.items():
        (root / speaker / "wav").mkdir(parents=True)
        (root / speaker / "transcript").mkdir(parents=True)
        for utt, text in utts.items():
            _wav(root / speaker / "wav" / f"{utt}.wav")
            if text is not None:
                (root / speaker / "transcript" / f"{utt}.txt").write_text(text, encoding="utf-8")
    return root


class TestL2ArcticLoader:
    def test_walks_speakers_and_maps_accent(self, arctic_root):
        loader = L2ArcticLoader(arctic_root)
        by_id = {s.sample_id: s for s in loader}

        assert loader.task_family is TaskFamily.ASR
        assert len(by_id) == 4
        aba = by_id["ABA-arctic_a0001"]
        assert aba.speaker_id == "ABA"
        assert aba.accent == "Arabic"
        assert aba.demographic == {"l1": "Arabic"}
        assert aba.label == "authors of the danger trail"
        assert by_id["HJK-arctic_a0001"].accent == "Korean"

    def test_missing_transcript_gives_none_label(self, arctic_root):
        by_id = {s.sample_id: s for s in L2ArcticLoader(arctic_root)}
        assert by_id["HJK-arctic_a0002"].label is None

    def test_skips_speakers_absent_from_tree(self, arctic_root):
        # Only ABA and HJK exist on disk; the other 22 mapped speakers are skipped.
        speakers = {s.speaker_id for s in L2ArcticLoader(arctic_root)}
        assert speakers == {"ABA", "HJK"}

    def test_accents_cover_multiple_l1s(self, arctic_root):
        accents = {s.accent for s in L2ArcticLoader(arctic_root)}
        assert accents == {"Arabic", "Korean"}

    def test_missing_root_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            list(L2ArcticLoader(tmp_path / "nope"))

    def test_load_audio_standardized(self, arctic_root):
        loader = L2ArcticLoader(arctic_root)
        audio, sr = loader.load_sample_audio(next(iter(loader)))
        assert sr == TARGET_SAMPLE_RATE
        assert audio.ndim == 1 and audio.dtype == np.float32

    def test_subsample_is_deterministic(self, arctic_root):
        loader = L2ArcticLoader(arctic_root)
        a = [s.sample_id for s in loader.subsample(2, seed=5)]
        b = [s.sample_id for s in loader.subsample(2, seed=5)]
        assert len(a) == 2 and a == b

    def test_registry_get_loader_returns_l2arctic(self, arctic_root):
        loader = get_loader("l2-arctic", root_dir=arctic_root)
        assert isinstance(loader, L2ArcticLoader)
        assert len(list(loader)) == 4

    def test_speaker_map_has_24_speakers_and_six_l1s(self):
        assert len(L2ArcticLoader.SPEAKER_L1) == 24
        assert set(L2ArcticLoader.SPEAKER_L1.values()) == {
            "Arabic", "Mandarin", "Hindi", "Korean", "Spanish", "Vietnamese",
        }
