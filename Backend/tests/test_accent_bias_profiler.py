"""Tests for the accent bias profiling core (LIT-168, FR15).

Synthetic L2-ARCTIC tree (same fixture shape as test_l2arctic_loader.py) plus
a fake TranscribeFn -- no real ASR model loaded, matching the project's
existing pattern of mocking `transcribe_whisper` rather than hitting a real
model in tests (see test_function_testing.py / test_performance_load.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.domain.accent_bias_profiler import (
    SampleWERResult,
    load_accent_cohorts,
    score_sample,
)
from app.infrastructure.dataset_ingestion import L2ArcticLoader


def _wav(path: Path, sr: int = 16_000) -> None:
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    sf.write(path, (0.1 * np.sin(2 * np.pi * 190 * t)).astype(np.float32), sr)


@pytest.fixture
def arctic_root(tmp_path: Path):
    """Three speakers across two L1s: ABA + YBAA (Arabic), HJK (Korean)."""
    root = tmp_path / "l2_arctic"
    layout = {
        "ABA": {"arctic_a0001": "authors of the danger trail"},
        "YBAA": {"arctic_a0001": "the boy was there", "arctic_a0002": "she had your dark suit"},
        "HJK": {"arctic_a0001": "ask her to bring these things"},
    }
    for speaker, utts in layout.items():
        (root / speaker / "wav").mkdir(parents=True)
        (root / speaker / "transcript").mkdir(parents=True)
        for utt, text in utts.items():
            _wav(root / speaker / "wav" / f"{utt}.wav")
            (root / speaker / "transcript" / f"{utt}.txt").write_text(text, encoding="utf-8")
    return root


class TestLoadAccentCohorts:
    def test_groups_by_accent(self, arctic_root):
        cohorts = load_accent_cohorts("l2-arctic", root_dir=arctic_root)
        assert set(cohorts) == {"Arabic", "Korean"}
        assert len(cohorts["Arabic"]) == 3  # ABA (1) + YBAA (2)
        assert len(cohorts["Korean"]) == 1

    def test_unbounded_returns_every_sample(self, arctic_root):
        cohorts = load_accent_cohorts("l2-arctic", root_dir=arctic_root)
        assert sum(len(v) for v in cohorts.values()) == 4

    def test_samples_per_cohort_caps_each_group_independently(self, arctic_root):
        cohorts = load_accent_cohorts("l2-arctic", samples_per_cohort=1, root_dir=arctic_root)
        # Arabic has 3 candidates, Korean has 1 -- both must be capped at 1,
        # not have the global stream exhausted before Korean is ever seen.
        assert len(cohorts["Arabic"]) == 1
        assert len(cohorts["Korean"]) == 1

    def test_samples_per_cohort_deterministic_for_seed(self, arctic_root):
        a = load_accent_cohorts("l2-arctic", samples_per_cohort=1, seed=7, root_dir=arctic_root)
        b = load_accent_cohorts("l2-arctic", samples_per_cohort=1, seed=7, root_dir=arctic_root)
        assert [s.sample_id for s in a["Arabic"]] == [s.sample_id for s in b["Arabic"]]


class TestScoreSample:
    def test_perfect_transcription_scores_zero_wer(self, arctic_root):
        meta = next(iter(L2ArcticLoader(arctic_root)))
        result = score_sample(meta, transcribe=lambda _: meta.label)
        assert result is not None
        assert result.wer == 0.0
        assert result.accent == meta.accent

    def test_wer_ignores_case_and_punctuation(self, arctic_root):
        meta = next(s for s in L2ArcticLoader(arctic_root) if s.label == "authors of the danger trail")
        result = score_sample(meta, transcribe=lambda _: "Authors of the danger trail!")
        assert result.wer == 0.0

    def test_wrong_transcription_scores_nonzero_wer(self, arctic_root):
        meta = next(iter(L2ArcticLoader(arctic_root)))
        result = score_sample(meta, transcribe=lambda _: "completely unrelated text here")
        assert result.wer > 0.0

    def test_skips_sample_with_no_transcript(self, tmp_path):
        (tmp_path / "l2_arctic" / "ABA" / "wav").mkdir(parents=True)
        (tmp_path / "l2_arctic" / "ABA" / "transcript").mkdir(parents=True)
        _wav(tmp_path / "l2_arctic" / "ABA" / "wav" / "arctic_a0001.wav")
        meta = next(iter(L2ArcticLoader(tmp_path / "l2_arctic")))
        assert meta.label is None
        assert score_sample(meta, transcribe=lambda _: "anything") is None

    def test_skips_sample_with_no_accent(self, arctic_root):
        meta = next(iter(L2ArcticLoader(arctic_root)))
        blank_accent = meta.__class__(**{**meta.__dict__, "accent": None})
        assert score_sample(blank_accent, transcribe=lambda _: meta.label) is None

    def test_transcribe_receives_the_audio_path(self, arctic_root):
        meta = next(iter(L2ArcticLoader(arctic_root)))
        seen_paths = []

        def fake_transcribe(path: str) -> str:
            seen_paths.append(path)
            return meta.label

        score_sample(meta, transcribe=fake_transcribe)
        assert seen_paths == [str(meta.audio_path)]

    def test_result_is_frozen_dataclass(self, arctic_root):
        meta = next(iter(L2ArcticLoader(arctic_root)))
        result = score_sample(meta, transcribe=lambda _: meta.label)
        assert isinstance(result, SampleWERResult)
        with pytest.raises(Exception):
            result.wer = 1.0  # frozen -- mutation must raise
