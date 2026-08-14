"""Tests for the group-wise WER diagnostic runner (LIT-182, FR15).

Same synthetic L2-ARCTIC fixture shape as test_accent_bias_profiler.py, plus
a fake TranscribeFn -- no real ASR model loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.domain.accent_bias_runner import (
    AccentBiasReport,
    CohortWERSummary,
    main,
    run_accent_bias_diagnostic,
)


def _wav(path: Path, sr: int = 16_000) -> None:
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    sf.write(path, (0.1 * np.sin(2 * np.pi * 190 * t)).astype(np.float32), sr)


@pytest.fixture
def arctic_root(tmp_path: Path):
    """Arabic (2 speakers, transcribed perfectly) vs. Korean (1 speaker,
    transcribed wrong) -- Korean should rank first as the worse cohort."""
    root = tmp_path / "l2_arctic"
    layout = {
        "ABA": {"arctic_a0001": "authors of the danger trail"},
        "YBAA": {"arctic_a0001": "the boy was there"},
        "HJK": {"arctic_a0001": "she had your dark suit"},
    }
    for speaker, utts in layout.items():
        (root / speaker / "wav").mkdir(parents=True)
        (root / speaker / "transcript").mkdir(parents=True)
        for utt, text in utts.items():
            _wav(root / speaker / "wav" / f"{utt}.wav")
            (root / speaker / "transcript" / f"{utt}.txt").write_text(text, encoding="utf-8")
    return root


def _perfect_except_korean(audio_path: str) -> str:
    # Fake ASR: echoes back a fixed "correct" transcript per speaker dir,
    # except Korean (HJK), which is deliberately garbled.
    if "HJK" in audio_path:
        return "totally wrong output"
    if "ABA" in audio_path:
        return "authors of the danger trail"
    return "the boy was there"


class TestRunAccentBiasDiagnostic:
    def test_report_shape(self, arctic_root):
        report = run_accent_bias_diagnostic(
            _perfect_except_korean, model_id="fake-model", root_dir=arctic_root
        )
        assert isinstance(report, AccentBiasReport)
        assert report.corpus == "l2-arctic"
        assert report.model_id == "fake-model"
        assert {c.accent for c in report.cohorts} == {"Arabic", "Korean"}
        assert len(report.sample_results) == 3

    def test_worst_cohort_ranks_first(self, arctic_root):
        report = run_accent_bias_diagnostic(
            _perfect_except_korean, root_dir=arctic_root
        )
        assert report.cohorts[0].accent == "Korean"
        assert report.cohorts[0].mean_wer > 0.0
        assert report.cohorts[-1].mean_wer == pytest.approx(0.0)

    def test_cohort_stats_are_correct(self, arctic_root):
        report = run_accent_bias_diagnostic(
            _perfect_except_korean, root_dir=arctic_root
        )
        arabic = next(c for c in report.cohorts if c.accent == "Arabic")
        assert arabic.sample_count == 2  # ABA + YBAA
        assert arabic.scored_count == 2
        assert arabic.mean_wer == 0.0
        assert arabic.min_wer == 0.0
        assert arabic.max_wer == 0.0

    def test_unscoreable_cohort_gets_none_stats_and_sorts_last(self, tmp_path):
        root = tmp_path / "l2_arctic"
        # ABA has a real transcript; HJK's utterance has no transcript file at
        # all, so its only sample is unscoreable.
        (root / "ABA" / "wav").mkdir(parents=True)
        (root / "ABA" / "transcript").mkdir(parents=True)
        _wav(root / "ABA" / "wav" / "arctic_a0001.wav")
        (root / "ABA" / "transcript" / "arctic_a0001.txt").write_text("hello world", encoding="utf-8")

        (root / "HJK" / "wav").mkdir(parents=True)
        (root / "HJK" / "transcript").mkdir(parents=True)
        _wav(root / "HJK" / "wav" / "arctic_a0001.wav")
        # no transcript file written for HJK -> label is None -> unscoreable

        report = run_accent_bias_diagnostic(lambda _: "hello world", root_dir=root)
        korean = next(c for c in report.cohorts if c.accent == "Korean")
        assert korean.scored_count == 0
        assert korean.mean_wer is None
        # Unscoreable cohort must not be ranked ahead of a scored one.
        assert report.cohorts[-1].accent == "Korean"

    def test_samples_per_cohort_is_forwarded(self, arctic_root):
        report = run_accent_bias_diagnostic(
            _perfect_except_korean, samples_per_cohort=1, seed=3, root_dir=arctic_root
        )
        for cohort in report.cohorts:
            assert cohort.sample_count <= 1


class TestReportJsonExport:
    def test_to_json_dict_round_trips_through_json(self, arctic_root):
        report = run_accent_bias_diagnostic(
            _perfect_except_korean, model_id="fake-model", root_dir=arctic_root
        )
        payload = json.dumps(report.to_json_dict())
        parsed = json.loads(payload)
        assert parsed["model_id"] == "fake-model"
        assert len(parsed["cohorts"]) == 2
        assert len(parsed["sample_results"]) == 3

    def test_none_stats_serialize_as_json_null_not_nan(self, tmp_path):
        root = tmp_path / "l2_arctic"
        (root / "ABA" / "wav").mkdir(parents=True)
        (root / "ABA" / "transcript").mkdir(parents=True)
        _wav(root / "ABA" / "wav" / "arctic_a0001.wav")
        # no transcript -> unscoreable -> None stats
        report = run_accent_bias_diagnostic(lambda _: "anything", root_dir=root)
        payload = json.dumps(report.to_json_dict())
        assert "NaN" not in payload
        parsed = json.loads(payload)
        assert parsed["cohorts"][0]["mean_wer"] is None


class TestCLI:
    def test_main_writes_report_to_output_file(self, arctic_root, tmp_path, monkeypatch):
        out_path = tmp_path / "report.json"
        # main() imports make_whisper_transcriber lazily from accent_bias_profiler
        # at call time (kept lazy so importing this module doesn't pull in
        # torch/transformers unless the CLI path actually runs) -- patch it at
        # its source so that call-time import picks up the fake.
        monkeypatch.setattr(
            "app.domain.accent_bias_profiler.make_whisper_transcriber",
            lambda model_id: _perfect_except_korean,
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "accent_bias_runner",
                "--model-id", "fake-model",
                "--output", str(out_path),
            ],
        )
        # root_dir isn't a CLI flag; patch get_loader's default via the corpus
        # registry kwarg path instead by monkeypatching load_accent_cohorts'
        # underlying loader root through the CORPUS_REGISTRY default dir.
        import app.infrastructure.dataset_ingestion as ingestion
        monkeypatch.setattr(ingestion.L2ArcticLoader, "DEFAULT_DIR", arctic_root)

        main()

        payload = json.loads(out_path.read_text())
        assert payload["model_id"] == "fake-model"
        assert len(payload["cohorts"]) == 2
