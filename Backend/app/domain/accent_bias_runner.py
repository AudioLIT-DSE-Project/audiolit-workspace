"""
Group-wise WER diagnostic runner (LIT-182, FR15 — Accent Bias Profiling).

Orchestrates LIT-168's accent-cohort batching and per-sample WER scoring
(`accent_bias_profiler.py`) into a full diagnostic pass: transcribes every
(or a bounded sample of) L2-ARCTIC utterance per accent cohort, aggregates
each cohort's WER statistics, and ranks cohorts worst-first into an
exportable JSON-ready report — per SAD §5.1 domain layer / §5.2 (Bias
Profiler and Faithfulness Auditor).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from .accent_bias_profiler import (
    SampleWERResult,
    TranscribeFn,
    load_accent_cohorts,
    score_sample,
)


@dataclass(frozen=True)
class CohortWERSummary:
    """Aggregated WER statistics for one accent cohort.

    Stats are ``None`` (not NaN) when a cohort has zero scoreable samples,
    so the report stays valid JSON for the dashboard chart the acceptance
    criteria call for -- a literal `NaN` token isn't standard JSON and most
    frontend JSON parsers reject it.
    """

    accent: str
    sample_count: int
    scored_count: int
    mean_wer: Optional[float]
    median_wer: Optional[float]
    stdev_wer: Optional[float]
    min_wer: Optional[float]
    max_wer: Optional[float]


@dataclass(frozen=True)
class AccentBiasReport:
    """Full diagnostic pass output -- ``cohorts`` ranked worst-WER first."""

    corpus: str
    model_id: str
    cohorts: List[CohortWERSummary]
    sample_results: List[SampleWERResult]

    def to_json_dict(self) -> dict:
        """Structured, JSON-serializable export for dashboard visualization."""
        return {
            "corpus": self.corpus,
            "model_id": self.model_id,
            "cohorts": [asdict(c) for c in self.cohorts],
            "sample_results": [asdict(r) for r in self.sample_results],
        }


def _summarize_cohort(accent: str, sample_count: int, results: List[SampleWERResult]) -> CohortWERSummary:
    wers = [r.wer for r in results]
    if not wers:
        return CohortWERSummary(
            accent=accent,
            sample_count=sample_count,
            scored_count=0,
            mean_wer=None,
            median_wer=None,
            stdev_wer=None,
            min_wer=None,
            max_wer=None,
        )
    return CohortWERSummary(
        accent=accent,
        sample_count=sample_count,
        scored_count=len(wers),
        mean_wer=statistics.mean(wers),
        median_wer=statistics.median(wers),
        stdev_wer=statistics.stdev(wers) if len(wers) > 1 else 0.0,
        min_wer=min(wers),
        max_wer=max(wers),
    )


def run_accent_bias_diagnostic(
    transcribe: TranscribeFn,
    corpus: str = "l2-arctic",
    model_id: str = "unknown",
    samples_per_cohort: Optional[int] = None,
    seed: int = 0,
    **loader_kwargs,
) -> AccentBiasReport:
    """Run the full group-wise WER diagnostic and return a ranked report.

    Batch-transcribes every accent cohort sequentially (technical step:
    "batch evaluation hooks that execute ASR transcriptions sequentially
    over selected non-native language directories"), scores each sample
    against its ground truth, and ranks cohorts by mean WER descending so
    the worst-performing accent group -- the actual bias signal -- sorts
    first. Cohorts with no scoreable samples are appended, unranked, after
    the scored ones rather than mixed in by an undefined comparison.
    """
    cohorts = load_accent_cohorts(
        corpus, samples_per_cohort=samples_per_cohort, seed=seed, **loader_kwargs
    )

    all_results: List[SampleWERResult] = []
    summaries: List[CohortWERSummary] = []
    for accent in sorted(cohorts):
        samples = cohorts[accent]
        cohort_results = [
            r for r in (score_sample(meta, transcribe) for meta in samples) if r is not None
        ]
        all_results.extend(cohort_results)
        summaries.append(_summarize_cohort(accent, len(samples), cohort_results))

    scored = [s for s in summaries if s.scored_count > 0]
    unscored = [s for s in summaries if s.scored_count == 0]
    scored.sort(key=lambda s: s.mean_wer, reverse=True)

    return AccentBiasReport(
        corpus=corpus,
        model_id=model_id,
        cohorts=scored + unscored,
        sample_results=all_results,
    )


def main() -> None:
    """CLI entrypoint: ``python -m app.domain.accent_bias_runner --model-id ...``."""
    import argparse

    from .accent_bias_profiler import make_whisper_transcriber

    parser = argparse.ArgumentParser(description="Run the group-wise accent-bias WER diagnostic.")
    parser.add_argument("--model-id", required=True, help="HF model id, e.g. openai/whisper-base")
    parser.add_argument("--corpus", default="l2-arctic")
    parser.add_argument("--samples-per-cohort", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="Write JSON report to this path instead of stdout")
    args = parser.parse_args()

    transcribe = make_whisper_transcriber(args.model_id)
    report = run_accent_bias_diagnostic(
        transcribe,
        corpus=args.corpus,
        model_id=args.model_id,
        samples_per_cohort=args.samples_per_cohort,
        seed=args.seed,
    )
    output_json = json.dumps(report.to_json_dict(), indent=2)
    if args.output:
        Path(args.output).write_text(output_json)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
