"""
Group-wise emotion-recognition accuracy (FR15, extended to the SER corpora).

`accent_bias_runner.py` measures word error rate per accent on L2-ARCTIC. The
emotion corpora have no accent label and no use for WER: their ground truth is
an emotion class, and what they do carry is who is speaking - race, sex and
ethnicity for CREMA-D, the speaker's language for ESD. This is the same
diagnostic for them: classify every sampled clip, score it against its label,
and rank the groups by accuracy so the worst-served group sorts first.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

from ..infrastructure.dataset_ingestion import SampleMetadata
from .accent_bias_profiler import load_accent_cohorts

logger = logging.getLogger(__name__)

#: The fields each corpus can be grouped by; the first is the default.
EMOTION_BIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "crema-d": ("race", "sex", "ethnicity"),
    "esd": ("language",),
}

#: The built-in SER model's toolbar key; `None` selects the default checkpoint.
_DEFAULT_SER_ALIASES = (None, "", "default", "wav2vec2")

PredictFn = Callable[[str], Optional[str]]
"""Classifies one audio file to an emotion label. Injected, like the accent
runner's transcriber, so the scoring can be tested without a loaded model."""


@dataclass(frozen=True)
class SampleEmotionResult:
    sample_id: str
    speaker_id: Optional[str]
    group: str
    reference: str
    predicted: Optional[str]
    correct: bool


@dataclass(frozen=True)
class GroupAccuracySummary:
    """Accuracy for one group. ``accuracy`` is ``None`` (not NaN) when the
    group has no scoreable sample, so the report stays valid JSON."""

    group: str
    sample_count: int
    scored_count: int
    correct_count: int
    accuracy: Optional[float]


@dataclass(frozen=True)
class EmotionBiasReport:
    """``cohorts`` ranked lowest accuracy first."""

    corpus: str
    model_id: str
    group_by: str
    cohorts: List[GroupAccuracySummary]
    sample_results: List[SampleEmotionResult]

    def to_json_dict(self) -> dict:
        return {
            "corpus": self.corpus,
            "model_id": self.model_id,
            "metric": "emotion_accuracy",
            "group_by": self.group_by,
            "cohorts": [asdict(c) for c in self.cohorts],
            "sample_results": [asdict(r) for r in self.sample_results],
        }


def resolve_group_field(corpus: str, group_by: Optional[str]) -> str:
    """The grouping field to use for ``corpus``; raises ``ValueError`` if the
    corpus has none or does not offer the one asked for."""
    fields = EMOTION_BIAS_GROUPS.get(corpus.lower())
    if not fields:
        raise ValueError(f"Emotion bias profiling is not available for corpus '{corpus}'.")
    if not group_by:
        return fields[0]
    if group_by not in fields:
        raise ValueError(
            f"Corpus '{corpus}' cannot be grouped by '{group_by}'. Choose from: {', '.join(fields)}."
        )
    return group_by


def score_emotion_sample(meta: SampleMetadata, group: str, predict: PredictFn) -> Optional[SampleEmotionResult]:
    """Classify one sample and compare it with its label; ``None`` if the
    sample has no label to compare against."""
    if not meta.label or not meta.label.strip():
        logger.warning("Skipping %s: no emotion label", meta.sample_id)
        return None
    reference = meta.label.strip().lower()
    predicted = predict(str(meta.audio_path))
    normalised = predicted.strip().lower() if isinstance(predicted, str) else None
    return SampleEmotionResult(
        sample_id=meta.sample_id,
        speaker_id=meta.speaker_id,
        group=group,
        reference=reference,
        predicted=normalised,
        correct=normalised == reference,
    )


def run_emotion_bias_diagnostic(
    predict: PredictFn,
    corpus: str,
    model_id: str = "unknown",
    group_by: Optional[str] = None,
    samples_per_cohort: Optional[int] = None,
    seed: int = 0,
    **loader_kwargs,
) -> EmotionBiasReport:
    """Classify a sample of every group and return a ranked accuracy report."""
    field = resolve_group_field(corpus, group_by)
    cohorts = load_accent_cohorts(
        corpus,
        samples_per_cohort=samples_per_cohort,
        seed=seed,
        group_key=lambda meta: (meta.demographic.get(field) or "").strip() or None,
        **loader_kwargs,
    )

    all_results: List[SampleEmotionResult] = []
    summaries: List[GroupAccuracySummary] = []
    for group in sorted(cohorts):
        samples = cohorts[group]
        results = [
            r for r in (score_emotion_sample(meta, group, predict) for meta in samples) if r is not None
        ]
        all_results.extend(results)
        correct = sum(1 for r in results if r.correct)
        summaries.append(
            GroupAccuracySummary(
                group=group,
                sample_count=len(samples),
                scored_count=len(results),
                correct_count=correct,
                accuracy=(correct / len(results)) if results else None,
            )
        )

    scored = sorted((s for s in summaries if s.scored_count > 0), key=lambda s: s.accuracy)
    unscored = [s for s in summaries if s.scored_count == 0]
    return EmotionBiasReport(
        corpus=corpus,
        model_id=model_id,
        group_by=field,
        cohorts=scored + unscored,
        sample_results=all_results,
    )


def make_ser_predictor(model_id: Optional[str]) -> PredictFn:
    """A predictor backed by the SER model `predict_ser` loads and caches."""
    from .model_loader_service import predict_ser

    selected = None if model_id in _DEFAULT_SER_ALIASES else model_id

    def _predict(audio_path: str) -> Optional[str]:
        return predict_ser(audio_path, model_id=selected).get("predicted_emotion")

    return _predict
