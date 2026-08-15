"""
Accent bias profiling core (LIT-168, FR15 — Accent Bias Profiling).

Batch-transcribes L2-ARCTIC utterances grouped by native language (L1) and
scores each against its ground-truth transcript with jiwer, per SAD §5.1
domain layer / §5.2 (Bias Profiler and Faithfulness Auditor).

This module owns the accent-cohort batching and the per-sample WER primitive.
LIT-182 (the group-wise WER diagnostic runner) is the orchestration layer
built on top of these that ranks cohort results into a disparity report.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import jiwer
import librosa
import numpy as np

from ..infrastructure.dataset_ingestion import SampleMetadata, get_loader

logger = logging.getLogger(__name__)

# ASR output includes casing/punctuation that a raw string comparison would
# penalize as "errors" even on an otherwise perfect transcription -- both
# sides are normalized the same way before scoring. jiwer's own
# `wer_standardize` transform stops short of removing punctuation attached to
# words (e.g. "cat," survives it unchanged), so it isn't enough on its own.
_WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


@dataclass(frozen=True)
class SampleWERResult:
    """One sample's transcription outcome against its L2-ARCTIC ground truth."""

    sample_id: str
    speaker_id: Optional[str]
    accent: str
    reference: str
    hypothesis: str
    wer: float


TranscribeFn = Callable[[str], str]
"""Transcribes one audio file path to text. Injected rather than hardcoded so
the scoring/batching primitives below don't depend on a specific ASR
entrypoint or a loaded model in tests -- callers pass a real transcriber
(see `make_whisper_transcriber`) or a fake."""


def load_accent_cohorts(
    corpus: str = "l2-arctic",
    samples_per_cohort: Optional[int] = None,
    seed: int = 0,
    **loader_kwargs,
) -> Dict[str, List[SampleMetadata]]:
    """Load ``corpus`` samples grouped by accent/L1.

    Without ``samples_per_cohort``, returns every sample per accent group.
    With it, reservoir-samples each cohort independently in a single pass
    over the corpus (the same algorithm `DatasetLoader.subsample` already
    uses, applied per group) -- a single global cap would let large cohorts
    exhaust it before smaller-represented accents are ever seen, silently
    starving them out of the profile.
    """
    loader = get_loader(corpus, **loader_kwargs)
    cohorts: Dict[str, List[SampleMetadata]] = {}

    if samples_per_cohort is None:
        for meta in loader.iter_metadata():
            cohorts.setdefault(meta.accent or "unknown", []).append(meta)
        return cohorts

    rng = random.Random(seed)
    seen: Dict[str, int] = {}
    for meta in loader.iter_metadata():
        key = meta.accent or "unknown"
        reservoir = cohorts.setdefault(key, [])
        i = seen.get(key, 0)
        seen[key] = i + 1
        if i < samples_per_cohort:
            reservoir.append(meta)
        else:
            j = rng.randint(0, i)
            if j < samples_per_cohort:
                reservoir[j] = meta
    return cohorts


def score_sample(meta: SampleMetadata, transcribe: TranscribeFn) -> Optional[SampleWERResult]:
    """Transcribe one sample and score it against its ground truth.

    Returns ``None`` (rather than raising) for samples this diagnostic can't
    score -- no ground-truth transcript, or no accent label -- since a batch
    run over hundreds of utterances shouldn't die on one bad row.
    """
    if not meta.label or not meta.label.strip():
        logger.warning("Skipping %s: no ground-truth transcript", meta.sample_id)
        return None
    if not meta.accent:
        logger.warning("Skipping %s: no accent/L1 label", meta.sample_id)
        return None

    hypothesis = transcribe(str(meta.audio_path))
    wer = jiwer.wer(
        meta.label,
        hypothesis,
        reference_transform=_WER_TRANSFORM,
        hypothesis_transform=_WER_TRANSFORM,
    )

    return SampleWERResult(
        sample_id=meta.sample_id,
        speaker_id=meta.speaker_id,
        accent=meta.accent,
        reference=meta.label,
        hypothesis=hypothesis,
        wer=wer,
    )


def make_whisper_transcriber(model_id: str) -> TranscribeFn:
    """Build a transcribe function backed by one cached Whisper pipeline.

    A profiling run transcribes many utterances in one pass;
    `model_loader_service.transcribe_whisper` rebuilds its HF pipeline on
    every call, which is fine for one-off request-path inference but far too
    slow across a batch, so this loads the pipeline once and reuses it.
    """
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
    )

    def _transcribe(audio_path: str) -> str:
        audio, _ = librosa.load(audio_path, sr=16_000)
        result = asr_pipeline(audio.astype(np.float32), chunk_length_s=30)
        return result["text"]

    return _transcribe
