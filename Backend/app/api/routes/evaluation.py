"""Faithfulness Audit + Accent Bias Dashboard exposure (LIT-231, FR15/FR16).

`POST /evaluation/faithfulness` wraps `perturbation_service.evaluate_downstream_degradation`
(LIT-183/184) - a real, measured degradation curve from actually masking audio
and re-running inference - not `evaluation_service.evaluate_batch_faithfulness_scores`,
which simulates the curve from a formula (see the LIT-212 Linear comment).

`POST /evaluation/accent-bias` just enqueues; the real work
(`accent_bias_task`) lives in `task_orchestrator.py` alongside the other
background tasks.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.api.dependencies import get_session_id
from app.domain.perturbation_service import evaluate_downstream_degradation
from app.domain.saliency_service import generate_saliency
from app.infrastructure.dataset_service import resolve_file
from app.orchestration.task_orchestrator import enqueue_accent_bias

router = APIRouter()
logger = logging.getLogger("audiolit.api.evaluation")


def _detect_model_type(model: str) -> str:
    """Best-effort SER-vs-ADD guess from a model id/name, overridable via `model_type`."""
    lowered = model.lower()
    if "deepfake" in lowered or "add" in lowered.split("-") or lowered == "add":
        return "add"
    return "ser"


class FaithfulnessRequest(BaseModel):
    model: str
    method: str = "gradcam"
    file_path: Optional[str] = None
    dataset: Optional[str] = None
    dataset_file: Optional[str] = None
    model_type: Optional[str] = None  # "ser" | "add"; inferred from `model` if omitted


@router.post("/evaluation/faithfulness")
async def evaluation_faithfulness(http_request: Request, request: FaithfulnessRequest) -> Dict[str, Any]:
    session_id = get_session_id(http_request)

    if request.file_path:
        resolved_path = Path(request.file_path)
    elif request.dataset and request.dataset_file:
        try:
            resolved_path = resolve_file(request.dataset, request.dataset_file, session_id)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))
    else:
        raise HTTPException(
            status_code=400,
            detail="Missing audio reference. Provide either 'file_path' or 'dataset' + 'dataset_file'.",
        )

    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {resolved_path}")

    try:
        saliency = await asyncio.to_thread(
            generate_saliency, str(resolved_path), request.model, request.method
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Saliency generation failed: {e}")

    series = saliency.get("series") or []
    if not series:
        raise HTTPException(
            status_code=422, detail="Saliency method returned no per-frame series to audit."
        )

    model_type = request.model_type or _detect_model_type(request.model)
    result = await asyncio.to_thread(
        evaluate_downstream_degradation,
        audio_path=str(resolved_path),
        attributions=series,
        model_type=model_type,
        model_id=request.model,
    )
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "Faithfulness audit failed"))
    return result


class AccentBiasRequest(BaseModel):
    model_id: str
    corpus: str = "l2-arctic"
    # None means "every sample in every cohort" (perturbation domain default);
    # this route defaults to a bounded run so a UI click doesn't accidentally
    # kick off a full-corpus pass - pass null explicitly to run everything.
    samples_per_cohort: Optional[int] = 10


class JobResponse(BaseModel):
    job_id: str
    websocket_url: str
    schema_version: str
    family_jobs: Dict[str, str]
    cache_key: Optional[str] = None


@router.post("/evaluation/accent-bias", response_model=JobResponse)
def evaluation_accent_bias(request: AccentBiasRequest) -> JobResponse:
    result = enqueue_accent_bias(
        model_id=request.model_id,
        corpus=request.corpus,
        samples_per_cohort=request.samples_per_cohort,
    )
    return JobResponse(**result.as_response())
