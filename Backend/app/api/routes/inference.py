"""FastAPI gateway routes for asynchronous inference (SRS FR3).

The gateway only enqueues and returns a job id - it never loads a model or runs
inference itself (SAD §5.1: "the gateway never loads AI models directly").
Job progress and results are served by `tasks.py`; this module deliberately does
not duplicate that surface.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...orchestration.task_orchestrator import (
    TaskFamily,
    enqueue_attribution,
    enqueue_multitask_analysis,
    enqueue_mutation,
)

logger = logging.getLogger("audiolit.api.inference")
router = APIRouter(prefix="/api", tags=["inference"])

class MultiTaskRequest(BaseModel):
    audio_ref: str = Field(..., description="Content-addressed audio ref")
    tasks: list[str] = Field(default_factory=lambda: ["asr", "ser", "add"])
    model_ids: dict[str, str] = Field(default_factory=dict)
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    cache_key: str | None = None

class AttributionRequest(BaseModel):
    audio_ref: str
    model_id: str
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    cache_key: str | None = None

class MutationRequest(BaseModel):
    audio_ref: str
    mutation: dict[str, Any]

class JobResponse(BaseModel):
    job_id: str
    websocket_url: str
    schema_version: str
    family_jobs: dict[str, str]
    cache_key: str | None = None

@router.post("/inference/multitask", response_model=JobResponse)
async def post_multitask(req: MultiTaskRequest) -> JobResponse:
    try:
        result = enqueue_multitask_analysis(
            audio_ref=req.audio_ref,
            tasks=[TaskFamily(t) for t in req.tasks],
            model_ids={TaskFamily(k): v for k, v in req.model_ids.items()},
            params={TaskFamily(k): v for k, v in req.params.items()},
            cache_key=req.cache_key,
        )
    except ValueError as exc:
        # An unknown or unsupported task family is a bad request, not a server
        # fault - it used to surface as a 500 from a KeyError.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JobResponse(**result.as_response())

@router.post("/inference/attribution", response_model=JobResponse)
async def post_attribution(req: AttributionRequest) -> JobResponse:
    result = enqueue_attribution(
        audio_ref=req.audio_ref, model_id=req.model_id, method=req.method, params=req.params, cache_key=req.cache_key
    )
    return JobResponse(**result.as_response())

@router.post("/inference/mutation", response_model=JobResponse)
async def post_mutation(req: MutationRequest) -> JobResponse:
    result = enqueue_mutation(audio_ref=req.audio_ref, mutation=req.mutation)
    return JobResponse(**result.as_response())
