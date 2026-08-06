"""FastAPI gateway routes for asynchronous inference (SRS FR3)."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.services.queue_service import (
    PROGRESS_CHANNEL_PREFIX, TaskFamily,
    enqueue_attribution, enqueue_multitask_analysis, enqueue_mutation,
    get_redis_connection, job_status,
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
    result = enqueue_multitask_analysis(
        audio_ref=req.audio_ref,
        tasks=[TaskFamily(t) for t in req.tasks],
        model_ids={TaskFamily(k): v for k, v in req.model_ids.items()},
        params={TaskFamily(k): v for k, v in req.params.items()},
        cache_key=req.cache_key,
    )
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

@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    return job_status(job_id)

@router.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    conn = get_redis_connection()
    channel = f"{PROGRESS_CHANNEL_PREFIX}:{job_id}".encode()
    ps = conn.pubsub()
    ps.subscribe(channel)
    loop = asyncio.get_running_loop()
    try:
        await websocket.send_text(json.dumps({"job_id": job_id, "stage": "subscribed"}))
        while True:
            msg = await loop.run_in_executor(None, ps.get_message, 1.0)
            if msg is not None and msg.get("type") == "message":
                await websocket.send_text(msg["data"].decode())
            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            ps.unsubscribe(channel)
            ps.close()
        except Exception:
            pass