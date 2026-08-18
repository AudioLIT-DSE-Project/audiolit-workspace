"""FastAPI gateway routes for asynchronous inference (SRS FR3).

The gateway only enqueues and returns a job id - it never loads a model or runs
inference itself (SAD §5.1: "the gateway never loads AI models directly").
Job progress and results are served by `tasks.py`; this module deliberately does
not duplicate that surface.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
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


class BatchWarmupRequest(BaseModel):
    dataset: str
    model: str = "whisper-base"
    tasks: list[str] = Field(default_factory=lambda: ["asr", "ser", "acoustic"])
    cooldown_ms: int = 100


@router.post("/inference/batch-warmup")
async def post_batch_warmup(req: BatchWarmupRequest):
    import uuid
    import json
    from app.orchestration.task_orchestrator import get_queue, WorkerFamily, run_batch_dataset_warmup_task, get_redis_connection

    job_id = f"warmup_{uuid.uuid4().hex[:12]}"
    try:
        conn = get_redis_connection()
        if conn:
            conn.set(f"job_progress_{job_id}", json.dumps({
                "completed": 0, "total": 100, "current_file": "Initializing...", "status": "running", "percent": 0.0
            }), ex=86400)
    except Exception as e:
        logger.warning(f"Could not initialize Redis progress for job {job_id}: {e}")

    try:
        q = get_queue(WorkerFamily.ASR)
        q.enqueue(
            run_batch_dataset_warmup_task,
            job_id,
            req.dataset,
            req.model,
            req.tasks,
            req.cooldown_ms,
            job_timeout=86400,
        )
    except Exception as e:
        logger.warning(f"Fallback to background thread for batch warmup: {e}")
        import asyncio
        asyncio.create_task(
            asyncio.to_thread(
                run_batch_dataset_warmup_task,
                job_id,
                req.dataset,
                req.model,
                req.tasks,
                req.cooldown_ms,
            )
        )

    return {"job_id": job_id, "status": "running", "message": "Batch warmup started"}


@router.get("/inference/progress/{job_id}")
async def get_job_progress(job_id: str):
    import json
    from app.orchestration.task_orchestrator import get_redis_connection

    try:
        conn = get_redis_connection()
        if not conn:
            return {"job_id": job_id, "status": "unknown", "completed": 0, "total": 0, "percent": 0.0}

        raw = conn.get(f"job_progress_{job_id}")
        if not raw:
            return {"job_id": job_id, "status": "not_found", "completed": 0, "total": 0, "percent": 0.0}

        data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        return data
    except Exception as e:
        return {"job_id": job_id, "status": "error", "error": str(e), "completed": 0, "total": 0, "percent": 0.0}


@router.post("/inference/cancel/{job_id}")
async def cancel_batch_job(job_id: str):
    from app.orchestration.task_orchestrator import get_redis_connection

    try:
        conn = get_redis_connection()
        if conn:
            conn.set(f"cancel_job_{job_id}", "1", ex=3600)
            logger.info(f"Cancellation signal sent for job {job_id}")
    except Exception as e:
        logger.warning(f"Could not send cancellation signal to Redis: {e}")

    return {"job_id": job_id, "status": "cancelled", "message": "Cancellation requested. Completed samples remain saved in cache."}


@router.post("/cache/clear")
@router.delete("/cache/clear")
async def clear_ml_cache():
    """Flush all cached ML results, saliency maps, acoustic profiles, and predictions."""
    from app.orchestration.task_orchestrator import get_redis_connection

    cleared_count = 0
    try:
        conn = get_redis_connection()
        if conn:
            patterns = ["result:*", "saliency_*", "acoustic_profile_*", "v2_*", "whisper*", "wav2vec2*"]
            for pattern in patterns:
                keys = conn.keys(pattern)
                if keys:
                    conn.delete(*keys)
                    cleared_count += len(keys)
            logger.info(f"Cleared {cleared_count} cached Redis keys")
    except Exception as e:
        logger.warning(f"Failed to clear Redis cache: {e}")
        return {"status": "error", "message": str(e), "cleared_keys": 0}

    return {"status": "ok", "message": "Cache cleared successfully", "cleared_keys": cleared_count}



