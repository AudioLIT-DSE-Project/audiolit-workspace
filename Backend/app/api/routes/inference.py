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


class CachedResultsRequest(BaseModel):
    audio_ref: str | None = None
    dataset: str | None = None
    dataset_file: str | None = None
    model: str = "whisper-base"


@router.post("/inference/cached-results")
async def get_cached_task_results(req: CachedResultsRequest):
    """Retrieve all cached task results (ASR, SER, ADD, acoustic) for a file reference."""
    import hashlib
    from app.infrastructure.dataset_service import resolve_audio_reference, load_metadata
    from app.infrastructure.redis import get_result
    from app.infrastructure import cache_keys as ck
    from app.orchestration.inference_service import ADD_MODEL_KEYS

    resolved_path = None
    if req.dataset and req.dataset_file:
        try:
            # Keyword arguments: the signature is
            # (file_path, dataset, dataset_file, session_id), so passing
            # (dataset, dataset_file, audio_ref) positionally resolved the
            # dataset NAME as a file path, every lookup missed, and the panel
            # silently fell back to ground-truth metadata.
            resolved_path = resolve_audio_reference(
                file_path=req.audio_ref,
                dataset=req.dataset,
                dataset_file=req.dataset_file,
            )
        except Exception:
            resolved_path = None

    file_ref = req.audio_ref or req.dataset_file or ""
    path_str = str(resolved_path) if resolved_path else file_ref
    file_path_hash = hashlib.md5(path_str.encode()).hexdigest()
    filename_hash = hashlib.md5(file_ref.split("/")[-1].split("\\")[-1].encode()).hexdigest() if file_ref else ""

    hash_candidates = [h for h in [file_path_hash, filename_hash] if h]
    # The content hash is what the saliency and acoustic families key on.
    if resolved_path is not None:
        try:
            hash_candidates.append(ck.content_hash(resolved_path))
        except OSError:
            pass

    async def first_hit(keys):
        for ns, key in keys:
            hit = await get_result(ns, key)
            if hit:
                return hit
        return None

    tasks: dict[str, Any] = {}
    hashes = tuple(hash_candidates)

    # ASR. The transcript family stores {"prediction": "<string>"}; handing the
    # wrapper straight to the UI left `asr.transcript` undefined, so the card
    # rendered blank while the data sat right there.
    asr = await first_hit(ck.transcript_keys(req.model, hashes))
    if asr is None:
        for h in hashes:
            asr = await get_result("predictions", h)
            if asr:
                break
    if asr:
        transcript = ck.as_transcript(ck.unwrap_prediction(asr))
        if transcript:
            tasks["asr"] = {"transcript": transcript, "tokens": []}

    # SER, ADD and acoustic all go through cache_keys. Hand-rolled spellings
    # here read `ser_{h}`, `add_{h}` and an unversioned `acoustic_profile_{h}`,
    # none of which any writer produces - so the Emotion Analytics and Deepfake
    # cards were starved even when the data was sitting in Redis under its real
    # key. One definition of a key family, or the readers drift from the writers.
    # Try the requested SER checkpoint first, then the default: a custom model's
    # prediction lives under its own key now and must not be answered by the
    # default model's entry.
    ser = await first_hit(ck.ser_keys(hashes, req.model)) or await first_hit(ck.ser_keys(hashes))
    if ser:
        tasks["ser"] = ck.unwrap_prediction(ser)

    for add_model in ADD_MODEL_KEYS:
        add = await first_hit(ck.deepfake_keys(add_model, hashes))
        if add:
            tasks["add"] = ck.unwrap_prediction(add)
            break

    acoustic = await first_hit(ck.acoustic_keys(hashes))
    if acoustic:
        tasks["acoustic"] = acoustic

    # Fallback to dataset metadata if ASR transcript is still missing
    if "asr" not in tasks and req.dataset and req.dataset_file:
        try:
            meta = load_metadata(req.dataset)
            clean_target = req.dataset_file.split("/")[-1].split("\\")[-1]
            for row in meta:
                path_val = str(row.get("path") or row.get("filepath") or row.get("file") or row.get("filename") or "")
                row_file = path_val.split("/")[-1].split("\\")[-1]
                if row_file == clean_target:
                    transcript = row.get("label") or row.get("transcript") or row.get("text") or row.get("sentence") or row.get("prediction")
                    if transcript:
                        tasks["asr"] = {"transcript": str(transcript), "tokens": []}
                    break
        except Exception:
            pass

    return {
        "audio_ref": file_ref,
        "tasks": tasks,
        "cached": len(tasks) > 0,
    }




