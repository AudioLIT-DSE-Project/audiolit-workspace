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

    import time
    job_id = f"warmup_{uuid.uuid4().hex[:12]}"
    try:
        conn = get_redis_connection()
        if conn:
            conn.set(f"job_progress_{job_id}", json.dumps({
                "completed": 0, "total": 100, "current_file": "Initializing...", "status": "running", "percent": 0.0,
                # The dataset is recorded on the job itself so a client that has
                # lost its job id (page reload, tab discard) can rediscover the
                # run and rebuild the progress banner without guessing which
                # dataset it belongs to. See GET /inference/warmup/active.
                "dataset": req.dataset,
                "model": req.model,
                "updated_at": time.time(),
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
            # The RQ job shares the warmup id so warmup_liveness() can tell a
            # run that is executing from one whose worker died under it.
            job_id=job_id,
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


@router.get("/inference/warmup/active")
async def list_active_warmups():
    """Warmup runs that are still in flight, so a client can reattach to one.

    The job id previously existed only in React state. A reload, a navigation,
    or the browser discarding a backgrounded tab dropped it, and because
    cancellation is addressed by job id, the run then became both invisible and
    uncancellable while continuing to consume CPU for up to its 24-hour job
    timeout. This endpoint lets a client that has lost the id find the run
    again instead of stranding it.

    Returns only non-terminal runs, newest progress first. `active_job_id` is a
    convenience for the common single-run case.
    """
    import json
    from app.orchestration.task_orchestrator import (
        get_redis_connection,
        reconcile_warmup_progress,
    )

    try:
        conn = get_redis_connection()
        if not conn:
            return {"active_job_id": None, "jobs": [], "status": "no_broker"}

        jobs = []
        # Bounded: scan_iter streams rather than materialising the keyspace, and
        # progress keys carry a 24h TTL so this set stays small.
        for key in conn.scan_iter(match="job_progress_*", count=100):
            key_s = key.decode("utf-8") if isinstance(key, bytes) else key
            raw = conn.get(key_s)
            if not raw:
                continue
            try:
                data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            except (ValueError, TypeError):
                continue  # a malformed record must not hide the healthy ones
            job_id = key_s[len("job_progress_"):]
            # Drops runs whose worker is gone (marking them terminal), which
            # previously sat here as "running" and were reattached forever.
            data = reconcile_warmup_progress(conn, job_id, data)
            if data.get("status") not in ("running", "cancelling"):
                continue
            data["job_id"] = job_id
            jobs.append(data)

        jobs.sort(key=lambda j: j.get("percent") or 0, reverse=True)
        return {
            "active_job_id": jobs[0]["job_id"] if jobs else None,
            "jobs": jobs,
            "status": "ok",
        }
    except Exception as e:
        logger.warning(f"Could not list active warmups: {e}")
        return {"active_job_id": None, "jobs": [], "status": "error", "error": str(e)}


@router.get("/inference/progress/{job_id}")
async def get_job_progress(job_id: str):
    import json
    from app.orchestration.task_orchestrator import (
        get_redis_connection,
        reconcile_warmup_progress,
    )

    try:
        conn = get_redis_connection()
        if not conn:
            return {"job_id": job_id, "status": "unknown", "completed": 0, "total": 0, "percent": 0.0}

        raw = conn.get(f"job_progress_{job_id}")
        if not raw:
            return {"job_id": job_id, "status": "not_found", "completed": 0, "total": 0, "percent": 0.0}

        data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        return reconcile_warmup_progress(conn, job_id, data)
    except Exception as e:
        return {"job_id": job_id, "status": "error", "error": str(e), "completed": 0, "total": 0, "percent": 0.0}


@router.post("/inference/cancel/{job_id}")
async def cancel_batch_job(job_id: str):
    """Cancel a warmup run and report the state it is actually in.

    This used to set a flag and unconditionally answer "cancelled", even when
    no worker was left to read the flag - so a run orphaned by a worker
    restart stayed "running" forever while the API claimed it had stopped.
    Now the response is ``cancelling`` (a live worker will stop at its next
    checkpoint) or ``cancelled`` (it was queued or orphaned, and is stopped
    now).
    """
    from app.orchestration.task_orchestrator import cancel_warmup, get_redis_connection

    try:
        conn = get_redis_connection()
        result = cancel_warmup(conn, job_id)
        logger.info("Cancellation for warmup %s -> %s", job_id, result.get("status"))
    except Exception as e:
        logger.warning(f"Could not cancel warmup {job_id}: {e}")
        raise HTTPException(status_code=503, detail=f"Could not reach the task broker: {e}")

    result["message"] = "Completed samples remain saved in cache."
    return result


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




