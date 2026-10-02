"""FastAPI WebSocket and HTTP long-polling routes for RQ task states (SRS FR3.2)."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from rq.job import JobStatus

from ...infrastructure.rq_connection import get_redis_connection
from ...orchestration.task_orchestrator import fetch_job, progress_channel

logger = logging.getLogger("audiolit.api.tasks")
router = APIRouter()

def _map_rq_status(job: Any) -> str:
    """Map internal RQ status to our frontend state contract."""
    if not job:
        return "FAILURE"
    status = job.get_status()
    if status == JobStatus.QUEUED:
        return "QUEUED"
    if status == JobStatus.STARTED:
        return "PROCESSING"
    if status == JobStatus.FINISHED:
        return "SUCCESS"
    if status == JobStatus.FAILED:
        return "FAILURE"
    if status == JobStatus.DEFERRED:
        return "QUEUED" # Waiting on dependency
    return "UNKNOWN"


TERMINAL_STATES = ("SUCCESS", "FAILURE")


def _job_result(job: Any) -> Any:
    """The job's return value in a form that survives JSON encoding.

    A task result is whatever the task function returned, and nothing stops one
    carrying raw bytes: the mutation result used to include the derived clip as
    WAV bytes, which made both this module's encoders raise on every finished
    mutation job. Bytes are dropped rather than base64'd - a result is a
    description of the work, and the audio itself is served by /upload/file.
    """
    if job is None or not job.is_finished:
        return None
    try:
        return jsonable_encoder(job.result, custom_encoder={bytes: lambda _: None})
    except Exception:
        logger.warning("task.result.unserialisable job_id=%s", job.id, exc_info=True)
        return None


def _job_error(job: Any) -> str | None:
    """The last line of the job's traceback: the exception, without the stack."""
    if job is None or not job.is_failed:
        return None
    lines = [line for line in str(job.exc_info or "").splitlines() if line.strip()]
    return lines[-1].strip() if lines else "Task failed"


def terminal_event(event: dict[str, Any], job: Any) -> dict[str, Any]:
    """Attach the job's outcome to a worker's SUCCESS/FAILURE progress event.

    The worker publishes only ``{"duration_s": ...}`` with a terminal stage -
    results can be megabytes and do not belong on a pub/sub channel. The client
    was reading that payload as the result, so every panel that waits on a job
    (accent bias, mutation, the multitask fan-in) received ``{"duration_s"}``
    in place of its data. The gateway already holds the job, so it adds the
    result here, in the same ``state`` + ``payload.result`` shape the socket's
    initial message uses.
    """
    stage = event.get("stage")
    payload = dict(event.get("payload") or {})
    if stage == "SUCCESS":
        payload["result"] = _job_result(job)
    elif stage == "FAILURE":
        payload["error"] = payload.get("error") or _job_error(job) or "Task failed"
    return {**event, "state": stage, "payload": payload}

@router.get("/api/tasks/{task_id}/status")
async def get_task_status(task_id: str) -> dict[str, Any]:
    """HTTP long-polling fallback for task state (SRS FR3.2)."""
    job = fetch_job(task_id)
    if not job:
        return {"task_id": task_id, "state": "UNKNOWN"}
    return {
        "task_id": task_id,
        "state": _map_rq_status(job),
        "result": _job_result(job),
        "error": _job_error(job),
    }

@router.websocket("/api/ws/tasks/{task_id}")
async def task_progress_ws(websocket: WebSocket, task_id: str) -> None:
    """
    Stream JSON state adjustments (QUEUED, PROCESSING, RETRYING, SUCCESS, FAILURE)
    relayed from the Redis pub/sub event layer keyed by job id.
    """
    await websocket.accept()
    conn = get_redis_connection()
    channel = progress_channel(task_id).encode()
    ps = conn.pubsub()
    ps.subscribe(channel)
    loop = asyncio.get_running_loop()

    try:
        # Send initial state immediately upon connection
        job = fetch_job(task_id)
        initial_state = _map_rq_status(job)
        await websocket.send_text(json.dumps({
            "task_id": task_id,
            "state": initial_state,
            "payload": {"result": _job_result(job), "error": _job_error(job)}
        }))

        # If job is already done, close socket after sending final state
        if initial_state in TERMINAL_STATES:
            await websocket.close()
            return

        while True:
            # Use run_in_executor so redis-py's blocking get_message doesn't block event loop
            msg = await loop.run_in_executor(None, ps.get_message, 1.0)
            if msg is not None and msg.get("type") == "message":
                try:
                    parsed = json.loads(msg["data"])
                except Exception:
                    parsed = None
                if isinstance(parsed, dict) and parsed.get("stage") in TERMINAL_STATES:
                    # A final state carries the outcome, then the socket closes.
                    await websocket.send_text(
                        json.dumps(terminal_event(parsed, fetch_job(task_id)))
                    )
                    await websocket.close()
                    break
                await websocket.send_text(msg["data"].decode())
            
            # Allow event loop to process sends / disconnects
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task {task_id}")
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {e}")
    finally:
        try:
            ps.unsubscribe(channel)
            ps.close()
        except Exception:
            pass
