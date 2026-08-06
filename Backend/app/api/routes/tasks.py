"""FastAPI WebSocket and HTTP long-polling routes for RQ task states (SRS FR3.2)."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from rq.job import JobStatus

from app.services.queue_service import (
    PROGRESS_CHANNEL_PREFIX, fetch_job, get_redis_connection
)

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

@router.get("/api/tasks/{task_id}/status")
async def get_task_status(task_id: str) -> dict[str, Any]:
    """HTTP long-polling fallback for task state (SRS FR3.2)."""
    job = fetch_job(task_id)
    if not job:
        return {"task_id": task_id, "state": "UNKNOWN"}
    return {
        "task_id": task_id,
        "state": _map_rq_status(job),
        "result": job.result if job.is_finished else None,
        "error": str(job.exc_info) if job.is_failed else None,
    }

@router.websocket("/api/ws/tasks/{task_id}")
async def task_progress_ws(websocket: WebSocket, task_id: str) -> None:
    """
    Stream JSON state adjustments (QUEUED, PROCESSING, RETRYING, SUCCESS, FAILURE)
    relayed from the Redis pub/sub event layer keyed by job id.
    """
    await websocket.accept()
    conn = get_redis_connection()
    channel = f"{PROGRESS_CHANNEL_PREFIX}:{task_id}".encode()
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
            "payload": {"result": job.result if job and job.is_finished else None}
        }))

        # If job is already done, close socket after sending final state
        if initial_state in ["SUCCESS", "FAILURE"]:
            await websocket.close()
            return

        while True:
            # Use run_in_executor so redis-py's blocking get_message doesn't block event loop
            msg = await loop.run_in_executor(None, ps.get_message, 1.0)
            if msg is not None and msg.get("type") == "message":
                await websocket.send_text(msg["data"].decode())
                # If the message indicates a final state, close the socket cleanly
                try:
                    parsed = json.loads(msg["data"])
                    if parsed.get("stage") in ["SUCCESS", "FAILURE"]:
                        await websocket.close()
                        break
                except Exception:
                    pass
            
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
