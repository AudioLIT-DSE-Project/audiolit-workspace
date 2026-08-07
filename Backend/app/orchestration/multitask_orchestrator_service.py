"""
Real ASR + SER multi-task fan-out/fan-in orchestrator (LIT-150 draft).

DRAFT -- LIT-150 is assigned to Rahim in Linear; this is a head start for
his review/takeover, not a claim on the issue.

Extends LIT-225's stub fan-out/fan-in prototype (fanout_orchestrator_service.py)
with real inference: ASR (Whisper), SER (Wav2Vec2 emotion) and ADD (Wav2Vec2
binary deepfake detection, LIT-128) all run as actual parallel RQ child jobs,
reusing the same job-dependency fan-in pattern (see docs/rq_fanout_pattern.md).

Deliberately NOT wired into the /upload or /inferences/run HTTP routes: doing
so changes their response contract from an inline result to a job id the
client polls for, which the frontend isn't built to handle yet (LIT-157, not
started). Wiring this into the actual request path is LIT-150/157's job.
"""

from __future__ import annotations

from typing import Any

from rq import Queue, get_current_job
from rq.job import Dependency, Job

from ..infrastructure.rq_connection import get_redis_connection
from .fanout_orchestrator_service import CHILD_JOB_RETRY, _publish_progress
from ..domain.model_loader_service import (
    predict_deepfake,
    predict_ser,
    transcribe_whisper_base,
    transcribe_whisper_large,
)

ASR_QUEUE_NAME = "multitask_asr"
SER_QUEUE_NAME = "multitask_ser"
ADD_QUEUE_NAME = "multitask_add"
AGGREGATOR_QUEUE_NAME = "multitask_aggregator"


def run_asr_job(file_path: str, model: str = "whisper-base") -> dict[str, Any]:
    """Real Whisper transcription, run as an RQ child job."""
    conn = get_redis_connection()
    job = get_current_job()
    if job is not None:
        _publish_progress(conn, job.id, "asr", 0.5)

    transcribe = transcribe_whisper_large if model == "whisper-large" else transcribe_whisper_base
    transcript = transcribe(file_path)

    if job is not None:
        _publish_progress(conn, job.id, "asr", 1.0)
    return {"task_name": "asr", "transcript": transcript}


def run_ser_job(file_path: str) -> dict[str, Any]:
    """Real Wav2Vec2 emotion prediction (LIT-206), run as an RQ child job."""
    conn = get_redis_connection()
    job = get_current_job()
    if job is not None:
        _publish_progress(conn, job.id, "ser", 0.5)

    prediction = predict_ser(file_path)

    if job is not None:
        _publish_progress(conn, job.id, "ser", 1.0)
    return {"task_name": "ser", **prediction}


def run_add_job(file_path: str) -> dict[str, Any]:
    """Real binary audio-deepfake detection (LIT-128), run as an RQ child job."""
    conn = get_redis_connection()
    job = get_current_job()
    if job is not None:
        _publish_progress(conn, job.id, "add", 0.5)

    prediction = predict_deepfake(file_path)

    if job is not None:
        _publish_progress(conn, job.id, "add", 1.0)
    return {"task_name": "add", **prediction}


def aggregate_multitask(child_job_ids: list[str]) -> dict[str, Any]:
    """Runs once all three child jobs finish or exhaust retries. Mirrors
    fanout_orchestrator_service.aggregate_children's failure handling: a
    failed sibling never loses the others' results.
    """
    conn = get_redis_connection()
    succeeded: dict[str, Any] = {}
    failed: dict[str, str] = {}

    for job_id in child_job_ids:
        job = Job.fetch(job_id, connection=conn)
        if job.is_failed:
            outcome = job.latest_result()
            exc_string = outcome.exc_string if outcome else None
            last_line = exc_string.strip().splitlines()[-1] if exc_string else "unknown error"
            failed[job_id] = last_line
        else:
            succeeded[job_id] = job.return_value()

    return {"succeeded": succeeded, "failed": failed}


def enqueue_multitask(file_path: str, model: str = "whisper-base") -> dict[str, Any]:
    """Dispatch ASR + SER + ADD as parallel child jobs, one queue per task
    family (SAD §6.1: "one worker for each kind of model... so each worker
    only ever needs to hold one model in memory"), then an aggregator job
    depending on all three.
    """
    conn = get_redis_connection()

    asr_job = Queue(ASR_QUEUE_NAME, connection=conn).enqueue(
        run_asr_job, file_path=file_path, model=model, retry=CHILD_JOB_RETRY
    )
    ser_job = Queue(SER_QUEUE_NAME, connection=conn).enqueue(
        run_ser_job, file_path=file_path, retry=CHILD_JOB_RETRY
    )
    add_job = Queue(ADD_QUEUE_NAME, connection=conn).enqueue(
        run_add_job, file_path=file_path, retry=CHILD_JOB_RETRY
    )

    child_jobs = [asr_job, ser_job, add_job]
    aggregator_job = Queue(AGGREGATOR_QUEUE_NAME, connection=conn).enqueue(
        aggregate_multitask,
        child_job_ids=[j.id for j in child_jobs],
        depends_on=Dependency(jobs=child_jobs, allow_failure=True),
    )

    return {
        "asr_job_id": asr_job.id,
        "ser_job_id": ser_job.id,
        "add_job_id": add_job.id,
        "aggregator_job_id": aggregator_job.id,
    }
