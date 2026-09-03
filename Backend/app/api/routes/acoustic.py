"""Acoustic Wave Profiler exposure (LIT-231, FR10).

Pure DSP - no model load, safe to run synchronously on the request path
(unlike model inference, which is why FR3's async gateway rule doesn't apply
here; SAD §5.1's "gateway never loads AI models" is about AI models
specifically). Wraps `acoustic_profiler_service.extract_acoustic_profile`,
which already returns a JSON-safe dict.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import soundfile as sf
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.api.dependencies import get_session_id
from app.domain.acoustic_profiler_service import extract_acoustic_profile
from app.infrastructure.dataset_service import resolve_audio_reference

router = APIRouter()
logger = logging.getLogger("audiolit.api.acoustic")


class AcousticProfileRequest(BaseModel):
    file_path: Optional[str] = None
    dataset: Optional[str] = None
    dataset_file: Optional[str] = None


@router.post("/acoustic/profile")
def acoustic_profile(http_request: Request, request: AcousticProfileRequest) -> dict:
    session_id = get_session_id(http_request)

    try:
        resolved_path = resolve_audio_reference(
            file_path=request.file_path,
            dataset=request.dataset,
            dataset_file=request.dataset_file,
            session_id=session_id,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {resolved_path}")

    from app.infrastructure.redis import get_result_sync, cache_result_sync
    from app.infrastructure import cache_keys as ck

    # Keys come from cache_keys so this route, the dataset warmup and any future
    # writer cannot drift apart. Hand-rolled duplicates are how the transcript
    # family ended up holding two incompatible payload shapes.
    keys = ck.acoustic_keys(ck.both_hashes(resolved_path))
    for ns, key in keys:
        cached = get_result_sync(ns, key)
        if cached:
            return cached

    try:
        audio, sr = sf.read(str(resolved_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load audio file: {e}")

    try:
        prof = extract_acoustic_profile(audio, sr)
        for ns, key in keys:
            cache_result_sync(ns, key, prof, ttl=86400)
        return prof
    except Exception as e:
        logger.error("Acoustic profiling failed for %s: %s", resolved_path, e)
        raise HTTPException(status_code=500, detail=f"Acoustic profiling failed: {e}")
