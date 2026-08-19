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
from app.infrastructure.dataset_service import resolve_file

router = APIRouter()
logger = logging.getLogger("audiolit.api.acoustic")


class AcousticProfileRequest(BaseModel):
    file_path: Optional[str] = None
    dataset: Optional[str] = None
    dataset_file: Optional[str] = None


@router.post("/acoustic/profile")
def acoustic_profile(http_request: Request, request: AcousticProfileRequest) -> dict:
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

    import hashlib
    from app.infrastructure.redis import get_result_sync, cache_result_sync
    file_path_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
    file_stat = resolved_path.stat()
    file_content_hash = hashlib.md5(f"{str(resolved_path)}_{file_stat.st_size}_{file_stat.st_mtime}".encode()).hexdigest()

    cached = get_result_sync("acoustic", f"acoustic_profile_{file_path_hash}") or get_result_sync("acoustic", f"acoustic_profile_{file_content_hash}")
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
        cache_result_sync("acoustic", f"acoustic_profile_{file_path_hash}", prof, ttl=86400)
        return prof
    except Exception as e:
        logger.error("Acoustic profiling failed for %s: %s", resolved_path, e)
        raise HTTPException(status_code=500, detail=f"Acoustic profiling failed: {e}")
