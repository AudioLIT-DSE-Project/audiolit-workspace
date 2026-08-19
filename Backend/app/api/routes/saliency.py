from fastapi import APIRouter, HTTPException, Body, Request
import asyncio
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel
from app.domain.saliency_service import generate_saliency
from app.infrastructure.dataset_service import resolve_file, resolve_audio_reference
from app.infrastructure.redis import get_result, cache_result
from app.api.dependencies import get_session_id

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
SALIENCY_SCHEMA_VERSION = "v3"  # bump to bust stale caches after logic changes (v3: gradcam vs integrated_gradients split)

class SaliencyRequest(BaseModel):
    model: str
    method: str = "gradcam"
    file_path: Optional[str] = None
    dataset: Optional[str] = None
    dataset_file: Optional[str] = None
    target_index: Optional[int] = None
    layer_name: Optional[str] = None
    no_cache: bool = False

class SaliencyResponse(BaseModel):
    """Mirrors exactly what `generate_saliency` returns.

    Every field below appears in all six return paths of
    `app/domain/saliency_service.py` (verified against the source, not assumed).
    Declaring a field the service never sets makes pydantic reject a perfectly
    good result with a 422 - which is what happened when this model was
    rewritten to `success`/`max_val`/`target_class`/`duration_s`/`sample_rate`.
    """

    model: str
    method: str
    segments: list
    total_duration: float
    series: Optional[list] = None
    base_spectrogram: Optional[list] = None
    saliency_matrix: Optional[list] = None
    emotion: Optional[str] = None
    # LIT-238 contract: a string enum value, not a dict.
    provenance: Optional[str] = None
    provenance_reason: Optional[str] = None

@router.post("/saliency/generate", response_model=SaliencyResponse)
async def generate_saliency_endpoint(http_request: Request, request: SaliencyRequest):
    if not request.model:
        raise HTTPException(status_code=400, detail="Model is required")
    
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
    
    import hashlib
    # Include file size and modification time for better cache key uniqueness
    file_stat = resolved_path.stat()
    file_content_hash = hashlib.md5(
        f"{str(resolved_path)}_{file_stat.st_size}_{file_stat.st_mtime}".encode()
    ).hexdigest()
    cache_key = f"saliency_{SALIENCY_SCHEMA_VERSION}_{request.model}_{request.method}_{file_content_hash}"
    
    if not request.no_cache:
        cached_result = await get_result("saliency", cache_key)
        if cached_result is not None:
            logger.info(f"Returning cached saliency for {resolved_path}")
            return SaliencyResponse(**cached_result)
    
    # Check if we have existing prediction data to reuse
    prediction_cache_key = f"{request.model}_{file_content_hash}"
    existing_prediction = await get_result(request.model, prediction_cache_key)
    
    try:
        result = await asyncio.to_thread(
            generate_saliency, 
            str(resolved_path), 
            request.model, 
            request.method,
            existing_prediction
        )
        
        await cache_result("saliency", cache_key, result, ttl=6*60*60)
        logger.info(f"Cached saliency for {resolved_path}")
        
        return SaliencyResponse(**result)
        
    except ValueError as e:
        logger.warning(f"Saliency generation bad request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating saliency: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Saliency generation failed: {str(e)}")

@router.get("/saliency/{method}/{model}/{file_id}")
async def get_saliency(method: str, model: str, file_id: str):
    # Match cache key lookup from generate_saliency_endpoint
    cache_key = f"saliency_{SALIENCY_SCHEMA_VERSION}_{model}_{method}_{file_id}"
    
    cached_result = await get_result("saliency", cache_key)
    if cached_result is None:
        # Also try legacy key format
        legacy_key = f"saliency_{model}_{method}_{file_id}"
        cached_result = await get_result("saliency", legacy_key)
        
    if cached_result is None:
        raise HTTPException(status_code=404, detail="Saliency not found in cache")
    
    return SaliencyResponse(**cached_result)
