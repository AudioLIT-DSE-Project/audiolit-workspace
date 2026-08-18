from __future__ import annotations
import logging
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from typing import List
import os
from app.infrastructure.dataset_service import (
    load_metadata,
    resolve_file,
    media_type_for,
)
from app.infrastructure import dataset_ingestion
from app.api.dependencies import get_session_id
router = APIRouter()
logger = logging.getLogger(__name__)

# Corpora with no loader registered yet are excluded - listing them would
# offer a dataset that 404s on every request. Empty as of LIT-236 (ESD was
# the last of the seven approved corpora left unwired); kept as a guard for
# whatever gets added to CORPUS_REGISTRY next.
_UNLOADABLE_CORPORA = {
    name for name, spec in dataset_ingestion.CORPUS_REGISTRY.items()
    if spec.loader_factory is None
}


@router.get("/datasets/list")
async def list_datasets() -> JSONResponse:
    """Built-in corpora available to select in the dataset dropdown (LIT-235).

    Distinct from GET /upload/dataset/list, which lists the session's custom
    (user-uploaded) datasets.
    """
    corpora = [
        name for name in dataset_ingestion.list_supported_corpora()
        if name not in _UNLOADABLE_CORPORA
    ]
    return JSONResponse(content={"datasets": sorted(corpora)})


@router.get("/{dataset}/metadata")
async def get_dataset_metadata(dataset: str, request: Request) -> JSONResponse:
    try:
        # URL decode the dataset parameter to handle colons in custom dataset names
        dataset = unquote(dataset)
        session_id = get_session_id(request)
        rows: List[dict] = load_metadata(dataset, session_id)
        return JSONResponse(content=rows)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read CSV: {e}")


@router.get("/{dataset}/file/{file_path:path}")
@router.head("/{dataset}/file/{file_path:path}")
@router.options("/{dataset}/file/{file_path:path}")
async def serve_dataset_file(dataset: str, file_path: str, request: Request):
    logger.info(f"serve_dataset_file called: dataset='{dataset}', file_path='{file_path}'")
    try:
        # URL decode the dataset parameter to handle colons in custom dataset names
        dataset = unquote(dataset)
        logger.info(f"After URL decode: dataset='{dataset}'")
        session_id = get_session_id(request)
        logger.info(f"Session ID: {session_id}")
        audio_path = resolve_file(dataset, file_path, session_id)
        logger.info(f"Resolved audio path: {audio_path}")
    except ValueError as e:
        # Unknown dataset
        logger.error(f"ValueError in serve_dataset_file: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError in serve_dataset_file: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    try:
        media_type = media_type_for(audio_path)
    except ValueError as e:
        # Unsupported media type
        raise HTTPException(status_code=415, detail=str(e))

    safe_name = audio_path.name
    file_size = audio_path.stat().st_size
    
    # Handle OPTIONS request for CORS preflight
    if request.method == "OPTIONS":
        return JSONResponse(
            content="",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                "Access-Control-Allow-Headers": "Range, Accept-Encoding, Origin, X-Requested-With, Content-Type, Accept, Authorization",
                "Access-Control-Allow-Credentials": "true",
            }
        )
    
    # Return audio file with Starlette FileResponse for non-blocking async streaming and native Range support
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "Range, Accept-Encoding, Origin, X-Requested-With, Content-Type, Accept, Authorization",
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Expose-Headers": "Content-Length, Content-Range, Accept-Ranges",
        "Content-Disposition": f"inline; filename=\"{safe_name}\"",
        "X-Content-Type-Options": "nosniff",
    }

    return FileResponse(
        path=audio_path,
        media_type=media_type,
        headers=headers,
    )
