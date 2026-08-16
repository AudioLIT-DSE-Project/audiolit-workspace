"""Model Registry exposure (LIT-231, FR1, SRS Use Case 5).

The gateway only validates and delegates - `model_registry_service.registry`
already does all the real work (Hub resolution, safetensors validation,
version pinning, hook-registration/layer discovery). This route is a thin
wrapper so the frontend's "add a custom Hugging Face model" flow and the
nav bar's hook-registration status have something to call.
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.domain.model_registry_service import ModelRegistryError, registry

router = APIRouter()
logger = logging.getLogger("audiolit.api.models")

# SRS Use Case 5: "An unsafe file, an unsupported model, or a download problem
# each produce a distinct, clear message" - map each typed error code to the
# HTTP status that best matches its cause instead of collapsing all three into
# one generic error.
_ERROR_STATUS = {
    "UNSUPPORTED_ARCHITECTURE": 422,
    "UNSAFE_ARTIFACT": 422,
    "HUB_UNAVAILABLE": 502,
}


class ResolveModelRequest(BaseModel):
    model_id: str
    revision: str = "main"


class ResolveModelResponse(BaseModel):
    model_id: str
    revision: str
    family: str
    weights_sha256: str
    available_layers: List[str]


@router.post("/models/resolve", response_model=ResolveModelResponse)
def resolve_model(request: ResolveModelRequest) -> ResolveModelResponse:
    """Resolve, safety-check, and load a Hugging Face model through the registry."""
    try:
        loaded = registry.get(request.model_id, revision=request.revision)
    except ModelRegistryError as e:
        status_code = _ERROR_STATUS.get(e.code, 400)
        raise HTTPException(status_code=status_code, detail={"code": e.code, "message": str(e)})
    except Exception as e:
        logger.error("Unexpected error resolving model %s: %s", request.model_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to resolve model: {e}")

    return ResolveModelResponse(
        model_id=loaded.model_id,
        revision=loaded.revision,
        family=loaded.family,
        weights_sha256=loaded.weights_sha256,
        available_layers=loaded.available_layers,
    )
