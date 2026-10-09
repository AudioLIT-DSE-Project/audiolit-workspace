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
from app.infrastructure import model_labels

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
    #: Class names in index order; empty unless the checkpoint has a
    #: classification head. The user's names when they have entered some.
    labels: List[str] = []
    #: The checkpoint itself only ships LABEL_0 .. LABEL_n, so the names have
    #: to be entered by hand before predictions mean anything.
    labels_are_placeholders: bool = False


class CancelModelRequest(BaseModel):
    model_id: str


class SetModelLabelsRequest(BaseModel):
    model_id: str
    revision: str = "main"
    #: Class names in index order; an empty list removes the stored names.
    labels: List[str]


class ModelLabelsResponse(BaseModel):
    model_id: str
    labels: List[str]
    labels_are_placeholders: bool


def _checkpoint_labels(model) -> List[str]:
    """The checkpoint's own class names in index order, [] for a non-classifier.

    Gated on the declared architecture because transformers gives *every*
    config a default two-entry id2label, including Whisper and CTC models.
    """
    config = getattr(model, "config", None)
    architectures = getattr(config, "architectures", None)
    id2label = getattr(config, "id2label", None)
    if not isinstance(architectures, (list, tuple)) or not isinstance(id2label, dict):
        return []
    if not any("Classification" in str(name) for name in architectures):
        return []
    return [str(id2label[key]) for key in sorted(id2label, key=int)]


def _load(model_id: str, revision: str):
    try:
        return registry.get(model_id, revision=revision)
    except ModelRegistryError as e:
        status_code = _ERROR_STATUS.get(e.code, 400)
        raise HTTPException(status_code=status_code, detail={"code": e.code, "message": str(e)})
    except Exception as e:
        logger.error("Unexpected error resolving model %s: %s", model_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to resolve model: {e}")


@router.post("/models/resolve", response_model=ResolveModelResponse)
def resolve_model(request: ResolveModelRequest) -> ResolveModelResponse:
    """Resolve, safety-check, and load a Hugging Face model through the registry."""
    loaded = _load(request.model_id, request.revision)
    checkpoint_labels = _checkpoint_labels(loaded.model)
    override = model_labels.get_label_override(loaded.model_id)
    if override and len(override) != len(checkpoint_labels):
        override = None

    return ResolveModelResponse(
        model_id=loaded.model_id,
        revision=loaded.revision,
        family=loaded.family,
        weights_sha256=loaded.weights_sha256,
        available_layers=loaded.available_layers,
        labels=override or checkpoint_labels,
        labels_are_placeholders=model_labels.are_placeholders(checkpoint_labels),
    )


@router.put("/models/labels", response_model=ModelLabelsResponse)
def set_model_labels(request: SetModelLabelsRequest) -> ModelLabelsResponse:
    """Store class names for a custom checkpoint that was published without them."""
    loaded = _load(request.model_id, request.revision)
    checkpoint_labels = _checkpoint_labels(loaded.model)
    if not checkpoint_labels:
        raise HTTPException(status_code=422, detail=f"'{loaded.model_id}' has no classification head to name.")

    placeholders = model_labels.are_placeholders(checkpoint_labels)
    if not request.labels:
        model_labels.set_label_override(loaded.model_id, None)
        return ModelLabelsResponse(
            model_id=loaded.model_id, labels=checkpoint_labels, labels_are_placeholders=placeholders
        )

    try:
        labels = model_labels.normalise_labels(request.labels, len(checkpoint_labels))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    model_labels.set_label_override(loaded.model_id, labels)
    return ModelLabelsResponse(model_id=loaded.model_id, labels=labels, labels_are_placeholders=placeholders)


@router.post("/models/cancel")
def cancel_model_resolution(request: CancelModelRequest):
    """Abort an active custom model resolution and purge temporary resources/memory."""
    success = registry.cancel_download(request.model_id)
    return {"status": "ok", "message": f"Resolution for '{request.model_id}' cancelled.", "cleaned": success}


@router.get("/models/active")
def get_active_downloads():
    """List ongoing model resolution & weight download tasks."""
    return {"active_downloads": registry.get_active_downloads()}

