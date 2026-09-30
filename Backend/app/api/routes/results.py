"""FastAPI routes for caching and retrieving ML results & forensic feature maps (LIT-152, FR3/FR7)."""
import hashlib
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

from app.core.redis import cache_manager

router = APIRouter()


class ForensicFeatureMap(BaseModel):
    """Pydantic validation schema for deepfake forensic feature maps & detection attributes (LIT-152)."""
    model_id: str = Field(..., description="Model identifier used for deepfake detection")
    audio_hash: str = Field(..., description="SHA-256 hash of the audio payload")
    predicted_label: str = Field(..., description="Classification label: bona-fide or spoof")
    synthetic_probability: float = Field(..., ge=0.0, le=1.0, description="Synthetic audio likelihood score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Top-class prediction confidence score")
    probabilities: Dict[str, float] = Field(..., description="Full class probability breakdown")
    verification_flags: Dict[str, bool] = Field(
        default_factory=lambda: {"authentic": True, "spoof_detected": False, "verified": True},
        description="Integrity and fraud verification flags"
    )
    timeline: Optional[List[Dict[str, Any]]] = Field(None, description="Confidence/probability timeline across frames")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional model and dataset attributes")


class ResultsResponse(BaseModel):
    cached: bool
    payload: Optional[Dict[str, Any]] = None
    forensic_map: Optional[ForensicFeatureMap] = None


@router.get("/results/{model}/{h}")
async def results_get(model: str, h: str):
    """Retrieve cached ML results and standardized forensic feature maps (LIT-152)."""
    key_str = f"audiolit:rest:{model}:{h}"
    redis_key = hashlib.sha256(key_str.encode()).hexdigest()
    payload = cache_manager.get(redis_key)

    if payload is None:
        return {"cached": False, "payload": None}

    forensic_map = None
    if isinstance(payload, dict) and "synthetic_probability" in payload and "predicted_label" in payload:
        try:
            predicted = payload.get("predicted_label", "bona-fide")
            forensic_map = ForensicFeatureMap(
                model_id=model,
                audio_hash=h,
                predicted_label=predicted,
                synthetic_probability=float(payload.get("synthetic_probability", 0.0)),
                confidence=float(payload.get("confidence", 0.0)),
                probabilities=payload.get("probabilities", {}),
                verification_flags=payload.get("verification_flags", {
                    "authentic": predicted == "bona-fide",
                    "spoof_detected": predicted == "spoof",
                    "verified": True,
                }),
                timeline=payload.get("timeline"),
                metadata=payload.get("metadata", {}),
            ).model_dump(by_alias=True)
        except Exception:
            forensic_map = None

    response_data = {"cached": True, "payload": payload}
    if forensic_map is not None:
        response_data["forensic_map"] = forensic_map
    return response_data


@router.post("/results/{model}/{h}")
async def results_put(model: str, h: str, payload: dict = Body(...)):
    """Store ML result in cache using msgpack/lz4 DAO and format forensic attributes (LIT-152)."""
    key_str = f"audiolit:rest:{model}:{h}"
    redis_key = hashlib.sha256(key_str.encode()).hexdigest()

    if isinstance(payload, dict) and "synthetic_probability" in payload and "verification_flags" not in payload:
        predicted = payload.get("predicted_label", "bona-fide")
        payload["verification_flags"] = {
            "authentic": predicted == "bona-fide",
            "spoof_detected": predicted == "spoof",
            "verified": True,
        }

    cache_manager.set(redis_key, payload)
    return {"ok": True, "model": model, "audio_hash": h}

