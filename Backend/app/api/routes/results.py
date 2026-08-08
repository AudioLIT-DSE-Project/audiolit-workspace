"""FastAPI routes for caching and retrieving ML results (REST API)."""
import hashlib
from fastapi import APIRouter, Body
from app.core.redis import cache_manager

router = APIRouter()

@router.get("/results/{model}/{h}")
async def results_get(model: str, h: str):
    """Retrieve a result from the cache."""
    key_str = f"audiolit:rest:{model}:{h}"
    redis_key = hashlib.sha256(key_str.encode()).hexdigest()
    
    # Retrieve using the new deserialization layer
    payload = cache_manager.get(redis_key)
    
    return {"cached": payload is not None, "payload": payload}

@router.post("/results/{model}/{h}")
async def results_put(model: str, h: str, payload: dict = Body(...)):
    """Store a result in the cache using the upgraded msgpack/lz4 DAO."""
    key_str = f"audiolit:rest:{model}:{h}"
    redis_key = hashlib.sha256(key_str.encode()).hexdigest()
    
    # Use the new serialization layer to store the JSON payload
    cache_manager.set(redis_key, payload)
    
    return {"ok": True}
