"""
Infrastructure Layer: Redis Multi-Dimensional Tensor Serialization (SAD §5.1 / §5.2).
Provides SHA-256 Cache-by-Hash retrieval and msgpack/NumPy tensor serialization.
"""
import os
import json
import time
import hashlib
import logging
import functools
from typing import Any, Callable, Dict, Optional
import redis
import msgpack
import numpy as np
import hashlib
import io
from fastapi import UploadFile, File, HTTPException, status

logger = logging.getLogger("audiolit.cache")

# Redis connection configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
CACHE_TTL = int(os.getenv("AUDIOLIT_CACHE_TTL", 60 * 60 * 24))  # 24 hours
def generate_audio_hash_from_bytes(audio_bytes: bytes) -> str:
    """
    Processing utility that reads arriving audio byte streams in chunks 
    to calculate a deterministic SHA-256 checksum string.
    """
    sha256_hash = hashlib.sha256()
    buffer = io.BytesIO(audio_bytes)
    
    # Process in 8KB chunks to handle large files without OOM
    chunk_size = 8192
    while True:
        chunk = buffer.read(chunk_size)
        if not chunk:
            break
        sha256_hash.update(chunk)
        
    return sha256_hash.hexdigest()


async def get_audio_hash_dependency(audio_file: UploadFile = File(...)) -> str:
    """
    FastAPI Dependency Interceptor: Computes the file hash before the payload 
    is passed to the FastAPI route handler or RQ queue.
    """
    sha256_hash = hashlib.sha256()
    chunk_size = 8192
    
    # Read arriving audio byte streams in chunks
    while True:
        chunk = await audio_file.read(chunk_size)
        if not chunk:
            break
        sha256_hash.update(chunk)
        
    # Reset file pointer to the beginning so subsequent route handlers can read the file
    await audio_file.seek(0)
    
    audio_hash = sha256_hash.hexdigest()
    logger.info(f"Computed SHA-256 audio hash: {audio_hash} for file: {audio_file.filename}")
    
    return audio_hash
class RedisCacheManager:
    """
    Data Access Object (DAO) for serializing, compressing, storing, and fetching
    complex multi-dimensional attribution matrices and attention tensors.
    """
    def __init__(self):
        self.client = redis.Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            db=REDIS_DB, 
            decode_responses=False,
            socket_connect_timeout=2,
            socket_timeout=5
        )
        
    def _generate_key(self, audio_ref: str, model_id: str, task: str, params: Dict) -> str:
        """Generates a deterministic SHA-256 cache key (SRS FR4)."""
        param_str = json.dumps(params, sort_keys=True)
        hash_input = f"{audio_ref}:{model_id}:{task}:{param_str}"
        sha256_hash = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
        return f"audiolit:tensor:{sha256_hash}"

    def _encode_numpy(self, obj: Any) -> Any:
        """Extensible encoder for msgpack to handle NumPy arrays natively."""
        if isinstance(obj, np.ndarray):
            return {
                b'__np__': True,
                b'dtype': obj.dtype.str.encode(),
                b'shape': obj.shape,
                b'data': obj.tobytes()
            }
        return obj

    def _decode_numpy(self, obj: Any) -> Any:
        """Extensible decoder for msgpack to reconstruct NumPy arrays."""
        if isinstance(obj, dict) and obj.get(b'__np__'):
            return np.frombuffer(obj[b'data'], dtype=obj[b'dtype'].decode()).reshape(obj[b'shape'])
        return obj

    def get(self, key: str) -> Optional[Any]:
        """Fetch and deserialize tensors from Redis."""
        packed_data = self.client.get(key)
        if packed_data is None:
            return None
        
        return msgpack.unpackb(packed_data, object_hook=self._decode_numpy, raw=False)

    def set(self, key: str, value: Any, ttl: int = CACHE_TTL) -> None:
        """Serialize and store tensors in Redis with LRU eviction TTL."""
        packed_data = msgpack.packb(value, default=self._encode_numpy, use_bin_type=True)
        self.client.set(key, packed_data, ex=ttl)

# Singleton instance
cache_manager = RedisCacheManager()

def cached_inference(audio_arg: str, model_arg: str, task_name: str, params_arg: str = "params"):
    """
    Decorator to intercept model inference paths (SAD §5.1).
    If a matching file hash is found, returns cached results instantly and 
    bypasses the heavy PyTorch computational graph completely.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract arguments based on parameter names
            audio_ref = kwargs.get(audio_arg) or args[0] if args else None
            model_id = kwargs.get(model_arg) or args[1] if len(args) > 1 else "default_model"
            params = kwargs.get(params_arg, {})
            
            cache_key = cache_manager._generate_key(audio_ref, model_id, task_name, params)
            
            # --- Cache Lookup ---
            start_time = time.time()
            cached_result = cache_manager.get(cache_key)
            fetch_duration_ms = (time.time() - start_time) * 1000
            
            if cached_result is not None:
                logger.info(
                    f"CACHE HIT: '{task_name}' for model '{model_id}' read from Redis in {fetch_duration_ms:.2f}ms. "
                    f"Bypassing PyTorch execution graph completely."
                )
                return cached_result
                
            # --- Cache Miss: Execute Heavy Computation ---
            logger.info(f"CACHE MISS: Executing PyTorch inference for '{task_name}'.")
            result = func(*args, **kwargs)
            
            # Store result back to Redis
            cache_manager.set(cache_key, result)
            return result
            
        return wrapper
    return decorator
