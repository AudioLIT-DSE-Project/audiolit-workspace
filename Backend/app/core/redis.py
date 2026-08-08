"""
Infrastructure Layer: Deterministic Cache-by-Hash Architecture (SAD §5.1 / §5.2).
Provides SHA-256 content-addressed retrieval, msgpack/NumPy serialization, 
lz4 compression, and deduplication locks.
"""
import os
import json
import time
import hashlib
import logging
import functools
import io
from typing import Any, Callable, Dict, Optional
import redis
import msgpack
import numpy as np
import lz4.frame
from fastapi import UploadFile, File

logger = logging.getLogger("audiolit.cache")

# Redis connection configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
CACHE_TTL = int(os.getenv("AUDIOLIT_CACHE_TTL", 60 * 60 * 24))  # 24 hours
CACHE_SCHEMA_VERSION = 1

def generate_audio_hash_from_bytes(audio_bytes: bytes) -> str:
    """
    Processing utility that reads arriving audio byte streams in chunks 
    to calculate a deterministic SHA-256 checksum string.
    """
    sha256_hash = hashlib.sha256()
    buffer = io.BytesIO(audio_bytes)
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
    while True:
        chunk = await audio_file.read(chunk_size)
        if not chunk:
            break
        sha256_hash.update(chunk)
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
        
    def _generate_key(self, audio_bytes: bytes, model_id: str, task: str, params: Dict) -> str:
        """
        Computes a deterministic cache key:
        audiolit: + SHA-256(audio_bytes ‖ model_id ‖ task ‖ canonical_params_json)
        """
        canonical_params = params.copy() if params else {}
        canonical_params["_cache_schema_version"] = CACHE_SCHEMA_VERSION
        
        # Sorted keys, no whitespace
        params_json = json.dumps(canonical_params, sort_keys=True, separators=(',', ':'))
        
        sha256_hash = hashlib.sha256()
        sha256_hash.update(audio_bytes)
        sha256_hash.update(model_id.encode('utf-8'))
        sha256_hash.update(task.encode('utf-8'))
        sha256_hash.update(params_json.encode('utf-8'))
        
        return f"audiolit:tensor:{sha256_hash.hexdigest()}"

    def _encode_numpy(self, obj: Any) -> Any:
        """Extensible encoder for msgpack to handle NumPy arrays natively."""
        if isinstance(obj, np.ndarray):
            return {
                b'__np__': True,
                b'dtype': obj.dtype.str.encode(),
                b'shape': list(obj.shape),
                b'data': obj.tobytes()
            }
        return obj

    def _decode_numpy(self, obj: Any) -> Any:
        """Extensible decoder for msgpack to reconstruct NumPy arrays."""
        if isinstance(obj, dict) and obj.get(b'__np__'):
            dtype = obj[b'dtype'].decode()
            shape = tuple(obj[b'shape'])
            return np.frombuffer(obj[b'data'], dtype=dtype).reshape(shape)
        return obj

    def _serialize(self, value: Any) -> bytes:
        """Serialize with msgpack, apply lz4 compression if > 1MB."""
        packed_data = msgpack.packb(value, default=self._encode_numpy, use_bin_type=True)
        if len(packed_data) > 1024 * 1024:
            return b'LZ4:' + lz4.frame.compress(packed_data)
        return b'RAW:' + packed_data

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize msgpack, decompressing lz4 if necessary."""
        if data.startswith(b'LZ4:'):
            unpacked = lz4.frame.decompress(data[4:])
        elif data.startswith(b'RAW:'):
            unpacked = data[4:]
        else:
            raise ValueError("Unknown serialization format in cache.")
        return msgpack.unpackb(unpacked, object_hook=self._decode_numpy, raw=False)

    def get(self, key: str) -> Optional[Any]:
        """Fetch and deserialize tensors. Treats corrupt entries as misses."""
        packed_data = self.client.get(key)
        if packed_data is None:
            return None
        try:
            return self._deserialize(packed_data)
        except Exception as e:
            logger.warning(f"Corrupt cache entry detected for key {key}: {e}. Deleting and treating as miss.")
            self.client.delete(key)
            return None

    def set(self, key: str, value: Any, ttl: int = CACHE_TTL) -> None:
        """Serialize and store tensors in Redis with LRU eviction TTL."""
        packed_data = self._serialize(value)
        self.client.set(key, packed_data, ex=ttl)

# Singleton instance
cache_manager = RedisCacheManager()

def cached_inference(audio_bytes_arg: str, model_arg: str, task_name: str, params_arg: str = "params"):
    """
    Decorator to intercept model inference paths.
    If a matching hash is found, returns cached results instantly. 
    On a miss, takes a deduplication lock and runs the cold pipeline.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            audio_bytes = kwargs.get(audio_bytes_arg) or args[0] if args else b""
            model_id = kwargs.get(model_arg) or args[1] if len(args) > 1 else "default_model"
            params = kwargs.get(params_arg, {})
            
            cache_key = cache_manager._generate_key(audio_bytes, model_id, task_name, params)
            
            # 1. Cache Lookup
            start_time = time.time()
            cached_result = cache_manager.get(cache_key)
            fetch_duration_ms = (time.time() - start_time) * 1000
            
            if cached_result is not None:
                logger.info(
                    f"CACHE HIT: '{task_name}' for model '{model_id}' read from Redis in {fetch_duration_ms:.2f}ms. "
                    f"Bypassing PyTorch execution graph completely."
                )
                return cached_result
                
            # 2. Cache Miss: Take deduplication lock (SET NX EX 60)
            lock_key = f"{cache_key}:lock"
            acquired_lock = cache_manager.client.set(lock_key, "locked", nx=True, ex=60)
            
            if not acquired_lock:
                logger.info(f"Cache miss, but another worker is computing '{task_name}'. Waiting for result...")
                time.sleep(1.0)
                cached_result = cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result
                logger.warning(f"Waited for lock but result still not available. Executing anyway.")
                
            # 3. Cold Pipeline Execution
            logger.info(f"CACHE MISS: Executing PyTorch inference for '{task_name}'.")
            result = func(*args, **kwargs)
            
            # 4. Cache and Release Lock
            cache_manager.set(cache_key, result)
            cache_manager.client.delete(lock_key)
            
            return result
            
        return wrapper
    return decorator
