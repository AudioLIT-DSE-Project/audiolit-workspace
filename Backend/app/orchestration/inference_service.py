import asyncio
import hashlib
import inspect
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import HTTPException

from app.domain.model_loader_service import (
    transcribe_whisper_base,
    transcribe_whisper_large,
    wave2vec,
    extract_whisper_embeddings,
    extract_wav2vec2_embeddings,
)
from app.infrastructure.dataset_service import resolve_file
from app.infrastructure.redis import get_result, cache_result

logger = logging.getLogger(__name__)

MODEL_FUNCTIONS = {
    "whisper-base": transcribe_whisper_base,
    "whisper-large": transcribe_whisper_large,
    "wav2vec2": wave2vec,
}


def _resolve_audio_path(
    file_path: Optional[str],
    dataset: Optional[str],
    dataset_file: Optional[str],
    session_id: Optional[str],
) -> Path:
    if file_path:
        return Path(file_path)
    if dataset and dataset_file:
        try:
            return resolve_file(dataset, dataset_file, session_id)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))
    raise HTTPException(
        status_code=400,
        detail="Missing audio reference. Provide either 'file_path' or 'dataset' + 'dataset_file'.",
    )


async def run_inference(
    model: str,
    file_path: Optional[str] = None,
    dataset: Optional[str] = None,
    dataset_file: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Run a model prediction, resolving/caching the audio and result.

    Shared by the /inferences/run route and the /upload route (previously
    upload.py imported this directly from the inferences route module —
    LIT-227 moves it here so neither route imports the other).
    """
    logger.info(
        "inference_service.run_inference model=%s file_path=%s dataset=%s dataset_file=%s session_id=%s",
        model,
        file_path,
        dataset,
        dataset_file,
        session_id,
    )

    func = MODEL_FUNCTIONS.get(model)
    if not func:
        # Check if resolved in ModelRegistry (LIT-231, FR1)
        from app.domain.model_registry_service import registry
        try:
            loaded_model = registry.get(model)
            if loaded_model.family == "whisper":
                func = transcribe_whisper_base
            elif loaded_model.family == "wav2vec2":
                func = wave2vec
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported model family: {loaded_model.family}")
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=400, detail=f"Invalid or unresolved model: {model} ({e})")

    resolved_path = _resolve_audio_path(file_path, dataset, dataset_file, session_id)
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {resolved_path}")

    file_content_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
    cache_key = f"v2_{model}_{file_content_hash}"

    cached_result = await get_result(model, cache_key)
    if cached_result is not None:
        logger.info(f"Returning cached result for {resolved_path}")
        return cached_result.get("prediction", cached_result)

    if inspect.iscoroutinefunction(func):
        prediction = await func(str(resolved_path))
    else:
        prediction = await asyncio.to_thread(func, str(resolved_path))

    await cache_result(model, cache_key, {"prediction": prediction}, ttl=6 * 60 * 60)
    logger.info(f"Cached prediction for {resolved_path}")

    return prediction


async def extract_single_embedding(
    model: str,
    file_path: Optional[str] = None,
    dataset: Optional[str] = None,
    dataset_file: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    """Extract (and cache) embeddings for a single audio file.

    Shared by the /inferences/embeddings/single route and the /upload route.
    Previously upload.py called the FastAPI endpoint function directly with
    a plain dict as its sole argument, which silently bound that dict to the
    endpoint's `http_request: Request` parameter and left `request` at its
    `Body(...)` sentinel default — embedding generation on upload was
    effectively always failing (swallowed by upload.py's broad except). This
    plain, typed function fixes that by construction.
    """
    resolved_path = _resolve_audio_path(file_path, dataset, dataset_file, session_id)
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {resolved_path}")

    file_content_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
    cache_key = f"{model}_embeddings_{file_content_hash}"

    cached_embeddings = await get_result(model, cache_key)
    if cached_embeddings is not None:
        embedding = cached_embeddings.get("embedding")
        logger.info(f"Using cached embeddings for {resolved_path}")
    else:
        if model.startswith("whisper"):
            model_size = "base" if "base" in model else "large"
            embedding = await asyncio.to_thread(extract_whisper_embeddings, str(resolved_path), model_size)
        elif model == "wav2vec2":
            embedding = await asyncio.to_thread(extract_wav2vec2_embeddings, str(resolved_path))
        else:
            raise HTTPException(status_code=400, detail=f"Embedding extraction not supported for model: {model}")

        await cache_result(model, cache_key, {"embedding": embedding.tolist()}, ttl=24 * 60 * 60)
        logger.info(f"Cached embeddings for {resolved_path}")

    if isinstance(embedding, list):
        embedding = np.array(embedding)

    return {
        "model": model,
        "file_path": str(resolved_path),
        "embedding": embedding.tolist(),
        "embedding_dim": len(embedding),
    }
