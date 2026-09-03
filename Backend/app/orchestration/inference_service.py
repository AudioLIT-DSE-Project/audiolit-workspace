import asyncio
import hashlib
import functools
import inspect
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import HTTPException

from app.domain.model_loader_service import (
    transcribe_whisper_base,
    wave2vec,
    predict_deepfake,
    extract_whisper_embeddings,
    extract_wav2vec2_embeddings,
    extract_add_embeddings,
)
from app.infrastructure.dataset_service import resolve_file
from app.infrastructure.redis import get_result, cache_result
from app.infrastructure import cache_keys as ck

logger = logging.getLogger(__name__)


def predict_melody_machine(audio_path: str):
    return predict_deepfake(audio_path, model_key="melody-machine")


def predict_wav2vec2_add(audio_path: str):
    return predict_deepfake(audio_path, model_key="wav2vec2-add")


MODEL_FUNCTIONS = {
    "whisper-base": transcribe_whisper_base,
    "wav2vec2": wave2vec,
    "melody-machine": predict_melody_machine,
    "wav2vec2-add": predict_wav2vec2_add,
}

ADD_MODEL_KEYS = ("melody-machine", "wav2vec2-add")


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


def _embedding_family(model: str) -> str:
    """Which extractor owns ``model``: ``add`` / ``wav2vec2`` / ``whisper``.

    Routing used to be three substring tests in an if/elif chain, which got two
    cases wrong. ``wav2vec2-add`` contains "wav2vec", so the deepfake models were
    handed to the emotion extractor - same 1024 dims, no error, wrong latent
    space. And a custom SER checkpoint whose name says neither "whisper" nor
    "wav2vec" (``myorg/custom-ser``) fell through to the *Whisper* extractor,
    which is how a custom emotion model ended up plotted in Whisper's space.

    The registry knows each model's family, so ask it. Substrings stay as the
    fast path for the built-in keys and as the fallback when the model is not
    registered, since guessing beats failing the request outright.
    """
    if model in ADD_MODEL_KEYS:
        return "add"

    lowered = model.lower()
    if "whisper" in lowered:
        return "whisper"
    if "wav2vec" in lowered:
        return "wav2vec2"

    try:
        from app.domain.model_registry_service import registry
        family = getattr(registry.get(model), "family", None)
        if family in ("whisper", "wav2vec2"):
            return family
    except Exception:
        logger.warning("Could not resolve family for %s; defaulting to whisper", model)
    return "whisper"


async def run_inference(
    model: str,
    file_path: Optional[str] = None,
    dataset: Optional[str] = None,
    dataset_file: Optional[str] = None,
    session_id: Optional[str] = None,
    force_refresh: bool = False,
):
    """Run a model prediction, resolving/caching the audio and result.

    Shared by the /inferences/run route and the /upload route (previously
    upload.py imported this directly from the inferences route module —
    LIT-227 moves it here so neither route imports the other).

    ``force_refresh`` skips the cache lookup and recomputes, then overwrites
    the cache entry with the fresh result — backs the per-row "Regenerate"
    button in AudioDataTable.tsx, which would otherwise just get the same
    cached prediction handed back unchanged (a deterministic model on an
    unchanged file always predicts the same thing; the point of that button
    is forcing a real, visible recompute, not silently no-op-ing).
    """
    logger.info(
        "inference_service.run_inference model=%s file_path=%s dataset=%s dataset_file=%s session_id=%s",
        model,
        file_path,
        dataset,
        dataset_file,
        session_id,
    )

    # Whisper is dispatched by family rather than by name so that any custom
    # checkpoint runs as itself. Binding the selected id here is what stops
    # `transcribe_whisper_base` from quietly transcribing with whisper-base
    # and caching the result under the requested model's key (FR1, FR4).
    func = MODEL_FUNCTIONS.get(model)
    if func is transcribe_whisper_base or (func is None and "whisper" in model.lower()):
        func = functools.partial(transcribe_whisper_base, model=model)
    elif not func:
        # Check if resolved in ModelRegistry (LIT-231, FR1)
        from app.domain.model_registry_service import registry
        try:
            loaded_model = registry.get(model)
            if loaded_model.family == "whisper":
                func = functools.partial(transcribe_whisper_base, model=model)
            elif loaded_model.family == "wav2vec2":
                # Bind the selection, exactly as the whisper branch does. Left
                # unbound, a custom SER checkpoint ran the project default and
                # cached the result under the custom model's own key - a wrong
                # prediction wearing the right name.
                func = functools.partial(wave2vec, model_id=model)
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

    cached_result = None if force_refresh else await get_result(model, cache_key)
    if cached_result is not None:
        logger.info(f"Returning cached result for {resolved_path}")
        prediction = ck.unwrap_prediction(cached_result)
        # An ASR-with-attention payload written into the transcript family by
        # an older dataset warmup. Callers of this family expect the bare
        # transcript, so flatten rather than hand back a dict they cannot use.
        if isinstance(prediction, dict) and "text" in prediction and "attention" in prediction:
            prediction = ck.as_transcript(prediction)
        return prediction

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
        family = _embedding_family(model)

        if family == "add":
            embedding = await asyncio.to_thread(extract_add_embeddings, str(resolved_path), model)
        elif family == "wav2vec2":
            # Pass the selection through: this call dropped `model`, so a custom
            # SER checkpoint got the default model's embeddings cached under the
            # custom model's key (FR1, FR4) - the same substitution already fixed
            # on the prediction path. "wav2vec2" is the family alias rather than
            # a hub id, so it stays None and resolves to the default checkpoint.
            ser_model_id = None if model == "wav2vec2" else model
            embedding = await asyncio.to_thread(
                extract_wav2vec2_embeddings, str(resolved_path), ser_model_id
            )
        else:
            embedding = await asyncio.to_thread(extract_whisper_embeddings, str(resolved_path), model)

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
