from pathlib import Path
from itertools import islice
from typing import Dict, List, Optional
import csv
import librosa
import logging
from . import dataset_ingestion
from .custom_dataset_service import (
    get_custom_dataset_manager,
    is_custom_dataset,
    parse_custom_dataset_name
)
from .settings import settings

logger = logging.getLogger(__name__)


# Resolve paths relative to repo structure: Backend/data/
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# CSV metadata files per dataset. Only common-voice is handled here directly -
# every other corpus (including ravdess, whose path used to be hardcoded to a
# stale "ravdess_subset" directory that no longer exists) falls back to
# dataset_ingestion.py's CORPUS_REGISTRY/DatasetLoader system below (LIT-235).
# That system was built by LIT-123/141/142/181/208 but was never actually
# reachable from these live routes until now.
DATASET_PATHS: Dict[str, Path] = {
    "common-voice": DATA_DIR / "common_voice_valid_dev" / "common_voice_valid_data_metadata.csv",
    "cv-valid-dev": DATA_DIR / "common_voice_valid_dev" / "common_voice_valid_data_metadata.csv",
}

# Base directories for dataset audio files
DATASET_BASE_DIRS: Dict[str, Path] = {
    "common-voice": DATA_DIR / "common_voice_valid_dev",
    "cv-valid-dev": DATA_DIR / "common_voice_valid_dev",
}


def _registry_metadata_rows(
    dataset: str, *, limit: Optional[int] = None, offset: int = 0
) -> List[Dict[str, str]]:
    """Metadata rows for any dataset_ingestion-registered corpus (LIT-235; LIT-237).

    Converts each SampleMetadata into the same dict-row shape the frontend
    already reads generically (checks "path"/"filepath"/"file"/"filename" in
    that order, per AudioDatasetPanel.tsx) so no frontend contract changes.

    Two FR2 fixes on top of the original LIT-235 version:

    - FR2.1: streams through ``validated_stream(deep=False)`` so rows whose
      audio file doesn't exist on disk are excluded rather than shown and
      then failing on playback. ``deep=False`` is an existence check only
      (no decode) — this runs on every row of a listing request, so it has
      to stay cheap; the full decode+silence check lives in the batch/eval
      path (``accent_bias_profiler.load_accent_cohorts``) instead.
    - FR2.2: capped to ``settings.DATASET_METADATA_ROW_CAP`` (or an explicit
      ``limit``) via a windowed ``islice`` instead of materializing the
      entire corpus into a list, which is what this function did before and
      is exactly what the "nothing materialized whole" acceptance line in
      the SRS forbids.
    """
    try:
        loader = dataset_ingestion.get_loader(dataset)
    except (ValueError, NotImplementedError) as e:
        raise ValueError(str(e))

    cap = limit if limit is not None else settings.DATASET_METADATA_ROW_CAP
    window = islice(loader.validated_stream(deep=False), offset, offset + cap)

    rows: List[Dict[str, str]] = []
    for sample in window:
        row: Dict[str, str] = {
            "filename": sample.audio_path.name,
            "label": sample.label or "",
        }
        if sample.speaker_id:
            row["speaker_id"] = sample.speaker_id
        if sample.accent:
            row["accent"] = sample.accent
        if sample.language:
            row["language"] = sample.language
        if sample.license:
            row["license"] = sample.license
        for key, value in sample.demographic.items():
            row.setdefault(key, value)
        duration = calculate_audio_duration(sample.audio_path)
        if duration:
            row["duration"] = str(duration)
        rows.append(row)
    return rows


def _registry_resolve_file(dataset: str, file_path: str) -> Path:
    """Resolve one file's path for a dataset_ingestion-registered corpus.

    Linear scan over the loader's catalog matching by filename - simple and
    correct for the demo-scale corpora this project uses; revisit with an
    index if a corpus grows large enough for this to matter.
    """
    try:
        loader = dataset_ingestion.get_loader(dataset)
    except (ValueError, NotImplementedError) as e:
        raise ValueError(str(e))

    target = Path(file_path).name
    for sample in loader.iter_metadata():
        if sample.audio_path.name == target:
            if not sample.audio_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {target}")
            return sample.audio_path
    raise FileNotFoundError(f"Dataset file not found: {target}")


import soundfile as sf

def calculate_audio_duration(audio_path: Path) -> float:
    """Calculate duration of audio file in seconds using fast header parsing"""
    try:
        info = sf.info(str(audio_path))
        return round(info.duration, 2)
    except Exception:
        try:
            duration = librosa.get_duration(path=str(audio_path))
            return round(duration, 2)
        except Exception as e:
            logger.warning(f"Could not calculate duration for {audio_path}: {e}")
            return 0.0

def load_metadata(
    dataset: str,
    session_id: Optional[str] = None,
    *,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Dict[str, str]]:
    """Load metadata for both global and custom datasets"""

    # Handle custom datasets
    if is_custom_dataset(dataset):
        if not session_id:
            raise ValueError("session_id is required for custom datasets")

        session_id_from_name, dataset_name = parse_custom_dataset_name(dataset)
        logger.info(f"Custom dataset metadata: session_id_from_name='{session_id_from_name}', current_session_id='{session_id}'")
        if session_id_from_name != session_id:
            logger.warning(f"Session ID mismatch in metadata: dataset has '{session_id_from_name}' but request has '{session_id}'")
            # Use the dataset's session ID instead
            manager = get_custom_dataset_manager(session_id_from_name)
            return manager.get_dataset_files_as_csv_format(dataset_name)

        manager = get_custom_dataset_manager(session_id)
        return manager.get_dataset_files_as_csv_format(dataset_name)

    # Handle global datasets (existing logic)
    ds = dataset.lower()
    if ds not in DATASET_PATHS:
        return _registry_metadata_rows(ds, limit=limit, offset=offset)

    csv_path = DATASET_PATHS[ds]
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found for: {dataset}")

    rows: List[Dict[str, str]] = []
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # normalize keys to lowercase; strip whitespace
                normalized = {str(k).strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                rows.append(normalized)
    except Exception:
        # Re-raise to let the route map to a 500
        raise

    return rows


def resolve_file(dataset: str, file_path: str, session_id: Optional[str] = None) -> Path:
    """Resolve file path for both global and custom datasets"""
    
    # Handle custom datasets
    if is_custom_dataset(dataset):
        if not session_id:
            raise ValueError("session_id is required for custom datasets")
        
        session_id_from_name, dataset_name = parse_custom_dataset_name(dataset)
        logger.info(f"Custom dataset: session_id_from_name='{session_id_from_name}', current_session_id='{session_id}'")
        if session_id_from_name != session_id:
            logger.warning(f"Session ID mismatch: dataset has '{session_id_from_name}' but request has '{session_id}'")
            # For debugging, let's check if the file exists with the dataset's session ID
            manager = get_custom_dataset_manager(session_id_from_name)
            try:
                return manager.resolve_file_path(dataset_name, file_path)
            except Exception as e:
                logger.error(f"Could not resolve file with dataset session ID: {e}")
                raise ValueError(f"Session ID mismatch for custom dataset. Dataset session: {session_id_from_name}, Request session: {session_id}")
        
        manager = get_custom_dataset_manager(session_id)
        return manager.resolve_file_path(dataset_name, file_path)
    
    # Handle global datasets (existing logic)
    ds = dataset.lower()
    if ds not in DATASET_BASE_DIRS:
        return _registry_resolve_file(ds, file_path)

    base_dir = DATASET_BASE_DIRS[ds]
    safe_name = Path(file_path).name
    audio_path = base_dir / safe_name
    if not audio_path.exists() or not audio_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {safe_name}")

    return audio_path


def resolve_audio_reference(
    file_path: Optional[str] = None,
    dataset: Optional[str] = None,
    dataset_file: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Path:
    """Resolve an audio reference into an absolute Path on disk.

    Supports:
    1. explicit dataset + dataset_file
    2. relative file_path with embedded dataset directory (e.g. 'cv-valid-dev/sample-000775.mp3')
    3. direct file_path (absolute or relative to CWD)
    """
    if dataset and dataset_file:
        try:
            return resolve_file(dataset, dataset_file, session_id)
        except (FileNotFoundError, ValueError):
            pass

    if file_path:
        p = Path(file_path)
        if p.is_absolute() and p.exists():
            return p

        # Check relative to DATA_DIR
        data_resolved = DATA_DIR / p
        if data_resolved.exists():
            return data_resolved.resolve()

        # If file_path contains dataset folder prefix (e.g. cv-valid-dev/sample-000775.mp3)
        parts = p.parts
        if len(parts) > 1:
            try:
                return resolve_file(parts[0], parts[-1], session_id)
            except Exception:
                pass

        if p.exists():
            return p.resolve()

        # Check directly under DATA_DIR (e.g. DATA_DIR / p.name)
        data_name_resolved = DATA_DIR / p.name
        if data_name_resolved.exists():
            return data_name_resolved.resolve()

        # Fallback try resolve_file with dataset if dataset was provided
        if dataset:
            try:
                return resolve_file(dataset, p.name, session_id)
            except Exception:
                pass

        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if dataset and dataset_file:
        return resolve_file(dataset, dataset_file, session_id)

    raise ValueError("Missing audio reference. Provide either 'file_path' or 'dataset' + 'dataset_file'.")


def media_type_for(audio_path: Path) -> str:
    ext = audio_path.suffix.lower()
    media_type_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
    }
    media_type = media_type_map.get(ext)
    if media_type is None:
        raise ValueError(f"Unsupported media type for extension: {ext}")
    return media_type
