import csv
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import librosa
import hashlib

logger = logging.getLogger(__name__)

# Base directory for session-based custom datasets
SESSIONS_BASE_DIR = Path("uploads/sessions")

# LIT-247 — ground-truth CSV column name candidates, tried in order (same
# flexible-header pattern as dataset_ingestion.py's ColumnMap, kept separate
# here since custom datasets aren't part of the CORPUS_REGISTRY machinery).
_GROUND_TRUTH_FILENAME_COLUMNS = ("filename", "file", "audio_file", "audio", "name")
_GROUND_TRUTH_TEXT_COLUMNS = ("ground_truth", "transcript", "label", "text", "sentence")


def normalize_filename_key(filename: str) -> str:
    """Match key for ground-truth lookups: lowercased, extension-stripped.

    Files can get renamed on upload collision (``sample.wav`` -> a second
    upload of the same name becomes ``sample_1.wav``), so matching is done
    against this normalized form of ``original_filename`` (what the user
    actually named the file), not the possibly-renamed on-disk ``filename``.
    Extension-stripped so a CSV listing ``sample`` or ``sample.wav`` both
    match a file the user uploaded as ``sample.WAV``.
    """
    return Path(filename).stem.strip().lower()


def parse_ground_truth_csv(raw: bytes) -> Dict[str, str]:
    """Parse a ground-truth CSV into ``{normalized_filename: text}``.

    Column names are resolved case-insensitively against
    ``_GROUND_TRUTH_FILENAME_COLUMNS``/``_GROUND_TRUTH_TEXT_COLUMNS`` (first
    match wins), the same tolerant-header approach
    ``dataset_ingestion.ColumnMap`` uses for the benchmark corpora, so a CSV
    exported as ``file,label`` or ``filename,transcript`` both work without
    the user needing to match an exact schema.

    Raises ``ValueError`` (not a silent empty result) when the CSV has no
    rows or neither expected column is present, so the route can return a
    typed 400 instead of the upload looking like it "succeeded" with zero
    matches.
    """
    try:
        text = raw.decode("utf-8-sig")  # -sig strips a BOM if present
    except UnicodeDecodeError as e:
        raise ValueError(f"Ground truth CSV must be UTF-8 encoded: {e}")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("Ground truth CSV has no header row")

    lower_fields = {f.strip().lower(): f for f in reader.fieldnames if f}
    filename_col = next((c for c in _GROUND_TRUTH_FILENAME_COLUMNS if c in lower_fields), None)
    text_col = next((c for c in _GROUND_TRUTH_TEXT_COLUMNS if c in lower_fields), None)
    if filename_col is None or text_col is None:
        raise ValueError(
            "Ground truth CSV must have a filename column "
            f"({', '.join(_GROUND_TRUTH_FILENAME_COLUMNS)}) and a ground-truth "
            f"column ({', '.join(_GROUND_TRUTH_TEXT_COLUMNS)}); found: "
            f"{', '.join(reader.fieldnames)}"
        )

    mapping: Dict[str, str] = {}
    for row in reader:
        raw_filename = (row.get(lower_fields[filename_col]) or "").strip()
        raw_text = (row.get(lower_fields[text_col]) or "").strip()
        if not raw_filename or not raw_text:
            continue
        mapping[normalize_filename_key(raw_filename)] = raw_text

    if not mapping:
        raise ValueError("Ground truth CSV had a valid header but no usable rows")
    return mapping

class CustomDatasetManager:
    """Manages session-based custom datasets"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_dir = SESSIONS_BASE_DIR / session_id
        self.datasets_dir = self.session_dir / "datasets"
        
        # Ensure directories exist
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def create_dataset(self, dataset_name: str) -> Dict:
        """Create a new custom dataset"""
        dataset_dir = self.datasets_dir / dataset_name
        
        if dataset_dir.exists():
            raise ValueError(f"Dataset '{dataset_name}' already exists in this session")
        
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        metadata = {
            "dataset_name": dataset_name,
            "created_at": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "files": [],
            "total_files": 0
        }
        
        metadata_file = dataset_dir / "dataset_metadata.json"
        with metadata_file.open("w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created custom dataset '{dataset_name}' for session {self.session_id}")
        return metadata
    
    def add_file_to_dataset(self, dataset_name: str, filename: str, file_data: bytes) -> Dict:
        """Add a file to an existing custom dataset"""
        dataset_dir = self.datasets_dir / dataset_name
        metadata_file = dataset_dir / "dataset_metadata.json"
        
        if not dataset_dir.exists() or not metadata_file.exists():
            raise ValueError(f"Dataset '{dataset_name}' does not exist")
        
        # Load existing metadata
        with metadata_file.open("r") as f:
            metadata = json.load(f)
        
        # Generate unique filename to avoid conflicts
        file_path = dataset_dir / filename
        counter = 1
        original_filename = filename
        name, ext = Path(filename).stem, Path(filename).suffix
        
        while file_path.exists():
            filename = f"{name}_{counter}{ext}"
            file_path = dataset_dir / filename
            counter += 1
        
        # Save the file
        with file_path.open("wb") as f:
            f.write(file_data)
        
        # Calculate audio metadata
        try:
            duration = librosa.get_duration(path=str(file_path))
            audio_data, sample_rate = librosa.load(file_path, sr=None)
        except Exception as e:
            logger.warning(f"Could not extract audio metadata for {filename}: {e}")
            duration = 0.0
            sample_rate = 0
        
        # Add file metadata
        file_metadata = {
            "filename": filename,
            "original_filename": original_filename,
            "duration": round(duration, 2),
            "sample_rate": sample_rate,
            "size": file_path.stat().st_size,
            "uploaded_at": datetime.utcnow().isoformat()
        }

        # LIT-247: if a ground-truth CSV was already uploaded for this
        # dataset (order-independent - CSV can arrive before the audio),
        # match this new file against it immediately rather than only at
        # CSV-upload time.
        gt_map = metadata.get("ground_truth_map", {})
        key = normalize_filename_key(original_filename)
        if key in gt_map:
            file_metadata["ground_truth"] = gt_map[key]

        metadata["files"].append(file_metadata)
        metadata["total_files"] = len(metadata["files"])

        # Save updated metadata
        with metadata_file.open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Added file '{filename}' to dataset '{dataset_name}' in session {self.session_id}")
        return file_metadata

    def set_ground_truth(self, dataset_name: str, mapping: Dict[str, str]) -> Dict:
        """LIT-247: store ``mapping`` (normalized filename -> ground truth) as
        this dataset's ground-truth map, replacing any previous one, and
        re-match it against every file currently in the dataset.

        Replace (not merge) semantics: the CSV just uploaded is treated as
        authoritative, so a file matched by an older CSV but absent from the
        new one loses its ``ground_truth`` rather than keeping a stale value
        silently. Returns a match summary the route can hand back to the
        caller so a typo or wrong file is visible immediately.
        """
        dataset_dir = self.datasets_dir / dataset_name
        metadata_file = dataset_dir / "dataset_metadata.json"

        if not dataset_dir.exists() or not metadata_file.exists():
            raise ValueError(f"Dataset '{dataset_name}' does not exist")

        with metadata_file.open("r") as f:
            metadata = json.load(f)

        metadata["ground_truth_map"] = mapping
        metadata["ground_truth_uploaded_at"] = datetime.utcnow().isoformat()
        summary = self._apply_ground_truth(metadata)

        with metadata_file.open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Applied ground truth to dataset '%s' in session %s: %s",
            dataset_name, self.session_id, summary,
        )
        return summary

    @staticmethod
    def _apply_ground_truth(metadata: Dict) -> Dict:
        """Set/clear ``ground_truth`` on every file per ``metadata``'s map.

        Mutates ``metadata["files"]`` in place. Split out from
        ``set_ground_truth`` so ``add_file_to_dataset`` doesn't need its own
        copy of the matching logic.
        """
        gt_map: Dict[str, str] = metadata.get("ground_truth_map", {})
        matched_keys = set()
        for file_info in metadata["files"]:
            key = normalize_filename_key(file_info["original_filename"])
            if key in gt_map:
                file_info["ground_truth"] = gt_map[key]
                matched_keys.add(key)
            else:
                file_info.pop("ground_truth", None)

        return {
            "matched_files": len(matched_keys),
            "total_files": len(metadata["files"]),
            "unmatched_csv_rows": sorted(set(gt_map) - matched_keys),
        }
    
    def list_datasets(self) -> List[Dict]:
        """List all custom datasets in the session"""
        if not self.datasets_dir.exists():
            return []
        
        datasets = []
        for dataset_dir in self.datasets_dir.iterdir():
            if dataset_dir.is_dir():
                metadata_file = dataset_dir / "dataset_metadata.json"
                if metadata_file.exists():
                    try:
                        with metadata_file.open("r") as f:
                            metadata = json.load(f)
                            datasets.append(metadata)
                    except Exception as e:
                        logger.warning(f"Could not read metadata for dataset {dataset_dir.name}: {e}")
        
        return datasets
    
    def get_dataset_metadata(self, dataset_name: str) -> Optional[Dict]:
        """Get metadata for a specific dataset"""
        dataset_dir = self.datasets_dir / dataset_name
        metadata_file = dataset_dir / "dataset_metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with metadata_file.open("r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not read metadata for dataset {dataset_name}: {e}")
            return None
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """Delete a custom dataset and all its files"""
        dataset_dir = self.datasets_dir / dataset_name
        
        if not dataset_dir.exists():
            return False
        
        # Remove all files in the dataset
        try:
            import shutil
            shutil.rmtree(dataset_dir)
            logger.info(f"Deleted custom dataset '{dataset_name}' from session {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Could not delete dataset {dataset_name}: {e}")
            return False
    
    def resolve_file_path(self, dataset_name: str, filename: str) -> Path:
        """Resolve the full path to a file in a custom dataset"""
        dataset_dir = self.datasets_dir / dataset_name
        file_path = dataset_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File '{filename}' not found in dataset '{dataset_name}'")
        
        return file_path
    
    def get_dataset_files_as_csv_format(self, dataset_name: str) -> List[Dict[str, str]]:
        """Get dataset files in the same format as global datasets (for compatibility)"""
        metadata = self.get_dataset_metadata(dataset_name)
        if not metadata:
            return []
        
        csv_format_files = []
        for file_info in metadata["files"]:
            row = {
                "filename": file_info["filename"],
                "duration": str(file_info["duration"]),
                "sample_rate": str(file_info.get("sample_rate", 0)),
                "size": str(file_info["size"]),
                "uploaded_at": file_info["uploaded_at"]
            }
            # LIT-247: AudioDataTable.tsx's "Ground Truth" column already
            # reads "ground_truth" generically off any dataset row - this is
            # the only change needed to make it show up for custom datasets.
            if file_info.get("ground_truth"):
                row["ground_truth"] = file_info["ground_truth"]
            csv_format_files.append(row)

        return csv_format_files


def get_custom_dataset_manager(session_id: str) -> CustomDatasetManager:
    """Factory function to create a CustomDatasetManager for a session"""
    return CustomDatasetManager(session_id)


def cleanup_session_datasets(session_id: str) -> bool:
    """Clean up all datasets for a session (called when session expires)"""
    session_dir = SESSIONS_BASE_DIR / session_id
    
    if not session_dir.exists():
        return True
    
    try:
        import shutil
        shutil.rmtree(session_dir)
        logger.info(f"Cleaned up session datasets for session {session_id}")
        return True
    except Exception as e:
        logger.error(f"Could not cleanup session {session_id}: {e}")
        return False


def is_custom_dataset(dataset_name: str) -> bool:
    """Check if a dataset name indicates a custom dataset"""
    return dataset_name.startswith("custom:")


def parse_custom_dataset_name(dataset_name: str) -> tuple[str, str]:
    """Parse custom dataset name format 'custom:session_id:dataset_name'"""
    if not is_custom_dataset(dataset_name):
        raise ValueError(f"Not a custom dataset name: {dataset_name}")
    
    parts = dataset_name.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid custom dataset name format: {dataset_name}")
    
    return parts[1], parts[2]  # session_id, dataset_name


def format_custom_dataset_name(session_id: str, dataset_name: str) -> str:
    """Format a custom dataset name for external use"""
    return f"custom:{session_id}:{dataset_name}"