"""
Multi-task dataset ingestion core (LIT-123, FR2 — Benchmark Dataset Management).

A single common interface over the seven approved benchmark corpora spanning
all three task families (ASR, SER, deepfake detection), per SAD §5.1
infrastructure layer (dataset-reading tools) and the LIT-106 dataset inventory.

This module is the *core*, not the per-corpus loaders. It provides:

  * ``SampleMetadata`` — one standardized record shape for every corpus, with
    accent/demographic fields parsed from each corpus's label catalog.
  * ``load_standardized_audio`` — decode any source clip to 16 kHz mono float32.
  * ``DatasetLoader`` — the common streaming / sub-sampling interface the child
    loaders subclass; ``CsvCatalogLoader`` is a ready generic implementation for
    the catalog-driven corpora.
  * ``CORPUS_REGISTRY`` — the seven approved corpora, each tagged with its task
    family and licence, resolved to a loader through ``get_loader``.

The concrete per-corpus loaders are separate child issues — LIT-141 (Common
Voice / LibriSpeech), LIT-142 (ASVspoof 2021), LIT-208 (CREMA-D / RAVDESS),
LIT-181 (L2-ARCTIC) — each of which registers a loader here.

Audio I/O is soundfile-only (project rule; torchaudio was removed in LIT-226);
resampling uses librosa's pure-numpy resampler.
"""

from __future__ import annotations

import csv
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Union

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Every corpus is standardized to this rate, single channel — the input contract
# shared by the Whisper (ASR) and Wav2Vec2 (SER/deepfake) model families.
TARGET_SAMPLE_RATE = 16_000

# Benchmark corpora live under Backend/data/ (gitignored — provisioned locally,
# not committed). Resolved the same way as the existing dataset_service.py:
# app/infrastructure/dataset_ingestion.py -> parents[2] == Backend/.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class TaskFamily(str, Enum):
    """The three interpretability task families the workbench serves."""

    ASR = "asr"
    SER = "ser"
    DEEPFAKE = "deepfake"


@dataclass(frozen=True)
class SampleMetadata:
    """One standardized sample record, identical in shape across all corpora.

    ``label`` carries the task-appropriate ground truth: the transcript for ASR,
    the emotion class for SER, and ``real`` / ``spoof`` for deepfake detection.
    ``accent`` and ``demographic`` are parsed from each corpus's catalog where
    present (they drive the FR15 accent-bias work downstream) and are ``None`` /
    empty when the corpus does not provide them.
    """

    dataset: str
    sample_id: str
    audio_path: Path
    task_family: TaskFamily
    label: Optional[str] = None
    speaker_id: Optional[str] = None
    accent: Optional[str] = None
    language: Optional[str] = None
    license: Optional[str] = None
    demographic: Dict[str, str] = field(default_factory=dict)
    extra: Dict[str, str] = field(default_factory=dict)


def load_standardized_audio(
    audio_path: Path | str, target_sr: int = TARGET_SAMPLE_RATE
) -> tuple[np.ndarray, int]:
    """Decode ``audio_path`` to mono float32 at ``target_sr``.

    Reads with soundfile (the project's only sanctioned audio I/O path),
    down-mixes multi-channel audio by averaging channels, and resamples with
    librosa only when the source rate differs from ``target_sr``. Returns the
    waveform and its (post-resample) sample rate so callers never have to guess.
    """
    audio_path = Path(audio_path)
    # always_2d gives a consistent (frames, channels) shape to reduce branching.
    data, source_sr = sf.read(str(audio_path), dtype="float32", always_2d=True)

    # Down-mix to mono by averaging channels.
    mono = data.mean(axis=1)

    if source_sr != target_sr:
        mono = librosa.resample(mono, orig_sr=source_sr, target_sr=target_sr)

    return np.ascontiguousarray(mono, dtype=np.float32), target_sr


# Reject empty or all-silence clips before they reach an evaluation batch
# (LIT-141 DoD: "remove corrupted frames or empty silence buffers"). RMS below
# this floor means effectively no signal.
SILENCE_RMS_FLOOR = 1e-4


def is_silent(audio: np.ndarray, rms_floor: float = SILENCE_RMS_FLOOR) -> bool:
    """True if the waveform is empty or below the silence RMS floor."""
    if audio.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return rms < rms_floor


class DatasetLoader(ABC):
    """Common interface every corpus loader exposes.

    Subclasses implement :meth:`iter_metadata` as a lazy generator so large
    corpora stream instead of being materialized; the base class layers
    sub-sampling and standardized audio loading on top of that one method.
    """

    def __init__(self, name: str, task_family: TaskFamily, license: Optional[str] = None):
        self.name = name
        self.task_family = task_family
        self.license = license

    @abstractmethod
    def iter_metadata(self) -> Iterator[SampleMetadata]:
        """Yield each sample's metadata lazily, in catalog order."""
        raise NotImplementedError

    def stream(self, limit: Optional[int] = None) -> Iterator[SampleMetadata]:
        """Stream metadata, optionally stopping after ``limit`` samples."""
        source = self.iter_metadata()
        return islice(source, limit) if limit is not None else source

    def subsample(self, n: int, seed: int = 0) -> List[SampleMetadata]:
        """Return up to ``n`` samples via reservoir sampling in a single pass.

        Bounds the working footprint on large corpora (Acceptance: "stream or
        sub-sample large corpora") without reading the whole catalog into memory,
        and is deterministic for a given ``seed`` so runs are reproducible.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        rng = random.Random(seed)
        reservoir: List[SampleMetadata] = []
        for i, meta in enumerate(self.iter_metadata()):
            if i < n:
                reservoir.append(meta)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = meta
        return reservoir

    def load_sample_audio(
        self, meta: SampleMetadata, target_sr: int = TARGET_SAMPLE_RATE
    ) -> tuple[np.ndarray, int]:
        """Decode one sample's audio to standardized 16 kHz mono float32."""
        return load_standardized_audio(meta.audio_path, target_sr=target_sr)

    def __iter__(self) -> Iterator[SampleMetadata]:
        return self.iter_metadata()


# A catalog field may be named differently across corpora (Common Voice's audio
# column is ``path``, a processed export may call it ``filename``), so each
# mapping accepts either a single column name or an ordered list of candidates —
# the first one present in a row wins.
ColumnRef = Union[str, Sequence[str]]


@dataclass
class ColumnMap:
    """Maps a corpus catalog's columns onto :class:`SampleMetadata` fields.

    Only ``filename`` and ``label`` are usually required; the rest are wired up
    per corpus (e.g. Common Voice exposes ``accent``/``gender``/``age``, RAVDESS
    encodes emotion in the filename so its loader overrides parsing entirely).
    Each field may be a single column name or a list of fallback candidates.
    """

    filename: ColumnRef = "filename"
    label: Optional[ColumnRef] = None
    sample_id: Optional[ColumnRef] = None
    speaker_id: Optional[ColumnRef] = None
    accent: Optional[ColumnRef] = None
    language: Optional[ColumnRef] = None
    demographic: tuple[str, ...] = ()


class CsvCatalogLoader(DatasetLoader):
    """Generic loader for corpora described by a CSV/TSV label catalog.

    Covers the catalog-driven corpora directly (Common Voice and RAVDESS ship
    metadata CSVs today) and serves as the base the child loaders extend. Column
    names are resolved case-insensitively so ``.tsv`` exports with mixed casing
    still line up.
    """

    def __init__(
        self,
        name: str,
        task_family: TaskFamily,
        catalog_path: Path | str,
        audio_base_dir: Path | str,
        column_map: ColumnMap,
        *,
        delimiter: str = ",",
        license: Optional[str] = None,
    ):
        super().__init__(name=name, task_family=task_family, license=license)
        self.catalog_path = Path(catalog_path)
        self.audio_base_dir = Path(audio_base_dir)
        self.column_map = column_map
        self.delimiter = delimiter

    def iter_metadata(self) -> Iterator[SampleMetadata]:
        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog for dataset '{self.name}' not found: {self.catalog_path}"
            )

        cmap = self.column_map
        with self.catalog_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=self.delimiter)
            for index, raw in enumerate(reader):
                row = {
                    str(k).strip().lower(): (v.strip() if isinstance(v, str) else v)
                    for k, v in raw.items()
                    if k is not None
                }

                filename = self._resolve(row, cmap.filename)
                if not filename:
                    logger.warning(
                        "Skipping row %d in %s: no value for column(s) %s",
                        index,
                        self.name,
                        cmap.filename,
                    )
                    continue

                demographic = {
                    col: row[col.lower()]
                    for col in cmap.demographic
                    if row.get(col.lower())
                }

                yield SampleMetadata(
                    dataset=self.name,
                    sample_id=self._resolve(row, cmap.sample_id) or filename,
                    audio_path=self.audio_base_dir / Path(filename).name,
                    task_family=self.task_family,
                    label=self._resolve(row, cmap.label),
                    speaker_id=self._resolve(row, cmap.speaker_id),
                    accent=self._resolve(row, cmap.accent),
                    language=self._resolve(row, cmap.language),
                    license=self.license,
                    demographic=demographic,
                )

    @staticmethod
    def _resolve(row: Dict[str, str], ref: Optional[ColumnRef]) -> Optional[str]:
        """Return the first non-empty value among ``ref``'s candidate columns."""
        if not ref:
            return None
        candidates = (ref,) if isinstance(ref, str) else ref
        for column in candidates:
            value = row.get(column.lower())
            if value:
                return value
        return None


class CommonVoiceLoader(CsvCatalogLoader):
    """Concrete loader for the Mozilla Common Voice validated-dev subset (ASR).

    Wired to the real catalog under ``Backend/data/common_voice_valid_dev/`` (see
    the existing ``dataset_service.py`` for the same paths). Common Voice carries
    the accent/age/gender labels the FR15 bias work depends on, and its
    transcript column is ``sentence``. Column names are given as candidate lists
    so both the raw Common Voice ``.tsv`` schema (``path``, ``accents``) and the
    repo's processed ``…_metadata.csv`` export (``filename``, ``accent``) load
    through the same mapping.

    Paths default to the real data location but are injectable for tests.
    """

    DEFAULT_DIR = DATA_DIR / "common_voice_valid_dev"
    DEFAULT_CATALOG = DEFAULT_DIR / "common_voice_valid_data_metadata.csv"

    COLUMN_MAP = ColumnMap(
        filename=("filename", "path"),
        label=("sentence", "transcript", "text"),
        speaker_id=("client_id", "speaker_id"),
        accent=("accent", "accents"),
        language=("locale", "language"),
        demographic=("age", "gender"),
    )

    def __init__(
        self,
        catalog_path: Optional[Path | str] = None,
        audio_base_dir: Optional[Path | str] = None,
        *,
        name: str = "common-voice",
    ):
        super().__init__(
            name=name,
            task_family=TaskFamily.ASR,
            catalog_path=catalog_path or self.DEFAULT_CATALOG,
            audio_base_dir=audio_base_dir or self.DEFAULT_DIR,
            column_map=self.COLUMN_MAP,
            license="CC0-1.0",
        )


class LibriSpeechLoader(DatasetLoader):
    """Loader for LibriSpeech (ASR; LIT-141).

    LibriSpeech has no single catalog: transcripts live in per-chapter
    ``<speaker>-<chapter>.trans.txt`` files (each line ``<utt-id> <TRANSCRIPT>``)
    beside the utterance ``.flac`` files, under ``<root>/<speaker>/<chapter>/``.
    This walks that tree. Speaker gender is read from the root
    ``SPEAKERS.TXT`` (``ID | SEX | SUBSET | MINUTES | NAME``) when present, and
    exposed as demographic metadata. Audio is standardised to 16 kHz mono.

    Paths default to the real data location but are injectable for tests.
    """

    DEFAULT_DIR = DATA_DIR / "librispeech"

    def __init__(
        self,
        root_dir: Optional[Path | str] = None,
        *,
        speakers_file: Optional[Path | str] = None,
        name: str = "librispeech",
    ):
        super().__init__(name=name, task_family=TaskFamily.ASR, license="CC-BY-4.0")
        self.root_dir = Path(root_dir or self.DEFAULT_DIR)
        self.speakers_file = Path(speakers_file) if speakers_file else self.root_dir / "SPEAKERS.TXT"

    def iter_metadata(self) -> Iterator[SampleMetadata]:
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"LibriSpeech root for '{self.name}' not found: {self.root_dir}"
            )
        genders = self._load_speaker_genders()

        for trans_file in sorted(self.root_dir.rglob("*.trans.txt")):
            with trans_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    utt_id, transcript = parts
                    speaker = utt_id.split("-")[0]
                    demographic = {"gender": genders[speaker]} if speaker in genders else {}
                    yield SampleMetadata(
                        dataset=self.name,
                        sample_id=utt_id,
                        audio_path=trans_file.parent / f"{utt_id}.flac",
                        task_family=TaskFamily.ASR,
                        label=transcript,
                        speaker_id=speaker,
                        license=self.license,
                        demographic=demographic,
                    )

    def _load_speaker_genders(self) -> Dict[str, str]:
        """Parse SPEAKERS.TXT (``ID | SEX | …``) into {speaker_id: gender}."""
        genders: Dict[str, str] = {}
        if not self.speakers_file.exists():
            return genders
        sex_map = {"M": "male", "F": "female"}
        with self.speakers_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(";"):  # header/comment lines
                    continue
                cols = [c.strip() for c in line.split("|")]
                if len(cols) >= 2 and cols[0]:
                    genders[cols[0]] = sex_map.get(cols[1].upper(), cols[1])
        return genders


@dataclass
class CorpusSpec:
    """A registry entry: what a corpus is, and how (or whether yet) to load it.

    ``loader_factory`` is ``None`` for corpora whose concrete loader is still a
    child issue — ``get_loader`` raises a clear ``NotImplementedError`` naming the
    owning issue rather than fabricating data, matching the anti-fabrication
    stance used elsewhere in the codebase.
    """

    name: str
    task_family: TaskFamily
    license: str
    loader_factory: Optional[Callable[..., DatasetLoader]] = None
    owner_issue: Optional[str] = None


# The seven approved corpora (LIT-106 inventory), one row per corpus, tagged with
# task family and licence. Concrete loaders are wired in by the child issues; the
# core owns the registry, the schema, and the standardiser.
CORPUS_REGISTRY: Dict[str, CorpusSpec] = {
    "common-voice": CorpusSpec("common-voice", TaskFamily.ASR, "CC0-1.0", loader_factory=CommonVoiceLoader, owner_issue="LIT-141"),
    "librispeech": CorpusSpec("librispeech", TaskFamily.ASR, "CC-BY-4.0", loader_factory=LibriSpeechLoader, owner_issue="LIT-141"),
    "crema-d": CorpusSpec("crema-d", TaskFamily.SER, "Open Database License", owner_issue="LIT-208"),
    "ravdess": CorpusSpec("ravdess", TaskFamily.SER, "CC-BY-NC-SA-4.0", owner_issue="LIT-208"),
    "esd": CorpusSpec("esd", TaskFamily.SER, "Research-only", owner_issue="LIT-208"),
    "l2-arctic": CorpusSpec("l2-arctic", TaskFamily.ASR, "Research-only", owner_issue="LIT-181"),
    "asvspoof-2021": CorpusSpec("asvspoof-2021", TaskFamily.DEEPFAKE, "ODC-By", owner_issue="LIT-142"),
}


def list_supported_corpora() -> List[str]:
    """Return the registered corpus names (the seven approved corpora)."""
    return list(CORPUS_REGISTRY.keys())


def get_corpus_spec(name: str) -> CorpusSpec:
    """Look up a corpus spec, case-insensitively."""
    key = name.strip().lower()
    if key not in CORPUS_REGISTRY:
        raise ValueError(
            f"Unknown corpus '{name}'. Supported: {', '.join(list_supported_corpora())}"
        )
    return CORPUS_REGISTRY[key]


def get_loader(name: str, **kwargs) -> DatasetLoader:
    """Instantiate the loader for ``name``, forwarding ``kwargs`` to its factory.

    Raises ``NotImplementedError`` (naming the child issue) for corpora whose
    concrete loader has not been contributed yet, so callers get an honest signal
    instead of silent empty data.
    """
    spec = get_corpus_spec(name)
    if spec.loader_factory is None:
        raise NotImplementedError(
            f"No loader registered for corpus '{spec.name}' yet — tracked by "
            f"{spec.owner_issue or 'a child issue of LIT-123'}."
        )
    return spec.loader_factory(**kwargs)
