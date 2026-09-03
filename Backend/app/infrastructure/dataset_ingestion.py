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
from dataclasses import dataclass, field, replace
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


#: FR2.3 / SAD C5 — corpora whose licence requires a user-visible notice on
#: load. Matches the four corpora FR2.3 names explicitly; kept as one place
#: so the API and frontend don't each maintain their own copy of this list.
NON_COMMERCIAL_CORPORA = frozenset({"ravdess", "l2-arctic", "esd", "asvspoof-2021"})


@dataclass(frozen=True)
class IntegrityReport:
    """One sample's FR2.1 integrity verdict.

    ``reason`` is only set when ``ok`` is False: ``"missing"`` (no file at
    ``audio_path``), ``"undecodable"`` (soundfile/librosa raised while
    reading it), or ``"silent"`` (decodes fine but is empty/below the RMS
    floor per :func:`is_silent`).
    """

    sample_id: str
    ok: bool
    reason: Optional[str] = None


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
        self._license_notice_logged = False

    @abstractmethod
    def iter_metadata(self) -> Iterator[SampleMetadata]:
        """Yield each sample's metadata lazily, in catalog order."""
        raise NotImplementedError

    def _maybe_log_license_notice(self) -> None:
        """FR2.3 / SAD C5 — log a licence notice once per loader instance.

        Concrete loaders for the four non-commercial corpora call this as the
        first line of their :meth:`iter_metadata`; centralized here so every
        one of them logs the same message instead of each hand-rolling it
        (this replaced a one-off implementation that only ``ASVspoofLoader``
        had).
        """
        if self._license_notice_logged or self.name not in NON_COMMERCIAL_CORPORA:
            return
        logger.warning(
            "%s is a non-commercial/research-use corpus (licence: %s) — "
            "SAD constraint C5 applies.",
            self.name,
            self.license or "unknown",
        )
        self._license_notice_logged = True

    def check_integrity(self, meta: SampleMetadata, *, deep: bool = True) -> IntegrityReport:
        """FR2.1 — validate one sample before it reaches a batch or a listing.

        ``deep=False`` only checks the file exists (cheap — safe to run on
        every row of a metadata listing). ``deep=True`` additionally decodes
        the audio and runs :func:`is_silent`, for batch/evaluation callers
        where a wrong WER/label attributed to the model is a worse outcome
        than the extra decode cost.
        """
        if not meta.audio_path.exists():
            return IntegrityReport(meta.sample_id, False, "missing")
        if not deep:
            return IntegrityReport(meta.sample_id, True)
        try:
            audio, _ = self.load_sample_audio(meta)
        except Exception:
            return IntegrityReport(meta.sample_id, False, "undecodable")
        if is_silent(audio):
            return IntegrityReport(meta.sample_id, False, "silent")
        return IntegrityReport(meta.sample_id, True)

    def validated_stream(
        self,
        limit: Optional[int] = None,
        *,
        deep: bool = True,
        on_reject: Optional[Callable[[IntegrityReport], None]] = None,
    ) -> Iterator[SampleMetadata]:
        """Stream only samples that pass :meth:`check_integrity` (FR2.1).

        ``limit`` counts accepted samples, not samples examined, so a caller
        asking for 50 valid clips actually gets 50 (skipping rejects) rather
        than getting fewer because some of the first 50 catalog rows were
        corrupt. ``on_reject`` lets a caller collect/log what was excluded
        instead of it disappearing silently.
        """
        accepted = 0
        for meta in self.iter_metadata():
            if limit is not None and accepted >= limit:
                return
            report = self.check_integrity(meta, deep=deep)
            if report.ok:
                accepted += 1
                yield meta
            elif on_reject is not None:
                on_reject(report)

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
        encoding: str = "utf-8",
    ):
        super().__init__(name=name, task_family=task_family, license=license)
        self.catalog_path = Path(catalog_path)
        self.audio_base_dir = Path(audio_base_dir)
        self.column_map = column_map
        self.delimiter = delimiter
        # "utf-8-sig" transparently strips a leading BOM if present and is
        # otherwise identical to "utf-8" -- corpora whose CSV export carries a
        # BOM on the header row (LIT-236) would otherwise have every row
        # silently skipped, since the BOM makes the first column name never
        # match its ColumnMap entry.
        self.encoding = encoding

    def iter_metadata(self) -> Iterator[SampleMetadata]:
        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog for dataset '{self.name}' not found: {self.catalog_path}"
            )
        self._maybe_log_license_notice()

        cmap = self.column_map
        with self.catalog_path.open("r", encoding=self.encoding, newline="") as fh:
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
# ASVspoof labels are the tokens "bonafide" / "spoof"; normalize to these.
BONA_FIDE = "bona-fide"
SPOOF = "spoof"
_ASVSPOOF_LABEL_TOKENS = {"bonafide": BONA_FIDE, "spoof": SPOOF}
ASVSPOOF_LICENSE = "ASVspoof 2021 DF — research use only"


class ASVspoofLoader(DatasetLoader):
    """Loader for the ASVspoof 2021 DF deepfake benchmark (LIT-142; FR7 eval set).

    The protocol / label file is whitespace-delimited with no header. The
    bona-fide vs spoof tag is read by scanning each line for the
    ``bonafide`` / ``spoof`` token, so both the ASVspoof 2019 LA CM protocol
    (``SPEAKER FILE - SYSTEM bonafide``) and the 2021 DF ``trial_metadata.txt``
    layout (``SPEAKER FILE codec source attack spoof …``) load through the same
    path. Audio is the per-utterance FLAC named by the file-id column,
    standardised to 16 kHz mono like every other loader.

    Research-use-only corpus (SAD constraint C5): a licence notice is logged the
    first time samples are read. Paths default to the real data location but are
    injectable for tests.
    """

    DEFAULT_DIR = DATA_DIR / "asvspoof2021_df"
    DEFAULT_PROTOCOL = DEFAULT_DIR / "trial_metadata.txt"

    def __init__(
        self,
        protocol_path: Optional[Path | str] = None,
        audio_base_dir: Optional[Path | str] = None,
        *,
        audio_ext: str = ".flac",
        file_col: int = 1,
        speaker_col: int = 0,
        name: str = "asvspoof-2021",
    ):
        super().__init__(name=name, task_family=TaskFamily.DEEPFAKE, license=ASVSPOOF_LICENSE)
        self.protocol_path = Path(protocol_path or self.DEFAULT_PROTOCOL)
        self.audio_base_dir = Path(audio_base_dir or self.DEFAULT_DIR)
        self.audio_ext = audio_ext
        self.file_col = file_col
        self.speaker_col = speaker_col

    def iter_metadata(self) -> Iterator[SampleMetadata]:
        if not self.protocol_path.exists():
            raise FileNotFoundError(
                f"ASVspoof protocol for '{self.name}' not found: {self.protocol_path}"
            )
        self._maybe_log_license_notice()

        with self.protocol_path.open("r", encoding="utf-8") as fh:
            for index, line in enumerate(fh, start=1):
                tokens = line.split()
                if not tokens:
                    continue

                label = self._extract_label(tokens)
                if label is None:
                    logger.warning(
                        "Skipping ASVspoof line %d: no bonafide/spoof token in %r",
                        index,
                        line.strip(),
                    )
                    continue
                if self.file_col >= len(tokens):
                    logger.warning(
                        "Skipping ASVspoof line %d: no file id at column %d",
                        index,
                        self.file_col,
                    )
                    continue

                file_id = tokens[self.file_col]
                speaker = tokens[self.speaker_col] if self.speaker_col < len(tokens) else None
                filename = file_id if Path(file_id).suffix else f"{file_id}{self.audio_ext}"

                yield SampleMetadata(
                    dataset=self.name,
                    sample_id=file_id,
                    audio_path=self.audio_base_dir / Path(filename).name,
                    task_family=TaskFamily.DEEPFAKE,
                    label=label,
                    speaker_id=speaker,
                    license=self.license,
                )

    @staticmethod
    def _extract_label(tokens: List[str]) -> Optional[str]:
        for token in tokens:
            normalized = _ASVSPOOF_LABEL_TOKENS.get(token.strip().lower())
            if normalized is not None:
                return normalized
        return None


# L2-ARCTIC's 24 speakers grouped by first language (L1). The accent is the
# whole point of this corpus (it drives the FR15 accent-bias work) and the
# speaker -> L1 mapping is fixed, so it's encoded here rather than parsed.
L2_ARCTIC_SPEAKER_L1 = {
    "ABA": "Arabic", "SKA": "Arabic", "YBAA": "Arabic", "ZHAA": "Arabic",
    "BWC": "Mandarin", "LXC": "Mandarin", "NCC": "Mandarin", "TXHC": "Mandarin",
    "ASI": "Hindi", "RRBI": "Hindi", "SVBI": "Hindi", "TNI": "Hindi",
    "HJK": "Korean", "HKK": "Korean", "YDCK": "Korean", "YKWK": "Korean",
    "EBVS": "Spanish", "ERMS": "Spanish", "MBMPS": "Spanish", "NJS": "Spanish",
    "HQTV": "Vietnamese", "PNV": "Vietnamese", "THV": "Vietnamese", "TLV": "Vietnamese",
}
L2_ARCTIC_LICENSE = "L2-ARCTIC — research use only"


class L2ArcticLoader(DatasetLoader):
    """Loader for the L2-ARCTIC non-native reading corpus (LIT-181; FR2/FR15).

    Each speaker lives in ``<root>/<SPEAKER>/`` with ``wav/arctic_*.wav`` and a
    matching per-utterance transcript at ``transcript/<utt>.txt``. The speaker's
    first language (accent) comes from the fixed ``SPEAKER_L1`` map and is
    exposed as ``accent`` / ``demographic["l1"]`` — the grouping the FR15 bias
    diagnostics slice on. Audio is standardised to 16 kHz mono for the ASR path.

    Research-use-only corpus; paths default to the real data location but are
    injectable for tests.
    """

    # Real data directory is "l2arctic" (no underscore) - was "l2_arctic" here,
    # a mismatch against the actual Backend/data/ layout that made this loader
    # unable to find its data by default (LIT-235).
    DEFAULT_DIR = DATA_DIR / "l2arctic"
    SPEAKER_L1 = L2_ARCTIC_SPEAKER_L1

    def __init__(self, root_dir: Optional[Path | str] = None, *, name: str = "l2-arctic"):
        super().__init__(name=name, task_family=TaskFamily.ASR, license=L2_ARCTIC_LICENSE)
        self.root_dir = Path(root_dir or self.DEFAULT_DIR)

    def iter_metadata(self) -> Iterator[SampleMetadata]:
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"L2-ARCTIC root for '{self.name}' not found: {self.root_dir}"
            )
        self._maybe_log_license_notice()
        for speaker in sorted(self.SPEAKER_L1):
            wav_dir = self.root_dir / speaker / "wav"
            if not wav_dir.is_dir():
                continue
            l1 = self.SPEAKER_L1[speaker]
            for wav_path in sorted(wav_dir.glob("*.wav")):
                utt_id = wav_path.stem
                transcript = self._read_transcript(
                    self.root_dir / speaker / "transcript" / f"{utt_id}.txt"
                )
                yield SampleMetadata(
                    dataset=self.name,
                    sample_id=f"{speaker}-{utt_id}",
                    audio_path=wav_path,
                    task_family=TaskFamily.ASR,
                    label=transcript,
                    speaker_id=speaker,
                    accent=l1,
                    license=self.license,
                    demographic={"l1": l1},
                )

    @staticmethod
    def _read_transcript(path: Path) -> Optional[str]:
        if path.exists():
            return path.read_text(encoding="utf-8").strip() or None
        return None


#: Canonical SER label vocabulary (LIT-208).
#:
#: Every SER corpus is normalised onto these names so ground truth is directly
#: comparable with model output. They deliberately match the label strings the
#: default SER checkpoint emits (LIT-224) -- note "fearful"/"surprised" rather
#: than "fear"/"surprise" -- because the whole point of this corpus is measuring
#: the model against known labels, and a vocabulary mismatch there silently
#: scores every fear clip wrong.
#:
#: "calm" is RAVDESS-only and has no counterpart in the classifier. It is kept
#: because it is genuine ground truth; an accuracy harness must decide whether
#: to exclude those clips rather than have the loader quietly drop them.
EMOTION_LABELS = (
    "angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised",
)

CREMA_D_LICENSE = "Open Database License (ODbL) - research use"
RAVDESS_LICENSE = "CC-BY-NC-SA-4.0 - non-commercial research use"

#: CREMA-D encodes emotion as the third underscore-delimited filename field.
_CREMA_D_EMOTION = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fearful",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}
_CREMA_D_INTENSITY = {"LO": "low", "MD": "medium", "HI": "high", "XX": "unspecified"}

#: RAVDESS encodes emotion as the third dash-delimited filename field.
_RAVDESS_EMOTION = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
_RAVDESS_INTENSITY = {"01": "normal", "02": "strong"}
_RAVDESS_STATEMENT = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}


class CremaDLoader(DatasetLoader):
    """Loader for the CREMA-D crowd-sourced emotional speech corpus (LIT-208; FR2/FR6).

    Audio lives flat in ``<root>/AudioWAV/`` with the label encoded in the
    filename: ``<actor>_<sentence>_<emotion>_<intensity>.wav``, e.g.
    ``1001_DFA_ANG_XX.wav``. Emotion is normalised onto ``EMOTION_LABELS``.

    Speaker demographics (age, sex, race, ethnicity) come from the corpus's
    ``VideoDemographics.csv`` when present -- CREMA-D is the one SER corpus that
    ships them, which makes it the natural input for the FR15 accent/demographic
    bias work later. The loader still works without that file; the demographic
    dict is simply sparse.
    """

    DEFAULT_DIR = DATA_DIR / "crema_d"
    AUDIO_SUBDIR = "AudioWAV"
    DEMOGRAPHICS_FILE = "VideoDemographics.csv"

    def __init__(self, root_dir: Optional[Path | str] = None, *, name: str = "crema-d"):
        super().__init__(name=name, task_family=TaskFamily.SER, license=CREMA_D_LICENSE)
        self.root_dir = Path(root_dir or self.DEFAULT_DIR)

    def iter_metadata(self) -> Iterator[SampleMetadata]:
        audio_dir = self.root_dir / self.AUDIO_SUBDIR
        if not audio_dir.is_dir():
            # Some CREMA-D distributions get extracted/copied flat, without
            # the official AudioWAV/ subfolder - fall back to the root itself
            # if it directly contains the .wav files (LIT-235), rather than
            # failing on a structural variant that still has real data.
            if self.root_dir.is_dir() and next(self.root_dir.glob("*.wav"), None) is not None:
                audio_dir = self.root_dir
            else:
                raise FileNotFoundError(
                    f"CREMA-D audio directory for '{self.name}' not found: {audio_dir}"
                )
        demographics = self._load_demographics()
        for wav_path in sorted(audio_dir.glob("*.wav")):
            parsed = self._parse_filename(wav_path.stem)
            if parsed is None:
                logger.warning("crema-d: skipping unparseable filename %s", wav_path.name)
                continue
            actor_id, sentence, emotion, intensity = parsed
            demo = dict(demographics.get(actor_id, {}))
            demo["intensity"] = intensity
            yield SampleMetadata(
                dataset=self.name,
                sample_id=wav_path.stem,
                audio_path=wav_path,
                task_family=TaskFamily.SER,
                label=emotion,
                speaker_id=actor_id,
                language="en",
                license=self.license,
                demographic=demo,
                extra={"sentence_code": sentence, "intensity": intensity},
            )

    @staticmethod
    def _parse_filename(stem: str) -> Optional[tuple[str, str, str, str]]:
        """``1001_DFA_ANG_XX`` -> (actor, sentence, canonical emotion, intensity)."""
        parts = stem.split("_")
        if len(parts) != 4:
            return None
        actor_id, sentence, emotion_code, intensity_code = parts
        emotion = _CREMA_D_EMOTION.get(emotion_code.upper())
        if emotion is None:
            return None
        return (
            actor_id,
            sentence,
            emotion,
            _CREMA_D_INTENSITY.get(intensity_code.upper(), intensity_code.lower()),
        )

    def _load_demographics(self) -> Dict[str, Dict[str, str]]:
        path = self.root_dir / self.DEMOGRAPHICS_FILE
        if not path.exists():
            return {}
        out: Dict[str, Dict[str, str]] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                actor = (row.get("ActorID") or "").strip()
                if not actor:
                    continue
                out[actor] = {
                    key: (row.get(column) or "").strip()
                    for key, column in (
                        ("age", "Age"), ("sex", "Sex"),
                        ("race", "Race"), ("ethnicity", "Ethnicity"),
                    )
                    if (row.get(column) or "").strip()
                }
        return out


class RavdessLoader(DatasetLoader):
    """Loader for the RAVDESS emotional speech corpus (LIT-208; FR2/FR6).

    Audio sits under per-actor directories (``<root>/Actor_01/*.wav``) with a
    seven-field dash-delimited filename, e.g. ``03-01-05-01-02-01-12.wav``:
    modality, vocal channel, emotion, intensity, statement, repetition, actor.
    Emotion (field 3) is normalised onto ``EMOTION_LABELS``.

    Only speech is yielded. RAVDESS also ships a *song* vocal channel (field 2 =
    ``02``), which is out of scope for an SER demo and would otherwise be scored
    as if it were speech. Actor gender follows the corpus convention that
    odd-numbered actors are male and even-numbered female.

    Non-commercial research licence (CC-BY-NC-SA-4.0), so the audio is not
    vendored -- ``Backend/data/`` is gitignored and paths are injectable.
    """

    DEFAULT_DIR = DATA_DIR / "ravdess"
    SPEECH_CHANNEL = "01"

    def __init__(
        self,
        root_dir: Optional[Path | str] = None,
        *,
        name: str = "ravdess",
        include_song: bool = False,
    ):
        super().__init__(name=name, task_family=TaskFamily.SER, license=RAVDESS_LICENSE)
        if root_dir:
            self.root_dir = Path(root_dir)
        else:
            self.root_dir = self.DEFAULT_DIR if self.DEFAULT_DIR.is_dir() else (DATA_DIR / "ravdess_subset")
        self.include_song = include_song

    def iter_metadata(self) -> Iterator[SampleMetadata]:
        if not self.root_dir.is_dir():
            raise FileNotFoundError(
                f"RAVDESS root for '{self.name}' not found: {self.root_dir}"
            )
        self._maybe_log_license_notice()
        # Actor_* subdirectories are the documented layout, but tolerate a flat
        # dump too -- both appear in the wild depending on how it was unzipped.
        wav_paths = sorted(self.root_dir.glob("Actor_*/*.wav")) or sorted(
            self.root_dir.glob("*.wav")
        )
        for wav_path in wav_paths:
            parsed = self._parse_filename(wav_path.stem)
            if parsed is None:
                logger.warning("ravdess: skipping unparseable filename %s", wav_path.name)
                continue
            channel, emotion, intensity, statement, repetition, actor = parsed
            if channel != self.SPEECH_CHANNEL and not self.include_song:
                continue
            yield SampleMetadata(
                dataset=self.name,
                sample_id=wav_path.stem,
                audio_path=wav_path,
                task_family=TaskFamily.SER,
                label=emotion,
                speaker_id=f"actor_{actor}",
                language="en",
                license=self.license,
                demographic={
                    "sex": "male" if int(actor) % 2 == 1 else "female",
                    "intensity": intensity,
                },
                extra={
                    "statement": _RAVDESS_STATEMENT.get(statement, statement),
                    "repetition": repetition,
                    "vocal_channel": "speech" if channel == self.SPEECH_CHANNEL else "song",
                },
            )

    @staticmethod
    def _parse_filename(stem: str) -> Optional[tuple[str, str, str, str, str, str]]:
        """``03-01-05-01-02-01-12`` -> (channel, emotion, intensity, statement, rep, actor)."""
        parts = stem.split("-")
        if len(parts) != 7:
            return None
        _, channel, emotion_code, intensity_code, statement, repetition, actor = parts
        emotion = _RAVDESS_EMOTION.get(emotion_code)
        if emotion is None or not actor.isdigit():
            return None
        return (
            channel,
            emotion,
            _RAVDESS_INTENSITY.get(intensity_code, intensity_code),
            statement,
            repetition,
            actor,
        )


#: ESD's raw emotion values are Title case and spell the fifth class
#: "Surprise" rather than the classifier vocabulary's "surprised" -- resolved
#: case-insensitively so a stray casing difference in the catalog doesn't
#: silently drop a class (LIT-236).
_ESD_EMOTION = {
    "angry": "angry",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprise": "surprised",
}
ESD_LICENSE = "Research-only"


class ESDLoader(CsvCatalogLoader):
    """Loader for the ESD (Emotional Speech Database) corpus (LIT-236; FR2/FR6).

    Catalog-driven like Common Voice, not filename-encoded like CREMA-D/RAVDESS:
    a single CSV (``filename,sample_id,speaker_id,emotion,emotion_cn,
    transcription,dataset``) lists every clip, with all ``.wav`` files flat in
    the corpus root.

    The shipped catalog carries a UTF-8 BOM on its header row, which the base
    class's default ``encoding="utf-8"`` does not strip -- the BOM merges into
    the first column name, so it never matches ``ColumnMap.filename`` and every
    row is silently skipped. Opting into ``encoding="utf-8-sig"`` here fixes
    that without changing behaviour for any other catalog corpus.

    Raw emotion values are Title case (``Angry``, ``Surprise``, ...); they're
    remapped onto the canonical ``EMOTION_LABELS`` vocabulary via
    ``_ESD_EMOTION`` after the base class resolves them, the same normalisation
    CREMA-D/RAVDESS do from their filename encodings -- otherwise ``"Surprise"``
    never matches the SER classifier's own ``"surprised"`` output and any
    accuracy scoring against this corpus is silently wrong.
    """

    DEFAULT_DIR = DATA_DIR / "esd"
    DEFAULT_CATALOG = DEFAULT_DIR / "esd_test_100_metadata.csv"

    COLUMN_MAP = ColumnMap(
        filename="filename",
        label="emotion",
        sample_id="sample_id",
        speaker_id="speaker_id",
    )

    def __init__(
        self,
        catalog_path: Optional[Path | str] = None,
        audio_base_dir: Optional[Path | str] = None,
        *,
        name: str = "esd",
    ):
        super().__init__(
            name=name,
            task_family=TaskFamily.SER,
            catalog_path=catalog_path or self.DEFAULT_CATALOG,
            audio_base_dir=audio_base_dir or self.DEFAULT_DIR,
            column_map=self.COLUMN_MAP,
            license=ESD_LICENSE,
            encoding="utf-8-sig",
        )

    def iter_metadata(self) -> Iterator[SampleMetadata]:
        for meta in super().iter_metadata():
            if meta.label is None:
                yield meta
                continue
            normalized = _ESD_EMOTION.get(meta.label.strip().lower())
            if normalized is None:
                logger.warning(
                    "esd: unrecognised emotion label %r for sample %s -- passing through unmapped",
                    meta.label,
                    meta.sample_id,
                )
                yield meta
                continue
            yield replace(meta, label=normalized)


def demo_clips_by_emotion(
    loader: DatasetLoader, per_class: int = 2, seed: int = 0
) -> Dict[str, List[SampleMetadata]]:
    """Pick up to ``per_class`` clips for each emotion present in ``loader``.

    Backs the mid-evaluation walkthrough ("a handful of demo clips per emotion
    class") and gives LIT-224's outstanding accuracy check a balanced, known-label
    sample instead of whatever the catalog happens to list first. Deterministic
    for a given ``seed``; streams rather than materialising the corpus.
    """
    if per_class < 0:
        raise ValueError("per_class must be non-negative")
    rng = random.Random(seed)
    seen: Dict[str, int] = {}
    picked: Dict[str, List[SampleMetadata]] = {}
    for meta in loader.iter_metadata():
        if meta.label is None:
            continue
        count = seen[meta.label] = seen.get(meta.label, 0) + 1
        bucket = picked.setdefault(meta.label, [])
        if len(bucket) < per_class:
            bucket.append(meta)
        else:
            # Reservoir-sample within the class so the choice is spread across
            # the corpus rather than biased to its first few actors.
            j = rng.randint(0, count - 1)
            if j < per_class:
                bucket[j] = meta
    return picked


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
    "crema-d": CorpusSpec("crema-d", TaskFamily.SER, CREMA_D_LICENSE, loader_factory=CremaDLoader, owner_issue="LIT-208"),
    "ravdess": CorpusSpec("ravdess", TaskFamily.SER, RAVDESS_LICENSE, loader_factory=RavdessLoader, owner_issue="LIT-208"),
    "esd": CorpusSpec("esd", TaskFamily.SER, ESD_LICENSE, loader_factory=ESDLoader, owner_issue="LIT-236"),
    "l2-arctic": CorpusSpec("l2-arctic", TaskFamily.ASR, L2_ARCTIC_LICENSE, loader_factory=L2ArcticLoader, owner_issue="LIT-181"),
    "asvspoof-2021": CorpusSpec("asvspoof-2021", TaskFamily.DEEPFAKE, ASVSPOOF_LICENSE, loader_factory=ASVspoofLoader, owner_issue="LIT-142"),
}


def list_supported_corpora() -> List[str]:
    """Return the registered corpus names (the seven approved corpora)."""
    return list(CORPUS_REGISTRY.keys())


def get_corpus_spec(name: str) -> CorpusSpec:
    """Look up a corpus spec, case-insensitively with alias normalization."""
    key = name.strip().lower().replace("_", "-")
    if key == "l2arctic":
        key = "l2-arctic"
    elif key == "cremad":
        key = "crema-d"
    elif key == "commonvoice":
        key = "common-voice"
    elif key == "asvspoof2021":
        key = "asvspoof-2021"

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


def is_corpus_available(name: str) -> bool:
    """FR2 §5.1 — does ``name`` have both a loader *and* provisioned data?

    ``list_supported_corpora``/``CORPUS_REGISTRY`` only tell you whether code
    is wired (a ``loader_factory``); they say nothing about whether
    ``Backend/data/<corpus>`` actually exists on this machine. A corpus that
    is wired but unprovisioned (LibriSpeech, until someone downloads it)
    would otherwise appear selectable and then 404 on first use. This does
    one lazy single-row probe rather than a full catalog walk.
    """
    try:
        loader = get_loader(name)
    except (ValueError, NotImplementedError):
        return False
    try:
        next(iter(loader.stream(limit=1)))
        return True
    except (FileNotFoundError, StopIteration):
        return False
    except Exception:
        logger.warning("Availability probe raised for corpus %r", name, exc_info=True)
        return False


def measure_footprint(data_dir: Optional[Path | str] = None) -> Dict[str, int]:
    """FR2.2 — bytes on disk per corpus directory under ``data_dir``.

    Walks ``Backend/data/<subdir>`` (one entry per provisioned corpus) rather
    than the registry, so it reports what is actually on disk today,
    including any corpus directory that isn't (yet) wired to a loader.
    """
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    if not base.is_dir():
        return {}
    usage: Dict[str, int] = {}
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        usage[entry.name] = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
    return usage
