# Chapter 14 — Datasets

`Backend/app/infrastructure/dataset_ingestion.py`, 1239 lines, loads seven
speech corpora with completely different formats and presents them all through
one interface.

Dataset code is unglamorous and it is where a research system quietly breaks.
A corpus whose emotion codes you mapped wrong produces a model evaluation that
is confidently, silently incorrect.

---

## 14.1 The seven corpora

```python
CORPUS_REGISTRY: Dict[str, CorpusSpec] = {
    "common-voice":  CorpusSpec("common-voice",  TaskFamily.ASR,      "CC0-1.0",         loader_factory=CommonVoiceLoader, owner_issue="LIT-141"),
    "librispeech":   CorpusSpec("librispeech",   TaskFamily.ASR,      "CC-BY-4.0",       loader_factory=LibriSpeechLoader, owner_issue="LIT-141"),
    "crema-d":       CorpusSpec("crema-d",       TaskFamily.SER,      CREMA_D_LICENSE,   loader_factory=CremaDLoader,      owner_issue="LIT-208"),
    "ravdess":       CorpusSpec("ravdess",       TaskFamily.SER,      RAVDESS_LICENSE,   loader_factory=RavdessLoader,     owner_issue="LIT-208"),
    "esd":           CorpusSpec("esd",           TaskFamily.SER,      ESD_LICENSE,       loader_factory=ESDLoader,         owner_issue="LIT-236"),
    "l2-arctic":     CorpusSpec("l2-arctic",     TaskFamily.ASR,      L2_ARCTIC_LICENSE, loader_factory=L2ArcticLoader,    owner_issue="LIT-181"),
    "asvspoof-2021": CorpusSpec("asvspoof-2021", TaskFamily.DEEPFAKE, ASVSPOOF_LICENSE,  loader_factory=ASVspoofLoader,    owner_issue="LIT-142"),
}
```

| Corpus | Task | What it is | Why it is here |
|---|---|---|---|
| Common Voice | ASR | Mozilla's crowd-sourced multilingual speech, with age/gender/accent metadata | baseline ASR, accent variety |
| LibriSpeech | ASR | read audiobooks, clean and aligned | clean benchmark, saliency validation |
| CREMA-D | SER | 7,442 clips, 91 actors, 6 emotions, with demographics | SER across demographics |
| RAVDESS | SER | acted emotional speech, 8 emotions, two intensities | emotion classification, pitch metrics |
| ESD | SER | 10 English + 10 Mandarin speakers, 5 emotions | cross-lingual emotion |
| L2-ARCTIC | ASR | 24 non-native speakers, 6 first languages, **identical English prompts** | accent-bias profiling |
| ASVspoof 2021 DF | deepfake | bona-fide and spoofed utterances | deepfake detection and evaluation |

**L2-ARCTIC is the one that makes accent-bias measurement valid.** All speakers
read the *same* English prompts, so text difficulty is controlled and a WER
difference between cohorts is attributable to accent rather than to content
(§2.9).

---

## 14.2 One schema for everything

```python
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
```

**`label` is polymorphic** — a transcript, an emotion, or `real`/`spoof`,
depending on `task_family`. That is a deliberate simplification: one field
rather than three mutually exclusive ones. The consumer knows which it is from
`task_family`.

**`frozen=True`** makes records immutable, so a consumer cannot corrupt a
shared record.

**`field(default_factory=dict)`** rather than `= {}`. This is a real Python
trap: a mutable default is created **once, at function definition**, and shared
by every instance. `= {}` would give every `SampleMetadata` the *same* dict —
mutate one sample's demographics and you mutate them all.

**`extra`** is the escape hatch for corpus-specific fields that do not
generalise, so adding a corpus never requires adding a field to the shared
schema.

---

## 14.3 Standardising audio

```python
def load_standardized_audio(audio_path, target_sr=TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Decode ``audio_path`` to mono float32 at ``target_sr``.

    Reads with soundfile (the project's only sanctioned audio I/O path),
    down-mixes multi-channel audio by averaging channels, and resamples with
    librosa only when the source rate differs from ``target_sr``. Returns the
    waveform and its (post-resample) sample rate so callers never have to guess.
    """
    audio_path = Path(audio_path)
    # always_2d gives a consistent (frames, channels) shape to reduce branching.
    data, source_sr = sf.read(str(audio_path), dtype="float32", always_2d=True)

    mono = data.mean(axis=1)

    if source_sr != target_sr:
        mono = librosa.resample(mono, orig_sr=source_sr, target_sr=target_sr)

    return np.ascontiguousarray(mono, dtype=np.float32), target_sr
```

The same normalisation as §11.2, at a different boundary. Three points:

**`always_2d=True` removes a branch.** Mono and stereo both come back 2-D, so
`data.mean(axis=1)` works for both without an `if`.

**Resample only when needed.** `librosa.resample` is not free, and resampling
16 kHz to 16 kHz is pure waste.

**`np.ascontiguousarray`** guarantees a contiguous buffer. `mean(axis=1)` and
`resample` usually return one, but "usually" is how you get a confusing
`torch.from_numpy` failure later (§11.2).

**Returning the rate** even though it is always `target_sr` means the caller
never has to assume. A function that returns what it guarantees is easier to
use correctly than one whose contract lives in a docstring.

---

## 14.4 Integrity validation

```python
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
```

`dtype=np.float64` inside `np.square` is not decoration. Squaring float32
values near the format's limits overflows to `inf`, and a mean of `inf` is
`inf`. Promoting to float64 first avoids it. A subtle numerical detail in a
four-line function.

```python
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
```

Three distinct reasons, because they call for different responses: `missing`
means your provisioning is incomplete, `undecodable` means a corrupt download,
`silent` means a genuine but useless recording.

```python
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
```

**Two validation depths, with the trade stated.** Listing 2000 rows for a table
cannot decode 2000 files — that is minutes. But an evaluation batch *must*,
because a corrupt file scored as a model error corrupts the measurement (§13.3).

Cheap for display, thorough for measurement. The cost/benefit differs, so the
depth differs, and the docstring says which is which.

---

## 14.5 Streaming, and why it is not optional

```python
class DatasetLoader(ABC):
    """Common interface every corpus loader exposes.

    Subclasses implement :meth:`iter_metadata` as a lazy generator so large
    corpora stream instead of being materialized; the base class layers
    sub-sampling and standardized audio loading on top of that one method.
    """

    @abstractmethod
    def iter_metadata(self) -> Iterator[SampleMetadata]:
        """Yield each sample's metadata lazily, in catalog order."""
        raise NotImplementedError
```

The **template method** pattern: subclasses implement exactly one method, and
the base class builds everything else on it. `validated_stream`, `stream`,
`subsample`, `load_sample_audio` and `__iter__` are all written once.

**Laziness is a hard requirement, not a style preference.** The working
footprint across all seven corpora is bounded at ~100 GB. LibriSpeech alone is
tens of gigabytes. `iter_metadata` must be a generator or the first call
exhausts memory.

```python
    def validated_stream(self, limit=None, *, deep=True, on_reject=None) -> Iterator[SampleMetadata]:
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
```

**`limit` counts accepted, not examined.** The obvious implementation —
`islice(stream, limit)` then filter — returns fewer than `limit` when some rows
are corrupt. Asking for 50 clips and getting 43 with no explanation is a
silently degraded evaluation.

**`on_reject` is a callback** rather than logging inside. The loader does not
know whether the caller wants a count, a log line, or a report. §13.3 uses it to
aggregate by reason.

```python
    def subsample(self, n: int, seed: int = 0) -> List[SampleMetadata]:
        """Return up to ``n`` samples via reservoir sampling in a single pass.

        Bounds the working footprint on large corpora without reading the whole
        catalog into memory, and is deterministic for a given ``seed`` so runs
        are reproducible.
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
```

Reservoir sampling (§13.3): uniform random *n* from a stream of unknown length,
one pass, O(n) memory. Seeded for reproducibility. `random.Random(seed)` is a
*local* generator, so it does not disturb the global random state that other
code may depend on.

---

## 14.6 Two loader strategies

### Catalog-driven

Common Voice, ESD and others ship a CSV or TSV listing every file with its
metadata. `CsvCatalogLoader` handles them, with a column mapping:

```python
# A catalog field may be named differently across corpora (Common Voice's audio
# column is ``path``, a processed export may call it ``filename``), so each
# mapping accepts either a single column name or an ordered list of candidates —
# the first one present in a row wins.
ColumnRef = Union[str, Sequence[str]]


@dataclass
class ColumnMap:
    """Maps a corpus catalog's columns onto :class:`SampleMetadata` fields."""
```

Accepting a *list* of candidate column names is what lets one loader handle
several distributions of the same corpus. Common Voice v7 and a processed
re-export name the audio column differently; `["path", "filename"]` covers
both without a subclass.

### Filename-driven

CREMA-D and RAVDESS encode metadata **in the filename**. No catalog at all.

```python
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
```

`1001_DFA_ANG_XX.wav` is actor 1001, sentence DFA, angry, unspecified
intensity. `03-01-05-01-01-01-12.wav` is RAVDESS's seven dash-delimited fields.

**These mapping tables are the highest-risk code in the module.** They are
small, boring, and impossible to validate from inside the system. Swap `"HAP"`
and `"SAD"` and every SER evaluation on CREMA-D is wrong, in a way no test can
catch — the code runs, the labels are plausible, the accuracy number is
meaningless.

The only defence is to check them against the corpus's own published
documentation, which is why each table carries a comment saying exactly which
filename field it decodes. If you build this, verify these tables against the
source documentation and treat any change to them as a correctness change.

Note also that RAVDESS's `_RAVDESS_STATEMENT` maps the statement code to the
actual sentence, which is what gives RAVDESS a usable ASR ground truth as well
as an emotion label.

### Tolerating layout variation

```python
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
```

Corpora arrive in variants — someone extracted the archive flat, or copied only
the WAVs. The loader checks for the official layout, then for a flat one, and
only then fails with a message naming the path it expected.

`next(glob(...), None)` checks for *at least one* match without materialising
the whole listing — the cheap existence test on a generator.

**Note what it does not do:** it does not search recursively or guess. Two
known layouts, then an honest failure. Unbounded guessing would eventually find
the wrong directory.

```python
            parsed = self._parse_filename(wav_path.stem)
            if parsed is None:
                logger.warning("crema-d: skipping unparseable filename %s", wav_path.name)
                continue
```

An unparseable filename is skipped with a warning, not a crash. Corpora contain
stray files — `.DS_Store`, a README, a re-encoded duplicate — and one of them
must not abort a 7,442-file load.

`sorted(audio_dir.glob("*.wav"))` gives deterministic order, so
`subsample(n, seed=0)` returns the same samples on every machine. Without the
sort, filesystem enumeration order varies and "reproducible" quietly stops
being true.

---

## 14.7 Licences

```python
NON_COMMERCIAL_CORPORA = frozenset({"ravdess", "l2-arctic", "esd", "asvspoof-2021"})
```

Four of the seven are research-use-only, and the requirement is that they
display a licence notice on load.

```python
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
            self.name, self.license or "unknown",
        )
        self._license_notice_logged = True
```

**Once per loader instance**, not once per sample. A 7,442-sample load must not
emit 7,442 identical warnings.

It was originally implemented in one loader only. Centralising it means adding
a corpus to `NON_COMMERCIAL_CORPORA` is sufficient — you cannot forget to
implement the notice, because there is nothing to implement.

The licence also reaches the UI. `datasets.py` exposes it:

```python
@router.get(...)
def _corpus_info(name):
    """FR2.3 — licence/task-family info for one corpus, by registry name."""
    ...
    return {"license": spec.license, "task_family": ..., "non_commercial": ...}
```

And `Frontend/src/components/dataset/DatasetLicenseNotice.tsx` renders it. A
log line satisfies an auditor; a visible notice satisfies the licence.

---

## 14.8 Registering a corpus, and refusing to fabricate

```python
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
```

```python
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
```

The registry can list a corpus **before** its loader exists, and asking for it
raises with the issue id. The alternative — returning an empty iterator — would
make "this corpus is not implemented" indistinguishable from "this corpus has
no samples", and an evaluation over zero samples reports 0.0 rather than an
error.

Same disposition as everything else: **refuse, and say why.**

### Name normalisation

```python
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
```

`L2_ARCTIC`, `l2arctic`, `L2-Arctic` and `l2-arctic` all resolve. Corpus names
appear in URLs, config files, directory names and issue descriptions, each with
its own convention.

The error **lists the supported names**, built from the registry. An error that
tells you the valid options is worth several minutes of someone's time.

---

## 14.9 Dataset footprint

```python
DATASET_FOOTPRINT_LIMIT_GB: float = 100.0
```

Checked at startup:

```python
@app.on_event("startup")
async def _warn_if_dataset_footprint_over_limit() -> None:
    """FR2.2 — surface it in the logs, not just via GET /datasets/footprint,
    the moment the provisioned corpora exceed the ~100 GB working bound."""
    try:
        usage = dataset_ingestion.measure_footprint()
        total_gb = sum(usage.values()) / (1024 ** 3)
        if total_gb > settings.DATASET_FOOTPRINT_LIMIT_GB:
            logger.warning(
                "Dataset working footprint is %.1f GB, over the configured %.1f GB "
                "limit (FR2.2). Per-corpus usage: %s",
                total_gb, settings.DATASET_FOOTPRINT_LIMIT_GB, usage,
            )
    except Exception:
        logger.warning("Could not measure dataset footprint at startup", exc_info=True)
```

**Warn, do not fail.** Exceeding the budget is a capacity concern, not a
correctness one — refusing to start would be worse than the problem.

The per-corpus breakdown is included, so the warning tells you *which* corpus
to prune rather than just that you are over.

The whole hook is wrapped so a measurement failure cannot prevent startup. A
startup hook that can crash the application is a liability.

```python
DATASET_METADATA_ROW_CAP: int = 2000
```

Caps `/{dataset}/metadata` so a request cannot materialise an entire large
corpus into a JSON response.

---

## 14.10 Custom datasets

`custom_dataset_service.py` lets a user upload their own corpus, scoped per
session:

```python
def cleanup_session_datasets(session_id: str) -> bool:
```

Session scoping means two users' uploads cannot collide, and `resolve_file`
takes a `session_id` for exactly this reason (§4.7). It is also the cleanup
boundary: a session's datasets can be removed as a unit.

---

## 14.11 Summary

- Seven corpora, one `SampleMetadata` schema. `label` is polymorphic by
  `task_family`; `extra` absorbs corpus-specific fields so adding a corpus never
  changes the schema.
- `field(default_factory=dict)`, never `= {}` — a mutable default is shared by
  every instance.
- All audio standardises to 16 kHz mono float32, contiguous, with the rate
  returned rather than assumed.
- `np.square(audio, dtype=np.float64)` avoids float32 overflow to `inf` in an
  RMS calculation.
- Integrity has two depths with the trade stated: existence-only for listings,
  full decode plus silence check for evaluation, because a corrupt file scored
  as a model error corrupts the measurement.
- `iter_metadata` must be a lazy generator; the base class builds streaming,
  validation and reservoir sampling on that one method.
- `limit` counts *accepted* samples, or a corrupt catalog silently shrinks your
  evaluation.
- `on_reject` is a callback, because the loader does not know what the caller
  wants to do with a rejection.
- Two loader strategies: catalog-driven with candidate column-name lists, and
  filename-driven with code tables.
- **The filename code tables are the highest-risk code in the module** — small,
  boring, unvalidatable from inside, and a swap makes every evaluation on that
  corpus silently wrong. Verify against the corpus's own documentation.
- Tolerate two known layouts, then fail with the path you expected. Do not
  guess recursively.
- Skip unparseable filenames with a warning; corpora contain stray files.
- `sorted(glob(...))` or "reproducible sampling" is not reproducible.
- Licence notices are centralised and logged once per loader; adding a corpus
  to a frozenset is the whole implementation.
- A registered corpus with no loader raises `NotImplementedError` naming its
  issue, because empty data is indistinguishable from unimplemented.
- Name lookup normalises case, underscores and known aliases, and the error
  lists the valid options.
- Footprint is warned, not enforced, with a per-corpus breakdown; the startup
  hook cannot crash the app.

Next: [Chapter 15 — The frontend](15-frontend.md).
