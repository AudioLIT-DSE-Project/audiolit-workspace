# FR2 — Benchmark Dataset Ingestion and Management: implementation audit

**Audited:** 2026-08-19 · **Branch:** `develop` · **Auditor:** Claude Code session

Source of truth for the requirement is `docs/SRS.md` §3.1.1 (lines 487–513).
Every claim below was verified against the actual tree on `develop`, not against
Linear status or `docs/ISSUE_PLAN.md` summaries.

---

## 1. What FR2 actually requires

> **Description:** extend ECHO 1.0's custom-dataset system to ingest and manage
> the seven approved benchmark corpora. Standardised loaders shall stream or
> sub-sample large corpora to keep the active working footprint bounded, and
> shall surface licence metadata on load.

| Clause | Text (abridged) |
| :-- | :-- |
| **FR2.1** | Each corpus shall have a loader, built on the inherited dataset service, that **validates file integrity**, reads per-item metadata (speaker, language, emotion label, bona-fide/synthetic tag), and returns a **streaming iterable**. |
| **FR2.2** | Active working footprint across all datasets shall not exceed **~100 GB**; loaders shall **stream or sub-sample** large corpora (LibriSpeech, RAVDESS). |
| **FR2.3** | Per-dataset **licence metadata shall be retained**; non-commercial corpora (RAVDESS, L2-ARCTIC, ESD, ASVspoof 2021 DF) shall **display a licence notice on load**. |

Related, and consistent: SRS line 843 (`Dataset working footprint | ~100 GB |
Streaming and sub-sampling loaders; nothing materialised whole`) and SAD line 232
/ constraint **C5** (`Research-only datasets … displays the appropriate licence
notices`).

---

## 2. Verdict at a glance

| Clause | Status | One-line reason |
| :-- | :-- | :-- |
| Seven corpora have loaders | ✅ **Done** | All 7 wired in `CORPUS_REGISTRY`, all with tests. |
| Per-item metadata parsing | ✅ **Done** | `SampleMetadata` carries speaker/language/label/accent/demographic. |
| Streaming iterable | ⚠️ **Done in the core, defeated at the API** | Loaders are generators; `dataset_service.py` materialises them whole on every request. |
| **FR2.1** integrity validation | ❌ **Not wired** | `is_silent()` exists and is unit-tested but is **called from zero production code paths**. |
| **FR2.2** footprint bound | ❌ **No mechanism** | `subsample()`/`stream(limit=)` have **no production callers**; no accounting, no cap, no config. |
| **FR2.3** licence retained | ⚠️ **Partial** | Retained in the domain model, **dropped** by the API serialiser. |
| **FR2.3** licence notice displayed | ❌ **Not implemented** | Zero licence references anywhere in `Frontend/src/`. |

**Bottom line:** the *ingestion core* (LIT-123 and its loader children) is
genuinely complete and good. The *management* half of FR2 — integrity gating,
footprint bounding, licence surfacing — is either unwired or absent. Three of the
sub-clauses cannot currently be demonstrated in a review or a demo.

---

## 3. What is implemented, and how

### 3.1 The ingestion core — `Backend/app/infrastructure/dataset_ingestion.py` (980 lines)

A single, well-factored module. Design:

- **`SampleMetadata`** (`:63-84`) — one frozen dataclass, identical shape for
  every corpus: `dataset`, `sample_id`, `audio_path`, `task_family`, `label`,
  `speaker_id`, `accent`, `language`, `license`, `demographic`, `extra`.
  `label` is polymorphic by task family: transcript (ASR), emotion class (SER),
  `bona-fide`/`spoof` (deepfake). This directly satisfies FR2.1's "reads per-item
  metadata (speaker, language, emotion label, bona-fide/synthetic tag)".
- **`load_standardized_audio`** (`:87-107`) — soundfile read → channel-average
  down-mix → librosa resample only when rates differ → contiguous float32 at
  16 kHz. soundfile-only, per project rule (torchaudio removed in LIT-226). ✅
- **`DatasetLoader`** ABC (`:124-174`) — subclasses implement one method,
  `iter_metadata()`, as a lazy generator. The base layers `stream(limit=)`
  (`:142-145`, `islice`) and `subsample(n, seed)` (`:147-165`, single-pass
  reservoir sampling, deterministic per seed) on top. Correct design for FR2.2.
- **`CsvCatalogLoader`** (`:203-291`) — generic catalog-driven loader with a
  `ColumnMap` (`:184-200`) supporting *ordered candidate column names*, so the
  raw Common Voice `.tsv` schema (`path`, `accents`) and the repo's processed
  `_metadata.csv` export (`filename`, `accent`) both resolve through one mapping.
  Column lookup is case-insensitive; `encoding` is injectable (ESD's BOM fix).
- **`CORPUS_REGISTRY`** (`:932-940`) — the seven approved corpora, each tagged
  with `TaskFamily` and licence string, resolved via `get_loader()` (`:967-980`).
  `get_corpus_spec()` (`:948-964`) normalises aliases (`l2arctic` → `l2-arctic`).
  Unwired corpora raise `NotImplementedError` naming the owning issue rather than
  returning empty data — the right anti-fabrication stance.

### 3.2 The seven loaders — all present and registered

| Corpus | Class | Line | Strategy | Issue |
| :-- | :-- | :-- | :-- | :-- |
| Common Voice | `CommonVoiceLoader` | `:294` | CSV/TSV catalog, dual schema | LIT-141 |
| LibriSpeech | `LibriSpeechLoader` | `:337` | walks `*.trans.txt` + `.flac`; gender from `SPEAKERS.TXT` | LIT-141 |
| ASVspoof 2021 DF | `ASVspoofLoader` | `:411` | protocol scan for `bonafide`/`spoof` token; handles 2021-DF **and** 2019-LA layouts | LIT-142 |
| L2-ARCTIC | `L2ArcticLoader` | `:519` | per-speaker `wav/` + `transcript/`; fixed 24-speaker → L1 accent map | LIT-181 |
| CREMA-D | `CremaDLoader` | `:614` | filename encoding `<actor>_<sent>_<emo>_<int>`; demographics CSV | LIT-208 |
| RAVDESS | `RavdessLoader` | `:709` | 7-field dash filename; **excludes the song channel**; gender by actor parity | LIT-208 |
| ESD | `ESDLoader` | `:815` | CSV catalog, `utf-8-sig` for BOM header | LIT-236 |

Two details worth calling out as genuinely good work, not boilerplate:

- **`EMOTION_LABELS`** (`:588-590`) deliberately matches the SER checkpoint's own
  output vocabulary (`fearful`/`surprised`, not `fear`/`surprise`). The comment
  explains why: a vocabulary mismatch silently scores every clip of that class
  wrong. ESD's Title-case `Surprise` is remapped for exactly this reason
  (`:805-811`, `:864-878`).
- **RAVDESS song-channel exclusion** (`:759-760`) — the song channel would
  otherwise be scored as speech.

### 3.3 Wiring into the live API (LIT-235)

- `GET /datasets/list` — `Backend/app/api/routes/datasets.py:29-40`. Lists
  registry corpora, filtered by `_UNLOADABLE_CORPORA` (`:23-26`).
- `GET /{dataset}/metadata` — `datasets.py:43-56` → `dataset_service.load_metadata`.
- `GET /{dataset}/file/{path}` — `datasets.py:59-118`. `FileResponse` with
  `Accept-Ranges: bytes`, so range requests work.
- Bridge into the registry: `dataset_service._registry_metadata_rows` (`:37-60`)
  and `_registry_resolve_file` (`:63-81`).
- Frontend consumes `/datasets/list` in `Toolbar.tsx:163-171`, with friendly
  labels at `:102-110`.

### 3.4 Inherited ECHO custom-dataset system (the base FR2 extends)

Intact and untouched: `custom_dataset_service.py` (226 lines) +
`api/routes/dataset_management.py` (316 lines) — create, upload, list, metadata,
files, delete, serve, session cleanup. FR2's "extends rather than replaces"
framing holds.

### 3.5 Test coverage — better than expected

| File | Lines | Covers |
| :-- | :-- | :-- |
| `test_dataset_ingestion.py` | 372 | standardisation, `CsvCatalogLoader`, `stream`/`subsample`, registry, Common Voice, ESD |
| `test_ser_corpora.py` | — | CREMA-D + RAVDESS (26 references) |
| `test_asvspoof_loader.py` | — | protocol parsing, subsample determinism |
| `test_l2arctic_loader.py` | — | speaker/L1 mapping, subsample |
| `test_librispeech_loader.py` | — | tree walk, `SPEAKERS.TXT`, **`is_silent` unit tests** |
| `test_datasets_routes.py` | 53 | `/datasets/list`, route-shadowing guard |
| `test_dataset_service.py` | 115 | metadata/resolve paths |

All seven loaders have tests. This is the strongest part of FR2's delivery.

---

## 4. Gaps, by clause

### ❌ 4.1 — FR2.1 "validates file integrity" is not wired (High)

`is_silent()` (`dataset_ingestion.py:113-121`) is implemented and has four unit
tests. A repo-wide grep for `is_silent|SILENCE_RMS_FLOOR` across `Backend/`
returns **only its definition and its tests** — no production caller.

Consequences:
- A truncated, zero-length, or all-silence clip flows into an evaluation batch
  and is scored as a genuine miss. On ASR that is a 100% WER row attributed to
  the model; on ADD it is a false spoof/bona-fide decision. Both silently
  corrupt FR15 bias numbers and FR7 deepfake metrics.
- No decode check either: nothing verifies a listed `audio_path` is readable
  before it enters a batch. `_registry_resolve_file` checks `.exists()` only at
  HTTP serve time (`dataset_service.py:78-80`), which is far too late and covers
  only the browser-playback path, not the inference path.

**What "integrity" should mean here:** (a) the file exists and decodes; (b) it is
not empty and not below the silence floor; (c) its duration is within a sane
band. None of the three is enforced.

### ❌ 4.2 — FR2.2 footprint bound has no mechanism at all (High)

- A grep for `footprint|quota|disk_usage|max_size|MAX_DATASET` across
  `Backend/app/` returns two *comments* and nothing executable.
- `subsample()` and `stream(limit=)` have **no production callers** — only tests,
  plus a passing mention in a comment in `accent_bias_profiler.py:71`.
- There is no configured cap, no per-corpus sample budget, and nothing that would
  notice or refuse if `Backend/data/` grew past 100 GB.

Today the tree is only 130 MB (`Backend/data/`: `asvspoof2021_df`,
`common_voice_valid_dev`, `crema_d`, `esd`, `l2arctic`, `ravdess`) so nothing is
*breached* — but the requirement asks for a bound, and there is none. The moment
anyone provisions full LibriSpeech (~60 GB) the clause becomes live with no guard
behind it.

**And the API actively works against the streaming design:**
`_registry_metadata_rows` (`dataset_service.py:37-60`) does
`for sample in loader.iter_metadata(): rows.append(...)` — it materialises the
**entire catalog** into a list on **every** `/{dataset}/metadata` request. That is
precisely what SRS line 843 forbids ("nothing materialised whole"). The lazy
generator design in the core is real; it is discarded one layer up.

### ⚠️ 4.3 — FR2.3 licence is retained but never surfaced (High)

**Retained:** ✅ — `CorpusSpec.license` (`:912-926`), `SampleMetadata.license`
(`:82`), and per-corpus constants (`ASVSPOOF_LICENSE`, `L2_ARCTIC_LICENSE`,
`CREMA_D_LICENSE`, `RAVDESS_LICENSE`, `ESD_LICENSE`).

**Displayed:** ❌ — three independent failures:

1. `_registry_metadata_rows` builds rows containing only `filename`, `label`,
   `speaker_id`, `accent` (`dataset_service.py:50-59`). It **drops** `license`,
   `language`, `demographic`, and `extra`. The licence never crosses the API
   boundary.
2. `GET /datasets/list` returns bare name strings (`datasets.py:36-40`) — no
   licence, no task family.
3. `grep -rni "licen[cs]e" Frontend/src/` returns **nothing**. There is no
   licence notice component, banner, badge, or modal anywhere in the UI.

The only licence signalling that exists is `ASVspoofLoader`'s one-time
`logger.warning` (`:453-458`) — server-side, into the log, invisible to the user.
The other three non-commercial corpora named in FR2.3 (**RAVDESS, L2-ARCTIC,
ESD**) emit nothing at all.

This is also a **SAD C5 compliance gap**, not just an FR gap: SAD line 232 states
the system "displays the appropriate licence notices." It does not.

---

## 5. Additional defects found during the audit

These are outside the literal clause text but sit inside FR2's blast radius.

### 5.1 LibriSpeech is offered in the UI but has no data (Medium — user-visible break)

`_UNLOADABLE_CORPORA` (`datasets.py:23-26`) filters on `spec.loader_factory is
None` — i.e. it tests *code wiring*, not *data availability*. `librispeech` has a
factory, so it passes the filter and appears in the dropdown. But
`Backend/data/librispeech/` **does not exist** on this machine, so
`LibriSpeechLoader.iter_metadata` raises `FileNotFoundError` (`:364-367`) →
`load_metadata` → 404 on `/{dataset}/metadata`.

Net effect: a user picks "LibriSpeech" from the dropdown and gets a failure. The
list should reflect what can actually be served.

### 5.2 Dead code: the RAVDESS duration branch can never execute (Low)

`dataset_service.py:136-151` is guarded by `if ds == "ravdess"`, but it sits
*after* the early return at `:120-121` (`if ds not in DATASET_PATHS: return
_registry_metadata_rows(ds)`), and `DATASET_PATHS` contains only `common-voice`
and `cv-valid-dev` (`:25-28`). So `ds` can never be `"ravdess"` at line 136.
Worse, if it somehow were, `DATASET_BASE_DIRS[ds]` at `:141` would `KeyError` —
`ravdess` is absent from that dict too.

Left over from the pre-LIT-235 hardcoded era. Should be deleted.

### 5.3 Duration is lost for every registry-backed corpus (Low–Medium)

`AudioDatasetPanel.tsx:19` declares `duration?: number`, and
`calculate_audio_duration` exists (`dataset_service.py:86-97`) — but
`_registry_metadata_rows` never emits it. So every corpus except Common Voice
shows no duration. Fixing 5.2 by deleting the dead branch should be paired with
emitting duration properly (lazily — see §6.3, it's an `sf.info` header read per
row, which must not happen inline on a full-catalog request).

### 5.4 Two code paths for Common Voice (Low)

`DATASET_PATHS`/`DATASET_BASE_DIRS` (`:25-34`) special-case `common-voice` with a
hand-rolled `csv.DictReader` walk (`:127-153`), duplicating what
`CommonVoiceLoader` already does — and the two produce *different row shapes*
(the special case passes every raw CSV column through; the registry path emits
four fixed keys). Collapsing onto the registry path removes a whole class of
"works for Common Voice, breaks for everything else" bugs.

### 5.5 `_registry_resolve_file` is O(N) per audio file (Medium — performance)

`dataset_service.py:63-81` re-instantiates the loader and linear-scans the entire
catalog to resolve **one** filename. Rendering a 100-row table therefore triggers
100 full catalog walks. For a directory-walking loader like LibriSpeech that is
100 recursive `rglob` traversals. The in-code comment acknowledges this
("revisit with an index if a corpus grows large enough") — with FR2.2 explicitly
contemplating LibriSpeech-scale corpora, it has.

### 5.6 `esd` missing from the frontend label map (Trivial)

`DATASET_LABELS` (`Toolbar.tsx:102-110`) has no `esd` entry, so it renders as the
raw registry name `esd` rather than `ESD`. Also note the hardcoded fallback list
at `Toolbar.tsx:~140` duplicates the registry — it will drift.

---

## 6. Remediation plan

Ordered by requirement risk. Each item is scoped to be independently
reviewable — one issue, one branch, one PR, per the project branch model.

### 6.1 — P0: Wire integrity validation into the load path (FR2.1)

**Where:** `dataset_ingestion.py` — the `DatasetLoader` base class.

Add a validation step callers opt into, rather than silently mutating existing
iteration semantics (existing tests assert current `iter_metadata` behaviour):

```python
@dataclass(frozen=True)
class IntegrityReport:
    sample_id: str
    ok: bool
    reason: Optional[str] = None      # "missing" | "undecodable" | "silent" | "too-short"

def validated_stream(
    self,
    limit: Optional[int] = None,
    *,
    on_reject: Optional[Callable[[IntegrityReport], None]] = None,
) -> Iterator[SampleMetadata]:
    """Stream only samples whose audio exists, decodes, and carries signal."""
    for meta in self.stream(limit=limit):
        report = self.check_integrity(meta)
        if report.ok:
            yield meta
        elif on_reject is not None:
            on_reject(report)

def check_integrity(self, meta: SampleMetadata) -> IntegrityReport:
    if not meta.audio_path.exists():
        return IntegrityReport(meta.sample_id, False, "missing")
    try:
        audio, _ = self.load_sample_audio(meta)
    except Exception:                       # soundfile raises several types
        return IntegrityReport(meta.sample_id, False, "undecodable")
    if is_silent(audio):
        return IntegrityReport(meta.sample_id, False, "silent")
    return IntegrityReport(meta.sample_id, True)
```

Then **make the evaluation/inference paths call `validated_stream`** — that is the
part that actually closes the clause. Grep for `iter_metadata` consumers in
`app/domain/evaluation_service.py` and `app/domain/accent_bias_runner.py` and
switch them over. Surface the rejection count in the run result so a user can see
"12 clips excluded (3 missing, 9 silent)" rather than absorbing them as model
error.

**Cheap-check variant:** `check_integrity` decodes the file, which is expensive
for a metadata listing. Offer a `deep=False` mode that uses `sf.info()` (header
read only — existence, duration, non-zero frames) for listing paths, and full
decode only on the evaluation path.

**Tests:** truncated file, zero-byte file, all-silence WAV, missing path,
happy path; plus an assertion that a rejected sample never reaches a batch.

### 6.2 — P0: Surface licence metadata end-to-end (FR2.3, SAD C5)

Three small changes, one PR:

**(a) Enrich `GET /datasets/list`** — `datasets.py:29-40`:

```python
corpora = [
    {
        "name": name,
        "task_family": spec.task_family.value,
        "license": spec.license,
        "non_commercial": name in NON_COMMERCIAL_CORPORA,
    }
    for name, spec in sorted(dataset_ingestion.CORPUS_REGISTRY.items())
    if name not in _UNLOADABLE_CORPORA
]
```

Keep a `datasets` key of bare names alongside for one release so `Toolbar.tsx`
does not break, or update both in the same PR (preferred — single repo).

Define the non-commercial set explicitly from FR2.3's own list, in
`dataset_ingestion.py` next to the licence constants:

```python
#: FR2.3 — corpora requiring a user-visible licence notice on load.
NON_COMMERCIAL_CORPORA = frozenset({"ravdess", "l2-arctic", "esd", "asvspoof-2021"})
```

**(b) Stop dropping licence in the row serialiser** — `dataset_service.py:50-59`:
add `license` and `language` to the emitted row (`demographic`/`extra` can stay
out of the table, but licence is the FR clause).

**(c) Display the notice** — new
`Frontend/src/components/dataset/DatasetLicenseNotice.tsx`, rendered when the
selected dataset is in the non-commercial set. A dismissible banner above the
dataset table stating corpus name + licence string + "research/non-commercial use
only". This is the component that actually satisfies "display a licence notice on
load" and SAD C5; without it the clause stays open no matter what the API returns.

**Also:** promote `ASVspoofLoader`'s one-off `logger.warning` (`:453-458`) into a
shared base-class hook so **all four** non-commercial corpora log consistently,
rather than one corpus doing it by hand.

### 6.3 — P1: Make the API honour the streaming design (FR2.2)

`_registry_metadata_rows` must stop materialising whole catalogs.

- **Paginate `/{dataset}/metadata`**: accept `?limit=&offset=` and drive it with
  `islice(loader.iter_metadata(), offset, offset + limit)`. Default `limit` to
  something demo-sane (200). Return `{"rows": [...], "limit":, "offset":,
  "has_more": bool}`.
- **Cap unbounded requests**: a request with no `limit` should apply a hard
  ceiling rather than walking a 60 GB corpus.
- Update `AudioDatasetPanel.tsx` (`:338`, `:395`) to pass the window and handle
  `has_more`.

### 6.4 — P1: Add a real footprint budget (FR2.2)

FR2.2 asks for a bound "across all datasets". Implement it as configuration plus
a check, not as a comment:

- Add to `app/infrastructure/settings.py`:
  `dataset_footprint_limit_gb: float = 100.0`, `dataset_default_sample_cap: int`.
- Add `dataset_ingestion.measure_footprint() -> dict[str, int]` — per-corpus
  bytes under `DATA_DIR`, summed.
- Expose `GET /datasets/footprint` returning per-corpus bytes, the total, and the
  limit — this is what makes the clause *demonstrable* in a review, which today
  it is not.
- Log a warning at startup if the total exceeds the limit.
- Have the evaluation path default to `subsample(settings.dataset_default_sample_cap)`
  for corpora flagged large, so the existing (good) sub-sampling code finally has
  a production caller.

**Provisioning note:** document in `docs/README.md` that LibriSpeech should be
provisioned as a bounded subset (e.g. `test-clean`, ~350 MB) rather than the full
distribution — FR2.2's "sub-sample large corpora (LibriSpeech, RAVDESS)" is as
much an ops instruction as a code one.

### 6.5 — P1: Fix the LibriSpeech-not-provisioned break (§5.1)

Make the exclusion filter test data availability, not just code wiring:

```python
def is_available(spec: CorpusSpec) -> bool:
    if spec.loader_factory is None:
        return False
    try:
        loader = spec.loader_factory()
        next(iter(loader.stream(limit=1)))   # one lazy probe, not a full walk
        return True
    except (FileNotFoundError, StopIteration, NotImplementedError):
        return False
```

Cache the result (a module-level dict refreshed at startup) so this is not
recomputed per request. Better still, return the corpus with an
`"available": false` flag and let the UI grey it out with "data not provisioned"
— more honest than silently hiding it, and it tells a developer why their corpus
vanished.

### 6.6 — P2: Cleanups

- Delete the dead RAVDESS duration branch (`dataset_service.py:136-151`) — §5.2.
- Collapse the Common Voice special case onto the registry path — §5.4. Removes
  `DATASET_PATHS`/`DATASET_BASE_DIRS` entirely and unifies row shape.
- Index-backed `_registry_resolve_file` — §5.5. Build `{filename: path}` once per
  corpus, cache it (invalidate on catalog mtime). Turns O(N) per file into O(1).
- Emit `duration` from the registry serialiser via `sf.info()` header read, only
  on paginated windows so it stays cheap — §5.3.
- Add `esd: "ESD"` to `DATASET_LABELS` and drop the hardcoded fallback corpus
  list in `Toolbar.tsx` — §5.6.

---

## 7. What is already good and should not be "improved"

Worth stating explicitly so a follow-up pass does not churn it:

- The `ColumnMap` candidate-list design is the right abstraction for corpora that
  ship in multiple schema variants. Leave it.
- Reservoir sampling in `subsample()` is correct, single-pass, and deterministic.
  The problem is that nothing calls it — not the algorithm.
- The `EMOTION_LABELS` ↔ SER-checkpoint vocabulary alignment and the ESD
  Title-case remap are subtle correctness work with the reasoning captured in
  comments. Do not "simplify" these without reading those comments.
- `get_loader` raising `NotImplementedError` with the owning issue name, instead
  of returning empty data, is the correct anti-fabrication behaviour and matches
  the project's stated stance.
- Structural-variant tolerance (CREMA-D flat vs `AudioWAV/`, RAVDESS flat vs
  `Actor_*/`) reflects how these corpora actually arrive in the wild.

---

## 8. Suggested issue breakdown

| Proposed issue | Clause | Priority | Est. | Files |
| :-- | :-- | :-- | :-- | :-- |
| Wire integrity validation into load + eval paths | FR2.1 | **P0** | M | `dataset_ingestion.py`, `evaluation_service.py`, `accent_bias_runner.py` |
| Surface licence metadata end-to-end + UI notice | FR2.3, SAD C5 | **P0** | M | `datasets.py`, `dataset_service.py`, new `DatasetLicenseNotice.tsx`, `Toolbar.tsx` |
| Paginate `/{dataset}/metadata`; stop materialising catalogs | FR2.2 | **P1** | M | `dataset_service.py`, `datasets.py`, `AudioDatasetPanel.tsx` |
| Footprint budget: settings + `measure_footprint()` + endpoint | FR2.2 | **P1** | S | `settings.py`, `dataset_ingestion.py`, `datasets.py` |
| Dataset availability probe in `/datasets/list` | §5.1 | **P1** | S | `datasets.py`, `Toolbar.tsx` |
| Dataset-service cleanup (dead branch, CV special case, resolve index) | §5.2/5.4/5.5 | **P2** | M | `dataset_service.py` |

All six are additive to a complete, tested core — none require reworking the
ingestion layer.

---

## 9. Verification notes

Claims in this document were checked directly against the tree, per
`CLAUDE.md`'s "verify against the actual source" rule:

- `grep -rn "is_silent|SILENCE_RMS_FLOOR" Backend --include="*.py"` → definition
  + tests only, no production caller.
- `grep -rn "subsample|\.stream(" Backend --include="*.py"` → tests only, plus
  one comment in `accent_bias_profiler.py:71`.
- `grep -rni "licen[cs]e" Backend/app/api Backend/app/domain Backend/app/orchestration` → **no matches**.
- `grep -rni "licen[cs]e" Frontend/src/` → **no matches**.
- `grep -rni "footprint|quota|disk_usage|max_size" Backend/app --include="*.py"` → two comments, no code.
- `ls Backend/data/` → six corpora present, `librispeech` absent; `du -sh` → 130M.
- `docs/SRS.md:487-513` read in full for the clause text.

**Correction to `CLAUDE.md`:** the repo-structure section states `app/core/` and
`app/services/` are "both gone". On `develop` today, `Backend/app/core/` still
contains `redis.py`, and `Backend/app/services/` still exists (empty but for
`__pycache__`). Not an FR2 issue, but it is exactly the doc-drift failure mode
`CLAUDE.md` itself warns about — worth a separate cleanup and a doc fix.

---

## 10. Summary

FR2's **ingestion** half is delivered to a high standard: seven loaders, one
coherent metadata schema, correct 16 kHz-mono standardisation, lazy generators,
deterministic sub-sampling, and real test coverage for every corpus.

FR2's **management** half is not. Integrity validation is written but unwired;
the footprint bound exists only as prose; licence metadata is retained in the
domain model but dropped at the API boundary and absent from the UI entirely. Two
of the three numbered sub-clauses (FR2.1's validation duty, FR2.2's bound) and
half of the third (FR2.3's display duty) cannot currently be demonstrated.

The good news: the core's design anticipated all three. `is_silent`, `subsample`,
`stream(limit=)`, and `SampleMetadata.license` are already there and already
tested. Closing FR2 is mostly a matter of **calling code that already exists**
from the paths that need it, plus one new frontend component.
