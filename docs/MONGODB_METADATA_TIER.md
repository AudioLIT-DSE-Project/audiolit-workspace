# MongoDB 6.0+ Metadata Tier — Operations & Degradation Guide

> **Documents the durable MongoDB metadata tier** (SRS §3.10, SAD §9, constraints
> C4/SR4). Implemented across **LIT-256** (store module + provisioning),
> **LIT-257** (durable write-through for Tier 2/3 records) and **LIT-258**
> (test tier, graceful degradation, this guide). Table of contents and errata in
> [`docs/README.md`](README.md).

## Two stores, two lifetimes

AudioLIT keeps **Redis** and **MongoDB** side by side (SAD §9) with deliberately
different lifetimes:

| Store | Holds | Lifetime | Losing it means |
|-------|-------|----------|-----------------|
| **Redis** | recomputable cache + job queue | transient, TTL'd | results are recomputed; the system keeps working |
| **MongoDB** | durable model / analysis / bias records | bias reports permanent; analyses 24 h | reproducible history and bias evidence — it must survive |

The two are separated by one rule (SAD §9): *if it can be recreated on demand it
goes in Redis; if it must be kept so a result can be reproduced or a bias finding
referred back to, it goes in MongoDB.* A consequence of the design is that
MongoDB stores only a **reference** (`redis_tensor_key`) to where a result is
cached, never the large result itself — so a small cache supports a large
history, and **no store ever holds audio bytes** (constraint C4 / SR4).

## Collections

Provisioned by `MetadataStore.ensure_schema()` with JSON-schema document
validation and indexes (SRS §3.10):

| Collection | Documents | ID / uniqueness | Retention | Notes |
|------------|-----------|-----------------|-----------|-------|
| `models` | a model registration + reproducibility record | unique `model_id` (`hf_model_id@revision`) | permanent | revision + `weight_digest` let a model be reproduced exactly |
| `audio_samples` | sample *metadata*, path-reference only | unique `sample_id` (SHA-256 of path) | permanent | `file_path_reference` — never `audio_bytes` (C4/SR4) |
| `analysis_results` | one record per task run | unique `analysis_id` (`cache_key:task`) | **TTL 24 h** | `redis_tensor_key`, not the tensor; TTL via index on `created_at` |
| `bias_reports` | one report per accent cohort | unique `report_id` | permanent | no TTL index — the SAD §9 "kept permanently" case |

Compound indexes back the cross-collection joins in the SRS figure
(`analysis_results`: `sample_id`+`model_id`; `bias_reports`: `model_id`+`cohort`).

## Write-through points (where records enter the tier)

Three call sites persist durable records:

1. **Model load** — `model_registry_service._record_model_load()` writes a
   `models` document every time `download_and_load()` finishes, recording the
   resolved revision and the SHA-256 digest of the safetensors weights.
2. **Analysis fan-in** — `task_orchestrator._write_analysis_metadata()` runs in
   the multi-task aggregator **before** the Redis cache write (SAD §6.2 write
   order). It records the sample once and one `analysis_results` document per
   finished task. Large array payloads (saliency/attention heatmaps, embedding
   vectors) are stripped by `_strip_array_fields` before persisting (`_MAX_PERSISTED_LIST_LEN`
   bound; anything NumPy/tensor-shaped is dropped) — tensors stay in Redis under
   `redis_tensor_key`.
3. **Accent-bias diagnostic** — `task_orchestrator._write_bias_report()` writes
   one permanent `bias_reports` document per cohort after an accent-bias run.

## Configuration

| Setting | Env var | Default | Meaning |
|---------|---------|---------|---------|
| `MONGO_URL` | `MONGO_URL` | `""` | **Empty = tier off.** `get_metadata_store()` returns `None` and every call site skips writes silently. |
| `MONGO_DB_NAME` | `MONGO_DB_NAME` | `audiolit` | database to read/write |
| `MONGO_ANALYSIS_TTL_HOURS` | `MONGO_ANALYSIS_TTL_HOURS` | `24` | analysis-record TTL (hours) |
| `MONGO_SERVER_SELECTION_TIMEOUT_MS` | `MONGO_SERVER_SELECTION_TIMEOUT_MS` | `1500` | bounds the degradation ping so a dead server can't hang a worker |

All in `Backend/app/infrastructure/settings.py` (pydantic-settings). Enable the
tier by setting `MONGO_URL` (e.g. `mongodb://127.0.0.1:27017`).

## Graceful degradation (SRS §3.3.1 / SAD §11.1)

MongoDB is a **supporting record store, not a request-path dependency**. There
are three distinct degraded states, all verified in `test_metadata_store.py`
and the write-through test classes (`LIT-258`):

| State | `available` | Writes | Reads |
|-------|-------------|--------|-------|
| **Configured off** (`MONGO_URL` empty) | — | never attempted (`get_metadata_store()` = `None`) | `None`/`[]` |
| **Server unreachable** | `False` (bounded ping) | `False` — logged, never raised | `None`/`[]` — logged, never raised |
| **Driver missing** (pymongo not installed) | `False` | `False` (preflight short-circuits) | `None`/`[]` |

Every write is wrapped in try/except at the write-through site too — a failed
*individual* record (e.g. the sample write) never blocks the *other* records of
the same fan-in, and a metadata failure can never fail the analysis itself. The
single area that intentionally does **not** degrade is provisioning:
`ensure_schema()` expects a live server at setup time (that's how a misconfigured
deployment announces itself), while all runtime reads/writes are degradation-safe.

## Testing

Unit tests use **mongomock**, mirroring how `fakeredis` backs the Redis tier —
no live server needed:

```bash
cd Backend
pytest tests/test_metadata_store.py              # store CRUD + degradation matrix
pytest tests/test_task_orchestrator.py           # fan-in write-through + strip/bias degrades
pytest tests/test_model_registry_service.py      # model-load write-through + degrade
```

The degradation matrix above is exercised by `TestGracefulDegradation`
(dead-server and missing-driver fakes), `TestWriteAnalysisMetadataEdgeCases`,
`TestBiasReportWriteDegrades` and the model-registry failing-store tests.

For a local smoke test against a real server, run a temporary instance and point
the tier at it:

```bash
docker run -d --name lit-mongo --rm -p 27017:27017 mongo:6
MONGO_URL=mongodb://127.0.0.1:27017 pytest tests/test_metadata_store.py
```

> Docker Compose integration for the full stack (including the MongoDB
> container from SAD §7) is tracked by **LIT-203** (containerisation), not here.

## Code map

| File | Role |
|------|------|
| `Backend/app/infrastructure/metadata_store.py` | `MetadataStore` — schemas, indexes, provisioning, CRUD, degradation |
| `Backend/app/infrastructure/settings.py` | configuration (section above) |
| `Backend/app/domain/model_registry_service.py` | `_record_model_load` write-through |
| `Backend/app/orchestration/task_orchestrator.py` | `_write_analysis_metadata`, `_write_bias_report`, `_strip_array_fields`, `_sample_id`, `_audio_sample_details` |
| `Backend/tests/test_metadata_store.py` | store-level tests + `TestGracefulDegradation` |
| `Backend/tests/test_task_orchestrator.py` | aggregation/bias write-through + edge-case degradation tests |
| `Backend/tests/test_model_registry_service.py` | model-load write-through + degradation test |