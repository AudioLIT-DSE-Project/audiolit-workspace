# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AudioLIT is an interpretability workbench for speech models: ASR (Whisper), speech emotion recognition (Wav2Vec2) and audio deepfake detection, with attribution overlays (Grad-CAM, Integrated Gradients, LIME, SHAP), perturbation, latent projection, accent-bias profiling and faithfulness auditing. FastAPI + Redis + RQ backend in `Backend/`, React 18 + TypeScript + Vite frontend in `Frontend/`. It extends the open-source ECHO 1.0 baseline, so some code and tests are inherited and say "LIT for Voice".

`README.md` and `DEPLOYMENT.md` are the client-facing install and operations guides. `DEPLOYMENT.md` is the reference for every environment variable and for the operational reasoning behind the compose settings.

## Branches and what may be committed

- `main` is the production tree shipped to a client. `develop` is the integration branch; feature branches are `feature/lit-<n>-...`, one per Linear issue (team `LIT`). CI runs on `main`, `develop` and `testing`.
- `release/*` branches and `main` are a **stripped** tree. `.github/workflows/main-hygiene.yml` fails any push or PR to `main` that contains `docs/`, `plans/`, `scratch/`, `scripts/` (repo root), `CLAUDE.md`, `GEMINI.md`, `AGENTS.md`, `.cursorrules`, `.DS_Store` or `newman-report.html`. **Do not commit this file on a branch headed for `main`.** `develop` carries its own tracked `CLAUDE.md` plus the SAD/SRS under `docs/`; code comments citing `SAD §…`, `SRS §…`, `FR…` and `LIT-…` refer to those documents, which are not present on a release tree.
- The same workflow requires `README.md` to keep `Requirements`, `Install` and `Verify` sections.

## Commands

### Full stack (Docker)

```bash
docker compose up --build -d                                              # web :8080, api :8000, worker, redis, mongo
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d   # workers get the GPU
curl -s http://127.0.0.1:8000/health/workers                              # must list asr, ser, add, xai, mutation
```

Open `http://127.0.0.1:8080`, not `localhost`: the web image is built with `VITE_API_BASE_URL=http://127.0.0.1:8000`, and the `SameSite=Lax` session cookie is dropped when the UI and API host names differ.

### Native development

```bash
# Redis :6379 and Mongo :27017 only
docker compose -f Backend/docker-compose.yml up -d

# Backend (Python 3.11 in the image, 3.10 in CI); run from Backend/
cd Backend
pip install -r requirements-dev.txt        # includes requirements.txt
uvicorn app.main:app --reload --port 8000
python -m app.orchestration.worker all     # or one family: asr | ser | add | xai | mutation

# Frontend (Node 20); dev server on :8080
cd Frontend
npm ci
npm run dev
```

With no `VITE_API_BASE_URL`, the frontend targets `http://localhost:8000`, so browse the dev server at `http://localhost:8080` to keep the cookie same-site. `MONGO_URL` is empty by default, which turns the metadata tier into logged no-ops.

### Backend tests

```bash
cd Backend
pytest                                                        # whole suite, fakeredis, no services needed
pytest tests/test_task_orchestrator.py                        # one file
pytest tests/test_task_orchestrator.py::TestName::test_case   # one test
pytest -m "not slow"                                          # markers are strict; declared in pytest.ini
REDIS_URL="redis://127.0.0.1:1/0" pytest -q                   # prove nothing depends on a real Redis, as in CI
```

- Tests must be run on the host from `Backend/`. The production image copies only `Backend/app` and installs only `requirements.txt`, so there is no pytest and no `tests/` inside the `api` container.
- CI has no Redis. `conftest.py` autouse-patches the async client (`app.infrastructure.redis.redis`) with fakeredis. Anything that reaches the synchronous RQ connection, including a direct call to a task function (they call `publish_progress`), needs the `broker` fixture pattern from `tests/test_task_orchestrator.py`, which patches `rq_connection._CONNECTION`. A locally running Redis hides this mistake.
- `pytest.ini` sets a 300 s per-test timeout because burst-mode `SimpleWorker` drains of dependency-gated jobs on fakeredis can hang. If an orchestrator test is flaky, run the children and call the aggregator function directly instead of draining it as a dependent job.
- `tests/README.md` and `tests/run_tests.py` are inherited from ECHO and only know four of the test files; use `pytest` directly.

### Frontend checks (the CI order)

```bash
cd Frontend
npm run typecheck      # tsc -p tsconfig.app.json; a bare `tsc --noEmit` checks zero files
npm run lint
npm test               # Jest + Testing Library
npx jest src/tests/XAIOverlayCanvas.test.tsx -t "name"    # one file / one test
npm run build
npm run test:e2e                                  # Playwright layout + quick-start, 3 browsers, no backend
npx playwright test --project=accessibility       # axe WCAG 2.1 AA, no backend
npm run test:e2e:dataflow                         # needs the full stack running; not in CI
npx playwright test e2e/layout.spec.ts --project=chromium
```

`vite build` does not typecheck and Jest only compiles files a test imports, so `npm run typecheck` is the only gate that catches a type error in an untested component.

### Other gates CI enforces

- `npm audit --omit=dev --audit-level=high` and `pip-audit` (the `lodash` override in `package.json` and several version floors in the requirements files exist to keep these clean).
- Both Docker images are built from the repo root and scanned with Trivy at CRITICAL.
- API contract: `npx newman run Backend/apitests/AudioLIT.postman_collection.json --env-var baseUrl=http://127.0.0.1:8000` against a running stack. Load tests: `Backend/loadtests/locustfile.py`.

## Architecture

### Backend layers (`Backend/app/`)

Each layer depends only on the ones below it: `api/routes` → `orchestration` → `domain` → `infrastructure`.

- `api/routes/` — FastAPI routers, registered in `main.py`. Most are mounted at the root; the async job routes in `inference.py` and `tasks.py` live under `/api`, and `dataset_management.py` is mounted under `/upload`. `datasets.py` defines `/{dataset}/metadata` and `/{dataset}/file/...`, catch-all patterns that a new single-segment route can collide with.
- `orchestration/task_orchestrator.py` — the single RQ fabric: queue config, worker class, task functions, enqueue API, job status, dataset-warmup lifecycle. Extend this module; a second, parallel queue module was built once by accident and had to be merged back (the two used different progress-channel prefixes, so jobs published by one were invisible to the other). `fanout_orchestrator_service.py` and `multitask_orchestrator_service.py` are the earlier fan-out/fan-in prototypes. `session_queue_service.py` is the per-session list behind `/queue` and is unrelated to RQ.
- `domain/` — model loading and inference (`model_loader_service.py`), custom Hugging Face model resolution (`model_registry_service.py`), attribution (`saliency_service.py`), perturbation, acoustic profiling, accent bias, faithfulness evaluation, and the provenance contract (`provenance.py`).
- `infrastructure/` — settings, the async Redis client and session helpers, the sync RQ connection, cache keys, dataset loaders, the Mongo metadata store, logging and metrics.
- `core/redis.py` — a content-addressed cache manager (SHA-256, msgpack, lz4) whose only production consumer is `api/routes/results.py`. Everything else caches through `infrastructure/redis.py` + `infrastructure/cache_keys.py`.

### Two request paths coexist

1. **Async jobs** (`/api/inference/multitask|attribution|mutation|batch-warmup`, `/evaluation/accent-bias`): the gateway enqueues and returns a job id. Workers publish progress on the Redis channel `audiolit:progress:<job_id>`; the gateway relays it over `WS /api/ws/tasks/{id}`, with `GET /api/tasks/{id}/status` as the polling fallback. `Frontend/src/hooks/useTaskStatus.ts` implements the client side (three WebSocket retries, then 2 s polling). A worker's terminal event carries no result; `api/routes/tasks.py` attaches `job.result` as `payload.result` when it relays `SUCCESS`, and the hook reads only that field. Job results must be JSON-serialisable (no bytes).
2. **Synchronous routes** (`/inferences/*`, `/saliency/generate`, `/perturb`, `/acoustic/profile`, `/evaluation/faithfulness`): inherited from ECHO, these run model code inside the API process and return the result inline. The frontend still uses many of them, so the API process does load models despite the "gateway never runs a model" framing in the docs.

The synchronous routes run model code on the thread pool (`asyncio.to_thread`) against one process-wide cached `nn.Module` per model, so attribution and inference serialise on a per-model `threading.Lock` (`lock_for_model` in `saliency_service.py`). Grad-CAM and Captum register hooks on the shared module, so unlocked concurrent calls capture each other's activations and gradients.

### Workers

- Five families, one queue and one process each: `asr`, `ser`, `add`, `xai` (GPU-bound, concurrency 1) and `mutation` (CPU, concurrency 2). `worker.py all` supervises one child process per family and respawns any that die.
- `AudioLITWorker` subclasses RQ's `SimpleWorker` so models stay loaded in-process between jobs. Switching to the forking `Worker` would reload the model on every job.
- GPU-bound families take a Redis lock `audiolit:worker-lock:<family>`, renewed while the worker lives. A restarted worker can wait up to 150 s for the old lock to lapse.
- Multi-task analysis is a fan-out of per-family child jobs plus an aggregator job gated with RQ `depends_on`; RQ has no built-in fan-in.
- Workers use `get_worker_redis_connection()` (long read timeout for blocking dequeue); the request path uses `get_redis_connection()`. Both are synchronous and separate from the async client in `infrastructure/redis.py`.

### Redis is cache and broker at once

Sessions, result cache, progress keys, RQ queues and worker locks share one instance. Compose sets `--maxmemory 2gb --maxmemory-policy volatile-lru` deliberately: `allkeys-lru` evicted RQ's queue lists and worker locks under cache pressure. Every cache, progress and session key must therefore carry a TTL, and RQ's own keys must not.

### Cache keys

`infrastructure/cache_keys.py` owns the key families and the **shape** of the value stored under each. Two hashes over the resolved path are in use (`path_hash` = md5 of the path; `content_hash` = md5 of path, size and mtime), and different routes read different ones, so a writer such as dataset warmup must populate both. Storing a differently shaped value under an existing family breaks consumers rather than causing a cache miss.

### Provenance

Every attribution payload carries `provenance` (`measured` | `fallback` | `unavailable`) and `provenance_reason`, built by `provenance_fields()`; `FALLBACK` without a reason raises. A failed attribution must surface as `fallback` (a signal-energy map) with its reason, never as an unlabelled stand-in, and the faithfulness auditor refuses to score one. The frontend renders the label through `components/ui/ProvenanceBadge.tsx`.

### Models and datasets

- Custom models are limited to Whisper and Wav2Vec2 architectures, must ship `safetensors` weights (checked before any download), and are pinned to a commit revision.
- Audio I/O is `soundfile` only, resampling via librosa; `torchaudio` was removed on purpose, as was Celery.
- `infrastructure/dataset_ingestion.py` holds `CORPUS_REGISTRY` and the `DatasetLoader` interface for the seven benchmark corpora; `dataset_service.py` is what the routes call and falls through to it. Loaders resolve the corpus root from their module path (`Backend/data/`, mounted at `/app/data` in containers). Corpora are not in git; `datasets.lock` pins the Hugging Face dataset revision.
- Uploads are stored under `uploads/` with UUID filenames and purged on a TTL (on each upload and at startup).
- Mongo (`infrastructure/metadata_store.py`) stores metadata only, never audio or tensors, and every call degrades to a no-op when it is unconfigured or unreachable. Inference must never depend on it.
- Logs are single-line JSON (`LOG_FORMAT=json`) and must not contain filenames, session ids or transcripts.

### Frontend (`Frontend/src/`)

- Single route: `pages/Index.tsx` renders `components/layout/MainLayout.tsx`, which owns nearly all workbench state (selected file, model, dataset, task toggles, warmup job) and passes it to the resizable panels in `components/panels/`. Shared state beyond that lives in `context/ModelRegistryContext.tsx` and `contexts/{Embedding,Playback}Context.tsx` (two directories, both in use).
- There is no API client layer. `lib/api.ts` exports only `API_BASE` (compiled in from `VITE_API_BASE_URL` at build time) and components call `fetch` directly. The WebSocket origin is derived from `API_BASE`, not `window.location`.
- `lib/audioSelection.ts` is the only place that decides whether the selected clip is an upload/live recording or a dataset row. ESLint (`no-restricted-syntax`) rejects inline copies of that predicate; use `isUploadedAudio()` / `isLiveRecordingFilename()`.
- `lib/melScale.ts` is the only Hz↔pixel mapping (librosa's Slaney mel scale) and `lib/spectrogramImage.ts` the only spectrogram renderer (lowest band at the bottom). Canvases that draw or select on a spectrogram use both.
- The Perturbation tab has one selection (`MutationRegion` in `components/analysis/PerturbationTools.tsx`), shown by both the waveform and the spectrogram. Checked perturbations carry it to the backend as `region` (`t_start_ms`/`t_end_ms`, optional `f_low_hz`/`f_high_hz`); `apply_region_perturbation` in `domain/perturbation_service.py` confines each one to it. No region means the whole clip. The engine resamples to 16 kHz, so the selector's frequency axis stops at 8 kHz.
- `lib/warmupJob.ts` persists the active dataset-warmup job id in `localStorage` so a reload can reattach; `GET /api/inference/warmup/active` is the server-side recovery path.
- `components/ui/` is vendored shadcn/ui (regenerated by its CLI); avoid hand-editing it. `@/` aliases `src/` in Vite, Jest and tsconfig.
- TypeScript is non-strict (`strict: false`, `noImplicitAny: false` in `tsconfig.app.json`); `no-explicit-any` is a warning.
