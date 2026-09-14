# AudioLIT — Comprehensive Testing & Evaluation Document

> **Document ID:** `docs/evaluation/TESTING_AND_EVALUATION.md`
> **Ticket Link:** `LIT-171` (sub-task of Phase 3 evaluation block, parent: `LIT-170`)
> **Scope:** Compilation of the committed test plan, executed test evidence, and
> data-science error analysis into one authoritative document the evaluator can
> audit against SRS FRs / NFRs.
> **Status:** Living document — updated as PRs land on `develop`.

---

## 1. Executive Summary

AudioLIT is validated by a four-layer test strategy:

1. **Backend unit / integration tests** (pytest) — domain services, routes,
   orchestration, ingestion, caching, and models via mocked inference.
2. **Frontend tests** (Jest) — component and interaction tests; **Playwright**
   E2E specs exercise the real browser against the running stack.
3. **Performance & load** — in-process pytest benchmarks
   (`TestPerformanceProfiling`), the Locust suite scored against SRS §3.4.1,
   and memory/VRAM profiling.
4. **Data-science evaluation** — accent-bias WER profiling (FR15) and
   attribution-faithfulness auditing (FR16), documented in
   [`DS_ERROR_ANALYSIS.md`](DS_ERROR_ANALYSIS.md).

Every push/PR to `main` and `develop` gates on the CI pipeline
(`.github/workflows/ci.yml`): frontend `lint + jest + build`, backend
`pytest`. Latency/FPS observations are synthesised by the LIT-189 metrics
layer (`GET /metrics/synthesis`, scored against SRS §3.4.1).

---

## 2. Test Plan Structure

The committed test plan is implemented by `Backend/tests/run_tests.py`, which
organises tests into categories mapped to Master Test Plan sections:

| Category | Master Test Plan section | File(s) | Priority |
|---|---|---|---|
| Data & Database Integrity | §3.1.1 | `test_data_integrity.py`, `test_hashing.py`, `test_redis_cache.py`, `test_results_cache.py`, `test_provenance.py` | Critical |
| Function Testing | §3.1.2 | `test_function_testing.py`, `test_model_registry_service.py`, `test_models_routes.py`, `test_ser_model.py`, `test_ser_checkpoint.py`, `test_deepfake_classifier.py` | Critical |
| Performance Profiling & Load | §3.1.4 / §3.1.5 | `test_performance_load.py`, `test_memory_profiling.py`, `test_metrics_synthesis.py`, `loadtests/locustfile.py`, `scripts/run_memory_profile.py` | Important |
| Security & Access Control | §3.1.6 | `test_security.py`, `test_session_cookie.py`, `test_debug_and_tasks_routes.py` | Important |

Additional suites track individual FRs: `test_dataset_ingestion.py` (FR2),
`test_asvspoof_loader.py` / `test_librispeech_loader.py` / `test_l2arctic_loader.py`
(FR2 loaders), `test_acoustic_profiler_service.py` + `test_acoustic_routes.py`
(FR10), `test_saliency_service.py` / `test_saliency_routes.py` /
`test_integrated_gradients.py` / `test_spectrogram_attribution.py` /
`test_grad_cam.py` (FR8/FR9), `test_faithfulness.py` / `test_auc_faithfulness.py`
/ `test_degradation_scoring.py` / `test_high_saliency_masking.py` (FR16),
`test_accent_bias_profiler.py` / `test_accent_bias_runner.py` (FR15),
`test_task_orchestrator.py` / `test_fanout_orchestrator.py` /
`test_multitask_orchestrator.py` / `test_queue.py` (FR3), and
`test_warmup_cache_contract.py` / `test_custom_model_fidelity.py` /
`test_inference_consistency.py` / `test_system_integration.py` /
`test_evaluation_scoring.py` / `test_evaluation_routes.py` /
`test_perturbation_service.py`.

### 2.1 Execution commands

```bash
# Backend: full suite (the CI gate)
cd Backend && pytest

# Category run via the plan driver
cd Backend && python tests/run_tests.py performance   # or data_integrity | function_testing | security

# Frontend: lint + jest + build (the other CI gate)
cd Frontend && npm run lint && npm test && npm run build

# E2E (needs the stack running)
cd Frontend && npx playwright test
```

---

## 3. Test Environment

| Component | Configuration |
|---|---|
| Backend runtime | Python 3.10 (CI) / 3.11 (local), FastAPI 0.111, RQ 2.10 + Redis, PyTorch ≥ 2.6 (CPU-only wheel in CI) |
| Frontend runtime | Node 20, React 18, TypeScript, Vite |
| Test doubles | `fakeredis` replaces the Redis client per test (`fake_redis` fixture); models are mocked at the service boundary; synthetic audio via the `sample_audio_data` / `sample_audio_file` fixtures |
| Isolated storage | `temp_dir` fixture; dataset paths are git-ignored (`Backend/data/`) |

Red means nothing: **CI installs the CPU-only torch wheel** and pins a verified
requirements set (`pytest>=8.3.3,<9`). Recorded pass count at that pin:
**587 passed, 6 skipped** (backend suite, verified on pytest 8.4.2).

---

## 4. Test Results Summary

### 4.1 Backend (pytest)

The CI `backend-test` job installs dependencies, verifies torch is the CPU
build, and runs `pytest`. Suites that exercised in this session pass
independently, e.g. the LIT-189 metrics suite (`test_metrics_synthesis.py`,
17 tests) and the route-debug suite used to validate app wiring
(`test_debug_and_tasks_routes.py`).

### 4.2 Frontend

The CI `frontend-lint-and-build` job runs `npm ci`, ESLint, Jest, and a
production Vite build. Component tests cover the canvas selection tools,
waveform viewer, XAI overlay and shared UI primitives.

### 4.3 E2E (Playwright)

`Frontend/e2e/dataflow.spec.ts` drives a real upload → inference → result
cycle with mocked model traffic; `layout.spec.ts` checks the shell renders the
committed panels (Acoustic / Accent Bias / Faithfulness) and the status bar.

### 4.4 Privacy / depth of evidence

Unit-level results are deliberately mocked; **the authoritative
system-level evidence is the Locust load run against a live stack** (see §5),
because in-process mocks cannot see queue depth, Redis round-trips, or the
per-model saliency lock.

---

## 5. Performance & Load (SRS §3.4.1)

### 5.1 Target table

| Operation | Target | Model-bound | Enforced by default |
|---|---|---|---|
| Cached tensor retrieval | < 10 ms | No | Yes |
| API response, cached request | < 200 ms | No | Yes |
| Cache miss → enqueue | < 50 ms | No | Yes |
| Cold ASR (Whisper-base, 15 s) | < 3 s | Yes | No (reported) |
| Multi-task inference (ASR+SER+ADD) | < 8 s cold | Yes | No (reported) |
| Attribution (IG / saliency) | < 8 s | Yes | No (reported) |
| Canvas mutation — UI | < 500 ms, 30–60 FPS | No | Yes |
| Canvas mutation — backend | < 2 s | Yes | No (reported) |
| Accent bias profiling | < 30 s | Yes | No (reported) |
| Faithfulness audit | < 15 s | Yes | No (reported) |
| Cold model download + hooks | < 60 s | Yes | No (reported) |

`SRS_PERFORMANCE_TARGETS` in `Backend/app/infrastructure/metrics_synthesis.py`
is the canonical copy of this table, shared by the synthesis route and the
LIT-189 tests. Model-bound targets are not enforced by default (CPU fallback
is proportionally slower, per the SRS's own caveat) unless
`LOADTEST_ENFORCE_MODEL_TARGETS=1`.

### 5.2 In-process benchmarks

`tests/test_performance_load.py` measures mocked inference timing, cache
latency, concurrent-request success rate, and memory growth bounds (e.g.
`MAX_MEMORY_GROWTH_MB = 300` growth bound, per the LIT-170 determinism fix).
`tests/test_memory_profiling.py` tracks CPU RAM leakage (tracemalloc) and VRAM
clearance on CUDA; `scripts/run_memory_profile.py` compiles a standalone
summary.

### 5.3 Locust (system-level)

```bash
cd Backend
locust -f loadtests/locustfile.py --host http://127.0.0.1:8000 \
       --headless -u 8 -r 2 -t 3m
```

A `test_stop` hook prints a p95-vs-budget table and fails the run on any
enforced breach (or on zero requests made, so an empty run cannot report
green). Split user classes keep fast cache-read traffic flowing while heavy
attribution/cold-inference users generate realistic contention
(`127.0.0.1`, not `localhost`, on Windows hosts to avoid IPv6 fallback timeouts).

---

## 6. Data-Science Evaluation & Error Analysis

The quantitative evaluation lives in [`DS_ERROR_ANALYSIS.md`](DS_ERROR_ANALYSIS.md):

- **FR15 — Accent bias:** group-wise WER across six L2-ARCTIC L1 cohorts
  (Arabic / Chinese / Hindi / Korean / Spanish / Vietnamese). Mean WER
  **0.1353**, bias-discrepancy index **Δ = 0.0670** (Vietnamese 0.165 vs
  Spanish 0.098). Dominant failure modes are phoneme deletion/substitution
  patterns per L1.
- **FR16 — Faithfulness:** deletion scoring via trapezoidal integration.
  Masking the top 30 % of salient spectrogram features drops mean confidence
  by **54.17 %**; mean deletion AUC **0.6443** — evidence attribution maps
  isolate decision-relevant regions.

Shared, reproducible evidence for these numbers is generated through the
committed runners: `app/domain/accent_bias_runner.py` streams cohorts through
ASR (SRS Use Case 6) and the faithfulness engine in
`app/domain/perturbation_service.py` (masking, `test_faithfulness.py`,
`test_auc_faithfulness.py`).

---

## 7. Gaps, Limitations & Known Issues

State happens on the live stack, not in CI mocks, so treat §4.1 results as
fast regression signals and §5.3 as the performance evidence.

- **Latency/throughput synthesis** (LIT-189) ships the aggregation layer;
  automatic sampling of the request path and persistence of historical series
  into the MongoDB metadata tier is the follow-on (LIT-255/256/257).
- **Structured JSON task logs / operational metrics export** is tracked by
  LIT-259 (structured logs on the task fabric, metrics export endpoint).
- Pre-existing (noted, unconfirmed): `health.py` binds the redis client by
  name (LIT-229); a frontend `useEffect` in `PredictionPanel.tsx` lacks an
  unmount cleanup. Neither affects the results reported here.
- Model-bound targets are hardware-dependent; a CPU-only run will legitimately
  report "over (model-bound, not enforced)" for attribution and cold
  inference. Enforce only on GPU hardware (`LOADTEST_ENFORCE_MODEL_TARGETS=1`).

---

## 8. Traceability to Committed Requirements

| SRS area | Evidence |
|---|---|
| FR1 Model registry / hooks | `test_model_registry_service.py`, `test_hook_manager_service.py`, `test_custom_model_fidelity.py` |
| FR2 Dataset ingestion | `test_dataset_ingestion.py`, loader suites, `test_dataset_service.py`, `test_dataset_management_routes.py`, `FR2_AUDIT.md`, `DATASET_INGESTION_QA.md` |
| FR3 Async fabric | `test_task_orchestrator.py`, `test_fanout_orchestrator.py`, `test_multitask_orchestrator.py`, `test_queue.py`, `rq_fanout_pattern.md` |
| FR4 Content-addressed cache | `test_hashing.py`, `test_redis_cache.py`, `test_results_cache.py` |
| FR7 ADD | `test_deepfake_classifier.py`, `test_asvspoof_loader.py` |
| FR8/FR9 XAI | `test_saliency_service.py`, `test_integrated_gradients.py`, `test_spectrogram_attribution.py`, `test_grad_cam.py` |
| FR10 Acoustic profiler | `test_acoustic_profiler_service.py`, `test_acoustic_routes.py` |
| FR12 Mutation canvas | `test_perturbation_service.py`, frontend canvas Jest/E2E suites, `test_metrics_synthesis.py` (FPS) |
| FR15 Accent bias | `test_accent_bias_profiler.py`, `test_accent_bias_runner.py`, `test_l2arctic_loader.py`, `DS_ERROR_ANALYSIS.md` |
| FR16 Faithfulness | `test_faithfulness.py`, `test_auc_faithfulness.py`, `test_degradation_scoring.py`, `test_high_saliency_masking.py` |
| SRS §3.4.1 Performance | `test_performance_load.py`, `locustfile.py`, `test_memory_profiling.py`, `test_metrics_synthesis.py` |
| SRS §3.4.2 Security (SR1–SR7) | `test_security.py`, `test_session_cookie.py`; remediation tracked in LIT-223 |

---

## 9. References

- [`DS_ERROR_ANALYSIS.md`](DS_ERROR_ANALYSIS.md) — FR15/FR16 evaluation numbers
- [`WALKTHROUGH_SCRIPT.md`](WALKTHROUGH_SCRIPT.md) — live demo staging & script (LIT-191)
- `../PR_REVIEW_CHECKLIST.md` — code-review gates
- `../ISSUE_PLAN.md` — dependency-ordered issue map with status
- `../SRS.md` — committed FRs / NFRs, §3.4 performance & security

---

*Update this document as suites grow or numbers change; it is the evaluator's
entry point from the FR map in `../ISSUE_PLAN.md`.*