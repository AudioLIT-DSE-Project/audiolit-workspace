# Chapter 18 — Testing and evaluation in practice: the full account

Chapter 16 explained *why* this system needs unusual testing and which
techniques work. This chapter is the record of what was actually done: every
tool, every test, how each was run, what the numbers were, what failed, what
went well, and what is still open.

Where a number appears here it was measured and the measurement is dated.
Where a result was later found to be wrong, both the wrong and the corrected
result are shown, because the correction is the more useful thing to learn
from.

---

## 18.1 The mission

The testing effort was framed around one question:

> Can a user trust what this software tells them about a model?

That framing drove everything. It is not "does the software work" — a broken
interpretability tool that merely *looks* like it works is the dangerous case,
not the crashing one. So the effort was weighted toward:

- **Correctness of measurement** over uptime.
- **Honesty of output** over completeness of features.
- **Comparing two things** over asserting one thing.

The explicit scope boundary: **only committed functionality is in scope.**
Stretch items were not tested, and the availability of the Hugging Face Hub, the
correctness of the upstream model weights, and the accuracy of the corpora's own
ground-truth labels were all declared out of scope — they are other people's
systems.

---

## 18.2 The complete tool inventory

### Backend

| Tool | Version | Purpose |
|---|---|---|
| **pytest** | 9.1.1 | the whole backend suite |
| **pytest-asyncio** | 1.4.0 | `async def` tests, `asyncio_mode = auto` |
| **pytest-timeout** | 2.4 | per-test ceiling so a hang becomes a failure |
| **pytest-cov** | 7.1.0 | line coverage |
| **httpx** | — | `AsyncClient` for in-process route tests |
| **fakeredis** | — | Redis substitute, autouse for every test |
| **mongomock** | — | MongoDB substitute |
| **pip-audit** | — | Python dependency vulnerability scanning |

### Frontend

| Tool | Version | Purpose |
|---|---|---|
| **Jest** | 29 | component and unit tests |
| **ts-jest** | 29.4.12 | TypeScript in Jest |
| **jsdom** | — | DOM environment for Jest |
| **Testing Library** | — | component queries |
| **ESLint** | 9 | static analysis |
| **Vite** | 5.4 | production build as a test in itself |
| **npm audit** | — | JavaScript dependency scanning |

### End-to-end and system

| Tool | Version | Purpose |
|---|---|---|
| **Playwright** | — | cross-browser E2E across Chromium, Firefox, WebKit |
| **axe-core** | — | WCAG 2.1 AA accessibility scanning, driven from Playwright |
| **Lighthouse** | 13.5.0 | whole-page audit: performance, a11y, best practices, SEO |
| **Postman + newman** | 6.2.2 | an *independent* API harness |
| **Locust** | — | load testing against a real running stack |
| **Trivy** | 0.36.0 | container image vulnerability scanning |
| **GitHub Actions** | — | CI: four jobs |

### Reference implementations used as oracles

| Tool | Used to verify |
|---|---|
| **librosa** | F0 and RMS, as the reference toolkit the requirement names |
| **Captum** | Integrated Gradients, LIME, GradientShap |
| **jiwer** | word error rate, against a hand-rolled implementation |

That last row is worth noting: **two independent WER implementations existed on
purpose**, and their disagreement is what found a significant measurement bug
(§18.9).

---

## 18.3 The eight testing techniques, as applied

The test plan organises the effort into eight categories. Each is named with its
objective, its oracle (how you know the answer is right), and its tools.

### 1. Data and database integrity

Redis key schemes, payload shapes, TTLs, MongoDB collections and indexes,
dataset catalog integrity.

**Oracle:** the documented value shape per key family. A writer storing the
right key with the wrong shape is the specific defect being hunted (§8.1).

**Where:** `test_warmup_cache_contract.py` (23 tests), `test_results_cache.py`
(14), `test_redis_cache.py` (7), `test_metadata_store.py` (26),
`test_mongo_isolated.py` (2), `test_hashing.py` (3), `test_data_integrity.py`
(8).

### 2. Function testing

Every route and every domain engine.

**Oracle:** for most, the documented contract. For the high-value checks, a
*differential* oracle — two calls that must agree, or must differ.

The highest-value function checks, named explicitly in the plan:

- safetensors-only ingestion, with rejection *before* deserialisation
- all three tasks dispatchable concurrently on one clip
- SER returning a full distribution over at least six categories
- ADD returning a binary judgement with a confidence
- **Grad-CAM being genuinely gradient-weighted and not equal to the Integrated
  Gradients output for the same input**
- fallback-derived attributions carrying a provenance flag
- F0, RMS and log-mel validated against librosa
- mutations preserving the original and returning correctly shaped 16 kHz mono
- per-cohort WER disparity over L2-ARCTIC
- the top-K deletion-score audit

Note the fifth one. "Grad-CAM must not equal IG for the same input" is a
differential assertion, and it is the only kind of check that catches a
dispatch silently falling through to one method.

### 3. User interface testing

**Oracle:** axe-core's WCAG 2.1 A and AA rule set, plus cross-engine layout
comparison.

**Where:** `e2e/layout.spec.ts`, `e2e/accessibility.spec.ts`,
`e2e/quickstart.spec.ts`, and the Jest component suites.

### 4. Performance profiling

**Oracle:** the engineering targets committed in the requirements — not numbers
invented for the test.

| Operation | Budget | Hardware-bound? |
|---|---|---|
| Cached prediction (full API response) | 200 ms | no |
| Cache miss → task enqueue | 50 ms | no |
| Cold ASR inference (Whisper-base, 15 s audio) | 3 s | **yes** |
| Interpretability attribution (IG/saliency) | 8 s | **yes** |
| Canvas mutation → backend result | 2 s | **yes** |

The hardware distinction is load-bearing and is taken from the requirements
themselves, which state that GPU figures assume an NVIDIA T4 or equivalent free
cloud tier. So:

```python
ENFORCE_MODEL_TARGETS = os.getenv("LOADTEST_ENFORCE_MODEL_TARGETS") == "1"
```

Model-bound targets are **reported but not enforced** on a CPU host, with the
reasoning stated in the load test's own docstring:

> *So the model-bound targets are expected to be missed on a CPU box, and
> failing the run for that would make this suite permanently red and therefore
> ignored. The infrastructure targets — cached response, enqueue
> acknowledgement — are not hardware-bound in the same way and are always
> enforced.*

**A gate that cannot be met is a gate that gets ignored.** Separating
"unmeetable on this hardware" from "genuinely failing" is what keeps the suite
credible.

### 5. Load testing

**Oracle:** the same budgets, under concurrency, against a real stack.

`Backend/loadtests/locustfile.py`, and its docstring explains why it exists
alongside the in-process performance tests:

> *That pytest module mocks the models and drives concurrency in-process, so it
> measures the code path and nothing else. It cannot see what actually degrades
> under load on this system: RQ queue depth, Redis round-trips, uvicorn's worker
> pool, and the per-model saliency lock that serialises attribution against
> plain inference. Those only appear when real HTTP requests contend for a real
> stack.*

Two details in that file are worth copying.

**Cache warming before measuring a cache hit:**

```python
@events.test_start.add_listener
def warm_the_cache(environment, **_kwargs):
    """Prime WARM_CLIPS so "cached prediction" measures a hit, not a cold run.

    Without this the first request per clip pays full inference cost and lands
    in the same statistic as the hits, turning a 200 ms target into an
    unmeetable one for reasons that have nothing to do with the cache.
    """
```

And the clip pool is **split** so the two tasks cannot interfere:

```python
# Split so "cached" and "cold" never collide: a cold task touching a warmed
# file would silently measure a cache hit.
WARM_CLIPS = _ALL_CLIPS[:3]
COLD_CLIPS = _ALL_CLIPS[3:]
```

**A statistic that mixes two populations measures neither.**

**The host name that cost two seconds per user:**

```python
# IPv4 loopback by name. On Windows `localhost` resolves to ::1 first and the
# backend binds IPv4 only (`--host 0.0.0.0`), so every NEW connection tries
# IPv6, finds nothing, and falls back only after the OS connect timeout:
# measured 2063 ms per fresh connection via `localhost` against 23 ms via
# `127.0.0.1` (a warm keep-alive connection is 13 ms either way). Locust opens
# a fresh connection per user, so the wrong host name would add two seconds to
# every user's first request and distort the tail of every statistic reported.
DEFAULT_HOST = "http://127.0.0.1:8000"
```

This is a *measurement artefact* of exactly the kind §18.10 is about — and it
was caught and documented rather than reported as a latency finding.

And a guard against reading percentiles off too little data:

```python
# Below this many observations a p95 is not a percentile, so a target is
# reported but not judged. Raise the run duration, not this number.
MIN_SAMPLES_TO_ENFORCE = int(os.getenv("LOADTEST_MIN_SAMPLES", "20"))
```

### 6. Security and access control

**Oracle:** the seven security requirements, plus the inherited-baseline
remediation list.

**Where:** `test_security.py` (25 tests), `test_session_cookie.py` (14),
`test_debug_and_tasks_routes.py` (3), `test_dataset_service.py` (22),
`test_upload_limits.py` (8), plus the two dependency scanners and Trivy.

### 7. Failover and recovery

**Oracle:** deterministic degraded responses. The strongest single oracle named
in the plan is *"the retryable against non-retryable classification, because
that is what makes recovery automatic."*

The scenario matrix:

| # | Scenario |
|---|---|
| F1 | run with the MongoDB tier off and call every store method |
| F2 | stop Redis while the API is serving |
| F3 | restart Redis and re-test **without restarting the API** |
| F4 | kill the workers abruptly and immediately restart them |
| F5 | apply the manual recovery |
| F6 | inspect what the worker health endpoint reports after a crash |

Success criteria: no dependency failure produces an unhandled 500; each
dependency recovers on restoration without restarting the application; a killed
worker can be restarted immediately; no in-flight job becomes permanently
unobservable.

And the stated limitation, which is an architectural decision rather than a
gap: *"There is no redundant infrastructure and no continuous SLA: recovery is
by cheap resubmission."*

**Where:** `test_worker_supervision.py` (8 tests), `test_queue.py` (11),
`test_task_orchestrator.py` (69), `test_warmup_lifecycle.py` (19).

### 8. Configuration testing

**Oracle:** a cross-configuration differential oracle — *"the same functional
suite must produce the same pass or fail outcome across configurations, and any
divergence is itself the finding rather than noise to average away."*

| Dimension | Configurations |
|---|---|
| Browsers | Chromium, Firefox, WebKit at three widths |
| Operating systems | macOS, Windows 11, Ubuntu (CI) |
| Runtimes | Python 3.10 (CI) vs 3.11 (local); Node 20 (CI) vs 26 (local) |
| Accelerator | CPU-only vs GPU |
| Redis | containerised vs unreachable |
| Topology | native development vs containerised stack |

With the honest caveat: *"CI deliberately installs the CPU-only torch wheel, so
CI never exercises the GPU path at all, and GPU coverage is necessarily manual
and local."*

---

## 18.4 The backend suite, by what it tests

61 files, 729 collected tests. Grouped by target:

### Orchestration and queueing — 118 tests

| File | Tests | Covers |
|---|---|---|
| `test_task_orchestrator.py` | 69 | the whole fabric: queues, locks, progress, enqueue, worker lifecycle |
| `test_warmup_cache_contract.py` | 23 | the cache shapes dataset warmup writes |
| `test_warmup_lifecycle.py` | 19 | warmup start, cancel, reattach, staleness |
| `test_queue.py` | 11 | session queue |
| `test_worker_supervision.py` | 8 | respawn, backoff ceilings |
| `test_multitask_orchestrator.py` | 6 | fan-out |
| `test_fanout_orchestrator.py` | 5 | fan-in aggregation |

`test_task_orchestrator.py` at 69 tests is the largest single file, which is
proportionate: it is the module every asynchronous operation passes through.

### Datasets — 105 tests

| File | Tests |
|---|---|
| `test_dataset_ingestion.py` | 42 |
| `test_dataset_management_routes.py` | 37 |
| `test_ser_corpora.py` | 26 |
| `test_dataset_service.py` | 22 |
| `test_librispeech_loader.py` | 13 |
| `test_l2arctic_loader.py` | 12 |
| `test_asvspoof_loader.py` | 10 |
| `test_datasets_routes.py` | 10 |
| `test_data_integrity.py` | 8 |

Per-corpus loader files exist because each corpus has its own catalog format
and its own filename encoding — and those code tables are the highest-risk code
in the module (§14.6).

### Models and inference — 82 tests

| File | Tests | Covers |
|---|---|---|
| `test_inference_consistency.py` | 24 | **the differential-oracle file** — most of D01–D07's guards live here |
| `test_model_registry_service.py` | 24 | safetensors gate, pinning, LRU, circuit breaker |
| `test_custom_model_fidelity.py` | 19 | a user-selected checkpoint runs as itself |
| `test_hook_manager_service.py` | 12 | hook attach/detach, family resolution |
| `test_deepfake_classifier.py` | 9 | ADD |
| `test_ser_checkpoint.py` | 9 | the pinned SER checkpoint's head and labels |
| `test_model_pipeline_cache.py` | 5 | pipeline reuse (counting stubs) |
| `test_ser_attention_pooling.py` | 4 | the memory fix |
| `test_ser_model.py` | 3 | SER basics |

`test_inference_consistency.py` deserves its name. It is where "the same clip
through the same model twice must agree" and "two different methods must
differ" live — the technique that found the most severe defects.

### XAI — 44 tests

| File | Tests |
|---|---|
| `test_saliency_service.py` | 17 |
| `test_grad_cam.py` | 11 |
| `test_saliency_routes.py` | 11 |
| `test_provenance.py` | 9 |
| `test_latent_projection.py` | 9 |
| `test_spectrogram_attribution.py` | 9 |
| `test_integrated_gradients.py` | 7 |

### Evaluation — 53 tests

| File | Tests |
|---|---|
| `test_evaluation_scoring.py` | 13 |
| `test_accent_bias_profiler.py` | 11 |
| `test_accent_bias_runner.py` | 8 |
| `test_perturbation_service.py` | 11 |
| `test_high_saliency_masking.py` | 4 |
| `test_faithfulness.py` | 4 |
| `test_auc_faithfulness.py` | 4 |
| `test_degradation_scoring.py` | 3 |

### Infrastructure, security, performance — remainder

| File | Tests |
|---|---|
| `test_metadata_store.py` | 26 |
| `test_security.py` | 25 |
| `test_function_testing.py` | 22 |
| `test_metrics_synthesis.py` | 17 |
| `test_results_cache.py` | 14 |
| `test_session_cookie.py` | 14 |
| `test_performance_load.py` | 13 |
| `test_acoustic_profiler_service.py` | 13 |
| `test_operational_metrics.py` | 12 |
| `test_redis_cache.py` | 7 |
| `test_upload_limits.py` | 8 |
| `test_models_routes.py` | 6 |
| `test_evaluation_routes.py` | 5 |
| `test_memory_profiling.py` | 3 |
| `test_system_integration.py` | 3 |
| `test_hashing.py` | 3 |
| `test_acoustic_routes.py` | 3 |
| `test_mongo_isolated.py` | 2 |
| `test_inferences_route.py` | 2 |

---

## 18.5 The frontend and E2E suites

### Jest — 67 tests across 10 files

| File | Tests | Covers |
|---|---|---|
| `src/tests/ui-components.test.tsx` | 21 | the shared primitives, including the slider ARIA fix |
| `src/components/audio/WaveformViewer.test.tsx` | 8 | wavesurfer integration, region drag |
| `src/tests/QuickStartDialog.test.tsx` | 8 | first-run dialog and its dismissal storage |
| `src/tests/WarmupReattach.test.tsx` | 7 | recovering warmup state across a reload |
| `src/tests/XAIOverlayCanvas.test.tsx` | 7 | the overlay, including colour-ramp maths |
| `src/tests/PerturbationTools.test.tsx` | 5 | mutation controls |
| `src/tests/SpectrogramGridSelector.test.tsx` | 3 | canvas region selection |
| `src/tests/WarmupCancel.test.tsx` | 3 | cancellation |
| `src/tests/WaveformViewer.test.tsx` | 3 | (second, older suite) |
| `src/components/audio/AudioUploadRecorderModal.test.tsx` | 2 | the recorder |

### Playwright — 14 checks across 4 specs

```ts
projects: [
  { name: "chromium", testIgnore: /(dataflow|accessibility)\.spec\.ts/ },
  { name: "firefox",  testIgnore: /(dataflow|accessibility)\.spec\.ts/ },
  { name: "webkit",   testIgnore: /(dataflow|accessibility)\.spec\.ts/ },
  { name: "accessibility", testMatch: /accessibility\.spec\.ts/ },
  { name: "dataflow",      testMatch: /dataflow\.spec\.ts/, timeout: 180_000 },
]
```

| Spec | Checks | Engines | Needs a backend |
|---|---|---|---|
| `layout.spec.ts` | 2 | all three | no |
| `quickstart.spec.ts` | 2 | all three | no |
| `accessibility.spec.ts` | 4 | chromium | no |
| `dataflow.spec.ts` | 6 | chromium | **yes** — Redis, API, workers |

The `dataflow` suite is the most valuable and the most expensive. Its six checks
are written as *product-level honesty assertions*, not as UI smoke tests:

```ts
test("the predicted-transcript column is never raw JSON", async ({ page }) => {
  ...
  expect(body).not.toContain("[object Object]");
  expect(body).not.toContain('{"prediction"');
  expect(body).not.toContain('{"text"');
});
```

That is the cache-shape defect (§8.1) asserted from the browser. If a writer
stores a dict where consumers expect a string, this test sees it as text on
screen.

```ts
test("Grad-CAM renders a map that is flagged measured, not a fallback", async ({ ... }) => {
  expect(body.saliency_matrix?.length ?? 0).toBeGreaterThan(0);
  expect(body.base_spectrogram?.length ?? 0).toBeGreaterThan(0);
  expect(["measured", "fallback", "unavailable"]).toContain(body.provenance);
  expect(...).toBe("measured");
});
```

The provenance contract, asserted end to end. This is the check that guards
against the worst defect class in the project.

```ts
test("word segments name words the transcript actually contains", async ({ ... }) => { ... });
test("a genuine speech clip is not reported as spoof at full confidence", async ({ ... }) => { ... });
test("the same clip and model return the same prediction twice", async () => {
  ...
  expect(second).toEqual(first);
  expect(typeof first).toBe("string");
  expect(String(first).trim().length).toBeGreaterThan(0);
});
```

The last one is a differential check at the system level: determinism, plus the
assertion that the result is a *non-empty string* — which catches both the
cache-shape defect and an empty-transcript failure.

### Postman/newman — 13 requests, 39 assertions

```bash
npx newman run Backend/apitests/AudioLIT.postman_collection.json \
  --env-var baseUrl=http://127.0.0.1:8000 \
  -r cli,htmlextra --reporter-htmlextra-export docs/evaluation/api-reports/newman-report.html
```

The collection is **read-only** and asserts *contract properties*, not status
codes. From the report:

> *It asserts contract properties rather than merely checking for a 200: that
> the health broker flag is a real boolean and not a truthy string, that the
> worker count agrees with the length of the worker list, that an unknown route
> returns 404 without leaking a stack trace, and that an unknown job identifier
> is reported as `not_found` rather than as a zero-progress running job that a
> client would poll forever.*

Every one of those is a real bug shape:

- `"true"` instead of `true` — a JavaScript client treats both as truthy, so a
  degraded broker reads as healthy.
- A worker count that disagrees with its own list — internal inconsistency.
- A stack trace in a 404 — information disclosure.
- An unknown job reported as "running at 0%" — a client polls forever.

**Why a second harness at all?** *"So that a defect in the primary pytest
harness cannot hide a defect in the product."* The pytest route tests share a
Python process, a serialisation layer and a set of assumptions with the
application. A separate tool speaking raw HTTP does not.

---

## 18.6 How each layer is run

```bash
# Backend, with Redis deliberately unreachable to match CI
cd Backend
REDIS_URL="redis://127.0.0.1:1/0" python -m pytest -q -rs

# One file, one test
pytest tests/test_task_orchestrator.py
pytest tests/test_task_orchestrator.py::TestEnqueue::test_x

# Skip the slow ones
pytest -m "not slow"
```

```bash
# The full local CI equivalent for the frontend
cd Frontend
npm ci && npm run lint && npm test && npm run build

# E2E: layout across three engines, no backend needed
npm run test:e2e

# Accessibility only
npx playwright test --project=accessibility

# Full-stack data flow — needs Redis, API and workers running
npm run test:e2e:dataflow
```

```bash
# Load, against a running stack
cd Backend
locust -f loadtests/locustfile.py --host http://127.0.0.1:8000 \
       --headless -u 8 -r 2 -t 3m
```

```bash
# The data science evaluation, reproducibly
cd Backend
python scripts/evaluate_models.py --asr-model-id openai/whisper-base \
  --skip-faithfulness --output ../docs/evaluation/results

python scripts/evaluate_models.py --manifest eval_manifest.json --model-type ser \
  --skip-accent-bias --top-k 0.1 0.3 0.5 --output ../docs/evaluation/results_fr16
```

The evaluation script also has a `--self-test` mode, so the runner itself can be
checked without a corpus.

**Two run conventions that are not optional:**

**Point Redis at a closed port before pushing.** *"CI has no Redis service, and
a locally-reachable Redis masks failures that only bite on CI."* Port 1 is
guaranteed closed.

**Stress-run the orchestrator tests.** *"`SimpleWorker(burst=True)` draining a
dependency-gated aggregator on fakeredis has hung pytest intermittently, and
twice masked a real CI hang."* ~15 iterations in a loop with a
background-and-kill timeout. Note `perl alarm` does not work for this and Python
resets `SIGALRM`, so the timeout has to be external.

---

## 18.7 The three evidence passes

Reporting all three was deliberate, *"because the differences between them are
findings in their own right."*

| Item | Pass 1, macOS | Pass 2, Windows | Pass 3, Windows |
|---|---|---|---|
| Date | 2026-09-18 | 2026-09-19 | 2026-09-20 |
| Host | Apple M3 Pro, 18 GB | Windows 11 | Windows 11 |
| Python | 3.11.15 | 3.11.0 | 3.11.0 |
| Node | 26.4.0 | 20 | 20 |
| Runtime | PyTorch 2.13.0, no CUDA | 2.13.0+cpu | 2.13.0+cpu |
| Backend | 716 passed, 6 skipped, **135 s** | 715 passed, 7 skipped, **383 s** | 717 passed, 7 skipped, **1134 s** |
| Jest | 55 passed, 7 suites | 55 passed, 7 suites | 62 passed, 8 suites |
| ESLint | 0 errors, 113 warnings | 0 errors, 110 warnings | 0 errors, 110 warnings |
| Vite build | 11.1 s | 46.9 s | 48.6 s |

Two findings fall straight out of that table, and both are stated in the report:

**The same suite takes 2.8× to 8.4× longer on Windows than on macOS.** That is
not a curiosity — it is the number you need in order to set any timeout. A
300-second per-test ceiling is generous on macOS and only comfortable on
Windows.

**Test counts grew across passes**, because tests were added during the cycle.
*"A count is only meaningful next to its date and commit."* A bare "717 tests
pass" is not a verifiable claim.

Pass 3's supporting services: Redis 7.4.10 and MongoDB 6.0.28 in Docker, the API
on 8000 with five RQ worker families registered, Vite on 8080. **The software
was started and exercised for real** — the metadata tier created all four
collections with their expiry and uniqueness indexes, and the workbench loaded
and listed dataset rows in a browser.

### The seven skips, each with a reason

| Skipped test | Reason |
|---|---|
| `test_fanout_orchestrator.py:192` | needs a forking start method, which Windows does not provide |
| `test_function_testing.py:303` | requires GPU and model resources |
| `test_memory_profiling.py:35` | VRAM test requires CUDA |
| `test_ser_checkpoint.py:139`, `:152`, `:161` | each downloads ~1.2 GB from the Hub; gated behind an env var |
| `test_task_orchestrator.py:502` | no broker reachable — the intended condition for this run |

**Every skip states a reason.** A skip without one is an untested requirement
wearing a green tick.

---

## 18.8 Results by category

From pass 3 (2026-09-20):

| Category | Tool | Result |
|---|---|---|
| Backend unit + integration | pytest, 729 collected | **717 passed, 7 skipped, 0 failed** |
| Frontend component | Jest, 8 suites | **62 passed, 0 failed** |
| Static analysis | ESLint | 0 errors, 110 warnings |
| Production build | Vite | succeeds in 48.6 s |
| Cross-browser layout | Playwright × 3 engines | **passed on all three** |
| Quick-start walkthrough | Playwright | passed *after* defect 4 was fixed |
| Accessibility | axe-core | **FAILS** — 19 critical nodes |
| API, second harness | newman | 10 requests, 25 assertions, 0 failures |
| Whole page | Lighthouse | Perf **40**, A11y 96, Best Practices 100, SEO 100 |
| Full-stack data flow | Playwright | **5 of 6** — one intermittent |
| Load | Locust | 470 requests, 0 failures |
| JS dependencies | npm audit | **15 advisories** in production deps |
| Python dependencies | pip-audit | **33 advisories** across 5 packages |
| Container images | Trivy | configured and running |

### Load results

**Ten concurrent users, 90 seconds, 470 requests, 0 failures.**

| Operation | Requests | Median | p95 | Budget | Result |
|---|---|---|---|---|---|
| Cached prediction | 319 | 15 ms | 58 ms | 200 ms | **Pass** |
| Enqueue multitask | 118 | 58 ms | 110 ms | 50 ms | **Fail** |
| Health | 33 | 11 ms | 52 ms | — | observation |

**Single user.**

| Operation | Requests | p95 | Budget | Result |
|---|---|---|---|---|
| Cached prediction | 35 | 23 ms | 200 ms | Pass |
| Enqueue multitask | 28 | 50 ms | 50 ms | **Pass, exactly at the limit** |

The reading: *"The cached read path has comfortable headroom, using about a
quarter of its budget even under concurrency. The enqueue path is the weak
point: it sits exactly on its 50 ms budget with one user and misses it at ten."*

And the methodological point: *"Both measurements were taken so that the finding
is not mistaken for a measurement artefact, and they agree on the direction."*
One measurement is an anecdote. Two, at different concurrency levels, agreeing
on direction, is a finding.

For reference, a Whisper Grad-CAM attribution measured **21–37 seconds per
clip** called directly on the CPU host. Against an 8-second budget that assumes
a GPU, which is why those targets are reported rather than enforced.

### Lighthouse, and the risk it quantified

| Metric | Value |
|---|---|
| First Contentful Paint | 2.1 s |
| Largest Contentful Paint | 2.7 s |
| **Total Blocking Time** | **1,000 ms** |
| Cumulative Layout Shift | 0.005 |
| Speed Index | 4.5 s |

> *The performance score of 40 is the first measured figure for a risk both
> earlier versions of this report carried but neither had quantified: the
> frontend ships as a single chunk of roughly 1.78 MB gzipped with no code
> splitting.*

A risk that had been *listed* in two previous reports became a *number* here.
That is the difference between a risk register and an engineering finding.

---

## 18.9 The two evaluation defects — the most consequential findings

This is the part of the effort that mattered most, because **both data-science
metrics were wrong when the cycle started, and neither was wrong in a way
anybody would have noticed by using the product.**

### Defect A — the speech model was guessing the language

**Symptom.** Accent-bias evaluation reported overall mean WER **0.6019** and a
bias discrepancy index of **1.1676**. Two cohorts had mean error rates above
1.0, *which is only possible when the model inserts more words than the
reference contains.*

That impossibility is what triggered the investigation. A WER above 1.0 is not
"very bad" — it is a signal that something structural is wrong.

**Investigation.** Per-sample results sorted worst-first. Two samples out of 120
stood out at **22.30 and 17.80**. Their transcriptions were not poor English —
they were **Vietnamese and Arabic text, repeating the same phrase many times.**
Two more came back in Indonesian or Malay.

**Root cause.** The profiler built the ASR pipeline without specifying a
language, so Whisper ran language identification per utterance. On heavily
accented English it selected the speaker's *first language*, transcribed into
that language, and fell into a repetition loop.

**Why it mattered.** *"The two worst cohorts were worst only because of one bad
clip each. The published ranking was not measuring accent difficulty for those
cohorts. It was measuring language misdetection."*

**Fix.** Decode language fixed to English — correct, because the corpus is read
English throughout and the language is *known* rather than something to guess.

**Effect.** Overall mean 0.6019 → **0.2587**. Discrepancy index 1.1676 →
**0.0481**. The two pathological samples dropped from 22.30 and 17.80 to
**0.20 and 0.30**.

### Defect B — two WER implementations disagreed

**Symptom.** After the first fix, the summary still did not add up. Every cohort
mean sat between 0.097 and 0.206, *yet the overall mean printed above the same
table read 0.2587 — higher than every value it was supposedly summarising.*

An average larger than every value it averages is arithmetically impossible.
That contradiction is the entire finding.

**Investigation.** The report drew its cohort table from one code path and its
headline figures from another. Compared directly:

| Cohort | Profiler path | Evaluation-service path |
|---|---|---|
| Arabic | 0.2060 | 0.2917 |
| Hindi | 0.0969 | 0.2436 |
| Korean | 0.1347 | 0.2544 |
| Mandarin | 0.1246 | 0.2466 |
| Spanish | 0.1413 | 0.2515 |
| Vietnamese | 0.1812 | 0.2645 |

They disagreed on **every** cohort, consistently, with the second always higher.

**Root cause.** The evaluation service lower-cased and split on whitespace but
**did not remove punctuation**. Whisper returns punctuation, so a correctly
recognised final word `"child."` failed to match the reference `"child"` and
counted as an error. The profiler already removed punctuation *and had a comment
explaining why*. The two had drifted apart.

**Why it mattered more than it looks** — and this is the subtlest result in the
whole effort:

> *The inflation was near-constant across cohorts, because every cohort's
> transcripts end in a full stop at a similar rate. Adding a near-constant to
> every cohort barely changes their order, but it does compress the gap between
> them relative to their size, and the bias discrepancy index is exactly that
> gap. The published index was 0.0481 when the real spread was 0.1091. **The
> metric whose entire job is to measure disparity was understating that
> disparity by more than half.**

And note the direction: the bug made the model look **fairer than it is.** A bug
that flatters your system is far less likely to be investigated than one that
embarrasses it.

**Fix.** Both sides normalise identically, **plus a test asserting the two code
paths agree on the same input**, so they cannot drift apart silently again.

### Combined effect

| Stage | Overall mean WER | Bias discrepancy index |
|---|---|---|
| Before either fix | 0.6019 | 1.1676 |
| After the language fix | 0.2587 | 0.0481 |
| After the WER fix | **0.1430** | **0.0779** |
| Previously documented elsewhere | 0.1353 | 0.0670 |

The corrected figures land close to numbers documented earlier from different
hardware, *"which is a good sign for both."* Cross-validation against an
independent earlier measurement.

### The final corrected results

**FR15 — accent bias.** 120 samples, 20 per cohort, L2-ARCTIC, Whisper-base:

| Cohort | Samples | Mean WER | Median WER |
|---|---|---|---|
| Arabic | 20 | 0.2060 | 0.1538 |
| Vietnamese | 20 | 0.1812 | 0.1603 |
| Spanish | 20 | 0.1413 | **0.0000** |
| Korean | 20 | 0.1347 | 0.1056 |
| Mandarin | 20 | 0.1246 | 0.0955 |
| Hindi | 20 | 0.0969 | 0.0385 |

Overall mean **0.1430**, bias discrepancy index **0.0779**.

Reading: the model is measurably worse on Arabic-accented English than on
Hindi-accented English, by about 7.8 percentage points of WER. *"That is a real
disparity and it is the finding the requirement exists to surface. It is not
large enough to call the system unusable for any cohort, but it is large enough
that a user should be told which cohorts were measured."*

**FR16 — faithfulness.** Three CREMA-D clips, SER model, masking at 10/30/50%:

- Mean deletion score: **0.2907**
- Mean deletion AUC: **0.1408**
- Audio scored: 3 of 3. Refused as fallback: 0

Reading: masking the most salient regions reduces confidence by ~29%, so the
attributions carry real information. And the honest limitation: *"This is a small
sample. It demonstrates that the audit pipeline works end to end; it is not a
final faithfulness figure for the product."*

### Three failure modes found in the corrected data

1. **Accented English can be misread as another language entirely.** *"The most
   severe failure mode found, because the output is confident, fluent and
   completely wrong, and a user who does not read Vietnamese would have no way
   to tell."*
2. **Cohort means are driven by a minority of hard clips.** Spanish has a
   *median* WER of 0.0000 — at least half its samples transcribed perfectly —
   yet a mean of 0.1413. Reporting only the mean would suggest uniform mediocrity
   where the truth is "usually perfect, occasionally poor". Both are reported.
3. **The cohort ranking changed after the fixes.** Vietnamese looked worst
   before; Arabic is worst after. *"Any conclusion drawn from the earlier
   ranking, including any statement about which speakers the system serves
   worst, was based on a measurement error."*

---

## 18.10 Measurement errors — three findings that were wrong

Separate from the defects. These are cases where the *measurement* was wrong and
was caught before anyone acted on it.

> *Three findings during this work were wrong on first measurement and were
> corrected before they reached a fix:*
>
> - *An enqueue latency reported at 2100 ms (42× budget) was an artefact of
>   `urllib` splitting headers and body across packets; `requests` measured
>   23.6 ms. Caught because the load test said 41 ms and the contradiction was
>   chased rather than averaged away.*
> - *Acoustic profiling "taking 13 s" was a cold-start run; warm it is 0.1 s.*
> - *A claimed ~1.8× saliency slowdown disappeared once the untouched control
>   path was measured and had moved by the same factor — it was machine load.*

Each would have caused work on a problem that did not exist. And each was caught
by **a contradiction**, not by better measurement technique:

- Two tools disagreeing by 50×.
- A number that did not match the warm case.
- A control that moved with the treatment.

> *This is recorded because it bears on how the rest of the table should be
> read: no single timing observation was treated as evidence anywhere above.*

### The synthetic benchmark that proved the opposite of the truth

| Handler shape | p50 | p95 |
|---|---|---|
| `async def` | 44 ms | 70–75 ms |
| `def` + threadpool | 70 ms | 120 ms |

> *The synthetic test had used a 200 ms stub. A real loopback round trip is
> 1.67 ms, and per-request thread dispatch costs more than that. The change was
> reverted and the test deleted, because a test asserting the slower shape is
> worse than no test.*
>
> *The lesson generalises: a synthetic benchmark whose parameters do not match
> production can prove the opposite of the truth, confidently.*

The benchmark was internally valid. It answered a question about a system that
did not exist. And the disposition matters: the test was **deleted**, not left
passing, because it would have actively prevented the correct implementation.

---

## 18.11 The full defect inventory

23 defects found and recorded. How they were found, in descending order of
yield:

> 1. **End-to-end replay** — running one audio clip through the same model twice
>    and diffing the two responses. This is what surfaced D04, D05 and D07; a
>    unit test that calls the function directly cannot see any of them.
> 2. **Load testing** — sustained enqueue traffic, which is the only thing that
>    made D08 and D09 reproducible.
> 3. **Reading the fix, not the symptom** — D06 was found by a test written for
>    D07 that then failed for an unrelated reason.

### D01–D12 (Phase 3 verification)

| ID | Severity | Symptom | Root cause |
|---|---|---|---|
| D01 | High | Whisper Grad-CAM flat for every clip | non-class-discriminative target; ReLU zeroed the whole map |
| D02 | High | LIME all-zero | Captum's default Lasso surrogate drove every coefficient to zero |
| D03 | Medium | LIME output was speckle | perturbed individual mel cells, not coherent time regions |
| D04 | **Critical** | SER attention shown for checkpoints that expose none | **a different model was silently substituted** and its attention returned as the selected model's |
| D05 | High | word timestamps disagreed with the transcript | a second decode's labels diverged, and the divergence was hidden |
| D06 | High | custom SER checkpoint fed through Whisper's extractor | family resolved by substring match; unknown ids defaulted to Whisper |
| D07 | **Critical** | selecting a second SER checkpoint returned the first one's prediction | the SER cache key omitted the model identifier |
| D08 | **Critical** | workers exited silently mid-job under load; jobs queued forever | worker reused the request-path Redis connection whose `socket_timeout` is shorter than RQ's 405 s blocking dequeue |
| D09 | High | after an unclean exit a family stayed permanently blocked | the stale-lock scan used a hardcoded `worker_lock:` while the constant spells it `worker-lock:` — matched nothing |
| D10 | Medium | ADD saliency offered only some working methods | a UI support table narrower than reality |
| D11 | Medium | Redis evicted warmed entries during ordinary warmup | `maxmemory 256mb` against a 2 GB requirement |
| D12 | **Critical** | a failed attribution rendered identically to a successful one | no provenance on attribution results |

D09 is worth a second look: a **hyphen against an underscore** in a
string literal. The scan pattern matched nothing, so no stale lock was ever
purged, so a crashed worker blocked its family forever. The fix derives the key
from the constant instead of writing it out again — which is the same lesson as
the two progress-channel prefixes (§7.1).

D10 is worth a second look too: *"The gate returned once via a `main`→branch
merge and was removed a second time."* A fix can be undone by a merge.

### D13–D18 (2026-09-28)

| ID | Symptom | Root cause |
|---|---|---|
| D13 | language misdetection on accented English | pipeline built without a language |
| D14 | every transcription re-paid the model load | `_get_whisper_pipeline` and its cache existed with **zero callers**; 5.22 s load against 1.89 s inference — **73% of every call** |
| D15 | backend suite hung outright, 3× in one session | redis-py's `Pipeline.__del__` → `reset()` → `UNWATCH` re-entering a non-re-entrant fakeredis socket; GC-timing dependent |
| D16 | enqueue p95 70–75 ms against a 50 ms budget | three family enqueues ran sequentially; a loopback Redis round trip is 1.67 ms and RQ issues ~16 commands per enqueue |
| D17 | 19 critical accessibility nodes | icon-only buttons with no accessible name; Radix puts `role="slider"` on the Thumb while `aria-label` went to the Root |
| D18 | accessibility suite reported a **false pass** | a fresh browser profile opened the first-run dialog, whose modal overlay hid the workbench from the accessibility tree |

D15 was misattributed to coverage instrumentation for three sessions. The report
records the correction honestly:

> *A second full run during this cycle hung at the same file with no coverage
> instrumentation at all. It stopped producing output at 58 percent and never
> finished. The log size was sampled twice with no change between samples, and
> the run was terminated. The attribution to coverage instrumentation therefore
> does not hold.*

And the consequence, stated plainly: *"A single green run is not sufficient
evidence for this suite."*

D18 is the one to internalise. **The accessibility suite passed because the page
under test was covered up.** And the report notes it was the *same* blind spot
as defect 5 in the same cycle — the dialog overlay was found blocking the data
flow suite and fixed there, *"and the accessibility suite was not checked for
the identical problem. It had it."*

Fixing D18 is what revealed D17. A false pass was hiding 19 critical
violations.

### D19–D23 (2026-09-29/30)

| ID | Symptom | Outcome |
|---|---|---|
| D19 | 33 Python advisories, 25 JS advisories | Python cleared; JS 25 → 4, production 0 |
| D20 | SR1's 15-minute duration cap unimplemented | duration computed, displayed, never compared |
| D21 | uploaded audio never purged | TTL documented, asserted in a comment, unimplemented |
| D22 | dependency scanning not in CI | the scan that found 33 advisories was a one-off, not a gate |
| D23 | every undecodable upload returned 500, not 422 | `logger` used but never defined; the `NameError` was wrapped by the outer handler |

D19 also contains a correction of an earlier claim in the log itself: the
production lodash advisory had been recorded as *unfixable* on the grounds that
no patched 4.x existed. `lodash 4.18.1` does exist, outside the advisory range,
and `npm audit` had been reporting `fixAvailable: true` the whole time. The
correction is recorded rather than the text quietly edited.

D23's lesson is not the missing import: **the branch had never executed.** The
upload happy path ran constantly; that error path had run zero times, in a route
that until recently had no tests at all.

### Three fixed defects with no guarding test

Listed explicitly rather than left as a silent gap:

| ID | Why there is no test | What a test would need |
|---|---|---|
| D09 | no test references `_cleanup_stale_worker_locks`; the bug is a string-literal mismatch, invisible to any test that imports the constant rather than re-typing it | write a stale lock using `WORKER_LOCK_PREFIX`, run the cleanup, assert the key is gone — **built from the constant**, or it reproduces the bug it is meant to catch |
| D10 | frontend; no test references the XAI method list | a component test asserting all four methods render for an ADD model, or a Playwright assertion — it regressed once already via a merge |
| D11 | `maxmemory` appears in no test file; it is deployment configuration, not application code | a config-lint check asserting compose's cap matches the specified figure |

That table is a model of how to report debt: each row says *why* the gap exists
and *exactly* what would close it.

---

## 18.12 What went well

**The differential-oracle approach found the severe defects.** D04, D05 and D07
— two Critical, one High — were all found by running one clip through the same
model twice and diffing. No unit test could have seen any of them.

**Load testing found what nothing else could.** D08 and D09 were only
reproducible under sustained enqueue traffic. Both were Critical/High and both
caused jobs to hang forever.

**Contradictions were chased, not averaged.** Every one of the three false
measurements, and both evaluation defects, was caught because two numbers
disagreed and somebody followed it up. The WER bug in particular was found by
an average that was larger than every value it averaged — an arithmetic
impossibility that a less curious reading would have shrugged at.

**Errors were corrected in public.** The coverage-hang attribution, the
accessibility pass, the lodash "unfixable" claim, and the report's own headline
result were each wrong and each corrected *in place with the reason*, rather
than silently edited. That is what makes the rest of the document credible.

**Provenance turned an invisible failure class into a visible one.** D12's fix
means a fallback explanation can no longer be mistaken for a measured one, and
the dataflow E2E asserts it from the browser.

**Tests were verified by breaking the code.** "Verified by reverting: 2 fail
with the right diagnostics." An untested test is not a test, and this was done
repeatedly rather than once.

**The second harness paid for itself.** The Postman collection asserts contract
properties — a boolean that is really a boolean, a count that matches its own
list — that the in-process tests structurally could not check.

**Hardware honesty kept the gates credible.** Separating hardware-bound targets
from infrastructure targets, and reporting the former without enforcing them,
is why the load suite is still worth running on a CPU box.

**Skips all state reasons.** Seven skips, seven reasons. None of them is a
requirement quietly going untested.

---

## 18.13 What went badly, and what is still open

**Both data-science metrics were wrong at the start of the cycle**, and one of
them was wrong in the flattering direction. Had nobody noticed, the project
would have published a bias figure understating the disparity by more than half.

**The accessibility suite reported a false pass**, certifying an inaccessible
page as accessible, and it had the *same* blind spot that had already been found
and fixed in a sibling suite days earlier. Finding one instance of a bug class
did not trigger a search for others.

**Line coverage was unavailable for a long time** because the suite hung, and
the hang was misattributed to the coverage tool for three sessions. The real
cause was a GC-timing re-entrancy deadlock.

**A green run was treated as evidence**, more than once, for a suite known to
hang intermittently.

**Three fixed defects still have no guarding test**, and one of them (D10) has
already regressed once via a merge.

### Open at the time of writing (2026-09-30)

| Item | Status |
|---|---|
| Backend suite | **1 failed**, 782 passed, 7 skipped — `test_security.py::test_file_type_validation` returns 500 where it expects 400. Under active investigation. |
| Coverage | 70%. `inferences.py` is 763 statements at **8%** — the largest untested surface. |
| Enqueue latency | on budget at 1 user, over it at 10. Pipelining the family enqueues cut it 46%, but it remains the weak path. |
| GPU-bound targets | unmeasured — no GPU host. Grad-CAM measured 21–37 s per clip on CPU against an 8 s GPU budget. |
| Frontend bundle | ~1.78 MB gzipped, no code splitting. Lighthouse Performance 40, Total Blocking Time 1,000 ms. |
| Firefox/WebKit E2E | never run locally (`spawn UNKNOWN` on Windows); CI is their first real execution. |
| Dataflow provenance check | intermittent. Failed once with `provenance: fallback`; three direct retries returned `measured`. **Recorded as open, not closed** — *"an intermittent failure in exactly the check that guards against fake explanations is not something to wave through."* |
| ML runtime audit | could not be audited — installed from a local build not on the public index. *"That is a gap, not a clean result."* |
| Faithfulness sample size | 3 clips. Demonstrates the pipeline works; not a product figure. |
| `vite` advisory | 1 high, dev-only, fix is a three-major bump. |

---

## 18.14 The meta-lessons

Ranked by how much trouble each would have saved:

**1. A measurement instrument can work perfectly and measure the wrong
quantity.** WER was computed correctly; the transcription was in Vietnamese. The
number was valid and meaningless. Nothing inside the metric could reveal it —
only reading the actual transcripts did.

**2. Chase contradictions.** Every false measurement and both metric defects
were found this way. Two tools disagreeing, an average above its own maximum, a
control that moved with the treatment. Do not average, do not pick the number
you expected.

**3. A bug that flatters your system is the hardest to find.** The WER
normalisation bug made the model look fairer. Nobody investigates good news.

**4. A test that passes for the wrong reason is worse than a failing test.** The
accessibility suite actively certified an inaccessible page.

**5. When you find a bug class, search for other instances.** The modal-overlay
blind spot was found and fixed in one suite while sitting undetected in another.

**6. Error paths are code.** D23 lived in a branch that had never once executed.

**7. A single green run is not evidence** for a suite with a known intermittent
hang.

**8. Report a count with its date and commit**, or it is not a verifiable claim.

**9. Separate "unmeetable on this hardware" from "failing."** A gate that cannot
be met gets ignored, and then it protects nothing.

**10. Derive, never re-type.** D09 (a hyphen vs an underscore) and the two
progress-channel prefixes are the same bug. Build the string from the constant.

**11. A synthetic benchmark whose parameters do not match production can
confidently prove the opposite of the truth.** And delete the test it produced.

**12. Report both mean and median** for a skewed distribution. Spanish: median
0.0000, mean 0.1413. One number alone would mislead in either direction.

**13. State your debt with what would close it.** The three untested defects
each name the test that is missing.

---

## 18.15 Where the artefacts live

| Artefact | Path |
|---|---|
| Test plan and report | `docs/testing/AudioLIT_Test_Plan_Report_Merged.md` |
| Master test plan | `docs/testing/AudioLIT_Master_Test_Plan.md` |
| Defect log | `docs/evaluation/DEFECT_LOG.md` |
| Evaluation results, FR15 | `docs/evaluation/results/` |
| Evaluation results, FR16 | `docs/evaluation/results_fr16/` |
| newman HTML report | `docs/evaluation/api-reports/newman-report.html` |
| Lighthouse HTML report | `docs/evaluation/api-reports/lighthouse-report.report.html` |
| Backend tests | `Backend/tests/` (61 files) |
| Postman collection | `Backend/apitests/AudioLIT.postman_collection.json` |
| Load test | `Backend/loadtests/locustfile.py` (on `testing`) |
| Evaluation runner | `Backend/scripts/evaluate_models.py` |
| Frontend tests | `Frontend/src/**/*.test.tsx`, `Frontend/src/tests/` |
| E2E | `Frontend/e2e/` |
| CI | `.github/workflows/ci.yml` |
| Demo runbook | `docs/DEMO_RUNBOOK.md` |

---

*This is the final chapter of the AudioLIT Handbook. For the condensed version
of its lessons, see [Chapter 16](16-testing.md).*
