# §4 Software Testing Results, Coverage & Test Suite Health

**Author:** Ravindu Pathirana (LIT-251) · **For:** LIT-171 Testing & Evaluation document, assembled by Rahim Iqbal
**Target-of-test branch:** `testing` (decision D4, `docs/testing/TEST_PLAN_DESIGN.md`)
**Commit under test:** `3a934ea` (tip of `origin/testing` after PR #136 merged `develop` into `testing`, bringing in LIT-189 and LIT-256; supersedes the earlier `3d92d0b` run below)
**Environment:** Python 3.11.15, Node 20, macOS (dev machine), CPU-only PyTorch, Redis 7 via Docker (also independently verified against Redis deliberately unreachable — see §4.1)

This section reports the **executed** state of the suite, not an estimate. Every count below was produced by a run listed with its exact command; none are copied from a prior report without re-verification. Re-run in full after `testing` advanced from `3d92d0b` to `3a934ea` (LIT-189 latency/FPS metrics + LIT-256 MongoDB tier-1 metadata store landed in between) to confirm the new code doesn't change the picture. Two things were found in the original pass that the previously-cited baseline didn't have and remain true here: a real, still-open hang (§4.4, now reproduced a third time), and a Redis-reachability-dependent skip count (§4.1) — both reported honestly rather than smoothed over.

---

## 4.1 Backend — `Backend/tests/` (55 files)

Two conditions were run, because the result is not identical between them — this is itself a finding, not noise.

**A. Redis deliberately unreachable** (`REDIS_URL="redis://127.0.0.1:1/0"` — the project's standard pre-push check, since CI has no Redis service):

```
$ cd Backend && REDIS_URL="redis://127.0.0.1:1/0" pytest --cov=app --cov-report=term --cov-report=html --cov-report=json -q --ignore=tests/test_multitask_orchestrator.py
=========== 637 passed, 6 skipped, 373 warnings in 136.14s (0:02:16) ===========
```

**B. Redis reachable** (real Redis 7 via `docker-compose up -d`, default `REDIS_URL`):

```
$ cd Backend && pytest -q -rs --ignore=tests/test_multitask_orchestrator.py
=========== 638 passed, 5 skipped, 373 warnings in 127.76s (0:02:07) ===========
```

Same one-test delta as before (637/6 vs 638/5) — the `test_task_orchestrator.py` broker-reachability check is unaffected by the LIT-189/LIT-256 additions, it simply sits alongside +29 more tests now.

**+29 tests since the last run** (608 → 637), entirely from the two features that landed in between: LIT-256's `test_metadata_store.py` (12 tests, MongoDB tier via `mongomock`) and LIT-189's `test_metrics_synthesis.py` (17 tests, SRS §3.4.1 target table). Zero regressions in any pre-existing test.

The difference is exactly one test: `test_task_orchestrator.py` has a check that skips with `"no broker reachable to inspect the request-path client"` when Redis is unreachable, and runs (and passes) when it is. **This means the standard pre-push check and a normal local run are not equivalent** — one test's coverage of the request-path client only happens with Redis up. Neither condition is wrong; both are reported so the difference isn't silently averaged away.

Both runs **exclude** `test_multitask_orchestrator.py` (6 tests) — see §4.4 for why, and its own result.

### Every skip, individually (condition A, 6 skips)

| Test | Reason | Category |
|---|---|---|
| `test_function_testing.py:303` | "Requires GPU/model resources" | Hardware-gated (no CUDA on this machine) |
| `test_memory_profiling.py:35` | "VRAM test requires CUDA" | Hardware-gated |
| `test_ser_checkpoint.py:139` | Hits the Hugging Face Hub, downloads ~1.2 GB; opt-in via `AUDIOLIT_HUB_TESTS=1` | Network/opt-in, by design |
| `test_ser_checkpoint.py:152` | Same as above | Network/opt-in, by design |
| `test_ser_checkpoint.py:161` | Same as above | Network/opt-in, by design |
| `test_task_orchestrator.py:295` | "no broker reachable to inspect the request-path client" | Redis-reachability-dependent (only under condition A) |

No skip in this list is hiding a failure — each is a named, conditional gate (hardware, network cost, or environment), not a silently-broken test. The three Hub-download tests (`TestAgainstTheRealHub` in `test_ser_checkpoint.py`) exist specifically to catch the class of defect this project has hit before (a randomly-initialized classification head silently shipping as if trained) — they're opt-in because they cost ~1.2 GB and network access per run, not because they're unimportant.

### Coverage (condition A; `pytest-cov`, `--cov=app`)

**Overall: 66% (6,358 statements, 2,138 missed)** — steady vs. the prior run (6,076/2,088 at `3d92d0b`); the new LIT-189/LIT-256 code added proportionally similar coverage to what it replaced as "uncovered baseline." Full HTML report: `Backend/htmlcov/index.html`, machine-readable: `Backend/coverage.json` (both generated fresh, not committed — regenerate with the command above).

| Layer (SAD §5.1) | Coverage | Notable low spots |
|---|---|---|
| `app/domain/` (ML/XAI engines) | mostly 77–100%; `model_loader_service.py` 59% | `model_loader_service.py` (760 stmts, 59%) — the largest domain file, real-model-download paths are the uncovered branches |
| `app/orchestration/` (task fabric) | 45–98%, mixed | `task_orchestrator.py` 60% (466 stmts), `worker.py` 51%, `session_queue_service.py` 45% — orchestration error/retry branches are the gap |
| `app/infrastructure/` (cache, datasets, settings) | mostly 68–100% | `dataset_service.py` 71%, `app/infrastructure/redis.py` 76%, **new:** `metadata_store.py` (LIT-256, MongoDB tier) 68% of 149 stmts, `metrics_synthesis.py` (LIT-189) 97% of 104 stmts |
| `app/api/routes/` (15 routers) | wide spread, 8–100% | `inferences.py` 8% (763 stmts, the largest file in the codebase) and `inference.py` 31%, `tasks.py` 28%, `health.py` 38% are the real gaps — these are the request-path routes whose heavy branches (streaming responses, WebSocket relay, multi-model dispatch) aren't exercised by the unit suite and depend on the E2E/dataflow layer instead |

**Honest read:** the domain/business-logic layers most responsible for the interpretability claims (saliency, acoustic profiling, evaluation) are well covered. The weakest coverage is concentrated in two large, request-path-heavy files (`inferences.py`, `inference.py`) whose branches are exercised by the E2E dataflow suite (§4.3) rather than unit tests — this is a real coverage gap for the unit layer specifically, not an unverified gap overall, and is reported as a limitation rather than hidden behind the 66% headline.

---

## 4.2 Frontend unit — `Frontend/src/tests/` + component tests

```
$ cd Frontend && npx jest --coverage
Test Suites: 6 passed, 6 total
Tests:       47 passed, 47 total
```

**Coverage (`--coverage`, all files):** 60.88% statements / 39.73% branch / 50.17% functions / 62.74% lines.

Strongest coverage: `lib/utils.ts` (100%), most `components/ui/` primitives (92.46% avg, several at 100%). Weakest: `AttentionVisualization.tsx` (5.67% — largely untested), `useTaskStatus.ts` (20.73% — the WebSocket/polling hook), `AccentBiasPanel.tsx` and `FaithfulnessAuditPanel.tsx` (~25–31% — both newer panels with light test coverage relative to `AudioDatasetPanel`/`EmbeddingPanel`). These three are named explicitly as coverage gaps rather than folded into the aggregate number.

```
$ npm run lint
✖ 111 problems (0 errors, 111 warnings)

$ npm run build
✓ built in 9.46s
```

Zero lint errors; all warnings are pre-existing `@typescript-eslint/no-explicit-any` / `react-hooks/exhaustive-deps` style, not new defects.

---

## 4.3 End-to-end — `Frontend/e2e/`

Two Playwright projects exist, deliberately split by what they need to run (`playwright.config.ts`):

**Layout** (`layout.spec.ts`, LIT-160) — backend-free, three browser engines:

```
$ npx playwright test --project=chromium --project=firefox --project=webkit
12 passed (11.8s)
```

4 test cases × 3 engines (chromium/firefox/webkit) = 12. All pass — no cross-browser responsive-layout regression.

**Data flow** (`dataflow.spec.ts`) — chromium-only, requires the full stack (backend + Redis + RQ workers) live:

```
$ npm run test:e2e:dataflow
✓ Dataset table to workspace › selecting a clip binds it to the datapoint editor (1.6s)
✓ Dataset table to workspace › the predicted-transcript column is never raw JSON (1.6s)
✓ Cache behaviour through the UI › the same clip and model return the same prediction twice (131ms)
✓ Saliency panel › Grad-CAM renders a map that is flagged measured, not a fallback (2.1s)
✓ Deepfake panel › a genuine speech clip is not reported as spoof at full confidence (2.2s)
-  Saliency panel › word segments name words the transcript actually contains
5 passed (2.9s)
```

5 passed, 1 skipped — run against a live backend + a real Docker Redis + a running worker process, real model inference. **Re-run twice on two different days against the same long-lived backend process**: the first run (against `3d92d0b`) hit cold models (15.9s–2.0m per test); this second run (against `3a934ea`) hit warm caches from the first run (131ms–2.2s). Both are genuine, neither is mocked — the dramatic speed-up between runs is itself evidence the cache is doing real work (FR4.4), not a sign anything was faked.

**Why this layer exists, not just the unit suite:** the unit suite can stay fully green while the running application serves an attribution no model actually produced, or a transcript no audio contained — because a unit test calls a function directly, and the defect lives in the wiring *between* functions, which only a real request through the real stack exercises. `dataflow.spec.ts` asserts on provenance and content (e.g., "flagged measured, not a fallback," "never raw JSON") rather than mere HTTP-200 presence, which is exactly the class of defect a presence-only check would miss.

---

## 4.4 Known test-suite-health issues

Three issues are documented here, not two — a prior note (LIT-171) said "two de-flaked tests"; a third, still-open one was found during this pass.

**1. `test_performance_load.py::test_memory_usage_monitoring` — fixed, stable.**
Asserted absolute process RSS, which measured whatever real models earlier tests in the same process had already loaded — order-dependent, not a real leak signal. Fixed (LIT-170, refined further on `testing`) to assert memory *growth* across the loop instead, with `gc.collect()` before measuring. Confirmed stable across repeated full-suite runs.

**2. Unseeded Grad-CAM mock in `test_inference_consistency.py` — fixed, stable.**
The saliency mock's weights were unseeded; Grad-CAM's ReLU zeroes the attribution map whenever the weighted sum lands negative everywhere, so the test passed alone and failed unpredictably in a full run. Fixed by seeding the fixture.

**3. `test_multitask_orchestrator.py::TestMultitaskFanOutFanIn::test_asr_failure_does_not_lose_ser_result` — reproducible hang, still open, not fixed here.**

Reproduced **three times independently**, across two separate testing passes on two different commits (`3d92d0b` and `3a934ea`), by three different invocation styles — under coverage instrumentation, standalone with `-v -s`, and standalone with `-v -s` again after the branch advanced. Same exact test, same exact point, every time:

```
$ python -u -m pytest -v -s tests/test_multitask_orchestrator.py
tests/test_multitask_orchestrator.py::TestRealAsrJob::test_calls_whisper_base PASSED
tests/test_multitask_orchestrator.py::TestRealSerJob::test_returns_real_emotion_prediction PASSED
tests/test_multitask_orchestrator.py::TestAddJob::test_returns_real_deepfake_prediction PASSED
tests/test_multitask_orchestrator.py::TestMultitaskFanOutFanIn::test_enqueue_wires_aggregator_deferred_on_children PASSED
tests/test_multitask_orchestrator.py::TestMultitaskFanOutFanIn::test_aggregates_asr_ser_add_once_on_success PASSED
tests/test_multitask_orchestrator.py::TestMultitaskFanOutFanIn::test_asr_failure_does_not_lose_ser_result
[hangs indefinitely — killed after 2+ minutes, each of 3 attempts]
```

5 of 6 tests in the file pass in under a second each, every time; this one specifically hangs, every time it's been tried in this environment (3 for 3, not "sometimes"). **Downgrading the earlier "intermittent" characterization** — a single sample size of one made that call prematurely; a 3/3 reproduction rate is closer to deterministic-in-this-environment than intermittent, though the cited Linear baseline (614/615 passed, implying all 6 passed in whatever CI/local run produced that number) shows it isn't universal either. The honest framing is: **environment-dependent, not confirmed intermittent** — someone should try it on a second machine before concluding either way.

The file's own code comment documents a *related but different* known race: "draining the dependency-gated aggregator in the same `SimpleWorker.work(burst=True)` pass as its children ... can hang the burst worker intermittently," and this test already applies that mitigation (drains only the child queues, calls `aggregate_multitask` separately). The hang happens **inside the children-only drain phase**, before the aggregator is ever touched — specifically in the one test where a child job (ASR) fails with retry explicitly disabled. This looks like a variant of the documented `SimpleWorker(burst=True)` + fakeredis fragility, not the exact race already named in the comment, so it's reported as a distinct, open finding rather than assumed to be the same issue.

**Practical effect on this report's headline numbers:** this file's 6 tests are excluded from §4.1's totals in both runs. This is why the suite is reported both with and without this file, rather than a single blended number that would hide a reproducible hang behind an average.

---

## 4.5 Summary table

| Layer | Command | Result | Notes |
|---|---|---|---|
| Backend (Redis unreachable) | `REDIS_URL=... pytest --cov=app -q --ignore=test_multitask_orchestrator.py` | 637 passed, 6 skipped | Coverage 66% (6,358 stmts) |
| Backend (Redis reachable) | `pytest -q -rs --ignore=test_multitask_orchestrator.py` | 638 passed, 5 skipped | One fewer skip (broker-dependent test runs) |
| Backend (isolated) | `pytest -v -s test_multitask_orchestrator.py` | 5/6 pass, 1 hangs (3/3 reproductions) | See §4.4 |
| Frontend unit | `npx jest --coverage` | 47/47 passed | Coverage 60.88%/39.73%/50.17%/62.74% (unchanged, no frontend diff) |
| Frontend lint | `npm run lint` | 0 errors, 111 warnings | Pre-existing warnings only |
| Frontend build | `npm run build` | Succeeds | — |
| E2E layout | `npm run test:e2e` | 12/12 (3 browsers) | Backend-free |
| E2E data flow | `npm run test:e2e:dataflow` | 5 passed, 1 skipped | Full stack live, real inference (warm cache this run) |

**Revision history:**
- 2026-09-14, commit `3d92d0b`: initial pass — 608/609 passed (Redis unreachable/reachable), 66% coverage (6,076 stmts), hang reproduced ×2.
- 2026-09-14/15, commit `3a934ea`: re-run after `testing` absorbed `develop` (PR #136: LIT-189, LIT-256) — 637/638 passed, 66% coverage (6,358 stmts, +29 tests, zero regressions), hang reproduced a 3rd time (now characterized as environment-dependent rather than confidently "intermittent," see §4.4).

All figures pinned to their stated commit, generated fresh for each pass — none copied forward without re-running.
