# Defect Log & Remediation Record

**Issue:** LIT-253 · **Companion to:** `docs/evaluation/TESTING_AND_EVALUATION.md`
**Scope:** defects found and remediated on the ASR / SER / ADD inference and
explanation paths during Phase 3 verification. Last updated 2026-09-18.

Every row below was verified against the repository: the commit hash resolves on
`develop`, the named test exists in the named file, and the "no guarding test"
section is the result of an explicit search, not an omission.

---

## 1. How these defects were found

Three sources, in descending order of yield:

1. **End-to-end replay** — running one audio clip through the same model twice
   and diffing the two responses. This is what surfaced D04, D05 and D07; a
   unit test that calls the function directly cannot see any of them.
2. **Load testing** (`Backend/loadtests/locustfile.py`) — sustained enqueue
   traffic, which is the only thing that made D08 and D09 reproducible.
3. **Reading the fix, not the symptom** — D06 was found by a test written for
   D07 that then failed for an unrelated reason.

The recurring shape is worth naming: **a contract asserted on one side and
never checked on the other** — cache key vs. payload shape, function signature
vs. call site, response model vs. service, lock-key spelling vs. constant.
None of these produce an exception; they produce a plausible wrong answer.

---

## 2. Defect table

| ID | Symptom | Root cause | Fix | Guarding test | Commit / PR | Severity |
|----|---------|------------|-----|---------------|-------------|----------|
| **D01** | Whisper Grad-CAM heat-map rendered flat/empty for every clip. | Attribution target was total encoder energy — a non-class-discriminative scalar. The pre-ReLU map came out uniformly negative, so the ReLU zeroed all of it. | `_build_whisper_gradcam_target()` scores the decoded transcript's token logits; the energy target is retained as an explicit, labelled fallback. | `test_the_whisper_target_is_the_transcript_not_encoder_energy`, `test_a_class_style_target_yields_a_varying_map`, `test_whisper_gradcam_target_scores_the_transcript` (`test_inference_consistency.py`) | `6e51cbd` / #127 | High |
| **D02** | LIME attributions were all-zero, so the panel showed no explanation at all. | Captum's default surrogate is `SkLearnLasso(alpha=0.01)`; on standardised mel features L1 drove every coefficient to exactly zero. | `_lime_surrogate()` returns `SkLearnRidge(alpha=0.01)` — L2 shrinks but does not annihilate. | `test_surrogate_is_not_the_default_lasso`, `test_surrogate_recovers_a_known_linear_signal` (`test_inference_consistency.py`) | `6e51cbd` / #127 | High |
| **D03** | LIME output was high-frequency speckle with no interpretable time structure. | Perturbation operated on individual mel cells, so no coherent region was ever occluded. | `_time_band_feature_mask()` groups frames into ≤32 contiguous time bands, passed as `feature_mask` at all three LIME call sites (Whisper, SER, ADD). | `test_time_band_mask_groups_frames_not_cells`, `test_time_band_mask_handles_a_2d_waveform_input`, `test_bands_are_capped_by_available_frames` (`test_inference_consistency.py`) | `6e51cbd` / #127 (Whisper, SER); `5aab4b0` / #132 (ADD) | Medium |
| **D04** | SER attention panel displayed a confident attention pattern for checkpoints that expose no attention at all. | `model_loader_service` silently substituted a *different* model (`jonatasgrosman/wav2vec2-large-xlsr-53-english`, and an "enhanced" `facebook/wav2vec2-base-960h`) and returned its attention as the selected model's. | Both substitutions removed. The loader now returns real attention or none, with an explicit `attention_is_fallback` flag and FR17 provenance. | `test_real_attention_reaches_the_caller`, `test_a_synthesised_pattern_is_never_reported_as_measured`, `test_no_other_checkpoint_is_loaded_to_fill_the_gap` (`test_inference_consistency.py`) | `bbd3aaa` / #127 | **Critical** |
| **D05** | Word-level timestamps disagreed with the transcript shown above them. | `transcribe_whisper_with_timestamps` ran a second decode; its word labels could legitimately diverge from the canonical transcript, and the divergence was hidden. | Segments are relabelled from the canonical transcript; a word-count mismatch sets `word_labels_diverged` rather than being force-aligned. | `test_segments_are_relabelled_from_the_canonical_transcript`, `test_mismatched_word_counts_are_flagged_not_forced`, `test_agreeing_decodes_are_left_alone` (`test_inference_consistency.py`) | `bbd3aaa` / #127 | High |
| **D06** | A custom SER checkpoint was fed through Whisper's embedding extractor, yielding meaningless embeddings. | Model family was resolved by substring match on the model id. `myorg/custom-ser` matches neither `"whisper"` nor `"wav2vec"`, so it fell through to the Whisper branch by default. | `_embedding_family()` resolves family via the model registry; an unknown id raises instead of defaulting. | `test_custom_ser_checkpoint_reaches_the_ser_extractor`, `test_add_model_does_not_get_the_ser_extractor`, `test_family_alias_is_not_passed_as_a_hub_id` (`test_inference_consistency.py`); `test_a_broken_custom_model_raises_rather_than_using_whisper_base` (`test_custom_model_fidelity.py`) | `f2a1540` / #127 | High |
| **D07** | Selecting a second SER checkpoint returned the first checkpoint's prediction. | The SER cache key omitted the model identifier, so all checkpoints collided on one key. | Cache keys are namespaced per checkpoint; the default model keeps its historical spelling so existing entries stay valid. | `test_ser_cache_keys_differ_per_model`, `test_default_keeps_its_historical_spelling`, `test_a_second_model_is_not_answered_by_the_first` (`test_custom_model_fidelity.py`); `test_a_second_model_does_not_overwrite_the_first` (`test_inference_consistency.py`) | `d40fa95` / #124 | **Critical** |
| **D08** | RQ workers exited silently mid-job under sustained load; jobs stayed queued forever with no error logged. | The worker reused the request-path Redis connection, which sets a fail-fast `socket_timeout`. The worker's blocking dequeue (`worker_ttl 420` ⇒ `dequeue_timeout 405 s`) outlasts that timeout, so the read raised and the worker died. | `get_worker_redis_connection()` returns a separate connection with `socket_timeout=None` and `socket_keepalive=True`; the request path keeps its fail-fast timeout. | `test_worker_read_timeout_outlasts_the_blocking_dequeue`, `test_request_path_connection_keeps_its_fail_fast_timeout` (`test_task_orchestrator.py`) | `5aab4b0` / #132 | **Critical** |
| **D09** | After an unclean worker exit, that model family stayed permanently blocked — every later job for it hung. | `_cleanup_stale_worker_locks` built its scan pattern from a hardcoded literal `audiolit:worker_lock:`, while `WORKER_LOCK_PREFIX` spells it with a hyphen. The scan matched nothing, so no stale lock was ever purged. | The key is derived from `WORKER_LOCK_PREFIX` instead of being written out again. | **None** — see §3 | `5aab4b0` / #132 | High |
| **D10** | The ADD saliency panel offered only a subset of the XAI methods that actually work for it. | A UI gate added in `bfc71dd` filtered methods by declared model support; the support table was narrower than reality, so working methods were hidden. | Gate removed; the four-method array restored. The gate returned once via a `main`→branch merge and was removed a second time. | **None** — see §3 | `5aab4b0` / #132, reverting `bfc71dd` / #127 | Medium |
| **D11** | Redis evicted cache entries during ordinary dataset warm-up, so warmed results were gone before they were read. | `docker-compose.yml` capped Redis at `--maxmemory 256mb`; SRS §3.4.3 specifies 2 GB. | `--maxmemory 2gb`. | **None** — see §3 | `5aab4b0` / #132 | Medium |
| **D12** | A failed attribution was rendered identically to a successful one — the user could not tell a real explanation from a fallback. | No provenance was attached to attribution results. | FR17 provenance contract: every attribution carries `measured` / `fallback` / `unavailable`, and the UI renders the three states distinctly. | `test_a_synthesised_pattern_is_never_reported_as_measured`, `test_prediction_only_calls_carry_no_provenance_noise` (`test_inference_consistency.py`); `test_result_carries_the_flag` (`test_custom_model_fidelity.py`) | `4abe530` / #118 | **Critical** |

All backend test files cited above live in `Backend/tests/`.

---

## 3. Defects with no guarding test

These three are fixed in code but **nothing would catch a regression**. Listed
explicitly rather than left as a silent gap; each is a candidate follow-up
issue.

| ID | Why there is no test | What a test would need |
|----|----------------------|------------------------|
| **D09** — worker lock prefix | Verified by search: no test in `Backend/tests/` references `_cleanup_stale_worker_locks`. The bug is a string-literal mismatch, invisible to any test that imports the constant rather than re-typing it. | Write a stale lock using `WORKER_LOCK_PREFIX`, run the cleanup, assert the key is gone. The test must build the key from the constant, or it reproduces the bug it is meant to catch. |
| **D10** — ADD XAI method gate | Frontend; no test references the XAI method list or `PredictionPanel`'s method array. | A component test asserting all four methods render for an ADD model — or better, a Playwright assertion in `Frontend/e2e/dataflow.spec.ts`, since this defect regressed once already via a merge. |
| **D11** — Redis memory cap | Verified by search: `maxmemory` appears in no test file. It is deployment configuration, not application code, so no unit test observes it. | A config-lint check asserting `docker-compose.yml`'s cap matches the SRS §3.4.3 figure. |

---

## 4. D13, since fixed

**D13 — the accent-bias diagnostic transcribed accented English as another
language.** Found 2026-09-18, **fixed 2026-09-20**, verified present on
`testing` on 2026-09-28. It is kept in its own section because the fix moved
published FR15 figures, so the before-and-after matters more than the row would
convey.

A second defect surfaced only once this one was fixed: two word-error-rate
implementations disagreed on every cohort, because one stripped punctuation and
the other did not. Both are recorded below.

| Field | Detail |
|-------|--------|
| **Symptom** | Cohort mean WER is wildly inflated: Vietnamese 1.3328 against a median of 0.1742, Arabic 1.1285 against 0.1603. A WER above 1.0 means insertions outnumbered the reference words. |
| **Root cause** | `make_whisper_transcriber` (`app/domain/accent_bias_profiler.py:151`) builds the HF pipeline with no `language` argument and calls it with no `generate_kwargs`, so Whisper runs language ID per utterance. On heavily accented English it selects the speaker's L1 and transcribes into that language, then the decoder enters a repetition loop. |
| **Evidence** | 2 of 120 samples exceed WER 1.0. `TLV-arctic_b0251` (reference "They must have been swept away by the chaotic currents") returns Vietnamese text repeating "xe bánh" ~20 times, WER **22.30**. `ABA-arctic_b0066` returns Arabic text repeating "أن أرغب", WER **17.80**. Two further samples at WER 1.00 come back in Indonesian/Malay. |
| **Verified fix** | Passing `generate_kwargs={"language": "en", "task": "transcribe"}` collapses both: `TLV-arctic_b0251` 22.30 → **0.20**, `ABA-arctic_b0066` 17.80 → **0.30**. Measured directly, not inferred. |
| **Impact** | Two samples (1.7 % of the corpus) move the overall mean WER from 0.1617 to 0.4931. The bias ranking itself is affected: Vietnamese and Arabic rank worst almost entirely because of one clip each. |
| **Severity** | **High** — the diagnostic is reported as an accent-bias measurement, and for two cohorts it is substantially measuring language misdetection instead. |
| **Guarding test** | `test_evaluation_scoring.py::TestWerNormalisation`, which also asserts the two word-error-rate paths agree on the same input, so they cannot drift apart again. |

This also bears on the figures in `TESTING_AND_EVALUATION.md` §6 (mean WER 0.1353,
Δ 0.0670), which a current run does not reproduce — see `docs/DEMO_RUNBOOK.md`
§8.2. Excluding the two outliers gives 0.1617 / 0.1542, closer to §6 but still
not equal, so the language defect explains part of the gap and does not fully
close it. Both belong in one follow-up issue.

---

## 5. Defects found and fixed on 2026-09-28

These were found by running the software, not by reading it. Each row states how
it was measured, because three of them looked like something else first.

| ID | Symptom | Root cause | Fix | Guarding test |
| -- | ------- | ---------- | --- | ------------- |
| **D14** | Every transcription re-paid the model load. The API log showed the weights loading over and over. | `_get_whisper_pipeline` and its `_pipeline_cache` existed with **zero callers**. `transcribe_whisper` built the pipeline inline instead, duplicating the same logic, so the cache was never populated. Dead code that nothing failed on, because the output was correct and only slow. | Call the existing helper. | `test_model_pipeline_cache.py`, 5 tests. Verified by reverting: 2 fail with the right diagnostics. |
| **D15** | Backend suite hung outright, no error, no progress. Seen 3 times in one session. Previously attributed to coverage instrumentation. | **A re-entrancy deadlock.** redis-py's `Pipeline.__del__` calls `reset()`, which sends UNWATCH. When the collector fires that finaliser while fakeredis is mid-command, the UNWATCH re-enters a socket that is not re-entrant. Intermittent because it depends on GC timing, hence on memory pressure, hence it never appears when the tests run alone. | Suspend collection across the worker drain in the two affected test modules, plus a 300 s per-test timeout so any future hang fails instead of stalling CI. | Full suite completed twice with no hang; the timeout makes a recurrence visible. |
| **D16** | Enqueue p95 70-75 ms against the SRS 3.4.1 budget of 50 ms at 10 concurrent users. | The three family enqueues ran sequentially. A loopback Redis round trip measured **1.67 ms** on Docker Desktop for Windows and RQ issues 16 commands per enqueue, so round-trip count, not work per command, was the cost. | Batch the three independent family enqueues into one Redis pipeline. The aggregator still follows, because its `depends_on` needs the jobs to exist. | Orchestrator suite, 78 passed. Aggregator verified still DEFERRED on 3 dependencies. |
| **D17** | 19 nodes at critical severity: 18 unnamed buttons, 1 unlabelled form element, plus 2 unnamed ARIA inputs and no main landmark. | Icon-only buttons with no accessible name. Separately, Radix puts `role="slider"` on the **Thumb** while `aria-label` was being passed to the **Root**, so every slider reported a violation even where a label had been supplied. | Labels derived from each control's own tooltip text rather than a generic string; slider label forwarded to the Thumb in the shared wrapper. | `accessibility.spec.ts`, 4 checks, zero serious or critical. |
| **D18** | Accessibility suite reported a false pass. | Every Playwright context is a fresh profile, so the first-run dialog opened and its modal overlay hid the workbench from the accessibility tree. The scan was examining the dialog. | Dismiss the dialog before scanning. | Fixing this is what exposed D17. |

### D19, dependency vulnerabilities, 2026-09-29

**Python: from 33 advisories across 5 packages to none.**

| Package | Was | Now | Note |
| ------- | --- | --- | ---- |
| starlette | 0.37.2, **14 advisories** | 1.7.0, clear | The ASGI layer under every request, so the one that mattered. FastAPI 0.111 pinned `starlette<0.38.0` while the fixes start at 0.40.0, so the framework had to move too: 0.111.0 to 0.141.1. |
| anyio | 4.4.0, 2 advisories | 4.15.1, clear | |
| accelerate | 1.14.0, 1 advisory | 1.15.0, clear | 1.14.0 had no fix available. |
| pip | 22.3, 14 advisories | 26.2.1, clear | Build tooling, never shipped. |
| pytest | 8.4.2, 2 advisories | 9.1.1, clear | The old `<9` ceiling existed because pytest-asyncio 0.23.7 pinned `pytest<9`; both moved together, to pytest-asyncio 1.4.0. |

`pip-audit` now reports **no known vulnerabilities**. Verified across the whole
upgrade: 775 passed, 7 skipped, 0 failed; the app starts; a real multitask
request returns a correct job envelope; and the Postman collection passes 39
assertions over 13 requests against the upgraded API.

One scare worth recording, because it was a measurement error and not a
regression: after the upgrade `len(app.routes)` dropped from 74 to 20 and
appeared to show the routers had failed to register. They had not. Starlette 1.x
nests routers instead of flattening them, so `app.routes` counts something
different. The OpenAPI document still lists all 63 endpoints. Counting the wrong
structure is not the same as losing routes.

**JavaScript production dependencies: from 15 advisories to 1.**

| Package | Was | Now |
| ------- | --- | --- |
| plotly.js and maplibre-gl | **2 critical** (XSS sanitiser bypass) | plotly.js 3.0.3 to 4.1.1, clear |
| react-router, react-router-dom | 2 moderate (open redirect, CVE-2025-) | 7.18.4, clear |
| lodash, postcss, nanoid, glob, minimatch, brace-expansion, picomatch, @remix-run/router | 10 high | cleared by a non-breaking `npm audit fix` |

Both majors were checked before being applied rather than after:
`react-plotly.js` peers on `plotly.js: >1.34.0`, so v4 satisfies it, and the
router usage is only `BrowserRouter`, `Routes`, `Route` and `useLocation`, all
stable across v6 to v7. Verified: typecheck clean, 75 Jest tests, build
succeeds, and 10 Playwright checks including the embedding plot that actually
renders through plotly.

**Both of these were reported as open here and have since been closed.** The
original text is replaced rather than annotated, because one of its two claims
was factually wrong and leaving it in place would preserve the error.

*lodash, 1 high, production. **Closed.*** `_.template` code injection, reached
through `recharts@2.13.0`. This was first recorded here as unfixable, on the
stated grounds that "lodash 4.17.21 is the last of the 4.x line and there is no
patched 4.x release". **That was wrong, and it was wrong because it was asserted
from memory instead of checked.** `npm view lodash versions` lists 4.17.23,
4.18.0 and 4.18.1; the advisory range is `<=4.17.23`, so 4.18.1 is patched.
`npm audit` had been saying `fixAvailable: true` the whole time, which is the
contradiction that should have been chased on the first pass. Fixed with an
`overrides` entry pinning `lodash: ^4.18.1` in `Frontend/package.json` —
recharts asks for `^4.17.21`, so it is satisfied and nothing else moves.
**`npm audit --omit=dev` now reports 0 vulnerabilities**, and the CI gate was
tightened from critical to high accordingly.

*25 dev-only advisories, 1 critical. **Down to 4, 0 critical.*** These are build
and test tooling and never reach the runtime image, which contains only `dist/`
behind nginx. **14 of them arrived with `newman` and `newman-reporter-htmlextra`,
added during this test effort**, carrying handlebars (the critical), node-forge,
underscore and others transitively — a self-inflicted increase in audit surface.
Both were removed from `devDependencies`; the collection is the artefact worth
keeping and the runner does not need to be a declared dependency, so the
documented command is now `npx newman run ...`. The capability is unchanged.

Remaining: **4 advisories, 3 moderate and 1 high, all dev-only.** The high is
`vite`, whose fix is a three-major bump to 8.3.1. It is reported rather than
gated, because a dev-server advisory does not reach a user and a major bump of
the build tool is its own change with its own testing.
`vite` also carries one high advisory whose fix is a three-major bump to 8.3.1,
which should be its own change.

### A fix that was measured and then withdrawn

The enqueue handlers were briefly changed from `async def` to `def`, on the
reasoning that a synchronous body with blocking Redis calls should run in
FastAPI's threadpool rather than on the event loop. A synthetic test appeared to
confirm it: five concurrent requests went from 1.02 s to 0.22 s.

Against a live Redis it was **worse**, repeatably:

| | Median | p95 |
| - | ------ | --- |
| `async def` | 44 ms | 70-75 ms |
| `def` plus threadpool | 70 ms | 120 ms |

The synthetic test had used a 200 ms stub. A real loopback round trip is 1.67 ms,
and per-request thread dispatch costs more than that. The change was reverted and
the test deleted, because a test asserting the slower shape is worse than no
test. The measurements are recorded in `inference.py` so the same reasoning is
not repeated.

The lesson generalises: a synthetic benchmark whose parameters do not match
production can prove the opposite of the truth, confidently.

### D20-D22, requirements with no implementation, 2026-09-29

Found by checking the **submitted** SRS and SAD line by line against the tree,
rather than by running the software. All three are the same shape: a requirement
that was half-implemented and therefore never failed a test, because the half
that existed was the visible half.

| ID | Requirement | What was actually there | Fix | Guarding test |
| -- | ----------- | ----------------------- | --- | ------------- |
| **D20** | SR1: every upload validated for MIME type, **size (<= 100 MB)**, **duration (<= 15 min)**, magic number and structure before hashing. | The size cap was enforced (LIT-160). The duration was computed, returned to the UI, and **never compared against anything**. A 40-minute 8 kHz mono clip is ~38 MB, so it passed the size gate and fanned out to five workers each holding the whole decoded array. | Reject over the cap with 413 and delete the file, `AUDIOLIT_MAX_UPLOAD_SECONDS` configurable, 0 disables. | `test_upload_limits.py::TestDurationCap`, 3 tests. Verified by disabling the check: the rejection test fails. |
| **D21** | SR4 and SAD §3.2 constraint C4: uploaded audio is transient and purged on a configurable TTL; only analysis records are kept. | The Mongo tier had a TTL and sessions had `SESSION_TTL_SECONDS`, but the **audio files themselves had neither**. The only deletion path was `DELETE /upload/{file_id}`, which the browser had to remember to call, so every closed tab and failed request left a clip on disk permanently. The constraint was documented, asserted in a code comment in `settings.py`, and unimplemented. | `purge_expired_uploads()`, run on each upload and at startup, `AUDIOLIT_UPLOAD_RETENTION_SECONDS` configurable, 0 keeps everything. | `test_upload_limits.py::TestRetentionSweep`, 4 tests including one that fails if the handler stops calling the sweep. Verified by removing the call. |
| **D22** | SR7: container images **and Python and JavaScript dependencies** vulnerability-scanned on every build. | Only the images were scanned (Trivy, CRITICAL). `pip-audit` and `npm audit` were run by hand for D19 and appeared nowhere in `.github/workflows/ci.yml`, so the scan that found 33 advisories was a one-off, not a gate. | `pip-audit` on the already-installed backend environment after pytest, `npm audit --omit=dev` in the frontend job. | CI itself. The npm gate is set at `critical` rather than `high`, because the tree carries one high advisory with no patched release (lodash, via recharts); a high gate would be permanently red and would stop reporting anything. |

The upload route had **no tests at all** before this, which is the more useful
finding: D20 and D21 both sat in the one entry point every other requirement
depends on. Two further SR claims were checked and **hold** — SR5's keys are
digests, and SR6's CORS is regex-restricted to localhost, not a wildcard — and
SR1's magic-number clause is met in a stronger form than written, since the
route decodes the file with librosa and rejects what will not decode, which a
header-byte check would not catch.

---

## 6. Verification-method note

Three findings during this work were **wrong on first measurement** and were
corrected before they reached a fix:

- An enqueue latency reported at 2100 ms (42× budget) was an artefact of
  `urllib` splitting headers and body across packets; `requests` measured
  23.6 ms. Caught because the load test said 41 ms and the contradiction was
  chased rather than averaged away.
- Acoustic profiling "taking 13 s" was a cold-start run; warm it is 0.1 s.
- A claimed ~1.8× saliency slowdown disappeared once the untouched control path
  was measured and had moved by the same factor — it was machine load.

This is recorded because it bears on how the rest of the table should be read:
no single timing observation was treated as evidence anywhere above.

---

## 7. Lessons

The defects that mattered most were not crashes. D04, D05, D06 and D07 all
returned a confident, well-formed, wrong answer, and D04 actively fabricated
evidence — it showed one model's attention as another's. In an interpretability
tool that is the worst available failure mode, because the output's whole
purpose is to be trusted as measurement. Unit tests calling functions directly
could not see any of them; each needed either a second call to compare against
or a real end-to-end path. Two structural conclusions follow: **never substitute
a model silently** — return nothing and label it, which is what the FR17
provenance contract now enforces; and **derive keys, never re-type them**, since
D07 and D09 are the same bug wearing different clothes. The three rows in §3 are
the honest remaining debt.
