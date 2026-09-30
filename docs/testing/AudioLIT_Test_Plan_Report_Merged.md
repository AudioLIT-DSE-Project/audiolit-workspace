# AudioLIT

# Test Plan Report

**Version 3.0, merged edition**

**Date:** 2026-09-20

**Team:** Tharusha Perera, Rahim Iqbal, Ravindu Pathirana

---

## Revision History

| Date | Version | Description | Author |
| ---- | ------- | ----------- | ------ |
| 2026-09-18 | 1.0 | Master Test Plan, first structure and first macOS evidence pass | Ravindu Pathirana |
| 2026-09-19 | 1.1 | Re-verification against `origin/testing`, MongoDB tier brought into scope, Windows evidence pass added | Tharusha Perera |
| 2026-09-20 | 2.0 | Expanded plan: requirement traceability matrix, failover scenario matrix, Postman and Lighthouse added, two-host comparison | Rahim Iqbal |
| 2026-09-20 | 3.0 | Merged edition. Combines the version 2.0 structure and traceability with measured data science results, data science error analysis, and the defects found and fixed on 2026-09-20. Corrects the attributed cause of the test suite hang. | Tharusha Perera |

---

## How this merged edition was assembled

Two separate documents existed. One was strong on structure, requirement
traceability and breadth of tooling. The other carried measured data science
results and the error analysis behind them. Neither was complete on its own.

This edition takes the structure, the target test item tables, the requirement
traceability matrix and the failover scenario matrix from version 2.0, and adds
sections 5, 6 and 7, which report what the data science parts actually scored,
why two of those scores were wrong, and what was fixed.

Two things were checked rather than copied across. Every one of the 40 test
files named in the traceability matrix was confirmed to exist in the repository,
and the FR11 coverage gap was confirmed by searching the test directory. One
claim was corrected, and it is explained in section 4.3.

---

## Table of Contents

1. Evaluation Mission and Test Motivation
2. Target Test Items
3. Test Approach
4. Software Testing Results
5. Evaluation of the Data Science Parts
6. Error Analysis of the Data Science Parts
7. Defects Found and Fixed in This Test Cycle
8. Deliverables
9. Risks, Dependencies, Assumptions and Constraints
10. References

---

# 1. Evaluation Mission and Test Motivation

## 1.1 Background

AudioLIT is an interpretability workbench for three audio machine learning
tasks: automatic speech recognition, speech emotion recognition, and audio
deepfake detection. It extends the open source ECHO 1.0 baseline. The backend is
FastAPI with Redis and RQ for background work, the frontend is React 18 with
TypeScript and Vite, and a MongoDB tier stores analysis metadata.

Speech and audio models are used to make consequential judgements: what a
speaker said, how they felt, and whether a recording is genuine. A workbench
that explains those judgements carries the weight of the judgement itself.

## 1.2 Why testing this product is not ordinary web testing

Four properties shape this test effort, and none of them apply to a general web
application.

**The product is an explanation, not only a prediction.** A wrong transcript is
a visible defect. A saliency map that looks reasonable but does not reflect what
the model actually did is an invisible defect, and it is worse, because the user
acts on it believing it is true.

**Most interpretability outputs have no ground truth.** There is no single
correct heatmap for a clip that a test can compare against. Ordinary input and
expected output checks are not enough on their own, so this plan relies on
relationships that must hold between two runs wherever a direct answer does not
exist.

**Inference is slow and not fully repeatable.** The cache is therefore not an
optional speed up that testing can ignore. The reproducibility claim depends on
it, and the requirements demand that identical requests return identical cached
responses.

**The inherited baseline is known to be defective.** ECHO 1.0 quietly replaced
failed attention extraction with a fabricated pattern and returned it in the
same shape as a real one. It also labelled a panel with the name of one
attribution method while running a different one underneath. Both are treated
here as first class regression targets, because a workbench built for faithful
interpretation cannot inherit unfaithful interpretation.

## 1.3 Mission for this evaluation

Three missions are adopted from the template.

1. **Verify a specification.** The committed functional requirements FR1 to FR4,
   FR6 to FR12 and FR15 to FR17, and the security requirements SR1 to SR7,
   behave as the SRS states.
2. **Find defects.** Particularly the class of defect this product is prone to,
   where a well formed output is confidently wrong.
3. **Assess risk before submission.** Report what was measured, on what
   hardware, and what remains unmeasured.

A fourth objective is added in this edition, because the submission requires it:
**measure the data science parts and analyse their errors**, rather than only
describing how they would be measured.

## 1.4 Scope boundary

Only committed functionality is in scope. Stretch items are out of scope,
including multi-model side by side comparison. There is no FR5, FR13 or FR14 in
the reconciled SRS, and this report does not invent rows to look complete.

---

# 2. Target Test Items

## 2.1 Build under test

| Item | Value |
| ---- | ----- |
| Branch | `origin/testing` at commit `d5ecf95` |
| Plus | The fixes listed in section 7, made during this test cycle |

## 2.2 Items produced by the project team

| Category | Target test items | Relative importance |
| -------- | ----------------- | ------------------- |
| API gateway | The FastAPI application and its routers, covering upload, inference, saliency, perturbation, acoustic profiling, evaluation, datasets, dataset management, models, results, session, tasks, metrics, health and debug | High. Every user facing capability passes through here. |
| Interpretability and model engines | `app/domain/`: model registry and loader, hook manager, saliency service, acoustic profiler, perturbation service, accent bias profiler, evaluation service, provenance | High. The interpretability claims of the product live here. |
| Orchestration fabric | `app/orchestration/task_orchestrator.py` and `worker.py`, with five worker families: asr, ser, add, xai, mutation | High. Historically the site of two duplicate module incidents. |
| Cache tier | `app/infrastructure/cache_keys.py` and `app/core/redis.py`, over a Redis 7 keyspace | High. The reproducibility guarantee depends on it. |
| Metadata tier | `app/infrastructure/metadata_store.py` over MongoDB 6, holding `models`, `audio_samples`, `analysis_results` and `bias_reports` | Medium to high. Durable state, so index and expiry behaviour matter. |
| Frontend | The workbench page and its panels, the saliency overlay canvas, waveform viewer, dataset table, embedding views, and the quick start walkthrough | High |

## 2.3 Items the product relies on

These are not produced by the team, but failures surface through them.

| Category | Items |
| -------- | ----- |
| Models under test | Whisper base for ASR, a Wav2Vec2 model for SER, and a Wav2Vec2 family deepfake detector |
| Corpora | Common Voice, LibriSpeech, RAVDESS, CREMA-D, L2-ARCTIC, ASVspoof 2021 DF, ESD |
| Third party libraries | PyTorch, Transformers, Captum, Librosa, soundfile, RQ, Redis, FastAPI, React, Vite |
| Processor and accelerator hardware | CPU only hosts and GPU capable hosts |
| Browser and screen | Chromium, Firefox and WebKit at desktop widths |

## 2.4 Explicitly out of scope

The availability of the Hugging Face model hub and the training time accuracy of
the pretrained models are outside the team's control. Browser engine internals
below the level the automation can observe are also out of scope.

---

# 3. Test Approach

This section describes how the items in section 2 are exercised to serve the
mission in section 1. The approach is automated first. As of this edition the
backend carries 729 collected pytest cases across 54 test files, the frontend
carries 62 Jest cases across 8 suites, and Playwright carries 28 cases across
five projects.

A note on how results were obtained. Every figure in section 4 came from running
the command shown beside it, on the environment stated there. Where a
measurement looked surprising it was checked a second way before being reported.
That discipline is not decorative: three earlier findings on this project turned
out to be artefacts of the measuring tool rather than faults in the product, and
one finding in this edition was corrected for the same reason.

## 3.1 Testing Techniques and Types

### 3.1.1 Data and Database Integrity Testing

The template assumes a SQL database with tables and an ORM. AudioLIT has no SQL
tier, so the section is reinterpreted as cache value integrity, metadata
document integrity, and corpus loader integrity.

| Row | Content |
| --- | ------- |
| **Technique Objective** | Exercise the Redis cache, the MongoDB metadata tier and the corpus loaders independently of the user interface, to find cache corruption, key collisions, wrong value shapes, missing indexes, or payload data leaking into the metadata tier. |
| **Technique** | Drive the cache manager directly against `fakeredis`, asserting round trip fidelity and that the value shape stored under each key family matches what its reading route expects. Drive the metadata store against `mongomock`, asserting all four collections exist with their indexes, that a re run refreshes one document rather than adding a duplicate, and that no collection ever holds audio bytes or tensors. Seed loaders with valid, truncated, wrong sample rate and structurally malformed audio. |
| **Oracles** | Deterministic for round trips and digests. Identical requests must yield byte identical cached responses. Metamorphic for warm against cold. `fakeredis` and `mongomock` are oracles for the application's own logic, not for real Redis eviction timing or real MongoDB index enforcement. |
| **Required Tools** | Redis 7 and MongoDB 6 in Docker, `fakeredis`, `mongomock`, `pymongo`, pytest, `redis-cli`, `mongosh` |
| **Success Criteria** | Every key family has at least one shape assertion. All four MongoDB collections exist with the correct unique and expiry indexes. The suite stays green with both Redis and MongoDB unreachable. |
| **Special Considerations** | No SQL tier, so SQL injection and schema normalisation do not apply. Tests must pass with no service containers running, because continuous integration has none. |

### 3.1.2 Function Testing

| Row | Content |
| --- | ------- |
| **Technique Objective** | Exercise ingestion, inference, attribution, acoustic profiling, mutation and auditing through the public API with valid and invalid input, to verify correct results, correct typed errors, and correct application of every committed rule (FR1 to FR4, FR6 to FR12, FR15 to FR17). |
| **Technique** | Route level tests through an async HTTP client against the live application, and domain level unit tests per engine. The highest value checks are the ones that confirm an output is what it claims to be: safetensors only ingestion with rejection before deserialisation (FR1), concurrent multi task dispatch on one clip (FR3), a full probability distribution for SER (FR6), binary verdict with confidence for ADD (FR7), Grad-CAM being genuinely gradient weighted and not equal to Integrated Gradients on the same input (FR8 against FR9), fallback attributions carrying a provenance flag (FR17), acoustic features validated against Librosa (FR10), mutations preserving the original and returning a correctly shaped 16 kHz mono clip (FR12), per cohort WER disparity over L2-ARCTIC (FR15), and the top K deletion score audit (FR16). The same surface is then run a second time from an independent harness, a Postman collection, so that a defect in the primary harness cannot hide a defect in the product. |
| **Oracles** | Deterministic for status codes, schema shape and label set membership. Tolerance based for numeric outputs against a Librosa reference and for WER within a delta. Metamorphic for the interpretability claims, because no ground truth saliency map exists. |
| **Required Tools** | pytest, pytest-asyncio, httpx, fakeredis, Postman |
| **Success Criteria** | Every committed requirement has at least one test that fails if the requirement is broken. |
| **Special Considerations** | Hub downloading tests are gated behind an environment variable so the ordinary run stays offline and fast. |

### 3.1.3 User Interface Testing

| Row | Content |
| --- | ------- |
| **Technique Objective** | Verify navigation, panel state, canvas interaction and playback synchronisation, and confirm that the data reaching a panel is the data that panel claims to show. |
| **Technique** | Component tests with Jest and React Testing Library. Cross browser layout tests with Playwright on Chromium, Firefox and WebKit at three widths, run without a backend so they stay fast. A separate full stack data flow suite that reads values off the network response rather than off the rendered pixels. A dedicated test for the first run quick start walkthrough. |
| **Oracles** | Deterministic. For the data flow suite the oracle is the response body. A heatmap appearing is not a passing condition; the heatmap being flagged as measured is. |
| **Required Tools** | Jest, React Testing Library, Playwright |
| **Success Criteria** | No layout overflow at any tested width, no uncaught page errors, and every panel under test showing values that match the response that fed it. |
| **Special Considerations** | The workbench holds a task WebSocket open and polls for progress, so the browser network never becomes idle. Tests must wait for a specific element rather than for network idle. This caused two real test failures in this cycle, recorded in section 7. |

### 3.1.4 Performance Profiling

| Row | Content |
| --- | ------- |
| **Technique Objective** | Measure response time for each class of operation against the SRS section 3.4.1 table, and state the hardware every figure was measured on. |
| **Technique** | Time each operation class separately, because the budgets differ: a cached read, a cache miss that enqueues work, a cold inference, an attribution, and an acoustic profile. Separate the single user case from the concurrent case. Profile memory as growth across repeated calls rather than absolute process size, which is host dependent. Measure the frontend with Lighthouse against the production build rather than the development server. Capture the exported operational metrics alongside the wall clock figures. |
| **Oracles** | The SRS targets, but only for the hardware they assume. This environment is CPU only and the targets assume a GPU, so model bound figures are reported as observations rather than as pass or fail. |
| **Required Tools** | Locust, Lighthouse, the exported operational metrics |
| **Success Criteria** | The targets that do not depend on model execution are met. |
| **Special Considerations** | A timing measured while another heavy job runs measures machine load, not the product. Every figure in section 4.4 was taken with the machine otherwise idle. |

### 3.1.5 Load Testing

| Row | Content |
| --- | ------- |
| **Technique Objective** | Put the system under concurrent load and observe whether response times, queue depth and failure rate stay acceptable past the ordinary working point, and confirm it degrades rather than collapses. |
| **Technique** | Drive the running stack over HTTP with a weighted mix of cached reads, job enqueues, attribution requests, cold predictions, acoustic profiling and health checks. Warm the cache for the clips used by the cached path before measuring, so that the statistic measures hits rather than first runs. Score the observed 95th percentile against the SRS budgets. Observe queue depth per family under the concurrency one GPU pin. |
| **Oracles** | Tolerance based against the SRS budgets, using the 95th percentile rather than the mean, because the targets describe what a user should reliably get. The decisive oracle is behavioural rather than numerical: past saturation the system must queue and degrade, never corrupt state, silently drop a job, or lose a progress channel. |
| **Required Tools** | Locust |
| **Success Criteria** | No request failures. Infrastructure targets met at the 95th percentile. No data loss or silent job drop at or beyond saturation. |
| **Special Considerations** | Concurrency is kept modest on CPU. Queueing past the GPU concurrency pin is correct behaviour, not a defect. Model bound targets are enforced only when an environment flag is set. |

### 3.1.6 Security and Access Control Testing

Mapped one to one against SR1 to SR7, plus the inherited items in the SRS.

| Row | Content |
| --- | ------- |
| **Technique Objective** | Confirm that one session cannot read another session's data, that debug routes are not exposed under production configuration, that uploads are validated, that model artefacts are format checked before deserialisation, and that known vulnerable dependencies are visible to the team. |
| **Technique** | Negative assertions, where the correct result is a refusal. A session must not fetch another session's dataset by identifier. An unlisted origin must be rejected. A corrupted audio file must be rejected rather than silently accepted. A non safetensors artefact must be refused before any deserialisation happens. Dependency scanning for both ecosystems and image scanning in the pipeline. |
| **Oracles** | Deterministic for the refusal tests. The advisory databases are the oracle for dependency scanning. |
| **Required Tools** | pytest, `npm audit`, `pip-audit`, Trivy in continuous integration, curl for raw header and CORS probing, safetensors for format verification |
| **Success Criteria** | All refusal tests pass, and every advisory is recorded with a severity even where it is not yet fixed. |
| **Special Considerations** | There is no authentication tier, so access control means session isolation rather than roles. The resource list names OWASP and CheckMarx for this row; OWASP guidance is used as the checklist, and the automated equivalent here is the dependency and image scanning. Manual exploratory probing remains open. |

### 3.1.7 Failover and Recovery Testing

The template frames this section around power loss and disk controllers. That is
not this system's failure surface. The real one is a dependency disappearing
while work is in flight.

| Row | Content |
| --- | ------- |
| **Technique Objective** | Remove a dependency and observe whether the system recovers without human intervention, with no data loss and no silently wrong answer, which is the one unacceptable outcome. |
| **Technique** | A scenario matrix. F1, run with the MongoDB tier off and call every store method. F2, stop Redis while the API is serving and restart it. F3, kill a worker mid job and restart it. F4, restart the API while a background job is running. F5, fill the cache past its memory cap and observe eviction. |
| **Oracles** | Deterministic. The health endpoint must report the true state. A restarted worker must pick up work. No in flight job becomes permanently unobservable. |
| **Required Tools** | Docker, pytest, the health endpoints |
| **Success Criteria** | No crash when a dependency is absent, and no permanently blocked worker family after an unclean exit. |
| **Special Considerations** | These scenarios are destructive and are run last, after all other evidence is captured, against a local stack only. There is no redundant infrastructure and no continuous service level agreement. The metadata tier degrades quietly when MongoDB is absent, which is correct behaviour but also means a misconfiguration can hide itself. See section 9. |

### 3.1.8 Configuration Testing

| Row | Content |
| --- | ------- |
| **Technique Objective** | Verify the product behaves the same across the browser engines, operating systems, language runtimes, accelerators and deployment topologies it is expected to run on, and identify any configuration dependent behaviour. |
| **Technique** | Browsers: Playwright across Chromium, Firefox and WebKit at three widths. Operating systems: macOS, Windows 11 and Ubuntu on continuous integration. Runtimes: Python 3.10 on the pipeline against 3.11 locally, Node 20 against Node 26. Accelerator: CPU only against GPU. Redis: containerised against unreachable. Topology: the native development setup against the containerised stack. |
| **Oracles** | A cross configuration differential oracle. The same functional suite must produce the same pass or fail outcome across configurations, and any divergence is itself the finding rather than noise to average away. |
| **Required Tools** | Continuous integration, Playwright browser projects, Docker Compose |
| **Success Criteria** | The suite passes on every declared configuration, and any configuration dependent difference is explained. |
| **Special Considerations** | The pipeline installs a CPU only build of the machine learning library, so it can never provide GPU path evidence. That must come from a developer machine with a GPU. |

### 3.1.9 Accessibility Testing

| Row | Content |
| --- | ------- |
| **Technique Objective** | Confirm the workbench meets WCAG 2.1 level AA, which the SRS commits to. |
| **Technique** | Axe-core through Playwright for the WCAG 2.1 A and AA rule sets, covering contrast, accessible names, form labels, landmark structure, page title and document language. Lighthouse for a whole page score against the production build. A keyboard walk that presses Tab repeatedly and records what receives focus. Violations rated serious or critical fail the build. Minor and moderate violations are reported but do not fail, so a new regression is not buried under pre existing low severity noise in an inherited interface. |
| **Oracles** | The WCAG 2.1 A and AA rule sets as implemented by axe-core. |
| **Required Tools** | `@axe-core/playwright` 4.13.0, `axe-core` 4.13.0, Lighthouse 13.5.0, Playwright |
| **Success Criteria** | No serious or critical violations on the workbench. Every committed panel has at least one automated test. No horizontal overflow at any tested width. |
| **Special Considerations** | Automated scanning finds only part of what is wrong. Axe-core also reports a separate incomplete list that needs human judgement, and those are recorded rather than counted as passes. Screen reader testing and the keyboard walk still need a manual pass, which has not been completed. |

---

# 4. Software Testing Results

## 4.1 Evidence passes

Three evidence passes exist. Reporting all three is deliberate, because the
differences between them are findings in their own right.

| Item | Pass 1, macOS | Pass 2, Windows | Pass 3, Windows, this edition |
| ---- | ------------- | --------------- | ----------------------------- |
| Date | 2026-09-18 | 2026-09-19 | 2026-09-20 |
| Host | Apple M3 Pro, 18 GB | Windows 11 | Windows 11 |
| Python | 3.11.15 | 3.11.0 | 3.11.0 |
| Node | 26.4.0 | 20 | 20 |
| Runtime | PyTorch 2.13.0, CUDA not available | 2.13.0+cpu | 2.13.0+cpu |
| Backend | 716 passed, 6 skipped, 135.04 s | 715 passed, 7 skipped, 382.80 s | 717 passed, 7 skipped, 1134.23 s |
| Jest | 55 passed, 7 suites | 55 passed, 7 suites | 62 passed, 8 suites |
| ESLint | 0 errors, 113 warnings | 0 errors, 110 warnings | 0 errors, 110 warnings |
| Build | Succeeds, 11.10 s | Succeeds, 46.9 s | Succeeds, 48.6 s |

Two observations follow directly from this table. The same suite takes between
2.8 and 8.4 times longer on the Windows host than on the macOS host, which
matters when setting any timeout. And the test counts grew across passes because
tests were added during the cycle, so a count is only meaningful next to its
date and commit.

Supporting services for pass 3: Redis 7.4.10 and MongoDB 6.0.28 in Docker, the
API on port 8000 with five RQ worker families registered, and the Vite server on
port 8080. The software was started and exercised for real. The metadata tier
created all four collections with their expiry and uniqueness indexes, and the
workbench loaded and listed dataset rows in a browser.

## 4.2 Results by category

| Category | Tool | Result |
| -------- | ---- | ------ |
| Backend unit and integration | pytest, 54 files, 729 collected | 717 passed, 7 skipped, 0 failed |
| Frontend component | Jest, 8 suites | 62 passed, 0 failed |
| Static analysis | ESLint | 0 errors, 110 warnings |
| Production build | Vite | Succeeds in 48.6 s |
| Cross browser layout | Playwright on three engines | Passed on all three |
| Quick start walkthrough | Playwright | Passed after the fix in section 7 |
| Accessibility | axe-core with Playwright | **Fails.** Contrast fixed, but 19 critical severity nodes remain. See section 4.7 |
| API verification, second harness | newman 6.2.2 with the Postman collection | 10 requests, 25 assertions, 0 failures |
| Whole page audit | Lighthouse 13.5.0, desktop preset, production build | Performance 40, Accessibility 96, Best Practices 100, SEO 100 |
| Full stack data flow | Playwright | 5 of 6 passed, one intermittent |
| Load | Locust | 470 requests, 0 failures |
| Dependency scanning, JavaScript | npm audit | 15 advisories in production dependencies |
| Dependency scanning, Python | pip-audit | 33 advisories across 5 packages |
| Container image scanning | Trivy in the pipeline | Configured and running |

## 4.3 Backend suite, and a correction about the hang

The suite is run with the Redis address pointed at an unreachable port,
deliberately matching the condition the pipeline runs under, because a locally
reachable Redis hides failures that only appear in the pipeline.

```
cd Backend
REDIS_URL="redis://127.0.0.1:1/0" python -m pytest -q -rs
```

Result for pass 3: **717 passed, 7 skipped, 0 failed, 0 errors**, in 18 minutes
54 seconds. All seven skips are environment gated and state a reason.

| Skipped test | Reason |
| ------------ | ------ |
| `test_fanout_orchestrator.py:192` | Needs a forking start method, which Windows does not provide |
| `test_function_testing.py:303` | Requires GPU and model resources |
| `test_memory_profiling.py:35` | Video memory test requires CUDA |
| `test_ser_checkpoint.py:139`, `:152`, `:161` | Each downloads about 1.2 GB from the Hugging Face Hub, gated behind an environment variable |
| `test_task_orchestrator.py:502` | No broker reachable, which is the intended condition for this run |

**A correction to the previous edition.** Version 2.0 recorded that a coverage
run hangs at the sixth test in `test_multitask_orchestrator.py` and attributed
the hang to coverage instrumentation, noting that the same six tests pass in an
ordinary run.

A second full run during this cycle **hung at the same file with no coverage
instrumentation at all**. It stopped producing output at 58 percent and never
finished. The log size was sampled twice with no change between samples, and the
run was terminated.

The attribution to coverage instrumentation therefore does not hold. The hang is
in the orchestrator tests themselves, and coverage merely made it easier to hit.
This matches the repository guidance, which records that a burst mode worker
draining a dependency gated aggregator against `fakeredis` has hung the test run
before and has twice hidden a real pipeline hang.

The practical consequence is stated plainly. **A single green run is not
sufficient evidence for this suite.** The orchestrator tests should be stress run
in a loop with a timeout before any release, and line coverage remains
unavailable until the hang is fixed rather than worked around.

## 4.4 Performance and load

Locust was run twice, at ten concurrent users and at a single user, so that
concurrency effects can be separated from the cost of the operation itself.

**Ten concurrent users, 90 seconds, 470 requests, 0 failures.**

| Operation | Requests | Median | 95th percentile | Budget | Result |
| --------- | -------- | ------ | --------------- | ------ | ------ |
| Cached prediction | 319 | 15 ms | 58 ms | 200 ms | Pass |
| Enqueue multitask | 118 | 58 ms | 110 ms | 50 ms | Fail |
| Health | 33 | 11 ms | 52 ms | Not specified | Observation |

**Single user.**

| Operation | Requests | 95th percentile | Budget | Result |
| --------- | -------- | --------------- | ------ | ------ |
| Cached prediction | 35 | 23 ms | 200 ms | Pass |
| Enqueue multitask | 28 | 50 ms | 50 ms | Pass, exactly at the limit |

The cached read path has comfortable headroom, using about a quarter of its
budget even under concurrency. The enqueue path is the weak point: it sits
exactly on its 50 ms budget with one user and misses it at ten, where the 95th
percentile is 110 ms. Both measurements were taken so that the finding is not
mistaken for a measurement artefact, and they agree on the direction.

Model bound operations were not enforced, because this host has no GPU and the
targets assume one. For reference, a Whisper Grad-CAM attribution measured 21 to
37 seconds per clip called directly on this CPU host.

## 4.5 Security scanning

**JavaScript production dependencies, 15 advisories: 2 critical, 10 high, 2
moderate, 1 low.**

| Severity | Package | Issue |
| -------- | ------- | ----- |
| Critical | maplibre-gl | Cross site scripting through a sanitiser bypass |
| Critical | plotly.js | Inherited from a dependency |
| High | @remix-run/router, react-router, react-router-dom | Cross site scripting through open redirects |
| High | lodash | Code injection through the template function |
| High | postcss | Cross site scripting in stringify output |
| High | nanoid, glob, minimatch, brace-expansion, picomatch | Denial of service, command injection and related issues |

**Python dependencies, 33 advisories across 5 packages.**

| Package | Advisories | Note |
| ------- | ---------- | ---- |
| starlette 0.37.2 | 14 | The web framework layer under FastAPI, so the most important to address |
| pip 22.3 | 14 | Tooling, not shipped with the product |
| anyio | 2 | |
| pytest 8.4.2 | 2 | Test tooling only |
| accelerate | 1 | |

The machine learning runtime could not be audited because it is installed from a
local build that is not on the public package index. That is a gap, not a clean
result.

None of these were fixed in this cycle. Upgrading the web framework layer and
the routing library both risk behaviour changes, so each needs its own change
with its own testing rather than being folded into a test report.

## 4.6 Full stack data flow

Five of six checks pass: clip selection binds to the editor, the transcript
column never shows raw JSON, word segments name words the transcript actually
contains, and a genuine speech clip is not reported as a deepfake at full
confidence.

One check is intermittent. The Grad-CAM provenance test failed in one run
because the saliency response returned provenance `fallback`, with the reason
that the attribution was empty or constant and encoder energy was shown instead.
The same attribution was then generated directly on the same clip three times
and returned provenance `measured` each time, with healthy variance in the
saliency matrix. The underlying attribution code is therefore working, and the
failure is timing or state related. It is recorded as open rather than closed,
because an intermittent failure in exactly the check that guards against fake
explanations is not something to wave through.

One related observation came out of that investigation. The application log
shows the speech model weights being reloaded on many requests rather than held
in memory, which is the main reason an attribution through the API is much
slower than the same attribution called directly.

## 4.7 Accessibility, and a correction to an earlier result

**The accessibility tier fails.** This corrects an earlier version of this
report, which recorded the tier as passing with no serious or critical
violations. That result was wrong, and the reason matters more than the result.

The scan ran against a fresh browser profile, so the first run quick start
dialog was open and its modal overlay covered the workbench. Axe scanned the
dialog. The controls underneath were inert and hidden from the accessibility
tree, so they were never examined, and the scan reported clean. The suite was
passing because the page under test was covered up.

This is the same blind spot as defect 5 in section 7. The dialog overlay was
found blocking the data flow suite and fixed there, and the accessibility suite
was not checked for the identical problem. It had it.

After dismissing the dialog before the scan, the real state is:

| Rule | Impact | Nodes | Meaning |
| ---- | ------ | ----- | ------- |
| `button-name` | **Critical** | 18 | Buttons have no discernible text, so a screen reader announces nothing useful |
| `label` | **Critical** | 1 | A form element has no label |
| `aria-input-field-name` | Serious | 2 | An ARIA input field has no accessible name |
| `landmark-one-main` | Moderate | 1 | The document has no single main landmark |
| `color-contrast` | Serious | 0 remaining | Was 8 nodes, fixed during this cycle, see defect 1 |

Two scan states are reported separately, because they find different things.
With the dialog open, only contrast violations are visible. With the dialog
dismissed, the workbench contributes the critical rows above. A report that
scans one state only will understate the problem, which is exactly what
happened.

The contrast fixes made during this cycle are real and are retained. The
headline was the part that was wrong.

Lighthouse scores accessibility at 96 for the same page. That is not a
contradiction: Lighthouse runs a subset of the axe rule set and weights it
differently, so a high Lighthouse accessibility score can coexist with critical
axe violations. Where the two disagree, the axe result is the one to act on.

## 4.8 API verification from a second harness

The route surface is exercised a second time from a different client, so that a
defect in the primary pytest harness cannot hide a defect in the product.

```
npx newman run Backend/apitests/AudioLIT.postman_collection.json \
  --env-var baseUrl=http://127.0.0.1:8000 \
  -r cli,htmlextra --reporter-htmlextra-export docs/evaluation/api-reports/newman-report.html
```

Result: **10 requests, 25 assertions, 0 failures**, total run 2 seconds, average
response time 122 ms.

The collection is read only. It asserts contract properties rather than merely
checking for a 200: that the health broker flag is a real boolean and not a
truthy string, that the worker count agrees with the length of the worker list,
that an unknown route returns 404 without leaking a stack trace, and that an
unknown job identifier is reported as `not_found` rather than as a
zero progress running job that a client would poll forever.

Report: `docs/evaluation/api-reports/newman-report.html`.

## 4.9 Whole page audit

Lighthouse 13.5.0, desktop preset, run against the production build served from
`dist/` rather than the development server.

| Category | Score |
| -------- | ----- |
| Performance | **40** |
| Accessibility | 96 |
| Best Practices | 100 |
| SEO | 100 |

| Metric | Value |
| ------ | ----- |
| First Contentful Paint | 2.1 s |
| Largest Contentful Paint | 2.7 s |
| Total Blocking Time | 1,000 ms |
| Cumulative Layout Shift | 0.005 |
| Speed Index | 4.5 s |

The performance score of 40 is the first measured figure for a risk both earlier
versions of this report carried but neither had quantified: the frontend ships
as a single chunk of roughly 1.78 MB gzipped with no code splitting. Total
Blocking Time of 1,000 ms is the dominant contributor, which is consistent with
a large single bundle being parsed and executed before the page becomes
interactive. Layout stability is good at 0.005.

Report: `docs/evaluation/api-reports/lighthouse-report.report.html`.

---

# 5. Evaluation of the Data Science Parts

Two data science requirements are measured: accent bias in ASR (FR15) and
attribution faithfulness (FR16). Both were regenerated for this edition using a
single reproducible command, so the numbers can be checked rather than taken on
trust.

## 5.1 Accent bias profiling, FR15

**Method.** Group wise word error rate across six first language cohorts from
the L2-ARCTIC corpus, 20 samples each, 120 samples in total, using Whisper base.
Word error rate is word level edit distance after both reference and hypothesis
are normalised identically, lower cased with punctuation removed. The bias
discrepancy index is the difference between the worst and best cohort mean.

```
cd Backend
python scripts/evaluate_models.py --asr-model-id openai/whisper-base --skip-faithfulness --output ../docs/evaluation/results
```

| Cohort | Samples | Mean WER | Median WER |
| ------ | ------- | -------- | ---------- |
| Arabic | 20 | 0.2060 | 0.1538 |
| Vietnamese | 20 | 0.1812 | 0.1603 |
| Spanish | 20 | 0.1413 | 0.0000 |
| Korean | 20 | 0.1347 | 0.1056 |
| Mandarin | 20 | 0.1246 | 0.0955 |
| Hindi | 20 | 0.0969 | 0.0385 |

- Overall mean word error rate: **0.1430**
- Bias discrepancy index: **0.0779**

**Reading of the result.** The model is measurably worse on Arabic accented
English than on Hindi accented English, and the gap between best and worst cohort
is about 7.8 percentage points of word error rate. That is a real disparity and
it is the finding the requirement exists to surface. It is not large enough to
call the system unusable for any cohort, but it is large enough that a user
should be told which cohorts were measured.

## 5.2 Attribution faithfulness auditing, FR16

**Method.** Deletion scoring. The most salient regions identified by the
attribution are masked, inference is run again on the masked audio, and the drop
in confidence is measured. If an attribution is faithful, removing what it marks
as important should reduce the model's confidence. The deletion area under curve
integrates that drop across masking levels. Attributions that came from a
fallback are refused rather than scored, because scoring a fallback measures the
fallback and not the model.

```
cd Backend
python scripts/evaluate_models.py --manifest eval_manifest.json --model-type ser --skip-accent-bias --top-k 0.1 0.3 0.5 --output ../docs/evaluation/results_fr16
```

Three CREMA-D clips with the speech emotion model, masking at 10, 30 and 50
percent:

- Mean deletion score, that is the mean confidence drop: **0.2907**
- Mean deletion area under curve: **0.1408**
- Audio scored: 3 of 3. Refused as fallback attribution: 0

**Reading of the result.** Masking the most salient regions reduces model
confidence by about 29 percent on average, so the attributions carry real
information about what the model uses. Nothing was refused, so all three
attributions were genuinely measured rather than fallbacks.

This is a small sample. It demonstrates that the audit pipeline works end to
end; it is not a final faithfulness figure for the product. A larger run across
more clips and the other two tasks is the next step.

---

# 6. Error Analysis of the Data Science Parts

This section is the most consequential part of this report. Both data science
metrics were wrong when this cycle started, and neither was wrong in a way that
would have been noticed by looking at the product.

## 6.1 First defect: the speech model was guessing the language

**Symptom.** The accent bias evaluation reported an overall mean word error rate
of 0.6019 and a bias discrepancy index of 1.1676. Two cohorts had mean error
rates above 1.0, which is only possible when the model inserts more words than
the reference contains.

**Investigation.** The per sample results were sorted worst first. Two samples
out of 120 stood out at 22.30 and 17.80. Their transcriptions were not poor
English. They were Vietnamese and Arabic text, repeating the same phrase many
times. Two further samples came back in Indonesian or Malay.

**Root cause.** The profiler built the speech recognition pipeline without
specifying a language, so the model ran language identification on every
utterance. On heavily accented English it selected the speaker's first language,
transcribed into that language, and then fell into a repetition loop.

**Why it mattered.** The two worst cohorts were worst only because of one bad
clip each. The published ranking was not measuring accent difficulty for those
cohorts. It was measuring language misdetection.

**Fix.** The decode language is fixed to English, which is correct because the
corpus is read English throughout and the language is known rather than
something to guess.

**Effect.** Overall mean moved from 0.6019 to 0.2587 and the discrepancy index
from 1.1676 to 0.0481. The two pathological samples dropped from 22.30 and 17.80
to 0.20 and 0.30.

## 6.2 Second defect: two word error rate calculations disagreed

**Symptom.** After the first fix the summary still did not add up. Every cohort
mean sat between 0.097 and 0.206, yet the overall mean printed above the same
table read 0.2587, higher than every value it was supposedly summarising.

**Investigation.** The report drew its cohort table from one code path and its
headline figures from another. Compared directly, they disagreed on every
cohort, consistently, with the second path always higher.

| Cohort | Profiler path | Evaluation service path |
| ------ | ------------- | ----------------------- |
| Arabic | 0.2060 | 0.2917 |
| Hindi | 0.0969 | 0.2436 |
| Korean | 0.1347 | 0.2544 |
| Mandarin | 0.1246 | 0.2466 |
| Spanish | 0.1413 | 0.2515 |
| Vietnamese | 0.1812 | 0.2645 |

**Root cause.** The evaluation service compared words after lower casing and
splitting on whitespace but without removing punctuation. The speech model
returns punctuation, so a correctly recognised final word such as "child."
failed to match the reference word "child" and was counted as an error. The
profiler already removed punctuation and had a comment explaining why. The two
had drifted apart.

**Why it mattered more than it looks.** The inflation was near constant across
cohorts, because every cohort's transcripts end in a full stop at a similar rate.
Adding a near constant to every cohort barely changes their order, but it does
compress the gap between them relative to their size, and the bias discrepancy
index is exactly that gap. The published index was 0.0481 when the real spread
was 0.1091. The metric whose entire job is to measure disparity was understating
that disparity by more than half.

**Fix.** Both sides now normalise identically. A test was added asserting that
the two code paths agree on the same input, so they cannot drift apart silently
again.

**Effect.** Overall mean moved from 0.2587 to 0.1430 and the discrepancy index
from 0.0481 to 0.0779.

## 6.3 Combined effect

| Stage | Overall mean WER | Bias discrepancy index |
| ----- | ---------------- | ---------------------- |
| Before either fix | 0.6019 | 1.1676 |
| After the language fix | 0.2587 | 0.0481 |
| After the word error rate fix | **0.1430** | **0.0779** |
| Previously documented in the evaluation document | 0.1353 | 0.0670 |

The corrected measurements are close to the figures documented earlier, which is
a good sign for both. The remaining difference is small and expected, since the
earlier figures were produced on different hardware and are not guaranteed to
have used an identical sample selection.

## 6.4 Failure modes observed in the data

Beyond the two defects, three patterns in the corrected results are worth
recording.

1. **Accented English can be misread as another language entirely.** This is the
   most severe failure mode found, because the output is confident, fluent and
   completely wrong, and a user who does not read Vietnamese would have no way to
   tell. The fix removes it for this corpus, but any deployment accepting speech
   in an unknown language will face it again.

2. **Cohort means are driven by a minority of hard clips.** Spanish has a median
   word error rate of 0.0000, meaning at least half its samples were transcribed
   perfectly, yet its mean is 0.1413. Reporting only the mean would suggest the
   model is uniformly mediocre on Spanish accented speech when it is usually
   perfect and occasionally poor. Both figures are therefore reported.

3. **The cohort ranking changed after the fixes.** Before, Vietnamese looked like
   the worst cohort. After, Arabic is. Any conclusion drawn from the earlier
   ranking, including any statement about which speakers the system serves worst,
   was based on a measurement error.

---

# 7. Defects Found and Fixed in This Test Cycle

Seven defects were found by running the software. Six are fixed, and each fix
has a test that fails without it where a test is possible.

| # | Defect | Where | Fix | Status |
| - | ------ | ----- | --- | ------ |
| 1 | Eight serious colour contrast violations against WCAG 2.1 AA. White text on the primary colour measured 3.80 to 1 and the amber warning text 2.95 to 1, both under the 4.5 to 1 minimum for normal text | `src/index.css`, `Toolbar.tsx`, `WarmupModal.tsx`, `WarmupStatusBanner.tsx`, `CustomDatasetManager.tsx` | Primary darkened from 32 to 26 percent lightness, destructive from 60 to 44 percent, amber text one shade darker. Each value computed to the threshold rather than chosen by eye. A first attempt at 27 percent measured 4.45 to 1 once browser alpha blending was accounted for, so a second iteration was needed | Fixed, covered by the new accessibility suite |
| 2 | The speech model guessed the language and transcribed accented English into other languages | `accent_bias_profiler.py` | Decode language fixed to English | Fixed, effect measured in section 6.1 |
| 3 | Word error rate counted punctuation as recognition errors, and two code paths disagreed on every cohort | `evaluation_service.py` | Both paths normalise identically | Fixed, with a test asserting the two agree |
| 4 | The quick start walkthrough test waited for the browser network to become idle, which never happens because the workbench holds a WebSocket open. It passed only when run without a backend and failed as soon as the full stack was running | `e2e/quickstart.spec.ts` | Waits for the workbench element instead | Fixed |
| 5 | The quick start dialog opened over the workbench on first load and its overlay intercepted pointer events, so the entire full stack data flow suite could not click anything. Five of six checks failed for this reason | `e2e/dataflow.spec.ts` | The dialog is dismissed before the page loads. It keeps its own coverage in its own suite | Fixed, four more checks now pass |
| 6 | The evaluation script could not read the attribution vector out of a saliency response, so every faithfulness item was refused and the stage reported nothing measurable | `scripts/evaluate_models.py` | The extractor reads the field the service actually returns, and falls back to flattening the saliency matrix | Fixed, faithfulness now measures |
| 7 | Grad-CAM returned a fallback instead of a measured attribution in one full stack run | Not yet located | Not fixed | Open, see section 4.6 |
| 8 | **The accessibility suite scanned a covered page and reported a false pass.** Every Playwright context is a fresh profile, so the quick start dialog opened and its overlay hid the workbench from the accessibility tree | `e2e/accessibility.spec.ts` | The dialog is dismissed before the scan, matching the fix already applied to the data flow suite | Fixed. The corrected scan then exposed defect 9 |
| 9 | 18 buttons have no discernible text and one form element has no label, both at critical severity, plus two ARIA inputs with no accessible name and no main landmark on the document | Workbench components, not yet narrowed to specific files | Not fixed | **Open.** Found only after defect 8 was fixed |
| 10 | Frontend Lighthouse performance score of 40, driven by 1,000 ms Total Blocking Time from a single unsplit bundle | `Frontend` build configuration | Not fixed | Open, see section 4.9 |

Defect 5 deserves a note. Both sides were individually correct. The quick start
dialog worked and had passing tests, and the data flow suite had passing tests.
They broke only in combination, and only with the full stack running, which is a
condition neither suite is normally run under. This is the third time on this
project that two separately green changes have failed together, so it should be
treated as an expected failure mode rather than bad luck.

---

# 8. Deliverables

## 8.1 Test Evaluation Summaries

Each automated run records the suite name, tests collected, passed, failed and
skipped, the wall clock duration, the commit it ran against, and the exact
command, so any result can be reproduced independently. Manual results are
recorded as a dated observation against the check they address, with the
reviewer named.

Automated suites run on every pull request. The data flow suite, the load tests
and the accessibility scan need a running stack and are run on demand.

## 8.2 Reporting on Test Coverage

The centre of coverage reporting is the requirement to test matrix below. Every
row was checked against the repository rather than assumed from the SRS. For
this edition all 40 test files named below were confirmed to exist, and the FR11
gap was confirmed by searching the test directory.

Line coverage is not available, and the reason is itself a finding. `pytest-cov`
is now installed, so coverage was attempted, but the run does not finish. See
section 4.3 for the corrected explanation.

| Requirement | Summary | Technique | Test files | Coverage |
| ----------- | ------- | --------- | ---------- | -------- |
| FR1 | Dynamic Hugging Face model ingestion, safetensors only | 3.1.2, 3.1.6 | `test_model_registry_service.py`, `test_models_routes.py`, `test_custom_model_fidelity.py` | 3 files |
| FR2 | Benchmark dataset ingestion and management | 3.1.1, 3.1.2 | `test_dataset_ingestion.py`, `test_dataset_service.py`, `test_datasets_routes.py`, `test_dataset_management_routes.py`, `test_l2arctic_loader.py`, `test_librispeech_loader.py`, `test_asvspoof_loader.py` | 7 files |
| FR3 | Asynchronous multi task inference | 3.1.2, 3.1.5, 3.1.7 | `test_task_orchestrator.py`, `test_multitask_orchestrator.py`, `test_fanout_orchestrator.py`, `test_queue.py` | 4 files, but see the note below |
| FR4 | Deterministic cache by hash | 3.1.1, 3.1.4 | `test_redis_cache.py`, `test_results_cache.py`, `test_hashing.py`, `test_warmup_cache_contract.py` | 4 files |
| FR6 | Speech Emotion Recognition | 3.1.2 | `test_ser_model.py`, `test_ser_corpora.py`, `test_ser_checkpoint.py` | 3 files, checkpoint tests partly Hub gated |
| FR7 | Audio Deepfake Detection | 3.1.2 | `test_deepfake_classifier.py`, `test_asvspoof_loader.py`, `test_degradation_scoring.py` | 3 files |
| FR8 | Spectrogram LIME and SHAP, Grad-CAM | 3.1.2 | `test_grad_cam.py`, `test_saliency_service.py`, `test_saliency_routes.py`, `test_spectrogram_attribution.py` | 4 files |
| FR9 | Integrated Gradients, label correction | 3.1.2 | `test_integrated_gradients.py`, `test_grad_cam.py` | 2 files. Regression critical: the two must be asserted as distinct outputs, not merely both present |
| FR10 | Acoustic wave profiling | 3.1.2 | `test_acoustic_profiler_service.py`, `test_acoustic_routes.py` | 2 files |
| FR11 | Latent projection explorer | 3.1.2, 3.1.3 | `test_latent_projection.py` | 1 file, 10 tests. **Was 0 files at the time of the original edition; see the addendum.** |
| FR12 | Canvas driven signal mutation | 3.1.2, 3.1.3 | `test_perturbation_service.py`, plus `PerturbationTools.test.tsx` and `SpectrogramGridSelector.test.tsx` | 1 backend file, thin for four sub clauses |
| FR15 | Accent bias profiling | 3.1.2, and measured in section 5.1 | `test_accent_bias_profiler.py`, `test_accent_bias_runner.py`, `test_l2arctic_loader.py`, `test_evaluation_routes.py` | 4 files, plus a measured result |
| FR16 | Attribution faithfulness auditing | 3.1.2, and measured in section 5.2 | `test_auc_faithfulness.py`, `test_faithfulness.py`, `test_evaluation_scoring.py`, `test_high_saliency_masking.py` | 4 files, plus a measured result |
| FR17 | Faithful attention extraction with fallback flag | 3.1.2 | `test_hook_manager_service.py`, `test_provenance.py` | 2 files, regression critical as FR9 |
| SR1 to SR7 | Security requirements | 3.1.6 | `test_security.py`, `test_session_cookie.py`, `test_debug_and_tasks_routes.py`, `test_dataset_service.py`, plus the dependency and image scans | Automated subset executed, manual probing open |

There is no FR5, FR13 or FR14 row, because the reconciled SRS does not define
them: they are vacated identifiers left by a renumbering, not requirements that
were dropped. The submitted SRS numbers the same requirements FR1 to FR14 with
no gaps, so its FR13 and FR14 are this matrix's FR15 and FR16, and its FR5 is
SER, committed here as FR6. The full mapping is erratum E4 in `docs/README.md`.
Multi-model side-by-side comparison, which internal notes call "dropped FR5", is
an unnumbered out-of-scope item in SRS section 4.4. This matrix does not invent
rows to look complete.

**Gaps this matrix exposes.**

1. ~~**FR11 has no dedicated backend test file.**~~ **Closed.** The route and the
   dimensionality reduction dependency both existed, and the frontend had the
   embedding context and panel, but nothing on the backend asserted the
   projection was correct. `test_latent_projection.py` now covers output shape,
   method selection, determinism, cluster separation and row order preservation.
   Row order is the one that matters most: the panel binds output rows to
   filenames by position, so a reorder attaches every point to the wrong file
   with no visible symptom.
2. **FR12 has thin backend coverage** relative to its four SRS sub clauses, which
   cover non destructive originals, Web Audio behaviour, shape and sample rate.
3. **FR3 can pass at the orchestration level while the ASR result itself is
   empty**, which is exactly the class of defect this product is prone to: the
   plumbing is verified, the payload is not.
4. **Line coverage is unavailable** because the coverage run does not complete.

---

# 9. Risks, Dependencies, Assumptions and Constraints

## 9.1 Risks

| Risk | Mitigation | If the risk happens |
| ---- | ---------- | ------------------- |
| The enqueue path misses its 50 ms target under concurrency | Profile the enqueue path before the demonstration and keep concurrency low | State the measured figure rather than quoting the target |
| Two critical and ten high severity advisories in shipped JavaScript dependencies, and fourteen in the web framework layer | Schedule the upgrades as their own change with their own testing | Record the advisories in the submission rather than implying a clean scan |
| No GPU in the test environment, so model bound targets are unmeasured | Run those measurements on a GPU host before submission | Report every figure with the hardware named, and never present a CPU number against a GPU target |
| The backend suite can hang in the orchestrator tests rather than fail, which looks like a slow run | Stress run the orchestrator tests in a loop with a timeout. Never treat one green run as proof | Quarantine the specific test and record it, rather than rerunning until it passes |
| Grad-CAM can fall back to encoder energy and be reported as a fallback | The provenance contract surfaces this to the user | Investigate defect 7 before relying on attributions in a demonstration |
| ~~FR11 has no backend test at all~~ **Closed** | Covered by `test_latent_projection.py` | Reporting the gap explicitly is what got it assigned and closed, rather than letting the matrix imply coverage that did not exist |
| The metadata tier degrades quietly when MongoDB is absent | Check the health endpoint and the collection list after starting the stack | A misconfigured deployment can look healthy while storing nothing, so verify rather than assume |
| The product reloads model weights on many requests | Accept for now, since a cached read is fast | Budget tens of seconds for any live attribution during a demonstration |
| Two separately correct changes can break in combination | Run the full stack suite after any merge, not only the fast suites | This has now happened three times, so treat it as expected |
| A data science metric can be wrong in a way the product does not show | Verify the metric itself before reporting any number it produces | Re-measure and publish the correction, as done in section 6 |

## 9.2 Dependencies

Docker for Redis and MongoDB. The Hugging Face cache or Hub for the models used.
The corpora provisioned locally, which are licence restricted and research use
only for RAVDESS, L2-ARCTIC, ESD and ASVspoof 2021 DF. GPU access for the model
bound performance figures.

## 9.3 Assumptions

Single tenant academic deployment with best effort availability and no
continuous service level agreement. No user authentication or role based access
tier exists or is planned.

## 9.4 Constraints

The video memory budget, which is why GPU family concurrency is pinned to one. A
safetensors only model loading policy, with no arbitrary deserialisation. The
dataset working footprint bound. The pipeline's CPU only build of the machine
learning library, which means the pipeline can never be the source of GPU path
evidence.

---

# 10. Addendum, 2026-09-29

This section records what changed after the body of the report was written. The
body is left as it stood, because a test report whose findings are quietly
edited to match a later, better state stops being evidence of anything. Where a
row above is now false, it is struck through and points here.

## 10.1 Findings closed since the body was written

**FR11 now has backend tests.** `test_latent_projection.py`, 10 tests over
`reduce_dimensions`: output shape per requested width, empty input, finiteness,
method selection raising rather than falling back to PCA, PCA and t-SNE
producing different projections, determinism for both, cluster separation
surviving the projection, and row order preservation. The last is the one a
wrong answer hides in, because a reordered projection still renders as a
plausible scatter plot with every point attached to the wrong file.

**Line coverage is now available: 70 %.** The body reported it as unavailable
because the run did not finish, and called that a finding. It was. The cause
turned out not to be the coverage plugin at all but a re-entrancy deadlock in
redis-py's pipeline finaliser against fakeredis, recorded as D15 in the defect
log. With that fixed the suite completes and coverage reports 70 %. The largest
untested surface is `app/api/routes/inferences.py`, 763 statements at 8 %.

**The Python dependency advisories are cleared.** The body reported fourteen
advisories in the web framework layer. `pip-audit` now reports none, after
Starlette 0.37.2 to 1.7.0 (which required FastAPI 0.111.0 to 0.141.1, since
0.111 pinned `starlette<0.38.0`), anyio, accelerate, pip and pytest. One high
**The JavaScript side is now clean in production too.** `npm audit --omit=dev`
reports 0 vulnerabilities. This is a correction as much as an improvement: the
body, and an earlier version of this addendum, recorded the production `lodash`
high as unfixable because "there is no patched 4.x release". There is —
4.18.1, outside the advisory's `<=4.17.23` range — and `npm audit` had been
reporting `fixAvailable: true` all along. An `overrides` entry pinning
`lodash: ^4.18.1` clears it without moving recharts. Separately, `newman` and
`newman-reporter-htmlextra` were removed from `devDependencies`, since they were
added by this test effort and brought 14 advisories including the only critical
one; the collection still runs through `npx newman`. **Total: 25 advisories
(1 critical, 13 high) down to 4 (0 critical, 1 high).** The remaining high is
`vite`, dev-only, fix is a three-major bump.

## 10.2 Gaps found by checking the submitted documents against the tree

Three requirements were implemented only in the half that was visible, so
nothing failed. All three are now closed, and all three were in the upload
route, which had no tests of any kind before this.

| Requirement | What was missing | Now |
| ----------- | ---------------- | --- |
| SR1's duration cap, 15 minutes | The size cap was enforced and the duration was computed, displayed, and never compared against anything. A 40-minute 8 kHz mono clip is about 38 MB, so it passed the size gate and fanned out to five workers. | Rejected with 413, file deleted, `AUDIOLIT_MAX_UPLOAD_SECONDS` configurable. 3 tests. |
| SR4 and constraint C4, transient audio purged on a configurable TTL | Sessions and the Mongo tier had TTLs. The audio files had none. The only deletion path was an explicit DELETE the browser had to remember to send, so every closed tab left a clip on disk permanently. The constraint was documented, asserted in a code comment, and unimplemented. | `purge_expired_uploads()` on each upload and at startup, `AUDIOLIT_UPLOAD_RETENTION_SECONDS` configurable. 4 tests. |
| SR7, Python and JavaScript dependencies scanned on every build | Only the container images were scanned. `pip-audit` and `npm audit` were run by hand and appeared nowhere in the workflow, so the scan that found 33 advisories was a one-off rather than a gate. | Both are now CI steps. |

The npm gate is set at **high** on production dependencies, which is what SR7
asks for and which passes cleanly. It was briefly set at critical instead, on
the false premise that the lodash high was unfixable; once that turned out to be
wrong the premise for the looser gate went with it. Dev dependencies are
deliberately not gated, and that is stated here rather than left as a silent
threshold: the remaining high is `vite`, and a build-tool advisory does not reach
a user.

**One requirement is checked and open, by design.** SR3 requires TLS, and SAD
section 3.2's last constraint requires browser-to-server traffic to be
encrypted. The repository terminates plain HTTP: `Frontend/nginx.conf` listens
on 8080 with no TLS block, and `COOKIE_SECURE` defaults to false. This is
correct for the deployment the product is actually built for, a single academic
host reached over localhost, and the cookie flag is a setting rather than a
hardcode, so a deployment behind a TLS terminator only has to set it. It is
recorded as open rather than claimed as met, because an operator putting this on
a network needs to know the encryption is theirs to add and not something the
application provides.

Two further security requirements were checked and hold. SR5's cache keys are
digests and carry no filenames or tokens. SR6's CORS is a regex restricted to
localhost, not the wildcard the inherited baseline used. SR1's magic-number
clause is met in a stronger form than written: the route decodes the file with
librosa and rejects what will not decode, which catches a valid header in front
of a corrupt body that a header-byte check would pass.

## 10.3 One correction to the body

The coverage matrix said "FR5, multi-model comparison, was moved to stretch
scope". FR5 in the submitted SRS is Speech Emotion Recognition, which is
committed and delivered as FR6. The identifier FR5 is vacated by a renumbering,
and multi-model comparison is an unnumbered out-of-scope item. The matrix's
conclusion was right, its reason was not. Erratum E4 in `docs/README.md` has the
full mapping, and it is worth reading before citing any FR id from a submitted
document, because the submitted SRS is inconsistent with itself: its requirement
list runs FR1 to FR14 while its own later sections already cite FR7, FR16.1 and
FR17.

---

# 11. References

Every version below was read from the installed environment on the Windows
evaluation host, and every link was checked on 20 September 2026. Entries marked
"macOS host, reported" were not verifiable from the Windows host and are carried
on the teammate's statement.

**Test frameworks and runners**

1. pytest 8.4.2, unit and integration testing for Python. https://docs.pytest.org
2. pytest-asyncio 0.23.7, asyncio support for pytest. https://pytest-asyncio.readthedocs.io
3. pytest-cov, coverage plugin. **Not installed on the Windows host.** Version 7.1.0 reported on the macOS host. https://pytest-cov.readthedocs.io
4. Jest 29.7.0, JavaScript testing framework. https://jestjs.io
5. jest-environment-jsdom 29.7.0, the browser-like environment used by the component tests. https://www.npmjs.com/package/jest-environment-jsdom (this package is published from the Jest repository; the jsdom library it wraps is a separate project with separate versioning, at https://github.com/jsdom/jsdom)
6. React Testing Library 16.3.2. https://testing-library.com/docs/react-testing-library/intro
7. Playwright 1.63.0, cross-browser automation with bundled Chromium, Firefox and WebKit. https://playwright.dev
8. Locust 2.46.5, load testing in Python. https://locust.io

**Test doubles and fixtures**

9. fakeredis 2.23.2, in-memory Redis substitute. https://github.com/cunla/fakeredis-py
10. mongomock 4.3.0, in-memory MongoDB substitute. https://github.com/mongomock/mongomock
11. httpx 0.27.0, the async HTTP client used by the route tests. https://www.python-httpx.org

**Accessibility and API verification**

12. axe-core 4.13.0, the accessibility rule engine. https://github.com/dequelabs/axe-core
13. @axe-core/playwright 4.13.0, the Playwright binding. https://github.com/dequelabs/axe-core-npm
14. Lighthouse 13.5.0, whole page auditing, run with the desktop preset against the production build. https://developer.chrome.com/docs/lighthouse
15. Postman collection format, `Backend/apitests/AudioLIT.postman_collection.json`. https://www.postman.com
16. newman 6.2.2, headless runner for Postman collections. https://github.com/postmanlabs/newman
17. newman-reporter-htmlextra 1.23.1, the HTML reporter used for the run report in section 4.8. https://github.com/DannyDainton/newman-reporter-htmlextra

**Static analysis and security scanning**

18. ESLint 9.35.0 on the Windows host; 9.9.0 reported on the macOS host. https://eslint.org
19. typescript-eslint 8.65.0. https://typescript-eslint.io
20. npm audit, dependency scanning for Node, run through npm 11.16.0. https://docs.npmjs.com/cli/commands/npm-audit
21. pip-audit, dependency scanning for Python. https://github.com/pypa/pip-audit
22. Trivy, container image scanning, run in continuous integration as `aquasecurity/trivy-action@v0.36.0`, verified at `.github/workflows/ci.yml` lines 85 and 92. https://github.com/aquasecurity/trivy

**Build, runtime and deployment**

23. Python 3.11.0 on the Windows host, 3.11.15 on the macOS host, 3.10 on the continuous integration runners, verified in `ci.yml`. https://www.python.org
24. Node.js 20 on the continuous integration runners and in the frontend build image, both verified in `ci.yml` and `Frontend/Dockerfile`; 26.4.0 reported on the macOS host. https://nodejs.org
25. Docker Engine. https://docs.docker.com
26. Docker Compose. https://docs.docker.com/compose
27. nginx, Alpine variant, serving the built frontend, verified as `nginx:alpine` in `Frontend/Dockerfile`. https://nginx.org
28. GitHub Actions, continuous integration. https://docs.github.com/actions

**System dependencies as installed**

29. FastAPI 0.111.0. https://fastapi.tiangolo.com
30. Starlette 0.37.2, the ASGI layer beneath FastAPI, named here because section 4.5 reports advisories against it. https://github.com/encode/starlette (the package's own documentation URL, `www.starlette.io`, did not resolve on the access date; the source repository is cited instead)
31. RQ 2.10.0, the Redis-backed task queue. https://python-rq.org
32. Redis 7, image `redis:7-alpine`, running 7.4.10. https://redis.io
33. MongoDB 6, image `mongo:6`, running 6.0.28. https://www.mongodb.com/docs
34. pymongo 4.18.1. https://pymongo.readthedocs.io
35. PyTorch 2.13.0. The Windows host runs the `+cpu` build, which has no accelerator backend. The macOS host runs the standard arm64 wheel, which carries the Metal Performance Shaders backend but no CUDA. Neither host exercised a CUDA path. https://pytorch.org
36. Transformers 5.15.0 on the Windows host; 5.14.1 reported on the macOS host. https://huggingface.co/docs/transformers
37. Captum 0.9.0, the attribution library. https://captum.ai
38. Librosa 0.11.0, audio analysis and the reference implementation for the acoustic checks. https://librosa.org
39. soundfile 0.14.0. https://python-soundfile.readthedocs.io
40. NumPy 1.26.4. https://numpy.org
41. React 18.3.1 and Vite 5.4.20. https://react.dev and https://vite.dev
42. TypeScript 5.6.3 on the Windows host; 5.5.3 reported on the macOS host. https://www.typescriptlang.org
43. jiwer, word error rate scoring, used by the accent bias profiler. https://github.com/jitsi/jiwer

**Standards and guidance**

44. Web Content Accessibility Guidelines (WCAG) 2.1, Level AA, W3C Recommendation, 5 June 2018. https://www.w3.org/TR/WCAG21
45. OWASP API Security Top 10, used as the checklist for section 3.1.6. https://owasp.org/API-Security
46. Rational Unified Process test plan template, the structural basis for this document, supplied as `docs/testing/Template for Test plan.docx`.

**Project documents**

- Software Requirements Specification, `docs/SRS.md`
- Software Architecture Document, `docs/SAD.md`
- Master Test Plan, `docs/testing/AudioLIT_Master_Test_Plan.md`
- Data Science Error Analysis, `docs/evaluation/DS_ERROR_ANALYSIS.md`
- Defect Log, `docs/evaluation/DEFECT_LOG.md`
- Generated evaluation results, `docs/evaluation/results/` and
  `docs/evaluation/results_fr16/`

**Project documents, continued**

53. AudioLIT project conventions and errata, `docs/README.md`.
54. AudioLIT issue plan and dependency map, `docs/ISSUE_PLAN.md`.
55. Master Test Plan design specification, `docs/testing/TEST_PLAN_DESIGN.md`.
56. Testing and evaluation document, `docs/evaluation/TESTING_AND_EVALUATION.md`.
57. Sample test plan report, Find Your Job, MPM Solutions, 2016, supplied as `docs/testing/Sample test plan report.pdf`.
58. ECHO 1.0 baseline, `AudioLIT-DSE-Project/ECHO`, forked from `AnasSAV/ECHO`.
59. Newman API run report, `docs/evaluation/api-reports/newman-report.html`.
60. Lighthouse audit, `docs/evaluation/api-reports/lighthouse-report.report.html`.

**Methods referenced in the test approach**

Page ranges and venues below were checked against the publishers rather than
taken from secondary sources.

61. R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 618 to 626. Verified against the CVF Open Access repository.
62. M. Sundararajan, A. Taly and Q. Yan, "Axiomatic Attribution for Deep Networks", Proceedings of the 34th International Conference on Machine Learning (ICML), PMLR volume 70, 2017, pp. 3319 to 3328. The Integrated Gradients method. Verified against PMLR.
63. M. T. Ribeiro, S. Singh and C. Guestrin, "Why Should I Trust You? Explaining the Predictions of Any Classifier", Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016, pp. 1135 to 1144. The LIME method. Verified against Crossref, DOI 10.1145/2939672.2939778.
64. T. Y. Chen, S. C. Cheung and S. M. Yiu, "Metamorphic testing: a new approach for generating next test cases", Technical Report HKUST-CS98-01, Department of Computer Science, Hong Kong University of Science and Technology, 1998. The basis for the metamorphic oracles in section 3.1. **Not independently verified.** The citation matches the form used widely in the literature, but the indexing services consulted were unavailable during this check, so the report number and year are carried unconfirmed rather than asserted.

---

*End of Test Plan Report, merged edition.*
