# AudioLIT

# Test Plan Report

**Version 1.0**

**Date:** 2026-09-20

**Prepared by:** Tharusha Perera

**Reviewed by:** Rahim Iqbal, Ravindu Pathirana

---

## Revision History

| Date | Version | Description | Author |
| ---- | ------- | ----------- | ------ |
| 2026-09-20 | 1.0 | First Test Plan Report. Covers software testing, evaluation of the data science parts, and error analysis of the data science parts. All results in this report were produced by running the software, not copied from earlier documents. | Tharusha Perera |

---

## Table of Contents

1. Evaluation Mission and Test Motivation
2. Target Test Items
3. Test Approach
   - 3.1 Testing Techniques and Types
     - 3.1.1 Data and Database Integrity Testing
     - 3.1.2 Function Testing
     - 3.1.3 User Interface Testing
     - 3.1.4 Performance Profiling
     - 3.1.5 Load Testing
     - 3.1.6 Security and Access Control Testing
     - 3.1.7 Failover and Recovery Testing
     - 3.1.8 Configuration Testing
     - 3.1.9 Accessibility Testing
4. Software Testing Results
5. Evaluation of the Data Science Parts
6. Error Analysis of the Data Science Parts
7. Defects Found and Fixed in This Test Cycle
8. Deliverables
9. Risks, Dependencies, Assumptions and Constraints
10. References

---

# 1. Evaluation Mission and Test Motivation

AudioLIT is an interpretability workbench for three audio tasks: automatic
speech recognition (ASR), speech emotion recognition (SER), and audio deepfake
detection (ADD). It extends the open source ECHO 1.0 baseline. The backend is
FastAPI with Redis and RQ, the frontend is React 18 with TypeScript and Vite,
and a MongoDB tier stores analysis metadata.

The mission of this test effort is different from a normal web application, and
that difference shapes everything below. AudioLIT does not only have to work. It
has to be believable. Its whole purpose is to show a user why a model produced
an answer, so an output that looks correct but is not actually measured is worse
than an output that is plainly missing. A heatmap that is really a fallback, an
attention map borrowed from a different model, or a bias number computed with a
broken metric will all render perfectly well on screen and mislead the person
reading them.

The testing objectives are therefore:

1. Confirm that each committed functional requirement behaves as specified.
2. Confirm that every explanation the product shows is either genuinely measured
   or is clearly labelled as a fallback.
3. Measure the data science parts, that is accent bias in ASR and attribution
   faithfulness, with metrics that are themselves verified.
4. Find the failures that unit tests cannot see, which on this project have all
   been contract mismatches between two sides of an interface.
5. Check the non functional requirements that the SRS commits to, including
   response time, accessibility and security.
6. Record what was not tested, and why, instead of leaving gaps unstated.

---

# 2. Target Test Items

The table lists the items identified as targets for testing.

| Group | Items | Criticality |
| ----- | ----- | ----------- |
| API surface | The FastAPI routers under `Backend/app/api/routes/`, covering upload, inference, saliency, perturbations, acoustic, evaluation, datasets, models, results, session, tasks, health and debug | High. Every user facing capability passes through here. |
| Domain and machine learning engines | `Backend/app/domain/`: model registry and loader, saliency service, acoustic profiler, perturbation service, accent bias profiler, evaluation service, provenance | High. The interpretability claims of the product live here. |
| Orchestration | `Backend/app/orchestration/task_orchestrator.py` and `worker.py`, with five worker families: asr, ser, add, xai, mutation | High. This is where background work is scheduled and where two duplicate module incidents happened previously. |
| Cache tier | `Backend/app/infrastructure/cache_keys.py` and `Backend/app/core/redis.py`, with a Redis 7 keyspace | High. Reproducibility depends on it. |
| Metadata tier | `Backend/app/infrastructure/metadata_store.py` with MongoDB 6, holding the collections `models`, `audio_samples`, `analysis_results` and `bias_reports` | Medium to high. Durable state, so index and expiry behaviour matter. |
| Frontend | The workbench page and its panels, including prediction, acoustic, saliency overlay, waveform, dataset table, embedding views, and the quick start walkthrough | High |
| Models under test | Whisper base for ASR, a Wav2Vec2 model for SER, and a Wav2Vec2 family deepfake detector | High |
| Corpora | Common Voice, RAVDESS, CREMA-D, L2-ARCTIC, ASVspoof 2021 DF, ESD | Medium. Licence restricted and sub sampled. |
| Environment | Windows 11 and Ubuntu, Python 3.10 and 3.11, Node 20, CPU only and GPU, Chromium, Firefox and WebKit | Medium |

**Out of scope for this report.** The training time accuracy of the pretrained
models, Hugging Face Hub availability, and browser engine internals below the
level the automation can observe. These are outside the team's control.

**Target of test.** Branch `origin/testing` at commit `d5ecf95`, plus the fixes
described in section 7, which were made during this test cycle.

---

# 3. Test Approach

The approach is automated first. The product already carries a large automated
suite, and this report adds the categories that were missing rather than
rewriting what works. Manual testing is reserved for the things a machine cannot
judge, such as whether an explanation is useful to a human reader.

A note on how results were obtained. Every number in section 4 came from running
the command shown next to it on the environment described there. Where a
measurement looked surprising it was checked a second way before being reported,
because on this project three earlier findings turned out to be artefacts of the
measuring tool rather than faults in the product.

## 3.1 Testing Techniques and Types

### 3.1.1 Data and Database Integrity Testing

| | |
| --- | --- |
| **Technique Objective** | Exercise the Redis cache and the MongoDB metadata tier independently of the user interface, to find cache corruption, key collisions, wrong value shapes, missing indexes, or payload data leaking into the metadata tier. |
| **Technique** | Drive the cache manager directly against `fakeredis`, checking that a value survives the encode and decode round trip, and that the value shape stored under each key family is the shape its reading route expects. Drive the metadata store against `mongomock`, checking that all four collections are created with their indexes, that a re run refreshes one document instead of adding a duplicate, and that no collection ever holds audio bytes or tensors. Load each corpus with valid, truncated and malformed audio. |
| **Oracles** | Deterministic for round trips. The same request must return a byte identical cached response. For the metadata tier the oracle is the schema itself, that is the set of collections and indexes the SRS specifies. `fakeredis` and `mongomock` are oracles for the application logic, not for real Redis eviction timing or real MongoDB index enforcement, so those are listed as pending. |
| **Required Tools** | Redis 7 and MongoDB 6 in Docker, `fakeredis`, `mongomock`, `pymongo`, `pytest`, `redis-cli`, `mongosh` |
| **Success Criteria** | Every cache key family has at least one shape assertion. All four MongoDB collections exist with the correct unique and expiry indexes. The suite stays green with both Redis and MongoDB unreachable. |
| **Special Considerations** | There is no SQL database and no ORM, so SQL injection and schema normalisation do not apply. The tests must pass with no service containers running, because the continuous integration pipeline has none. |

### 3.1.2 Function Testing

| | |
| --- | --- |
| **Technique Objective** | Exercise ingestion, inference, attribution, acoustic profiling, mutation and auditing through the public API, with valid and invalid input, to confirm correct results and correct typed errors. |
| **Technique** | Route level tests against the live FastAPI application using an async HTTP client, and unit tests for each domain engine. The checks with the most value are the ones that confirm an output is what it claims to be: that Grad-CAM is genuinely gradient weighted and not identical to another method on the same input, that a fallback attribution carries a provenance flag, that a custom model is explained by itself and not by the default model, and that a second model selection does not return the first model's cached answer. |
| **Oracles** | Deterministic where a correct answer exists, for example a typed error for an unsupported model. Metamorphic where it does not, for example the same input must give the same output on a second call. |
| **Required Tools** | `pytest`, `pytest-asyncio`, `httpx`, `fakeredis` |
| **Success Criteria** | Every committed functional requirement has at least one test that fails if the requirement is broken. |
| **Special Considerations** | Tests that would download model weights from the Hugging Face Hub are gated behind an environment variable so the normal run stays offline and fast. |

### 3.1.3 User Interface Testing

| | |
| --- | --- |
| **Technique Objective** | Confirm navigation, panel state, canvas interaction and playback behave correctly, and that data reaching a panel is the data that panel claims to show. |
| **Technique** | Component tests with Jest and React Testing Library. Cross browser layout tests with Playwright on Chromium, Firefox and WebKit at three desktop viewport sizes, run without a backend so they stay fast. A separate full stack data flow suite that reads values off the network response rather than off the rendered pixels, because a panel can render a well laid out picture of the wrong thing. A dedicated test for the first run quick start walkthrough. |
| **Oracles** | Deterministic. For the data flow suite the oracle is the response body, for example the provenance field must read `measured`, not merely that a heatmap appeared. |
| **Required Tools** | Jest, React Testing Library, Playwright |
| **Success Criteria** | No layout overflow at any tested viewport, no uncaught page errors, and every panel under test shows values that match the API response that fed it. |
| **Special Considerations** | The workbench keeps a WebSocket open and polls for progress, so the browser network never becomes idle. Tests must wait for a specific element instead of waiting for network idle. This caused two real test failures during this cycle, described in section 7. |

### 3.1.4 Performance Profiling

| | |
| --- | --- |
| **Technique Objective** | Measure response time for each class of operation against the targets in SRS section 3.4.1, and state the hardware every figure was measured on. |
| **Technique** | Time each operation class separately, because they have different budgets: a cached read, a cache miss that enqueues background work, a cold model inference, an attribution, and an acoustic profile. Separate the single user case from the concurrent case, since they answer different questions. |
| **Oracles** | The SRS targets are the oracle, but only for the hardware they assume. This environment is CPU only, and the SRS targets assume a GPU, so model bound figures are reported as observations and not as pass or fail. |
| **Required Tools** | Locust, and the operational metrics the application now exports |
| **Success Criteria** | Targets that do not depend on model execution, that is the cached read and the enqueue path, meet their budgets. |
| **Special Considerations** | Model bound targets are not enforced on CPU. A timing measured while another heavy job is running measures machine load, not the product, so every figure in this report was taken with the machine otherwise idle. |

### 3.1.5 Load Testing

| | |
| --- | --- |
| **Technique Objective** | Observe behaviour under concurrent use, and find the point where a target stops being met. |
| **Technique** | Locust with three user classes rather than one class with weighted tasks, so that each operation gets enough samples for a meaningful 95th percentile. A warm up phase runs first so that cold model loading does not contaminate the measurement. |
| **Oracles** | The same SRS targets as section 3.1.4, with a minimum sample count before a target is enforced, so that a percentile is never computed from a handful of requests. |
| **Required Tools** | Locust |
| **Success Criteria** | No failed requests, and the non model bound targets met at the tested concurrency. |
| **Special Considerations** | The tested concurrency is modest and reflects an academic deployment with a small number of users, not a production service. |

### 3.1.6 Security and Access Control Testing

| | |
| --- | --- |
| **Technique Objective** | Confirm that one session cannot read another session's data, that debug routes are not exposed in production configuration, that uploads are validated, and that known vulnerable dependencies are visible to the team. |
| **Technique** | Negative assertions, that is the correct result is a refusal. A session must not be able to fetch another session's dataset by identifier. An unlisted origin must be rejected by the cross origin policy. A corrupted audio file must be rejected rather than silently accepted. Dependency scanning for both package ecosystems, and container image scanning in the pipeline. |
| **Oracles** | Deterministic for the refusal tests. For dependency scanning the advisory databases are the oracle. |
| **Required Tools** | `pytest`, `npm audit`, `pip-audit`, Trivy in continuous integration |
| **Success Criteria** | All refusal tests pass, and every dependency advisory is recorded with a severity even where it is not yet fixed. |
| **Special Considerations** | There is no user authentication in this product, so access control means session isolation, not roles and permissions. Manual exploratory probing is still outstanding. |

### 3.1.7 Failover and Recovery Testing

| | |
| --- | --- |
| **Technique Objective** | Confirm the system degrades sensibly when a dependency is missing, and recovers when it returns. |
| **Technique** | Run the application and the suite with Redis unreachable, and with MongoDB absent, and confirm the product still starts and reports its state honestly rather than crashing. Kill a worker during a job and confirm that a restart is safe, which depends on the stale worker lock being purged at startup. |
| **Oracles** | Deterministic. The health endpoint must report the true state, and a restarted worker must pick up work. |
| **Required Tools** | Docker, `pytest`, the health endpoints |
| **Success Criteria** | No crash when a dependency is absent, and no permanently blocked worker family after an unclean exit. |
| **Special Considerations** | The metadata tier is designed to degrade quietly when MongoDB is absent. That is correct behaviour, but it also means a misconfiguration can hide itself, which is discussed in section 9. |

### 3.1.8 Configuration Testing

| | |
| --- | --- |
| **Technique Objective** | Confirm correct operation across the supported operating systems, language runtimes, browsers and accelerator configurations. |
| **Technique** | Run the full suite on more than one host operating system, run the browser tests on three engines, and compare outcomes. Any difference between configurations is the finding. |
| **Oracles** | A differential oracle. The same suite must produce the same pass or fail result on every supported configuration. |
| **Required Tools** | Continuous integration, Playwright browser projects, Docker Compose |
| **Success Criteria** | The suite passes on every declared configuration, and any configuration dependent difference is explained rather than averaged away. |
| **Special Considerations** | Continuous integration installs a CPU only build of the machine learning library, so the pipeline can never provide evidence about the GPU path. That evidence has to come from a developer machine with a GPU. |

### 3.1.9 Accessibility Testing

This category was added during this test cycle. It was listed as outstanding in
the earlier plan and had no automated coverage at all.

| | |
| --- | --- |
| **Technique Objective** | Confirm the workbench meets WCAG 2.1 level AA, which the SRS commits to in section 3.2.3. |
| **Technique** | Automated scanning with axe-core driven by Playwright, covering colour contrast, accessible names for images and controls, form labels, landmark structure, page title and document language. Violations rated serious or critical fail the build. Violations rated minor or moderate are reported but do not fail, so that a new regression is not hidden inside pre existing low severity noise in an inherited interface. |
| **Oracles** | The WCAG 2.1 A and AA rule sets as implemented by axe-core. |
| **Required Tools** | `@axe-core/playwright`, `axe-core`, Playwright |
| **Success Criteria** | No serious or critical violations on the workbench. |
| **Special Considerations** | Automated scanning finds roughly a third of real accessibility problems. A clean scan is a floor and not a certificate. Keyboard order, focus visibility and screen reader wording still need a manual pass, which has not been done. |

---

# 4. Software Testing Results

## 4.1 Test environment

| Item | Value |
| ---- | ----- |
| Operating system | Windows 11 |
| Python | 3.11.0 |
| Node | 20 |
| Machine learning runtime | torch 2.13.0 with CPU only build, no GPU present |
| Redis | 7.4.10 in Docker, container `audiolit-redis`, port 6379 |
| MongoDB | 6.0.28 in Docker, container `audiolit-mongo`, port 27017 |
| Application | uvicorn on port 8000, five RQ workers running, Vite dev server on port 8080 |
| Branch | `origin/testing` at `d5ecf95`, plus the fixes in section 7 |

The software was started and exercised for real. The API reported healthy, the
five worker families registered, the metadata tier created all four collections
with the correct indexes, and the workbench loaded and listed dataset rows in a
browser.

## 4.2 Results by category

| Category | Tool | Result |
| -------- | ---- | ------ |
| Backend unit and integration | pytest, 54 test files | See section 4.3 |
| Frontend component | Jest, 8 suites | 62 passed, 0 failed |
| Static analysis | ESLint | 0 errors, 110 warnings |
| Production build | Vite | Succeeded in 48.6 seconds |
| Cross browser layout | Playwright, Chromium, Firefox, WebKit | Passed on all three engines |
| Quick start walkthrough | Playwright | Passed after the fix in section 7 |
| Accessibility | axe-core with Playwright | 4 checks, all passed after the fixes in section 7 |
| Full stack data flow | Playwright | 5 of 6 passed. One intermittent failure, see section 4.6 |
| Load | Locust | 470 requests, 0 failures |
| Dependency scanning, JavaScript | npm audit | 15 advisories in production dependencies |
| Dependency scanning, Python | pip-audit | 33 advisories across 5 packages |
| Container image scanning | Trivy in continuous integration | Already configured |

## 4.3 Backend suite

The backend suite was run with the Redis address pointed at an unreachable port,
deliberately matching the condition the continuous integration pipeline runs
under. A locally reachable Redis hides failures that only appear in the
pipeline.

Command:

```
cd Backend
REDIS_URL="redis://127.0.0.1:1/0" python -m pytest -q -rs
```

Result: **717 passed, 7 skipped, 0 failed, 0 errors**, in 18 minutes 54 seconds.

Five further tests were added during this cycle to guard the word error rate fix
described in section 6.2. Those were verified in their own file, which reports 13
passed, so the current total is 729 collected.

**A second full run of the same suite hung and did not finish.** It stopped
making progress at `tests/test_multitask_orchestrator.py`, at 58 percent, and
produced no further output. This is a known intermittent problem on this
project, recorded in the repository guidance: a burst mode worker draining a
dependency gated aggregator against `fakeredis` has hung the test run before,
and has twice hidden a real pipeline hang. It is reported here because it
happened during this cycle, not as a new defect. The practical consequence is
that a single green run is not sufficient evidence for this suite, and the
orchestrator tests should be stress run in a loop with a timeout before any
release.

All seven skips are gated by the environment and state a reason. None is a
silently disabled test.

| Skipped test | Reason |
| ------------ | ------ |
| `test_fanout_orchestrator.py:192` | Needs a forking start method, which Windows does not provide |
| `test_function_testing.py:303` | Requires GPU and model resources |
| `test_memory_profiling.py:35` | Video memory test requires CUDA |
| `test_ser_checkpoint.py:139`, `:152`, `:161` | Each downloads about 1.2 GB from the Hugging Face Hub, gated behind an environment variable |
| `test_task_orchestrator.py:502` | No message broker reachable, which is the intended condition for this run |

## 4.4 Performance and load

Locust was run twice: once with ten concurrent users, and once with a single
user, so that concurrency effects can be separated from the cost of the
operation itself.

**Ten concurrent users, 90 seconds, 470 requests, 0 failures.**

| Operation | Requests | Median | 95th percentile | Budget | Result |
| --------- | -------- | ------ | --------------- | ------ | ------ |
| Cached prediction | 319 | 15 ms | 58 ms | 200 ms | Pass |
| Enqueue multitask | 118 | 58 ms | 110 ms | 50 ms | Fail |
| Health | 33 | 11 ms | 52 ms | Not specified | Observation |

**Single user, same operations.**

| Operation | Requests | 95th percentile | Budget | Result |
| --------- | -------- | --------------- | ------ | ------ |
| Cached prediction | 35 | 23 ms | 200 ms | Pass |
| Enqueue multitask | 28 | 50 ms | 50 ms | Pass, at the limit |

The cached read path has comfortable headroom, using about a quarter of its
budget even under concurrency. The enqueue path is the weak point. It sits
exactly on its 50 ms budget with a single user and misses it at ten users, where
the 95th percentile is 110 ms, slightly over twice the target. This is reported
as a genuine finding rather than a measurement artefact, because the single user
baseline was measured separately and the two agree on the direction.

Model bound operations were not enforced, because this machine has no GPU and
the SRS targets assume one. For reference, a Whisper Grad-CAM attribution
measured 21 to 37 seconds per clip when called directly on this CPU host.

## 4.5 Security scanning

**JavaScript production dependencies, npm audit: 15 advisories.** 2 critical, 10
high, 2 moderate, 1 low.

| Severity | Package | Issue |
| -------- | ------- | ----- |
| Critical | maplibre-gl | Cross site scripting through a sanitiser bypass |
| Critical | plotly.js | Inherited from a dependency |
| High | @remix-run/router, react-router, react-router-dom | Cross site scripting through open redirects |
| High | lodash | Code injection through the template function |
| High | postcss | Cross site scripting in stringify output |
| High | nanoid, glob, minimatch, brace-expansion, picomatch | Denial of service, command injection and related issues |

**Python dependencies, pip-audit: 33 advisories across 5 packages.**

| Package | Advisories | Note |
| ------- | ---------- | ---- |
| starlette 0.37.2 | 14 | This is the web framework layer under FastAPI, so it is the most important one to address |
| pip 22.3 | 14 | Tooling, not shipped with the product |
| anyio | 2 | |
| pytest 8.4.2 | 2 | Test tooling only |
| accelerate | 1 | |

The machine learning runtime could not be audited because it is installed from a
local build that is not on the public package index. That is a gap, not a clean
result, and is recorded as such.

None of these were fixed in this cycle. Upgrading the web framework layer and
the routing library both risk behaviour changes, so they need their own change
with their own testing rather than being folded into a test report.

## 4.6 Full stack data flow

Five of six checks pass: clip selection binds to the editor, the transcript
column never shows raw JSON, word segments name words the transcript actually
contains, and a genuine speech clip is not reported as a deepfake at full
confidence.

One check is intermittent. The Grad-CAM provenance test failed in one run
because the saliency response came back with provenance `fallback` and the
reason "attribution was empty or constant, showing encoder energy, not
attribution". The same attribution was then generated directly on the same clip
three times and returned provenance `measured` each time, with healthy variance
in the saliency matrix. So the underlying attribution code is working, and the
failure is either timing related or specific to the state the application was in
during that run. It is recorded as open rather than closed, because an
intermittent failure in exactly the check that guards against fake explanations
is not something to dismiss.

One related observation came out of that investigation. The application log
shows the speech model weights being reloaded on many requests rather than being
held in memory. This is the main reason an attribution through the API is much
slower than the same attribution called directly.

---

# 5. Evaluation of the Data Science Parts

Two data science requirements are measured: accent bias in ASR, and attribution
faithfulness. Both were regenerated for this report using a single reproducible
command, so the numbers can be checked rather than taken on trust.

## 5.1 Accent bias in ASR

**Method.** Group wise word error rate across six first language cohorts from
the L2-ARCTIC corpus, 20 samples each, 120 samples in total, using Whisper base.
Word error rate is computed with word level edit distance after both the
reference and the hypothesis are normalised the same way, that is lower cased
with punctuation removed. The bias discrepancy index is the difference between
the worst and the best cohort mean.

Command:

```
cd Backend
python scripts/evaluate_models.py --asr-model-id openai/whisper-base --skip-faithfulness --output ../docs/evaluation/results
```

**Results.**

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
English than on Hindi accented English, and the gap between the best and worst
cohort is about 7.8 percentage points of word error rate. That is a real
disparity and it is the kind of finding the requirement exists to surface. It is
not a large enough gap to call the system unusable for any cohort, but it is
large enough that a downstream user should be told which cohorts were measured.

## 5.2 Attribution faithfulness

**Method.** Deletion scoring. The most salient regions identified by the
attribution are masked, inference is run again on the masked audio, and the drop
in confidence is measured. If an attribution is faithful then removing what it
marks as important should reduce the model's confidence. The deletion area under
curve integrates that drop across several masking levels. Attributions that came
from a fallback are refused rather than scored, because scoring a fallback
measures the fallback and not the model.

Command:

```
cd Backend
python scripts/evaluate_models.py --manifest eval_manifest.json --model-type ser --skip-accent-bias --top-k 0.1 0.3 0.5 --output ../docs/evaluation/results_fr16
```

**Results** on three CREMA-D clips with the speech emotion model, masking at 10,
30 and 50 percent:

- Mean deletion score, that is the mean confidence drop: **0.2907**
- Mean deletion area under curve: **0.1408**
- Audio scored: 3 of 3. Refused as fallback attribution: 0

**Reading of the result.** Masking the most salient regions reduces model
confidence by about 29 percent on average, so the attributions are carrying real
information about what the model is using. Nothing was refused, which means all
three attributions were genuinely measured rather than fallbacks.

This is a small sample and should be treated as a demonstration that the audit
pipeline works end to end, not as a final faithfulness figure for the product. A
larger run across more clips and both other tasks is the obvious next step.

---

# 6. Error Analysis of the Data Science Parts

This section is the most important part of this report. Both data science
metrics were wrong when this cycle started, and neither was wrong in a way that
would have been noticed by looking at the product.

## 6.1 First defect: the speech model was guessing the language

**Symptom.** The accent bias evaluation reported an overall mean word error rate
of 0.6019 and a bias discrepancy index of 1.1676. Two cohorts, Vietnamese and
Arabic, had mean error rates above 1.0, which is only possible when the model
inserts more words than the reference contains.

**Investigation.** The per sample results were sorted worst first. Two samples
out of 120 stood out with word error rates of 22.30 and 17.80. Their
transcriptions were not poor English. They were Vietnamese and Arabic text,
repeating the same phrase many times. Two further samples came back in
Indonesian or Malay.

**Root cause.** The profiler built the speech recognition pipeline without
specifying a language, so the model ran language identification on every
utterance. On heavily accented English it selected the speaker's first language,
transcribed into that language, and then fell into a repetition loop.

**Why it mattered.** The two worst cohorts were the two worst only because of one
bad clip each. The published ranking was therefore not measuring accent
difficulty at all for those cohorts. It was measuring language misdetection.

**Fix.** The decode language is now fixed to English, which is correct because
the corpus is read English throughout and the language is known rather than
something to guess.

**Effect.** Overall mean word error rate moved from 0.6019 to 0.2587, and the
discrepancy index from 1.1676 to 0.0481. The two pathological samples dropped
from 22.30 and 17.80 to 0.20 and 0.30 respectively.

## 6.2 Second defect: two different word error rate calculations disagreed

**Symptom.** After the first fix, the summary still did not add up. Every cohort
mean in the results table sat between 0.097 and 0.206, yet the overall mean
printed above the same table read 0.2587, which is higher than every single
value it was supposedly summarising.

**Investigation.** The report was drawing its cohort table from one code path
and its headline numbers from a different one. Comparing them directly showed
they disagreed on every cohort, not by a constant offset that could be ignored
but consistently, with the second path always higher.

| Cohort | Profiler path | Evaluation service path |
| ------ | ------------- | ----------------------- |
| Arabic | 0.2060 | 0.2917 |
| Hindi | 0.0969 | 0.2436 |
| Korean | 0.1347 | 0.2544 |
| Mandarin | 0.1246 | 0.2466 |
| Spanish | 0.1413 | 0.2515 |
| Vietnamese | 0.1812 | 0.2645 |

**Root cause.** The evaluation service compared words after lower casing and
splitting on whitespace, but without removing punctuation. The speech model
returns punctuation, so a correctly recognised final word such as "child."
failed to match the reference word "child" and was counted as an error. The
profiler already removed punctuation before comparing, and had a comment
explaining exactly why. The two had simply drifted apart.

**Why it mattered more than it looks.** The inflation was roughly constant across
cohorts, because every cohort's transcripts end in a full stop at a similar rate.
Adding a near constant to every cohort barely changes their ordering but it does
compress the gap between them relative to their size, and the bias discrepancy
index is exactly that gap. The published index was 0.0481 when the real spread
was 0.1091. The metric whose entire job is to measure disparity was understating
that disparity by more than half.

**Fix.** Both sides now normalise identically, that is lower case and remove
punctuation before comparison.

**Effect.** Overall mean word error rate moved from 0.2587 to 0.1430, and the
discrepancy index from 0.0481 to 0.0779. A test was added that checks the two
code paths agree on the same input, so they cannot drift apart again silently.

## 6.3 Combined effect and comparison with the previously documented figures

| Stage | Overall mean WER | Bias discrepancy index |
| ----- | ---------------- | ---------------------- |
| Before either fix | 0.6019 | 1.1676 |
| After the language fix | 0.2587 | 0.0481 |
| After the word error rate fix | **0.1430** | **0.0779** |
| Previously documented in the evaluation document | 0.1353 | 0.0670 |

The corrected measurements are close to the figures the team documented earlier,
which is a good sign for both. The remaining difference is small and is expected,
since the earlier figures were produced on different hardware and are not
guaranteed to have used an identical sample selection.

## 6.4 Failure modes observed in the data

Beyond the two defects, three patterns in the corrected results are worth
recording.

1. **Accented English can be misread as another language entirely.** This is the
   most severe failure mode found, because the output is confident, fluent and
   completely wrong, and a user who does not read Vietnamese would have no way
   to tell. The fix removes it for this corpus, but any deployment that accepts
   speech in an unknown language will face it again.

2. **Cohort means are driven by a minority of hard clips.** Spanish has a median
   word error rate of 0.0000, meaning at least half of its samples were
   transcribed perfectly, yet its mean is 0.1413. A small number of difficult
   clips carries the whole cohort. Reporting only the mean would suggest the
   model is uniformly mediocre on Spanish accented speech when it is in fact
   usually perfect and occasionally poor. Both figures are therefore reported.

3. **The cohort ranking changed after the fixes.** Before, Vietnamese looked like
   the worst cohort. After, Arabic is. Any conclusion drawn from the earlier
   ranking, including any statement about which speakers the system serves
   worst, was based on a measurement error.

---

# 7. Defects Found and Fixed in This Test Cycle

Seven defects were found by running the software. Six are fixed, and each fix has
a test that fails without it where a test is possible.

| # | Defect | Where | Fix | Status |
| - | ------ | ----- | --- | ------ |
| 1 | Eight serious colour contrast violations against WCAG 2.1 AA. White text on the primary colour measured 3.80 to 1 and the amber warning text measured 2.95 to 1, both under the 4.5 to 1 minimum | `src/index.css`, `Toolbar.tsx`, `WarmupModal.tsx`, `WarmupStatusBanner.tsx`, `CustomDatasetManager.tsx` | Primary colour darkened from 32 to 26 percent lightness, the destructive colour from 60 to 44 percent, and the amber text shade moved one step darker. Each value was computed to the contrast threshold rather than chosen by eye | Fixed, covered by the new accessibility suite |
| 2 | The speech model guessed the language and transcribed accented English into other languages | `accent_bias_profiler.py` | The decode language is fixed to English | Fixed, effect measured in section 6.1 |
| 3 | Word error rate counted punctuation as recognition errors, and two code paths disagreed on every cohort | `evaluation_service.py` | Both paths now normalise identically | Fixed, with a test asserting the two paths agree |
| 4 | The quick start walkthrough test waited for the browser network to become idle, which never happens because the workbench holds a WebSocket open. It passed only when run without a backend and failed as soon as the full stack was running | `e2e/quickstart.spec.ts` | Waits for the workbench element instead | Fixed |
| 5 | The quick start dialog opened over the workbench on first load and its overlay intercepted pointer events, so the entire full stack data flow suite could not click anything. Five of six checks failed for this reason | `e2e/dataflow.spec.ts` | The dialog is dismissed before the page loads. It keeps its own coverage in its own suite | Fixed, four more checks now pass |
| 6 | The evaluation script could not read the attribution vector out of a saliency response, so every faithfulness item was refused and the stage reported nothing measurable | `scripts/evaluate_models.py` | The extractor now reads the field the service actually returns, and falls back to flattening the saliency matrix | Fixed, faithfulness now measures |
| 7 | Grad-CAM returned a fallback instead of a measured attribution in one full stack run | Not yet located | Not fixed | Open, see section 4.6 |

Defect 5 is worth a note for the team. Both sides were individually correct. The
quick start dialog worked and had passing tests, and the data flow suite had
passing tests. They only broke in combination, and only when the full stack was
running, which is a condition neither suite is normally run under. This is the
third time on this project that two separately green changes have failed
together.

---

# 8. Deliverables

## 8.1 Test evaluation summaries

Every automated run in this report records the suite name, the counts of tests
collected, passed, failed and skipped, the wall clock duration, the commit it ran
against, and the exact command, so that any result can be reproduced
independently.

Automated suites run on every pull request through continuous integration. The
data flow suite, the load tests and the accessibility scan need a running stack
and are run on demand.

## 8.2 Reporting on test coverage

| Area | Automated coverage | Gaps |
| ---- | ------------------ | ---- |
| Backend routes and domain logic | 54 test files, 729 collected, 717 passing in the completed run | GPU paths, live model downloads, and an intermittent hang in the orchestrator tests |
| Frontend components | 8 Jest suites, 62 tests | |
| Cross browser layout | 3 engines, 3 viewports | Mobile viewports not covered |
| Full stack data flow | 6 checks | One intermittent, see section 4.6 |
| Accessibility | 4 automated checks at WCAG 2.1 AA | Keyboard navigation and screen reader wording need a manual pass |
| Performance and load | Cached read and enqueue paths measured | Model bound targets need GPU hardware |
| Security | Session isolation, upload validation, dependency and image scanning | Manual exploratory probing not done |
| Data integrity | Cache key shapes and MongoDB schema | Live Redis eviction timing and live MongoDB expiry not measured |

---

# 9. Risks, Dependencies, Assumptions and Constraints

| Risk | Mitigation | If the risk happens |
| ---- | ---------- | ------------------- |
| The enqueue path misses its 50 ms target under concurrency | Profile the enqueue path before the demonstration, and keep the number of concurrent users low | State the measured figure honestly rather than quoting the target |
| Two critical and ten high severity advisories in shipped JavaScript dependencies, and fourteen in the web framework layer | Schedule the upgrades as their own change with their own testing | Record the advisories in the submission rather than implying a clean scan |
| No GPU in the test environment, so the model bound performance targets are unmeasured | Run those measurements on a machine with a GPU before submission | Report every figure with the hardware named, and never present a CPU number against a GPU target |
| An attribution can fall back to encoder energy and be reported as a fallback | The provenance contract already surfaces this to the user | Investigate defect 7 before relying on attributions in a demonstration |
| The metadata tier degrades quietly when MongoDB is absent | Check the health endpoint and the collection list after starting the stack | A misconfigured deployment can look healthy while storing nothing, so verify rather than assume |
| The product reloads model weights on many requests | Accept for now, since a cached read is fast and a cold attribution is not on the critical path | Budget tens of seconds for any live attribution during a demonstration |
| Two separately correct changes can break in combination | Run the full stack suite after any merge, not just the fast suites | This has now happened three times, so treat it as expected rather than unlucky |
| The backend suite can hang in the orchestrator tests instead of failing, which looks like a slow run rather than a problem | Stress run the orchestrator tests in a loop with a timeout, and never treat one green run as proof | Quarantine the specific test and record it, rather than rerunning until it passes |

**Dependencies.** Docker for Redis and MongoDB, the Hugging Face Hub for first
time model downloads, and licence compliance for the research use corpora.

**Assumptions.** A single tenant academic deployment with no continuous service
level agreement, and no user authentication tier.

**Constraints.** A video memory budget that pins the GPU bound worker families to
one job at a time, a safetensors only model loading policy, and a working data
footprint bound on the corpora.

---

# 10. References

**Testing tools used in this report**

- pytest, https://pytest.org/
- pytest-asyncio, https://pytest-asyncio.readthedocs.io/
- fakeredis, https://github.com/cunla/fakeredis-py
- mongomock, https://github.com/mongomock/mongomock
- Jest, https://jestjs.io/
- React Testing Library, https://testing-library.com/react
- Playwright, https://playwright.dev/
- axe-core and @axe-core/playwright, https://github.com/dequelabs/axe-core
- Locust, https://locust.io/
- pip-audit, https://pypi.org/project/pip-audit/
- npm audit, https://docs.npmjs.com/cli/commands/npm-audit
- Trivy, https://trivy.dev/
- jiwer, https://github.com/jitsi/jiwer

**Project documents**

- Software Requirements Specification, `docs/SRS.md`
- Software Architecture Document, `docs/SAD.md`
- Master Test Plan, `docs/testing/AudioLIT_Master_Test_Plan.md`
- Data Science Error Analysis, `docs/evaluation/DS_ERROR_ANALYSIS.md`
- Defect Log, `docs/evaluation/DEFECT_LOG.md`
- Generated evaluation results, `docs/evaluation/results/` and
  `docs/evaluation/results_fr16/`

**Standards**

- Web Content Accessibility Guidelines 2.1, https://www.w3.org/TR/WCAG21/
- OWASP API Security Top 10, https://owasp.org/API-Security/

---

*End of Test Plan Report.*
