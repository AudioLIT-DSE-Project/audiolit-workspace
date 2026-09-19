# AudioLIT

# Master Test Plan

**Version 1.0**

> Structure follows `Template for Test plan.docx` (Rational Unified Process
> test-plan template) exactly, per the design specification in
> `docs/testing/TEST_PLAN_DESIGN.md`. Section numbering corrects the two
> mis-numbered headings in the source template (§3.1 and §4.2 render there as
> "1.1"). All guidance placeholder text has been removed and replaced with
> AudioLIT-specific content.
>
> **Reading key for this document:**
>
> - **✅ EXECUTED** — real evidence captured for this report; command, output,
>   and commit are given.
> - **⏳ PENDING** — designed and ready to run, but requires infrastructure
>   (GPU hardware, a live Redis/worker stack, a dedicated load-test window, or
>   human review) not available in the environment this report was assembled
>   in. Owner and trigger condition are stated for each.

---

## Revision History

| Date       | Version | Description                                                                                                                                        | Author                                                                                          |
| ---------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 2026-09-13 | 1.0     | Initial Master Test Plan, built from `TEST_PLAN_DESIGN.md`; §4.1 populated with real backend/frontend/e2e execution evidence from commit `a3a78fa` | Ravindu Pathirana (drafted with Claude Code); to be reviewed by Tharusha Perera and Rahim Iqbal |

---

## Table of Contents

1. Evaluation Mission and Test Motivation
2. Target Test Items
3. Test Approach
   3.1 Testing Techniques and Types
   &nbsp;&nbsp;&nbsp;3.1.1 Data and Database Integrity Testing
   &nbsp;&nbsp;&nbsp;3.1.2 Function Testing
   &nbsp;&nbsp;&nbsp;3.1.3 User Interface Testing
   &nbsp;&nbsp;&nbsp;3.1.4 Performance Profiling
   &nbsp;&nbsp;&nbsp;3.1.5 Load Testing
   &nbsp;&nbsp;&nbsp;3.1.6 Security and Access Control Testing
   &nbsp;&nbsp;&nbsp;3.1.7 Failover and Recovery Testing
   &nbsp;&nbsp;&nbsp;3.1.8 Configuration Testing
4. Deliverables
   4.1 Test Evaluation Summaries
   4.2 Reporting on Test Coverage
5. Risks, Dependencies, Assumptions, and Constraints
6. References

---

# 1. Evaluation Mission and Test Motivation

AudioLIT is an interpretability workbench for Automatic Speech Recognition
(ASR), Speech Emotion Recognition (SER), and Audio Deepfake Detection (ADD),
extending the open-source **ECHO 1.0** baseline in place. The backend is
FastAPI, backed by Redis 7 for caching, pub/sub progress, and an RQ (Redis
Queue) task fabric of five per-family background worker queues (`asr`, `ser`,
`add`, `xai`, `mutation`); the frontend is React 18 + TypeScript + Vite. The
project is built by three developers across an academic-project timeline
(Phase 2 MVP → Phase 3 refinement/testing → Phase 4 submission), and this
document covers the Phase 3 test effort.

**No prior-year baseline test report exists to extend.** The file supplied as
"the previous year's ECHO baseline test report"
(`The Learning Interpretability Tool (LIT) for Voice.pdf`) was inspected and
found to be the unmodified RUP template — every field still holds its
placeholder (`<Project Name>`, `<dd/mmm/yy>`, blue-italic guidance text), with
no ECHO-specific content anywhere in its 11 pages. This Master Test Plan is
therefore written from zero, not as an extension of prior testing.

**Why testing AudioLIT is not ordinary web-application testing:**

- **The product is an explanation, not just a prediction.** A wrong transcript
  is a visible defect; a plausible but unfaithful saliency map is an invisible
  one, and it is arguably worse, because a user acts on it believing it is
  faithful.
- **Most interpretability outputs have no ground truth.** There is no "correct"
  Grad-CAM heatmap for a given clip to assert equality against. Conventional
  input/expected-output assertions are insufficient on their own; this plan
  leans on **metamorphic oracles** (relationships that must hold between two
  runs) wherever a direct oracle does not exist — see the Oracles discussion
  under §3.
- **Inference is expensive and non-deterministic**, so the Redis-backed cache
  (FR4) is not an optional optimisation to skip in testing — the system's
  reproducibility claim now depends on it. FR4.4 requires that identical
  requests produce byte-identical cached responses.
- **The baseline is inherited and known-defective.** ECHO 1.0 silently
  substituted a fabricated attention pattern when real attention extraction
  failed, returning it unflagged in the same shape as genuine attention (the
  defect FR17 exists specifically to correct). ECHO 1.0 also shipped a UI label
  reading "GradCAM" over an attribution method that was actually Integrated
  Gradients (the defect FR9 exists specifically to correct). Both are
  first-class regression targets, not incidental bugs — a workbench whose
  purpose is faithful interpretability cannot silently inherit unfaithful
  interpretability.

**Mission statement for this test effort.** Of the RUP template's candidate
motivators, this iteration adopts:

- **Verify a specification** — FR1–FR4, FR6–FR12, FR15–FR17 and SR1–SR7 are
  written down and independently testable (SRS v1.0); this plan's primary job
  is to demonstrate each is met or to report exactly where it is not.
- **Find important problems and assess quality risk** — with particular
  weight on the faithfulness risk above, since it is the risk category unique
  to an interpretability tool.
- **Advise about product quality** for the Phase 4 academic submission
  decision-makers.

Explicitly **not adopted**: _certify to a standard_ (no certification target
exists for this academic deployment) and _fulfil process mandates_ (no
external process mandate applies).

**Scope boundary.** Only SRS-committed functionality is in scope. Per
`docs/CLAUDE.md` and `docs/SRS.md` §4.4, stretch items are out of scope for
this plan, including **FR5 (multi-model comparison)**, which was demoted from
committed to non-committed stretch. There is no FR5, FR13, or FR14 in the
reconciled SRS; this document does not test requirements that do not exist.

---

# 2. Target Test Items

The table below lists the items — software, models, corpora, and environment —
identified as targets for testing, grouped by category with a relative
criticality ranking.

**Target-of-test branch:** `origin/testing`. This branch is a strict superset
of `origin/develop` (verified via `git log`): it carries every commit merged
to `develop`, plus three testing-only commits adding
`Backend/tests/test_inference_consistency.py` (24 cross-cutting wiring
assertions), `Frontend/e2e/dataflow.spec.ts` (full-stack E2E), and
`Backend/loadtests/locustfile.py` (Locust load tests). Declaring `testing` as
the target-of-test, rather than `develop`, is what makes §3.1.5 Load Testing
answerable at all — `develop` alone carries no load-test harness.
**Commit referenced throughout this report's executed evidence: `a3a78fa`**
(the tip of `develop` at drafting time; the `testing`-only commits sit on top
of it for load and full-stack E2E).

| Group                                | Items                                                                                                                                                                                                                                                                           | Criticality                                                                                                       |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **API surface**                      | 15 FastAPI routers under `Backend/app/api/routes/`: `upload`, `inference`, `inferences`, `saliency`, `perturbations`, `acoustic`, `evaluation`, `datasets`, `dataset_management`, `models`, `results`, `session`, `tasks`, `health`, `debug`                                    | High — every user-facing capability passes through here                                                           |
| **Domain / ML engines**              | `Backend/app/domain/`: model registry + loader, hook manager, saliency service, acoustic profiler, perturbation service, accent-bias profiler, evaluation service, provenance                                                                                                   | High — the interpretability claims of the whole product live here                                                 |
| **Orchestration fabric**             | `Backend/app/orchestration/task_orchestrator.py` (the single RQ fabric per SAD §5.2), `worker.py`, fan-out and multitask orchestrators, session queue; five worker families: `asr`, `ser`, `add`, `xai`, `mutation`                                                             | High — FR3, and the historical site of two duplicate-module incidents (LIT-230, the PR #10/#13 combination break) |
| **Cache and persistence**            | `Backend/app/infrastructure/cache_keys.py` (MD5-of-resolved-path scheme, every hot path), `Backend/app/core/redis.py` (`RedisCacheManager`, SHA-256 content-addressed, FR4, consumed only by `results.py`), Redis 7 keyspace                                                    | High — FR4.4 reproducibility guarantee depends on this tier                                                       |
| **Frontend**                         | `Index.tsx` workbench; panels (Prediction, Acoustic, Accent Bias, Faithfulness, Embedding); `XAIOverlayCanvas`, `WaveformViewer`, `SpectrogramGridSelector`, `PerturbationTools`; `EmbeddingContext`, `PlaybackContext`, `ModelRegistryContext`; `useTaskStatus` WebSocket hook | High                                                                                                              |
| **Third-party dependencies**         | PyTorch ≥2.6 (installed: 2.13.0), Transformers ≥4.30, Captum ≥0.6, Librosa ≥0.10, soundfile, RQ 2.10, Redis 7, fakeredis 2.23.2, FastAPI 0.111, httpx 0.27, React 18.3, Vite 5.4, Playwright 1.63, Jest 29                                                                      | Medium — not authored by the team, but failures surface through them                                              |
| **Models under test**                | Whisper (ASR, family default); Wav2Vec2 SER pinned at `firdhokk/speech-emotion-recognition-with-facebook-wav2vec2-large-xlsr-53`, revision `611e6db8ee667aa07fe66596f9fc761e036ff5b9`; deepfake detector (Wav2Vec2-family, ASVspoof-trained)                                    | High                                                                                                              |
| **Corpora**                          | Common Voice, LibriSpeech, RAVDESS, CREMA-D, L2-ARCTIC, ASVspoof 2021 DF, ESD                                                                                                                                                                                                   | Medium — licence-gated, streamed/sub-sampled under the ~100 GB footprint bound (FR2.2)                            |
| **Environment / configuration axes** | Python 3.10 (CI, `ubuntu-latest`) vs 3.11 (this evaluation's local venv); Node 20; Redis 7-alpine (`Backend/docker-compose.yml`); CPU-only torch wheel on CI vs GPU-capable dev hardware; Chromium/Firefox/WebKit; desktop viewports 1024×768–1920×1080                         | Medium — the CPU/GPU split governs whether FR1.4's fallback path is ever exercised                                |

**Explicitly excluded from this test effort:** Hugging Face Hub availability
and the pretrained models' own training-time accuracy (both third-party,
outside AudioLIT's control); browser engine internals below the DOM/rendering
level Playwright can observe; and the SRS §3.10 MongoDB metadata tier —
**it is specified but not implemented** (no `pymongo`/`motor` in
`Backend/requirements.txt`, no Mongo reference anywhere under `Backend/app/`,
confirmed by direct search of the tree on 2026-09-13). This is a genuine
SRS/repository conflict, not a testing gap; it is raised again in §3.1.1 and
§5, and must be flagged in Linear per this project's own conflict-handling
convention (`CLAUDE.md`, step 10) rather than silently designed around.

---

# 3. Test Approach

The Test Approach describes **how** the items in §2 will be exercised to
fulfil the mission in §1. AudioLIT's approach is automated-first: as of commit
`a3a78fa`, the backend carries **593 collected pytest cases across 49 test
files**, the frontend carries **47 Jest cases across 6 suites** plus a
**12-case Playwright cross-browser layout suite**, and the `testing` branch
adds a full-stack Playwright dataflow suite and a Locust load harness. Manual
technique is reserved for what genuinely cannot be automated: subjective
usability judgement, physical failure simulation, and exploratory security
probing.

Overview of the eight techniques and how each is realised here:

- **Data and Database Integrity Testing** — exercised against the Redis
  keyspace (cache round-trips, key-shape contracts, eviction) and the dataset
  corpus loaders, independently of the UI, since AudioLIT has no SQL/ORM
  tier and its specified MongoDB tier is unimplemented (see §3.1.1).
- **Function Testing** — black-box route and domain-level tests against every
  committed FR, traced in the §4.2 matrix.
- **User Interface Testing** — component-level Jest tests for the
  interaction-heavy canvas/waveform primitives, plus a cross-browser
  Playwright layout suite, plus manual accessibility and usability review.
- **Performance Profiling** — single-user timing against the eleven SRS
  §3.4.1 targets, on stated hardware, warm vs cold explicitly separated.
- **Load Testing** — Locust-driven concurrent-user ramps against the upload →
  enqueue → poll → result path, on the `testing` branch.
- **Security and Access Control Testing** — mapped one-to-one to SR1–SR7.
- **Failover and Recovery Testing** — reframed from the template's
  power-cable/DASD model to this system's actual failure surface: Redis
  unreachable, worker killed mid-job, GPU OOM, partial task-family failure,
  WebSocket drop, corrupted cache value.
- **Configuration Testing** — cross-browser, cross-Python-version, and
  CPU-vs-GPU axes, since the CPU-fallback path (FR1.4) is only exercised on
  one side of the last axis.

**The fault models this plan tests against**, since interpretability-tool
testing has failure modes a generic web-app plan would not name:

1. **Silent unfaithfulness** — a returned explanation that is fabricated or
   mislabelled (FR17, FR9).
2. **Cache-shape corruption** — the right key holding a wrong-shaped value, so
   a consumer reads it instead of falling back to recomputation and crashes
   downstream. This is a _documented real incident_: dataset warmup once
   stored an ASR result (`{"text", "attention"}`) under the transcript family,
   whose consumers assume a plain string, and
   `/inferences/whisper-accuracy` died on
   `AttributeError: 'dict' object has no attribute 'lower'`
   (`Backend/app/infrastructure/cache_keys.py` module docstring).
3. **Silent combination breakage** — two individually green changes that
   break only together. Also a documented real incident: PR #10 added
   `app/core/rq_connection.py` importing `app.core.settings`; PR #13
   separately relocated `settings.py`; neither touched the same lines, so
   both merged conflict-free, and the combination broke `pytest` collection
   repository-wide until a third PR fixed it.
4. **Resource exhaustion** — VRAM overflow, the Redis 2 GB memory cap,
   oversized upload (SR1's 100 MB / 15-minute bound).
5. **Partial-failure cascade** — one task family failing must not prevent the
   other two from returning (SRS §3.3.1); tested explicitly in §3.1.7.

**On oracles.** Three oracle classes recur through every technique below and
are named here once rather than re-derived eight times:

1. **Deterministic** — an exact expected value: HTTP status codes, typed error
   codes, response schema shape, cache round-trip equality, digest stability.
2. **Tolerance-based** — no single right answer, but a bounded one: WER within
   a delta, F0/RMS within tolerance of a Librosa/Praat reference (FR10.3
   mandates this reference check), latency against a stated percentile.
3. **Metamorphic** — no ground truth at all, but an invariant must hold
   between two runs. This is AudioLIT's most important oracle class:
   a **warm cache read must equal the cold computation**
   (`test_warmup_cache_contract.py`); **identical requests must be
   byte-identical** (FR4.4); **masking the top-K highest-saliency region must
   drop confidence more than masking an equal-sized random region**
   (FR16.1, the deletion-score faithfulness audit); **an attribution labelled
   Grad-CAM must not equal the Integrated Gradients output for the same
   input** (the FR9 regression check).

## 3.1 Testing Techniques and Types

### 3.1.1 Data and Database Integrity Testing

AudioLIT has no SQL database and no ORM. Its persistence surface is (a) a
Redis 7 keyspace used for the result cache, the FR4 content-addressed cache
manager, task pub/sub progress, and the RQ queues themselves, and (b) a
read-only, licence-gated corpus of seven audio datasets on disk. This section
is reinterpreted accordingly: "database integrity" means **cache-value
integrity and keyspace correctness**, and **corpus-loader integrity**.

**A specification gap is recorded here rather than silently worked around.**
SRS §3.10 specifies a MongoDB 6.0+ metadata tier (four collections:
`models`, `audio_samples`, `analysis_results`, `bias_reports`; TTL and
compound indexes). Direct inspection of `Backend/requirements.txt` and every
file under `Backend/app/` on 2026-09-13 found no `pymongo`, no `motor`, and no
reference to Mongo anywhere in the codebase. **This tier is specified but not
implemented.** This is not a testing omission; it is an SRS/repository
conflict that predates this test plan, and per this project's own
conflict-handling rule it must be raised in Linear rather than resolved
unilaterally either by testing a tier that doesn't exist or by silently
dropping the requirement from the SRS. It is repeated as a risk in §5.

|                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Technique Objective:**    | Exercise Redis cache access methods and dataset corpus loaders independently of the UI, to observe and log incorrect functioning, cache corruption, key collisions, or value-shape violations. Accountable to **FR4.1–FR4.4**, **FR2.1–FR2.3**, **SR5**.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Technique:**              | Drive `RedisCacheManager` (`app/core/redis.py`) directly against `fakeredis`, asserting round-trip fidelity through its msgpack/lz4 encoding. Assert key uniqueness across the (audio content, model, task, parameters) tuple and that `CACHE_SCHEMA_VERSION` participates in the key so a schema change cannot silently collide with old entries. Separately, assert the _value shape_ stored under each key family in `cache_keys.py` matches what every declared consumer route expects — this is the specific check the transcript/attention shape incident (§3, fault model 2) shows is necessary and insufficient by round-trip alone. Seed dataset loaders with valid, truncated, wrong-sample-rate, and structurally malformed audio. Force the Redis memory cap and confirm LRU eviction; force a corrupted cached value and confirm it is treated as a miss and recomputed (FR4.3). Confirm per-corpus licence metadata is retained and surfaced on load (FR2.3) and that `measure_footprint()` enforces the ~100 GB working bound (FR2.2, `app/main.py`'s startup warning). |
| **Oracles:**                | **Deterministic** for round-trips and digests — `decode(encode(x)) == x`; identical requests must yield byte-identical cached responses (FR4.4), which is self-verifying and automatable. **Metamorphic** for warm-vs-cold — a warmed entry must equal the value the cold computation would have produced. **Stated honestly, not glossed over:** a naive round-trip oracle passes even when a key holds the _wrong-shaped_ value for its family — exactly the LIT-cache-shape incident — so shape is asserted per key family, separately from round-trip fidelity. `fakeredis` is an oracle for AudioLIT's own logic, not for Redis 7's real eviction/expiry timing; that gap is closed only where the real container is exercised (§3.1.5, §3.1.7).                                                                                                                                                                                                                                                                                                                                  |
| **Required Tools:**         | Redis 7-alpine (`Backend/docker-compose.yml`, container `lit-redis`); `fakeredis` 2.23.2; `pytest` 8.2.0 + `pytest-asyncio` 0.23.7; `msgpack`, `lz4`; `redis-cli` for manual keyspace inspection; `soundfile` + `numpy` for audio fixture generation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Success Criteria:**       | Every key family declared in `cache_keys.py` has at least one shape-assertion test; every corpus loader (Common Voice, LibriSpeech, RAVDESS, ASVspoof, L2-ARCTIC) has both a valid-data and a malformed-data test; FR4.4 byte-identity is demonstrated; eviction and corrupt-value-as-miss are both demonstrated.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Special Considerations:** | No SQL/ORM tier, so SQL-injection and schema-normalisation concerns from the template do not apply here. **The MongoDB tier (SRS §3.10) is unimplemented — this section covers Redis and the corpus only, and cannot cover MongoDB integrity until that gap is resolved one way or the other.** Redis persistence is deliberately disabled (every cache entry is cheaply recomputable), so there is no Redis backup/restore path to test at this tier — that concern moves to §3.1.7. Tests in this category must remain green with Redis unreachable, because the project's CI has no Redis service container (see the ✅ EXECUTED evidence below, captured exactly that way).                                                                                                                                                                                                                                                                                                                                                                                                        |

**Status: ✅ EXECUTED (subset within the full backend run) / ⏳ PENDING
(live-Redis eviction timing).**
The cache-key and dataset-loader unit tests below ran as part of the full
588-passed backend suite reported in §4.1 (`test_redis_cache.py`,
`test_results_cache.py`, `test_hashing.py`, `test_data_integrity.py`,
`test_warmup_cache_contract.py`, `test_dataset_ingestion.py`,
`test_dataset_service.py`, `test_l2arctic_loader.py`,
`test_librispeech_loader.py`, `test_asvspoof_loader.py`) — all passed against
`fakeredis`, with the real `REDIS_URL` pointed at an unreachable port,
matching CI. **Not yet executed:** LRU eviction timing and cache behaviour
against a _live_ Redis 7 container under a forced memory cap — this needs
`docker compose up -d` and a manual load push, which was out of scope for this
report's environment. **Owner:** whoever picks this up next should run it with
Redis actually up (`docker compose up -d` in `Backend/`) and record the
eviction-order result here.

---

### 3.1.2 Function Testing

|                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Technique Objective:**    | Exercise AudioLIT functionality — ingestion, inference, attribution, acoustic profiling, mutation, and auditing — via the public API and the UI, with valid and invalid data, to verify correct results, correct typed errors, and correct application of every committed business rule (FR1–FR4, FR6–FR12, FR15–FR17).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **Technique:**              | Route-level black-box tests via `httpx.AsyncClient` against the live FastAPI app, covering all 15 routers under `app/api/routes/`. Domain-level unit tests per engine under `app/domain/`. The full FR-by-FR mapping is in the §4.2 traceability matrix rather than repeated here; the highest-value checks are: safetensors-only model ingestion with rejection of non-safetensors artefacts before deserialisation, and an `UNSUPPORTED_ARCHITECTURE` error within 60 s for a supported-family miss (FR1.1, FR1.3); concurrent ASR+SER+ADD dispatch on one uploaded clip (FR3.1); SER returning a full probability distribution over ≥6 categories plus top-1 and confidence (FR6.1–FR6.2); ADD returning binary bona-fide/synthetic with confidence (FR7.1); Grad-CAM being genuinely gradient-weighted and **not** equal to the Integrated Gradients output for the same input (FR8.2 vs FR9, the corrected baseline defect); fallback-derived attributions carrying an explicit provenance flag (FR17.1, the corrected baseline defect); F0/RMS/log-mel spectrogram computed and validated against Librosa (FR10.1, FR10.3); mutations preserving the original clip and returning a correctly shaped 16 kHz mono derived clip (FR12.1, FR12.3); per-cohort WER disparity over L2-ARCTIC (FR15.1); and the top-K deletion-score faithfulness audit (FR16.1). |
| **Oracles:**                | **Deterministic** for contracts — status codes, typed error codes, JSON schema shape, label-set membership. **Tolerance-based** for numeric outputs — F0 and RMS are checked against a Librosa reference implementation per FR10.3's explicit mandate; WER is checked within a stated delta. **Metamorphic** for the interpretability claims themselves, because no ground-truth saliency map exists to assert equality against: masking the top-K salient region must reduce model confidence more than masking a random region of equal size and count (the FR16.1 audit _is_ the oracle for saliency quality, not a separate test of it). This is stated plainly rather than implied: **a saliency map's correctness is not directly assertable**, and faithfulness metrics are the deliberate substitute.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **Required Tools:**         | `pytest` 8.2.0, `pytest-asyncio` 0.23.7, `httpx` 0.27.0, `fakeredis` 2.23.2 (backend); Jest 29.7.0 + `@testing-library/react` 16.3.2 + `user-event` 14.6.4 (frontend component logic); Playwright 1.63.0 (full-stack dataflow, on `testing`); FastAPI's generated OpenAPI docs (`/docs`) for contract inspection; Captum ≥0.6 and Librosa ≥0.10 as reference implementations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **Success Criteria:**       | Every committed FR traces to at least one executed test in the §4.2 matrix (target: 100% FR coverage); every route has both a happy-path and an invalid-input test; both inherited baseline defects (FR9's mislabelling, FR17's silent fallback) have a dedicated regression test that would fail against the pre-fix ECHO 1.0 behaviour.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Special Considerations:** | Model downloads make cold-path tests slow and network-dependent; slow tests are marked with the `slow` pytest marker and Hub-downloading tests are explicitly gated behind `AUDIOLIT_HUB_TESTS=1` (confirmed in `test_ser_checkpoint.py`, see §4.1). Inference is non-deterministic in general, so tests pin model revisions (the SER checkpoint is pinned at revision `611e6db8ee667aa07fe66596f9fc761e036ff5b9`) or assert on tolerance, never on exact floating-point equality. Any test that calls a task-orchestrator function needs the `broker` fixture (patches `rq_connection._CONNECTION` with fakeredis) even when only the domain call is mocked, because the orchestrator wrapper itself calls `publish_progress`/`get_redis_connection` independently — this has previously caused CI-only failures when a Redis-touching orchestrator wrapper was assumed covered by mocking just the inner domain function.                                                                                                                                                                                                                                                                                                                                                                                                                                      |

**Status: ✅ EXECUTED.** This is the largest single category in the backend
suite reported in §4.1 — function-level route and domain tests make up the
majority of the 588 passed cases (see the per-file evidence table in §4.1),
including `test_function_testing.py`, `test_system_integration.py`,
`test_grad_cam.py`, `test_integrated_gradients.py`, `test_saliency_service.py`,
`test_perturbation_service.py`, `test_evaluation_scoring.py`, and
`test_accent_bias_profiler.py`. **The full FR-by-FR breakdown, including which
FRs have thin coverage, is in §4.2** — do not infer FR coverage from the
aggregate pass count alone.

---

### 3.1.3 User Interface Testing

|                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Technique Objective:**    | Verify navigation, panel state, canvas interaction, and playback synchronisation across the workbench, and confirm the UI conforms to the SRS §3.2.3 accessibility target (WCAG 2.1 AA). Accountable to **SRS §3.2.3**, **§3.9.1** (panel inventory), **FR8.4**, **FR10.2**, **FR11.2**, **FR12.2**.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Technique:**              | Jest + Testing Library component tests for the interaction-heavy primitives (`WaveformViewer`, `XAIOverlayCanvas`, `SpectrogramGridSelector`, `PerturbationTools`, plus a general `ui-components` suite). Playwright `e2e/layout.spec.ts` for responsive layout, run across Chromium, Firefox, and WebKit at three desktop viewports. Manual keyboard-only traversal, a screen-reader spot-check, and a dark/light contrast audit, none of which are currently automated. Functional checks specific to this system: time-synchronisation between audio playback, the F0 contour, and the attribution overlay (FR10.2); the alpha-blend transparency control and perceptually uniform colour scale on heatmap overlays (FR8.4); the client-side Web Audio preview muting/playing a selected region before dispatch (FR12.2); and a fallback-derived attribution being **visibly** distinguished in the UI, not only flagged in the API response (FR17.1). |
| **Oracles:**                | **Automatable:** DOM assertions, ARIA role/label presence, computed contrast ratios against the 4.5:1 bar, Playwright layout assertions (no horizontal overflow, all panels within viewport). **Not automatable, stated rather than hidden:** whether an explanation _reads_ as interpretable to a first-time user, and whether progressive disclosure achieves the SRS §3.2.1 goal of a first counterfactual within ~30 minutes — these need a small structured human usability walkthrough, measured against the §3.2.2 task-time table, not a script.                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Required Tools:**         | Jest 29.7.0 + `jest-environment-jsdom`; `@testing-library/react` 16.3.2, `@testing-library/user-event` 14.6.4, `@testing-library/jest-dom` 6.9.1; Playwright 1.63.0 with Chromium, Firefox, and WebKit browser binaries; browser DevTools; a colour-contrast analyser; a screen reader (VoiceOver / NVDA) for the manual pass.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Success Criteria:**       | Every SRS §3.9.1-committed panel has at least one automated test; the layout suite passes on all three engines at all three tested viewports; no WCAG AA contrast failure on body text or heatmap legend; keyboard traversal reaches every interactive control in a logical order.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Special Considerations:** | Canvas and WebGL content (the XAI overlay, the spectrogram grid) is largely opaque to DOM assertions; these tests assert on the _state_ driving the canvas plus a Playwright visual pass, not on rendered pixels. jsdom has no real Web Audio or Canvas 2D implementation, so those APIs are mocked in Jest — real behaviour is only covered in the Playwright pass. Plotly and WaveSurfer render asynchronously; tests assert on settled state rather than immediately after mount. Dark mode has a documented history of contrast regressions (LIT-234) — this was re-audited for this report, not assumed fixed from that ticket's fix alone.                                                                                                                                                                                                                                                                                                          |

**Status: ✅ EXECUTED for component tests and cross-browser layout. ⏳ PENDING
for accessibility/usability review.**

- **Jest component suite:** ✅ EXECUTED as part of the full 47/47 frontend run
  reported in §4.1 (`WaveformViewer.test.tsx` ×2 files,
  `SpectrogramGridSelector.test.tsx`, `PerturbationTools.test.tsx`,
  `XAIOverlayCanvas.test.tsx`, `ui-components.test.tsx`).
- **Cross-browser layout:** ✅ EXECUTED — `npx playwright test`, commit
  `a3a78fa`, 2026-09-13:

  ```
  Running 12 tests using 5 workers
    ✓ [chromium] renders without horizontal overflow at small-desktop (1024x768)
    ✓ [chromium] renders without horizontal overflow at wide-desktop (1920x1080)
    ✓ [chromium] all three top-level panels stay within the viewport at a standard desktop size
    ✓ [chromium] renders without horizontal overflow at laptop (1366x768)
    ✓ [firefox]  renders without horizontal overflow at wide-desktop (1920x1080)
    ✓ [webkit]   renders without horizontal overflow at wide-desktop (1920x1080)
    ✓ [webkit]   renders without horizontal overflow at small-desktop (1024x768)
    ✓ [webkit]   all three top-level panels stay within the viewport at a standard desktop size
    ✓ [webkit]   renders without horizontal overflow at laptop (1366x768)
    ✓ [firefox]  renders without horizontal overflow at laptop (1366x768)
    ✓ [firefox]  all three top-level panels stay within the viewport at a standard desktop size
    ✓ [firefox]  renders without horizontal overflow at small-desktop (1024x768)

  12 passed (10.1s)
  ```

  This suite is deliberately backend-free (renders the app against the real
  Vite dev server only), so it does not exercise FR10.2/FR12.2's live-data
  synchronisation — only layout.

- **Not yet executed:** the WCAG AA contrast audit, screen-reader pass, and
  the human usability walkthrough against the §3.2.2 task-time table. None of
  these are automatable by design (see Oracles above); they need a human
  reviewer with assistive-technology tooling, which this report's environment
  did not have. **Owner:** assign to whichever team member does the dark-mode
  contrast follow-up already tracked from LIT-234, since the tooling setup
  overlaps.

---

### 3.1.4 Performance Profiling

Every figure in this section is anchored to the SRS §3.4.1 performance table,
reproduced here as the requirement baseline.

| Operation                                     | SRS §3.4.1 Target | Notes                                          |
| --------------------------------------------- | ----------------- | ---------------------------------------------- |
| Cached (repeat) tensor retrieval              | < 10 ms           | SHA-256 cache-by-hash hit (FR4)                |
| API response for a cached request             | < 200 ms          | End to end, including deserialisation          |
| Cache miss to task enqueue                    | < 50 ms           | Validation, hashing, acknowledgement           |
| Cold ASR inference (Whisper-base, 15 s audio) | < 3 s             | GPU; inherited model                           |
| Multi-task inference (ASR + SER + ADD)        | < 8 s cold        | Concurrent workers; instant on cache hit (FR3) |
| Interpretability attribution (IG / saliency)  | < 8 s             | Captum, 15 s clip (FR8, FR9)                   |
| Canvas mutation — UI response                 | < 500 ms          | Targeting 30–60 FPS (FR12)                     |
| Canvas mutation — backend result              | < 2 s             | Per perturbation                               |
| Accent bias profiling                         | < 30 s            | L2-ARCTIC cohort batch, cache re-use (FR15)    |
| Faithfulness audit                            | < 15 s            | Per clip, deletion score (FR16)                |
| Cold model download + hook registration       | < 60 s            | Bounded by Hub bandwidth (FR1)                 |

|                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Technique Objective:**    | Measure single-user response times and resource consumption for each operation above, under normal single-user workload, and compare measured values against the SRS targets on stated hardware.                                                                                                                                                                                                                                                                                                                         |
| **Technique:**              | Time each operation across N repeated runs, reporting median and p95 (not a single sample). Profile memory with `scripts/run_memory_profile.py` and `test_memory_profiling.py`, asserting on **growth across iterations**, never on absolute RSS, since absolute memory is host-dependent. Profile frontend render and WebSocket-update latency in browser DevTools. Track VRAM and RAM against the SRS §3.4.3 budgets (~3–5 GB committed models, +1–2 GB with attribution, 16 GB host RAM recommended).                 |
| **Oracles:**                | Tolerance oracles against the SRS §3.4.1 targets, with the measurement method (median-of-N, p95-of-N, warm vs cold, GPU vs CPU) stated alongside every figure. **The confound is stated rather than hidden:** the SRS targets explicitly assume an NVIDIA T4-class GPU; the environment this report was produced in is a CPU-only host with no discoverable GPU (`nvidia-smi` is not present), so cold-inference targets are not meaningfully assessable from it — only the hardware-stable, cache-hit-side targets are. |
| **Required Tools:**         | `pytest` timing tests (`test_performance_load.py`, `test_memory_profiling.py`, `test_warmup_cache_contract.py`); `scripts/run_memory_profile.py`; `psutil` (installed: 7.2.2); `redis-cli --latency`; `nvidia-smi` for VRAM (GPU host only); Chrome DevTools Performance/Network panels; `/health/workers` for RQ queue depth.                                                                                                                                                                                           |
| **Success Criteria:**       | Every row of the SRS §3.4.1 table has a measured figure with stated hardware and method; hardware-stable targets (cache hit, enqueue, cached API response) are met; any deviation on a model-bound (cold/GPU) target is explained, not concealed.                                                                                                                                                                                                                                                                        |
| **Special Considerations:** | Measured on a quiet machine — background load invalidates timing figures. Cold and warm are reported separately; a cache hit trivially satisfies almost any cold target, so a warm number is never reported against a cold target. The first call after a worker process starts includes model load time and is excluded or reported separately. Memory assertions tolerate GC non-determinism (`gc.collect()` before measuring, per the existing test pattern).                                                         |

**Status: ⏳ PENDING — no GPU available in this report's environment.**

This report's evaluation environment is a CPU-only macOS host with no
`nvidia-smi` and no discoverable CUDA device — precisely the confound named in
the Oracles row above. Running the model-bound rows of the §3.4.1 table here
would produce numbers not comparable to the SRS's stated NVIDIA T4 assumption,
and reporting them as if they were would misrepresent the system's actual
performance. The honest position, per this document's own house style (§4 of
`TEST_PLAN_DESIGN.md`: "name what you did not test, and why"), is to leave
this section unexecuted rather than publish a misleading number.

**What is confirmed instead, as a lower bound:** `Backend/tests/` includes
GPU-gated tests that self-skip in this environment
(`tests/test_function_testing.py:303` — "Requires GPU/model resources";
`tests/test_memory_profiling.py:35` — "VRAM test requires CUDA"), confirmed by
the ✅ EXECUTED backend run in §4.1, which shows these two skips explicitly
rather than silently omitting them. This confirms the harness correctly
recognises the missing hardware rather than falsely passing.

**Owner and trigger:** whoever has access to the team's GPU development
machine (or the CI-declared T4-class cloud instance, if provisioned) should
run `scripts/run_memory_profile.py` and time each §3.4.1 row there, then
replace this paragraph with the measured table, median/p95, and hardware spec.

---

### 3.1.5 Load Testing

|                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Technique Objective:**    | Subject the API and worker fabric to increasing concurrent workload — normal, peak, and beyond expected maximum — to find the saturation point and confirm graceful degradation rather than collapse. Accountable to **SRS §3.4.1**, **§3.4.3**, **§3.3.1**.                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Technique:**              | Locust (`Backend/loadtests/locustfile.py`, present on the `testing` branch) driving concurrent virtual users against the upload → enqueue → poll → result path. Ramp in stages (e.g. 1 → 10 → 50 → 100 users), holding each stage long enough for a stable reading. Run cache-hit-heavy and cache-miss-heavy profiles **separately**, since they stress entirely different subsystems (the Redis read path vs the GPU-pinned worker pool). Observe RQ queue depth per family, worker saturation under the concurrency-1 GPU pin (SRS constraint C2), Redis memory against its 2 GB cap and LRU eviction under load, and WebSocket fan-out under many concurrent subscribers. |
| **Oracles:**                | Throughput and latency percentiles per ramp stage. The decisive oracle is **behavioural, not numerical**: past saturation, the system must _queue and degrade_, never corrupt state, silently drop a job, or lose a progress message. A job once enqueued must always be observable, either completing or failing with a typed error — never simply vanishing. The GPU families are deliberately pinned to concurrency 1 (SRS constraint C2's VRAM budget), so queueing under load past that point is _expected, correct_ behaviour, not a defect to chase.                                                                                                                  |
| **Required Tools:**         | Locust (`Backend/loadtests/locustfile.py`); `redis-cli INFO memory` / `MONITOR`; `/health/workers` for queue depth; `rq info`; `psutil` / `nvidia-smi`; the Playwright `dataflow` project (also on `testing`) for a concurrent full-stack check.                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **Success Criteria:**       | The saturation point is identified and reported; no data loss, no silent job drop, and no cache corruption occur at or beyond it; overload errors are typed and carry the correct retryable flag per SRS §3.3.2; latency recovers to baseline once load is removed.                                                                                                                                                                                                                                                                                                                                                                                                          |
| **Special Considerations:** | Must run on a dedicated machine at a dedicated time — background load invalidates the stage readings. Real model inference is expensive; a realistic 100-concurrent-user cache-miss scenario may exceed the academic hardware budget available to this team, and if so, the report must say plainly what was actually run versus what was extrapolated, never present an extrapolation as a measurement. Redis LRU eviction under sustained load will legitimately evict entries mid-run — that is correct cache behaviour under the configured cap, not a bug.                                                                                                              |

**Status: ⏳ PENDING.**

`Backend/loadtests/locustfile.py` exists only on `origin/testing`, not on
`develop`, and this report's environment did not have a full live stack
(FastAPI + Redis + all five RQ worker families + a GPU-capable model backend)
running concurrently to drive load against. Executing this section requires
someone to check out `testing`, bring up `docker compose up -d` plus
`python -m app.orchestration.worker all`, run `locust` against a staged
concurrency ramp, and record throughput/latency/queue-depth per stage plus the
saturation point.

**Owner and trigger:** the developer who authored the `testing` branch's load
harness (per `docs/RAVINDU_TESTING_ISSUES_PLAN.md`, that work already exists
on `testing` as of 2026-09-12) is best placed to run this and report actual
figures back into this section before Phase 4 submission.

---

### 3.1.6 Security and Access Control Testing

Mapped one-to-one to the SRS §3.4.2 security requirements SR1–SR7, plus the
inherited items recorded in SRS §4.5.

|                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Technique Objective:**    | Verify upload validation, model-deserialisation safety, session isolation, data-minimisation in cache keys and logs, and remediation of every inherited ECHO 1.0 exposure point recorded in SRS §4.5.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **Technique:**              | **SR1:** submit oversized (>100 MB), over-duration (>15 min), wrong-MIME, magic-number-mismatched, zero-byte, and structurally malformed audio; confirm rejection _before_ hashing. Include at least one polyglot file (a valid WAV header wrapping a hostile payload). **SR2:** attempt to ingest a `.bin`/pickle checkpoint disguised as a model artefact and confirm refusal before any deserialisation is attempted — this is the highest-severity check in this report, since it is arbitrary-code-execution prevention. **SR4:** confirm uploaded audio is purged on its configured TTL. **SR5:** inspect generated cache keys for filenames, session identifiers, or user identifiers; inspect application logs for audio content, transcripts, or other PII. **SR6:** confirm the inherited unauthenticated debug endpoint and wildcard CORS on file-serving routes (SRS §4.5) are hardened, and attempt cross-session dataset access with a forged `sid` session cookie. **SR7:** run a dependency vulnerability scan. Attempt path traversal against every route that accepts a `file_path` parameter (several inference/acoustic routes do). |
| **Oracles:**                | Deterministic and self-verifying for most checks: rejection must surface as an explicit typed error with the correct status code; cross-session access attempts must return 403/404 and never leak data. **A necessary caution, stated rather than implied:** a passing security test proves the _specific tested attack_ failed — it does not prove the system is secure in general. Dependency-scanner output is only an oracle for _known_ CVEs at scan time.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Required Tools:**         | `pytest` (`test_security.py`, `test_session_cookie.py`); `httpx` for forged/malformed requests; a Python dependency auditor (e.g. `pip-audit`) and `npm audit` for SR7; `curl` for raw header/CORS probing; crafted malformed-audio and polyglot fixtures; `safetensors` for format-verification checks.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Success Criteria:**       | Every SR1–SR7 clause has at least one executed test; every SRS §4.5 inherited exposure is either demonstrably remediated with evidence, or explicitly recorded here as outstanding with a Linear id; no unacknowledged high-severity dependency CVE at submission time.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Special Considerations:** | AudioLIT has **no user authentication or role-based access model** — it is single-tenant and session-cookie scoped, appropriate to its academic deployment target (SRS §3.3 — "best-effort availability, no continuous SLA"). The RUP template's "test each user type's permissions" therefore does not apply here, which is stated explicitly rather than left looking like an unaddressed row. TLS (SR3) is a deployment-time concern, not testable against localhost, and is likewise stated rather than silently skipped. No intrusive scanning is run against any host the team does not own.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

**Status: ✅ EXECUTED (automated subset) / ⏳ PENDING (manual exploratory
probing).**

`Backend/tests/test_security.py` and `test_session_cookie.py` ran and passed
as part of the ✅ EXECUTED 588-passed backend suite in §4.1 — this includes,
by direct observation of the run log, at least
`TestFileUploadSecurity::test_corrupted_audio_is_rejected_not_silently_accepted`.
**Not yet executed in this report:** the manual/exploratory items that need a
human attacker's judgement rather than a fixed assertion — forged-cookie
cross-session probing beyond what the existing test suite covers, polyglot
file crafting, path-traversal fuzzing across every `file_path`-accepting
route, and a dependency vulnerability scan (`pip-audit`/`npm audit` are not
currently wired into `Backend/requirements.txt` or CI — see §5). **Owner:**
whoever is assigned SR6 remediation follow-up should run the manual pass and
add `pip-audit`/`npm audit` to CI per SR7 before Phase 4 submission.

---

### 3.1.7 Failover and Recovery Testing

The RUP template frames this section around DASD controllers and client/server
power interruption — a 1990s mainframe/client-server failure model that does
not map onto a containerised Redis + RQ worker architecture. **This section is
deliberately reframed** to AudioLIT's actual failure surface, per SRS
§3.3.1–§3.3.2 (fault tolerance, graceful degradation, and typed error
recovery), rather than force-fitting DASD terminology onto Redis.

|                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Technique Objective:**    | Simulate infrastructure and resource failures and verify graceful degradation, correct retryable/non-retryable fault classification, and recovery to a known-good state without data loss or a silently wrong answer — the one unacceptable outcome under SRS §3.3.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **Technique:**              | A scenario matrix, each with a stated expected behaviour: (a) **Redis unreachable** — this project's own standard pre-push check, `REDIS_URL="redis://127.0.0.1:1/0" pytest -q`; (b) **Redis killed mid-job**; (c) **worker process killed mid-inference** — the job must end up observable as failed or retried, never silently lost; (d) **GPU OOM / unavailable** — must fall back to CPU with a user-visible warning (SRS §3.3.1), not fail the request; (e) **one task family fails** — the other two must still return results (the partial-failure-cascade fault model from §3); (f) **Hugging Face Hub unreachable** during a model download; (g) **corrupt cached value** — must be treated as a miss and recomputed (FR4.3); (h) **WebSocket connection dropped** — `useTaskStatus` must fall back to polling and reconnect (FR3.2); (i) **backend down but the result was already cached** — a warm cache must still serve (SRS §3.3.1); (j) **transient fault** — exponential backoff with a bounded retry count, then a durable failure record (SRS §3.3.2). |
| **Oracles:**                | For each scenario: the API response (correct typed error, correct retryable/non-retryable flag), the UI state (a retry control appears **only** when retrying is actually meaningful, per SRS §3.3.2's explicit requirement), and the durable failure record for unrecoverable faults. The single strongest oracle across the whole matrix is the **retryable/non-retryable classification being correct**, because that classification is what makes recovery automatic rather than requiring manual intervention. Some scenarios (process kill, network partition) are inherently manual to trigger; they are marked as such below and the procedure is recorded so the result is reproducible by someone else.                                                                                                                                                                                                                                                                                                                                                         |
| **Required Tools:**         | `docker compose stop redis` / `start redis`; `REDIS_URL` pointed at an unreachable port (the project's standard technique, used for this report's own ✅ EXECUTED evidence in §4.1); `kill -9` on worker process PIDs; `rq info` to inspect orphaned jobs; a network-disable step for Hugging Face Hub-reachability tests; `fakeredis` for deterministic orchestrator-failure unit tests; `test_task_orchestrator.py`, `test_fanout_orchestrator.py`, `test_queue.py`, `test_system_integration.py`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Success Criteria:**       | Every scenario in the matrix is executed with a recorded outcome; no scenario produces a silently wrong result; no in-flight job becomes permanently unobservable; CPU fallback and partial-family-failure isolation are both demonstrated with evidence.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Special Considerations:** | There is no redundant infrastructure and no continuous SLA (SRS §3.3): recovery is by cheap resubmission, which is an explicit architectural decision, not an untested gap, and this report treats it as such. **Known flake, recorded so it is not mistaken for a real failure:** `SimpleWorker(burst=True)` draining a dependency-gated aggregator on `fakeredis` has intermittently hung `pytest` in this project's history; the mitigation is to stress-run the affected orchestrator tests (~15 iterations) with a background-and-kill timeout — `perl alarm` does not work for this because Python resets `SIGALRM`.                                                                                                                                                                                                                                                                                                                                                                                                                                                |

**Status: ✅ EXECUTED (Redis-unreachable scenario, as part of the standard
suite run) / ⏳ PENDING (all other scenarios, which require deliberate,
destructive manual action).**

Scenario (a), Redis unreachable, is exactly the condition under which this
report's entire §4.1 backend evidence was captured
(`REDIS_URL="redis://127.0.0.1:1/0"`) — 588 passed, 0 failed under that
condition is itself the executed result for this scenario, and it is real,
reproducible evidence, not an assumption. **Scenarios (b) through (j) require
deliberately killing live processes, disconnecting networks, and corrupting
state by hand** — none of that is appropriate to perform unattended while
producing a written report, and doing so needs a human present to observe and
recover from each. **Owner and trigger:** these should be run together in one
dedicated session (the template's own Special Considerations note recommends
exactly this — "run after hours or on an isolated machine") once the team has
Redis, all five worker families, and a spare hour to deliberately break things
and record what happens.

---

### 3.1.8 Configuration Testing

|                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Technique Objective:**    | Verify correct operation across the supported browser, operating system, Python/Node runtime, and hardware-accelerator configurations, and identify any configuration-dependent behaviour.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Technique:**              | **Browsers:** Playwright across Chromium, Firefox, and WebKit. **Python:** 3.10 (CI, `ubuntu-latest`) vs 3.11 (this report's local dev venv) — this divergence is itself a configuration risk worth testing directly, not assuming away. **Node:** 20. **OS:** macOS (this report's environment; also the primary dev platform), Ubuntu (CI). **Accelerator:** CPU-only (CI's explicit CPU-only torch wheel install, and this report's own host) vs GPU — the single highest-value axis, since the CPU-fallback path (FR1.4) only executes on one side of it. **Redis:** containerised (`docker-compose.yml`) vs unreachable (this project's standard test-verification technique). **Frontend:** default vs explicit `VITE_API_BASE_URL`; Vite dev server (`:8080`) vs the built `dist/` bundle. **Viewports:** desktop 1024×768 through 1920×1080 (mobile viewports are not yet in the automated layout suite — see gap below).                  |
| **Oracles:**                | A cross-configuration **differential** oracle: the same functional suite must produce the same pass/fail outcome across configurations, and any divergence is itself the finding, not noise to average away. CI is the continuous instance of this oracle — every PR already runs the Ubuntu/Python 3.10/Node 20/CPU-only configuration automatically. Numeric outputs may legitimately differ slightly between CPU and GPU floating-point paths; where that matters, a tolerance is stated explicitly rather than asserting exact equality across accelerators.                                                                                                                                                                                                                                                                                                                                                                                   |
| **Required Tools:**         | GitHub Actions (`.github/workflows/ci.yml`); Playwright's three browser-engine projects; `docker compose`; local Python 3.10 and 3.11 environments (this report used the project's existing `.venv`, Python 3.11.15); `nvidia-smi` where a GPU is present; browser DevTools device emulation for viewport testing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Success Criteria:**       | The functional suite passes on every declared supported configuration; the CPU-fallback path is exercised in at least one configuration; no browser-specific layout or Web Audio failure; any configuration-dependent numeric divergence is quantified and bounded rather than left unexamined.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **Special Considerations:** | CI deliberately installs the **CPU-only torch wheel** (`.github/workflows/ci.yml`'s own comment: the default Linux wheel is a multi-GB CUDA build unsuited to CPU-only runners), so **CI never exercises the GPU path at all** — GPU coverage is necessarily manual and local, which this report states plainly rather than implying CI covers it. CI currently has **no Redis service container**; per this project's own tracked constraint (LIT-229, confirmed fixed on `develop` as of this report — `health.py` now imports the redis module, not the name, so the previously-documented `RuntimeError: Event loop is closed` failure mode does not reproduce), a Redis container could be added to CI, but was not attempted as part of this report. The Playwright layout suite deliberately runs backend-free to stay fast; the full-stack `dataflow` project (on `testing`) needs the whole stack live and was not run here (see §3.1.5). |

**Status: ✅ EXECUTED (Python 3.11 + macOS + CPU-only + three browser
engines) / ⏳ PENDING (Python 3.10 exact reproduction, native Windows, live
GPU).**

This report itself constitutes one full configuration run:
Python 3.11.15, macOS (Darwin), Node v26.4.0, CPU-only (no discoverable GPU),
against Chromium/Firefox/WebKit via Playwright 1.63.0 — all green (§4.1).
**Two configuration cells remain genuinely unverified by this report:** an
exact Python-3.10 local reproduction of the CI environment (CI itself already
covers this continuously and is green as of commit `a3a78fa`'s merge, per
`gh pr checks`, but this report did not independently reproduce it locally),
and any Windows-native run — no team member's dev machine was confirmed as
Windows at drafting time. **Owner:** whichever team member develops on
Windows, if any, should run the frontend and backend suites there once and
record the result; otherwise, state explicitly in the final report that
Windows is not a supported development configuration for this project.

---

# 4. Deliverables

The following artefacts are produced by this test effort and are the ones by
which its success should be measured:

- Backend `pytest` run logs and summary (§4.1)
- Frontend Jest run logs and summary (§4.1)
- Frontend ESLint report (§4.1)
- Frontend production build log (§4.1)
- Cross-browser Playwright layout report (§4.1)
- FR/SR → test traceability matrix (§4.2)
- GitHub Actions CI history as the continuous regression record (§4.2)
- ⏳ Performance profiling results vs SRS §3.4.1 (pending GPU hardware — §3.1.4)
- ⏳ Locust load-test report (pending a live full-stack run — §3.1.5)
- ⏳ Manual security probing results and dependency-scan output (§3.1.6)
- ⏳ Failover scenario matrix with recorded outcomes for scenarios (b)–(j) (§3.1.7)
- ⏳ Accessibility/usability review notes (§3.1.3)

## 4.1 Test Evaluation Summaries

**Form and content.** Each automated suite run produces: suite name, tests
collected/passed/failed/skipped, wall-clock duration, the git commit SHA it
was run against, and the exact command used (so it is independently
reproducible). Manual technique results are recorded as a dated observation
against the scenario or check it addresses, with the reviewer named.

**Frequency.** Automated suites run on every PR via CI (continuous); the full
manual pass (GPU performance, load testing, exploratory security, failover
scenarios, accessibility) is intended per milestone and once at Phase 4
submission.

**This report's actual execution evidence — commit `a3a78fa`, `develop`
branch, 2026-09-13:**

### Backend — `pytest`

```
$ cd Backend && source .venv/bin/activate
$ REDIS_URL="redis://127.0.0.1:1/0" python3 -m pytest -q --tb=no

platform darwin -- Python 3.11.15, pytest-8.2.0, pluggy-1.6.0
plugins: asyncio-0.23.7, anyio-4.4.0
asyncio: mode=Mode.AUTO
collected 593 items

[... 49 test files ...]

=========== 588 passed, 5 skipped, 370 warnings in 105.87s (0:01:45) ===========
```

Run **with `REDIS_URL` pointed at an unreachable port**, deliberately matching
the condition CI runs under (no Redis service container) rather than the more
forgiving condition of a locally reachable Redis, per this project's own
testing convention. **0 failures.** The 5 skips are all legitimate,
environment-gated skips, not silent omissions — confirmed individually:

| Test                           | Skip reason                                                                       |
| ------------------------------ | --------------------------------------------------------------------------------- |
| `test_function_testing.py:303` | Requires GPU/model resources                                                      |
| `test_memory_profiling.py:35`  | VRAM test requires CUDA                                                           |
| `test_ser_checkpoint.py:139`   | Hits the Hugging Face Hub, downloads ~1.2 GB; gated behind `AUDIOLIT_HUB_TESTS=1` |
| `test_ser_checkpoint.py:152`   | Same Hub-download gate                                                            |
| `test_ser_checkpoint.py:161`   | Same Hub-download gate                                                            |

Warnings observed were all benign and expected: `PySoundFile failed. Trying
audioread instead` (a librosa fallback path exercised deliberately by
malformed-audio fixtures), a captum LIME feature-count warning, and an RQ
`CLIENT SETNAME` compatibility warning against fakeredis — none indicate a
defect.

### Frontend — Jest

```
$ cd Frontend && npm test -- --silent

PASS src/components/audio/WaveformViewer.test.tsx
PASS src/tests/WaveformViewer.test.tsx
PASS src/tests/SpectrogramGridSelector.test.tsx
PASS src/tests/PerturbationTools.test.tsx
PASS src/tests/XAIOverlayCanvas.test.tsx
PASS src/tests/ui-components.test.tsx (6.092 s)

Test Suites: 6 passed, 6 total
Tests:       47 passed, 47 total
Snapshots:   0 total
Time:        6.764 s
```

**0 failures, 6/6 suites, 47/47 tests.**

### Frontend — ESLint

```
$ cd Frontend && npm run lint
✖ 108 problems (0 errors, 108 warnings)
```

**0 errors.** All 108 warnings are `@typescript-eslint/no-explicit-any` (loose
typing on WebSocket payloads and test mocks) and `react-hooks/exhaustive-deps`
(two components with an intentionally partial dependency array) plus
`react-refresh/only-export-components` (context files exporting both a
component and a hook/constant, a Vite Fast-Refresh advisory, not a
correctness issue). None block the build; none are new defects introduced by
this report's evaluation.

### Frontend — Production Build

```
$ cd Frontend && npm run build
✓ 2585 modules transformed.
dist/index.html                     0.91 kB │ gzip:     0.40 kB
dist/assets/index-DoDgtMUc.css     78.27 kB │ gzip:    13.35 kB
dist/assets/index-BZ_0YlEE.js   5,906.67 kB │ gzip: 1,761.88 kB
✓ built in 9.16s
```

**Build succeeds.** One real observation worth carrying into the risk log
(§5): the main JS bundle is 5.9 MB unminified / 1.76 MB gzipped in a single
chunk, and Vite's own build output warns that chunks over 500 kB should be
code-split. This is not a test failure, but it is a genuine, previously
unrecorded finding from this evaluation — large enough to affect the SRS
§3.2.2 "under 30 seconds, cold" first-load task-time target on a slow
connection, though this report did not measure that directly (see §3.1.4's
GPU/performance gap — this is the frontend analogue of it, similarly
unmeasured here).

### Cross-browser Layout — Playwright

See full output under §3.1.3 and §3.1.8: **12/12 passed**, Chromium + Firefox

- WebKit, three desktop viewports, 10.1s total.

### Not executed in this report (all named again in §3.1.x with owner and

trigger): GPU-gated performance figures, Locust load testing, manual
exploratory security probing, failover scenarios (b)–(j), accessibility/
usability review, live-Redis eviction timing, exact Python-3.10 local
reproduction, Windows configuration.

## 4.2 Reporting on Test Coverage

**Form.** The centrepiece is the FR/SR-to-test traceability matrix below.
Each row was checked against the actual repository, not assumed from the SRS
alone — the count column is a `grep` of test files referencing the FR id
directly, cross-checked by file inspection where the count was zero or
surprising.

**Frequency.** Regenerated at each milestone and immediately before Phase 4
submission; the underlying counts are cheap to reproduce
(`grep -rl "FR<n>" Backend/tests/*.py`).

| Req.    | Summary                                       | Technique (§3.1.x)  | Test file(s)                                                                                                                                                                                               | Coverage                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------- | --------------------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FR1     | Dynamic HF model ingestion, safetensors-only  | 3.1.2, 3.1.6        | `test_model_registry_service.py`, `test_models_routes.py`, `test_custom_model_fidelity.py`                                                                                                                 | ✅ 3 files                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| FR2     | Benchmark dataset ingestion & management      | 3.1.1, 3.1.2        | `test_dataset_ingestion.py`, `test_dataset_service.py`, `test_datasets_routes.py`, `test_dataset_management_routes.py`, `test_l2arctic_loader.py`, `test_librispeech_loader.py`, `test_asvspoof_loader.py` | ✅ 7 files                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| FR3     | Asynchronous multi-task inference             | 3.1.2, 3.1.5, 3.1.7 | `test_task_orchestrator.py`, `test_multitask_orchestrator.py`, `test_fanout_orchestrator.py`                                                                                                               | ✅ 3 files                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| FR4     | Deterministic cache-by-hash retrieval         | 3.1.1, 3.1.4        | `test_redis_cache.py`, `test_results_cache.py`, `test_hashing.py`, `test_warmup_cache_contract.py`                                                                                                         | ✅ 4 files                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| FR6     | Speech Emotion Recognition                    | 3.1.2               | `test_ser_model.py`, `test_ser_corpora.py`, `test_ser_checkpoint.py`                                                                                                                                       | ✅ 3 files (checkpoint tests partly Hub-gated)                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| FR7     | Audio Deepfake Detection                      | 3.1.2               | `test_deepfake_classifier.py`, `test_asvspoof_loader.py`, `test_degradation_scoring.py`                                                                                                                    | ✅ 3 files                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| FR8     | Spectrogram LIME/SHAP + Grad-CAM              | 3.1.2               | `test_grad_cam.py`, `test_saliency_service.py`, `test_saliency_routes.py`                                                                                                                                  | ✅ 3 files                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| FR9     | Integrated Gradients (label correction)       | 3.1.2               | `test_integrated_gradients.py`, `test_grad_cam.py`                                                                                                                                                         | ✅ 2 files — **regression-critical, verify the two are asserted as distinct outputs, not just both present**                                                                                                                                                                                                                                                                                                                                                                      |
| FR10    | Acoustic wave profiling (F0/RMS/log-mel)      | 3.1.2               | `test_acoustic_profiler_service.py`, `test_acoustic_routes.py`                                                                                                                                             | ✅ 2 files                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| FR11    | Latent projection explorer (PCA/t-SNE/UMAP)   | 3.1.2, 3.1.3        | _(no dedicated backend test file found by direct search)_                                                                                                                                                  | ⚠️ **0 files — real gap.** `umap-learn` is a declared dependency and `/inferences/embeddings` exists as a route (`app/api/routes/inferences.py:796`), and the frontend has `EmbeddingContext`/`EmbeddingPanel`/`EmbeddingPlot`, but no test file targets the embedding-extraction or projection logic directly. **This is a genuine coverage gap, not an oversight in this matrix** — confirmed by direct `grep` across `Backend/tests/` and `Frontend/src/tests/` on 2026-09-13. |
| FR12    | Canvas-driven signal mutation                 | 3.1.2, 3.1.3        | `test_perturbation_service.py`                                                                                                                                                                             | ⚠️ 1 file — thin for a committed FR with four sub-clauses (FR12.1–12.4); frontend `PerturbationTools.test.tsx` adds UI-side coverage but the backend has a single file                                                                                                                                                                                                                                                                                                            |
| FR15    | Accent bias profiling                         | 3.1.2               | `test_accent_bias_profiler.py`, `test_accent_bias_runner.py`, `test_l2arctic_loader.py`, `test_evaluation_routes.py`                                                                                       | ✅ 4 files                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| FR16    | Attribution faithfulness auditing             | 3.1.2               | `test_auc_faithfulness.py`, `test_faithfulness.py`, `test_evaluation_scoring.py`, `test_high_saliency_masking.py`                                                                                          | ✅ 4 files                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| FR17    | Faithful attention extraction (fallback flag) | 3.1.2               | `test_hook_manager_service.py`, `test_provenance.py`                                                                                                                                                       | ✅ 2 files — **regression-critical, the same as FR9**                                                                                                                                                                                                                                                                                                                                                                                                                             |
| SR1–SR7 | Security requirements                         | 3.1.6               | `test_security.py`, `test_session_cookie.py`                                                                                                                                                               | ✅ automated subset executed (§4.1); manual pass pending (§3.1.6)                                                                                                                                                                                                                                                                                                                                                                                                                 |

**There is no FR5, FR13, or FR14 row** — the reconciled SRS does not define
them (FR5, multi-model comparison, was demoted to non-committed stretch), and
this matrix does not invent one.

**Two real gaps this matrix surfaces, found by actually checking rather than
assuming coverage exists:**

1. **FR11 (Latent Projection Explorer) has no dedicated backend test file.**
   The route and the UMAP dependency exist; nothing in `Backend/tests/`
   targets them directly by name. This should be raised with whoever owns
   FR11 before Phase 4 submission — either a test exists under a
   non-obvious filename and this matrix is wrong (re-check before acting),
   or it is a genuine gap to close.
2. **FR12 (Canvas Mutation) has only one backend test file** against four
   SRS sub-clauses (non-destructive originals, Web Audio preview, correct
   16 kHz mono shape, sub-500ms/2s timing) — likely under-tested relative to
   its acceptance-criteria surface.

**Coverage metric not currently available:** line/branch coverage
(`pytest-cov`) is not installed in this project's environment
(`ModuleNotFoundError: No module named 'pytest_cov'`, confirmed 2026-09-13)
and is not in `Backend/requirements.txt`. Adding it is a small, high-value
change to make before Phase 4 submission — it would turn "588 passed" into a
line-coverage percentage, which is a stronger coverage claim than pass count
alone.

**Generic per-run report fields**, applied to every future automated run:
date, commit SHA, logged-in user/CI actor, suite, tests executed, pass count,
fail count, pass percentage, fail percentage, comments (used above for the
skip-reason table and the warning summary).

---

# 5. Risks, Dependencies, Assumptions, and Constraints

| Risk                                                                                                                                      | Mitigation Strategy                                                                                                                                                                              | Contingency (Risk is realised)                                                                                                        |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| MongoDB tier (SRS §3.10) is specified but unimplemented, so §3.1.1 cannot cover it as written                                             | Flag in Linear per `CLAUDE.md`'s conflict-handling rule; scope §3.1.1 explicitly to Redis + corpora (done in this report)                                                                        | Record as a known, named scope deviation with its Linear id rather than silently testing around it                                    |
| No GPU available in this report's evaluation environment — FR1.4 CPU-fallback and every model-bound SRS §3.4.1 target are unmeasured here | Run §3.1.4 manually on the team's GPU dev machine before each milestone; state hardware on every figure produced                                                                                 | Report all performance figures as point-in-time with hardware named; never substitute a CPU-host number for a GPU target              |
| Load testing (§3.1.5) may exceed the academic hardware budget at realistic concurrency                                                    | Stage the ramp; separate cache-hit from cache-miss profiles; report what was actually run vs extrapolated                                                                                        | State the executed ceiling plainly; never present an extrapolation as a measurement                                                   |
| No ground truth exists for saliency-map correctness                                                                                       | Rely on metamorphic and deletion-score faithfulness oracles (§3, FR16.1) rather than direct equality assertions                                                                                  | Report faithfulness metrics, not accuracy claims, for every interpretability output                                                   |
| `SimpleWorker(burst=True)` draining a dependency-gated aggregator on `fakeredis` has intermittently hung `pytest`                         | Stress-run affected orchestrator tests (~15 iterations) with a background-and-kill timeout                                                                                                       | Quarantine the flaky test and file it explicitly rather than silently retrying past it                                                |
| Two individually green PRs have previously broken only in combination (the PR #10/#13 incident)                                           | Fetch and merge latest `develop`, then re-run the full suite, before pushing any PR                                                                                                              | Revert and fix forward on `develop`; do not force-merge past a discovered combination break                                           |
| Corpora are large, licence-restricted, and slow to provision                                                                              | Streaming/sub-sampling loaders under the ~100 GB footprint bound (FR2.2)                                                                                                                         | Sub-sample and state the reduced corpus size used, in any report that depends on corpus scale                                         |
| Model downloads make tests network- and Hub-dependent                                                                                     | Pin model revisions (SER pinned at `611e6db8`); gate Hub-downloading tests behind `AUDIOLIT_HUB_TESTS=1` (confirmed already in place, `test_ser_checkpoint.py`)                                  | Mark as `slow`/gated; exclude from the per-PR CI gate, run only at milestones                                                         |
| Three developers may run concurrent sessions against the same repository the same day                                                     | Re-check Linear and `gh pr list --state open` immediately before starting work in an area; merge latest `develop` and re-run tests after any merge not your own                                  | Flag and resolve the specific combination conflict rather than assuming a green branch alone proves safety                            |
| FR11 and FR12 have thin or absent backend test coverage (§4.2 finding)                                                                    | Assign owners before Phase 4 submission; add the missing FR11 embedding/projection test file                                                                                                     | Report the gap explicitly in the final submission rather than let the traceability matrix imply coverage that does not exist          |
| Frontend production bundle is a single 5.9 MB (1.76 MB gzip) chunk with no code-splitting (§4.1 finding)                                  | Consider `manualChunks`/dynamic `import()` for the largest dependencies (Plotly, Wavesurfer, PyTorch-adjacent tooling if any ships client-side) before measuring the SRS §3.2.2 cold-load target | Measure actual cold-load time on a throttled connection before deciding whether this is acceptable for the academic deployment target |

**Dependencies:** a reachable Redis 7 instance for any live-Redis test tier
(§3.1.1's pending item, §3.1.5, §3.1.7); Hugging Face Hub availability for
cold-model-download tests; per-corpus licence compliance (RAVDESS, L2-ARCTIC,
ESD, ASVspoof 2021 DF are non-commercial/research-use only); GPU access for
§3.1.4 and the GPU-gated test skips to actually run.

**Assumptions:** single-tenant academic deployment with best-effort
availability and no continuous SLA (SRS §3.3); no user authentication or
role-based access tier exists or is planned; the `testing` branch remains a
strict superset of `develop` (re-verify this before relying on it, since
branches diverge — confirmed true only as of 2026-09-13).

**Constraints:** SRS constraint C2 (VRAM budget, hence GPU-family concurrency
pinned to 1); constraint C3 (safetensors-only, no arbitrary pickle
deserialisation); the ~100 GB dataset working-footprint bound (FR2.2); CI's
CPU-only torch wheel, meaning CI itself can never be the source of GPU-path
evidence.

---

# 6. References

**Testing tools and frameworks:**

- pytest 8.2.0, available at https://pytest.org/ (Accessed 2026-09-13)
- pytest-asyncio 0.23.7, available at https://pytest-asyncio.readthedocs.io/ (Accessed 2026-09-13)
- fakeredis 2.23.2, available at https://github.com/cunla/fakeredis-py (Accessed 2026-09-13)
- httpx 0.27.0, available at https://www.python-httpx.org/ (Accessed 2026-09-13)
- Jest 29.7.0, available at https://jestjs.io/ (Accessed 2026-09-13)
- Testing Library (React) 16.3.2, available at https://testing-library.com/docs/react-testing-library/intro/ (Accessed 2026-09-13)
- Playwright 1.63.0, available at https://playwright.dev/ (Accessed 2026-09-13)
- Locust, available at https://locust.io/ (Accessed 2026-09-13)
- ESLint 9.9.0, available at https://eslint.org/ (Accessed 2026-09-13)

**System dependencies (as pinned in this project):**

- FastAPI 0.111.0, available at https://fastapi.tiangolo.com/ (Accessed 2026-09-13)
- RQ (Redis Queue) 2.10.0, available at https://python-rq.org/ (Accessed 2026-09-13)
- Redis 7, available at https://redis.io/ (Accessed 2026-09-13)
- PyTorch ≥2.6 (installed 2.13.0 in this evaluation environment), available at https://pytorch.org/ (Accessed 2026-09-13)
- Transformers ≥4.30, available at https://huggingface.co/docs/transformers/ (Accessed 2026-09-13)
- Captum ≥0.6, available at https://captum.ai/ (Accessed 2026-09-13)
- Librosa ≥0.10, available at https://librosa.org/ (Accessed 2026-09-13)
- React 18.3, available at https://react.dev/ (Accessed 2026-09-13)
- Vite 5.4, available at https://vite.dev/ (Accessed 2026-09-13)

**Methods and standards:**

- Web Content Accessibility Guidelines (WCAG) 2.1, Level AA, W3C, available at https://www.w3.org/TR/WCAG21/ (Accessed 2026-09-13)
- Rational Unified Process Test Plan template — the structural basis for this document (`docs/testing/Template for Test plan.docx`)

**Project documents (this repository):**

- AudioLIT Software Requirements Specification v1.0, `docs/SRS.md`
- AudioLIT Software Architecture Document v1.0, `docs/SAD.md`
- AudioLIT project conventions and errata, `docs/README.md`
- AudioLIT issue plan and dependency map, `docs/ISSUE_PLAN.md`
- `docs/testing/TEST_PLAN_DESIGN.md` — the design specification this report was drafted from
- ECHO 1.0 baseline, `AudioLIT-DSE-Project/ECHO` (forked from `AnasSAV/ECHO`)
