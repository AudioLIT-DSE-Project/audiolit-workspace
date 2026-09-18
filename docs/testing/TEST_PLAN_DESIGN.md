# AudioLIT Master Test Plan — Design Specification

**What this document is:** the blueprint for writing `AudioLIT Master Test Plan v1.0`.
It does not contain the report's prose. It specifies, section by section, **what
must appear in each section**, which AudioLIT requirement each section is
accountable to, and which real repository artefact provides the evidence.

**Who writes from it:** Tharusha Perera, Rahim Iqbal, Ravindu Pathirana.

**Status:** design draft, 2026-09-13. Section 0 lists decisions that must be
settled before drafting starts.

---

## 0. Decisions to settle before anyone drafts

These are genuine forks. Resolve them as a team, record the answer here, then write.

| # | Decision | Why it is open | Recommendation |
|---|----------|----------------|----------------|
| D1 | **Plan or report?** The template is a *Test Plan* (forward-looking: "tests *will* be run"). The brief says "test report". | The two have different tenses and different §4 content. A plan describes intent; a report presents executed results. | Title it **"Master Test Plan"** (as the sample does) and write §1–§3 in plan voice, but make **§4 Deliverables carry the executed evidence** — real pytest/Jest/Playwright output, counts, pass rates. This satisfies both readings and matches how the sample used §4 (it pasted real PHPUnit and profiler screenshots). |
| D2 | **§3.1.1 is "Data and Database Integrity Testing" but AudioLIT has no SQL database.** | The template assumes a DBMS with tables, ORM, and SQL. AudioLIT's persistence is Redis 7 (cache + queues + pubsub) and a read-only dataset corpus on disk. | **Keep the template's section title unchanged** (marking scheme follows the template) and reinterpret the *content* as Redis keyspace integrity + cache-value integrity + dataset corpus integrity. Say so explicitly in the section's opening paragraph so the examiner sees the mapping was deliberate, not a gap. |
| D3 | **MongoDB.** SRS §3.10 specifies a MongoDB 6.0+ metadata tier with four collections, TTL indexes, and compound indexes. **It is not implemented** — no `pymongo`/`motor` in `Backend/requirements.txt`, no Mongo reference anywhere in `Backend/app/`. | §3.1.1 cannot claim to test a tier that does not exist, and cannot silently omit a committed SRS requirement either. | State it plainly in §3.1.1 and again in §5 as a risk: the MongoDB tier is specified but unimplemented, so database-integrity testing covers Redis and the corpus only. **Flag it in Linear before the report is submitted** — per `CLAUDE.md`, a SRS/repo conflict gets raised, not silently resolved. |
| D4 | **Which branch is the target-of-test?** `develop` carries the app; `testing` carries `dataflow.spec.ts`, `locustfile.py`, and `test_inference_consistency.py`. | §3.1.5 Load Testing has no evidence at all on `develop`. | Declare the target-of-test as **`testing`** (a strict superset of `develop`) and say so in §2. Otherwise §3.1.5 is unevidenced. |
| D5 | **Celery.** SRS §3.10 still calls Redis "the Celery broker". | Celery is removed project-wide (`CLAUDE.md`, hard rule). | Never mention Celery in the report. Cite RQ. Note the SRS wording as a documented erratum if a reader might catch it. |

---

## 1. Source documents inspected, and what each contributes

| Source | Verdict | What we take from it |
|--------|---------|----------------------|
| `Template for Test plan.docx` | **The binding structure.** Classic RUP/Rational test-plan template. | Exact section list, exact section titles, and the fixed six-row technique table. Non-negotiable. |
| `Sample test plan report.pdf` (Find Your Job, MPM Solutions, 2016, 20 pp.) | **The depth and tone calibration.** A completed instance of the same template. | How much prose per cell, the §3 bulleted preamble before §3.1, real tool screenshots in §4, the four-row risk table, the plain URL reference list. |
| `The Learning Interpretability Tool (LIT) for Voice.pdf` | ⚠️ **Not a filled-in report.** | Nothing. See warning below. |

> ### ⚠️ The "ECHO baseline test report" is the blank template
>
> All 11 pages of `The Learning Interpretability Tool (LIT) for Voice.pdf` are
> the unmodified RUP template: `<Project Name>`, `<Iteration/ Master> Test Plan`,
> `Version <1.0>`, every cell still carrying the blue-italic `[guidance text]`,
> the revision-history table still holding `<dd/mmm/yy>` and `<name>`.
> There is no ECHO content in it — no target test items, no techniques, no
> results.
>
> **Consequence for us:** there is no prior-year baseline to inherit, extend, or
> differentiate against. AudioLIT's test plan is written from zero. Budget for
> that. If a genuine ECHO test report exists elsewhere, find it before drafting —
> it would change §1's framing (we would then be reporting on *extending* a
> tested baseline rather than establishing one).
>
> One incidental gain: the file confirms the template's own internal numbering
> bug — it labels "Testing Techniques and Types" as **1.1** (should be 3.1) and
> "Reporting on Test Coverage" as **1.1** (should be 4.2). The sample report
> silently fixed both. **We fix both too.**

---

## 2. Mandatory structure

Reproduce exactly. Do not add, remove, reorder, or rename top-level sections.
Depth is ours to choose; structure is not.

```
Title page        — AudioLIT / Master Test Plan / Version 1.0
Revision History  — table: Date | Version | Description | Author
Table of Contents — with page numbers

1.  Evaluation Mission and Test Motivation
2.  Target Test Items
3.  Test Approach
    3.1   Testing Techniques and Types      [template mis-numbers this 1.1 — fix]
      3.1.1  Data and Database Integrity Testing
      3.1.2  Function Testing
      3.1.3  User Interface Testing
      3.1.4  Performance Profiling
      3.1.5  Load Testing
      3.1.6  Security and Access Control Testing
      3.1.7  Failover and Recovery Testing
      3.1.8  Configuration Testing
4.  Deliverables
    4.1   Test Evaluation Summaries
    4.2   Reporting on Test Coverage         [template mis-numbers this 1.1 — fix]
5.  Risks, Dependencies, Assumptions, and Constraints
6.  References
```

### The technique table — the report's core unit

Each of §3.1.1–§3.1.8 is a short framing paragraph followed by **one table with
exactly these six rows, in this order, with these labels**:

| Row | What belongs in it | What does *not* |
|-----|--------------------|-----------------|
| **Technique Objective:** | What behaviour we are trying to provoke and observe, tied to named FR/SR/NFR ids. | Tool names. |
| **Technique:** | The mechanics — the actual commands, fixtures, harnesses, and steps. Name real files. | Aspirations. |
| **Oracles:** | *How we know a result is right.* The pass/fail decision procedure, and honestly, its limits. | Restating the technique. |
| **Required Tools:** | Bulleted, versioned, real. | Tools we do not own (no Rational Quantify, no Canned Heat). |
| **Success Criteria:** | The bar for declaring the technique adequately applied. | Per-test assertions. |
| **Special Considerations:** | Constraints, intrusiveness, what we could not do and why. | Filler. |

**The Oracles row is where this report is won or lost.** The sample's weakest
cells are its oracles ("the developer can know whether the system behaves as
expected"). AudioLIT is an ML system, so its oracle problem is real and
interesting — say so. Three oracle classes to name explicitly and use throughout:

1. **Deterministic oracles** — exact expected value. Cache round-trips, hash
   stability, HTTP status codes, typed error codes, schema shape.
2. **Statistical / tolerance oracles** — no single right answer, but a bounded
   one. WER within a delta, F0 within tolerance of a Librosa/Praat reference
   (FR10.3), latency percentiles.
3. **Metamorphic / relational oracles** — no ground truth at all, but an
   invariant must hold between two runs. *This is AudioLIT's most valuable
   oracle.* Examples: a warm cache read must equal the cold computation
   (`test_warmup_cache_contract.py`); identical requests must be byte-identical
   (FR4.4); masking high-saliency regions must drop confidence more than masking
   random regions (FR16.1); an attribution labelled Grad-CAM must not be
   Integrated Gradients (FR9, the LIT-147 defect).

---

## 3. Section-by-section content specification

### §1 Evaluation Mission and Test Motivation

**Length:** ~1 page. Four moves, in order.

1. **What AudioLIT is.** An interpretability workbench for ASR, SER, and Audio
   Deepfake Detection, extending the open-source ECHO 1.0 baseline. FastAPI +
   Redis + RQ backend, React 18 + TypeScript + Vite frontend, five per-family
   background worker queues.
2. **Why testing this system is not ordinary web-app testing.** This is the
   paragraph that distinguishes our report from the sample's. Make these points:
   - The system's *product* is an explanation. A wrong transcript is a visible
     bug; a **plausible but unfaithful saliency map is an invisible one**, and it
     is worse, because a user acts on it.
   - Most outputs have **no ground truth**, so conventional assertion-based
     testing is insufficient and metamorphic oracles are load-bearing.
   - Inference is **non-deterministic and expensive**, so the cache is not an
     optimisation we may skip testing — correctness now depends on it (FR4.4).
   - The baseline is **inherited and known-defective**: ECHO 1.0 silently
     substituted fabricated attention when real extraction failed (FR17), and
     shipped a UI label reading "GradCAM" over Integrated Gradients (FR9). We
     inherited the code that produced both. Regression testing against inherited
     defects is a first-class mission here.
3. **The mission statement.** Select from the template's menu and justify each
   choice. Recommended: *verify a specification* (FR1–FR17, SR1–SR7 are written
   and testable), *find important problems and assess quality risks* (the
   faithfulness risk above), *advise about product quality* (Phase 4 academic
   submission). Explicitly **de-scope** "certify to a standard" — there is no
   certification target — and say why.
4. **Scope boundary.** Committed scope only. SRS §4.4 stretch items (including
   FR5 multi-model comparison) are **not** under test. There is no FR5, FR13, or
   FR14 — do not invent them.

---

### §2 Target Test Items

**Length:** ~1 page, grouped and ranked. Give each group a criticality
(High/Medium/Low) with a one-line justification — the template asks for relative
importance and the sample omitted it, so this is cheap differentiation.

| Group | Items | Criticality |
|-------|-------|-------------|
| **API surface** | 15 routers under `Backend/app/api/routes/` — `upload`, `inference`, `inferences`, `saliency`, `perturbations`, `acoustic`, `evaluation`, `datasets`, `dataset_management`, `models`, `results`, `session`, `tasks`, `health`, `debug` | High |
| **Domain / ML engines** | `app/domain/` — model registry + loader, hook manager, saliency service, acoustic profiler, perturbation, accent-bias profiler, evaluation, provenance | High — the interpretability claims live here |
| **Orchestration fabric** | `app/orchestration/task_orchestrator.py` (the single RQ fabric), `worker.py`, fan-out / multitask orchestrators, five queues: `asr`, `ser`, `add`, `xai`, `mutation` | High — FR3, and the site of two prior duplicate-module incidents |
| **Cache and persistence** | `app/infrastructure/cache_keys.py` (MD5-of-path scheme, hot paths), `app/core/redis.py` (`RedisCacheManager`, SHA-256 content-addressed, FR4), Redis 7 keyspace | High — FR4.4 reproducibility |
| **Frontend** | `Index.tsx` workbench, panels (Prediction, Acoustic, Accent Bias, Faithfulness, Embedding), `XAIOverlayCanvas`, `WaveformViewer`, `SpectrogramGridSelector`, `PerturbationTools`, contexts, `useTaskStatus` WS hook | High |
| **Third-party depended-upon** | PyTorch ≥2.6, Transformers ≥4.30, Captum ≥0.6, Librosa ≥0.10, soundfile, RQ 2.10, Redis 7, FastAPI 0.111, React 18.3, Vite 5.4 | Medium — not ours, but our failures surface through them |
| **Models under test** | Whisper (ASR), Wav2Vec2 SER pinned at `firdhokk/...xlsr-53` rev `611e6db8`, deepfake detector | High |
| **Corpora** | Common Voice, LibriSpeech, RAVDESS, CREMA-D, L2-ARCTIC, ASVspoof 2021 DF, ESD | Medium |
| **Environment** | Python 3.10 (CI) / 3.11 (local), Node 20, Redis 7-alpine, CPU-only CI runners vs GPU dev | Medium — see §3.1.8 |

State the **target-of-test branch** here (decision D4).

State explicitly what is **excluded**: Hugging Face Hub availability, the
pretrained models' own training-time accuracy, and browser engine internals.

---

### §3 Test Approach — preamble

Before §3.1, mirror the sample: a **bulleted overview naming all eight techniques
with 2–4 lines each** on how that technique is realised for AudioLIT. This is the
examiner's roadmap. Then add two short paragraphs the sample lacks:

- **The test pyramid as actually built.** ~567 backend test functions across 49
  `pytest` files; 41 frontend Jest + Playwright cases; one cross-browser layout
  suite; one full-stack dataflow suite and a Locust load harness on `testing`.
  Automated-first, with named manual exceptions.
- **The fault models we are testing against.** Name them, because the template
  asks for fault/failure models and the sample ignored the request entirely:
  1. *Silent unfaithfulness* — a returned explanation that is fabricated or
     mislabelled (FR17, FR9).
  2. *Cache-shape corruption* — right key, wrong value shape, so the consumer
     reads it instead of recomputing and crashes downstream. This is a real
     incident documented in `cache_keys.py`: warmup stored an ASR dict under the
     transcript family and `/inferences/whisper-accuracy` died on
     `'dict' object has no attribute 'lower'`.
  3. *Silent combination breakage* — two individually-green changes that break
     only together (PR #10 + PR #13 broke pytest collection repo-wide with no
     git conflict).
  4. *Resource exhaustion* — VRAM overflow, Redis memory cap, oversized upload.
  5. *Partial-failure cascade* — one task family failing taking the others with
     it, which SRS §3.3.1 forbids.

---

### §3.1.1 Data and Database Integrity Testing

Open by stating the reinterpretation (decision D2) and the MongoDB gap (D3).

| Row | Content to write |
|-----|------------------|
| **Objective** | Exercise Redis keyspace operations and dataset loaders independently of the UI, to detect cache corruption, key collisions, shape violations, and corpus integrity failures. Accountable to **FR4.1–FR4.4**, **FR2.1–FR2.3**, **SR5**. |
| **Technique** | Drive `RedisCacheManager` directly with `fakeredis`; assert round-trip fidelity through msgpack/lz4. Assert key uniqueness across the (audio, model, task, params) tuple and that `CACHE_SCHEMA_VERSION` participates in the key. Seed loaders with valid, malformed, truncated, and wrong-sample-rate audio. Verify LRU eviction under a forced memory cap and that a corrupt value is treated as a miss and recomputed. Verify per-corpus licence metadata survives load (FR2.3) and that `measure_footprint()` enforces the ~100 GB bound (FR2.2). Name the files: `test_redis_cache.py`, `test_results_cache.py`, `test_hashing.py`, `test_data_integrity.py`, `test_warmup_cache_contract.py`, `test_dataset_ingestion.py`, `test_dataset_service.py`, `test_l2arctic_loader.py`, `test_librispeech_loader.py`, `test_asvspoof_loader.py`. |
| **Oracles** | Deterministic for round-trips and digests: `decode(encode(x)) == x`, and the same input yields a byte-identical cached response (FR4.4) — self-verifying, automatable. **Metamorphic for warm-vs-cold**: a warmed entry must equal the cold computation. **State the known oracle weakness honestly:** a right key holding a wrong-shaped value passes a naive round-trip oracle, so value *shape* is asserted separately per key family — this is exactly the defect `cache_keys.py` exists to prevent. Also state that `fakeredis` is an emulator, so it is an oracle for our logic, not for Redis 7's own eviction behaviour. |
| **Required Tools** | Redis 7-alpine via `Backend/docker-compose.yml`; `fakeredis` 2.23.2; `pytest` 8.2 + `pytest-asyncio`; `msgpack`, `lz4`; `redis-cli` for manual keyspace inspection; `soundfile` + `numpy` for fixture generation. |
| **Success Criteria** | Every key family in `cache_keys.py` has at least one shape-assertion test; every corpus loader has integrity and malformed-input tests; FR4.4 byte-identity demonstrated; eviction and corrupt-value-as-miss both demonstrated. |
| **Special Considerations** | No SQL/ORM layer, so no SQL-injection or schema-normalisation concerns at this tier. **MongoDB (SRS §3.10) is unimplemented — declare it.** Redis persistence is deliberately disabled (every entry recomputable), so there is no backup/restore path to test here — that moves to §3.1.7. Tests must pass with Redis unreachable, because CI has no Redis service. |

---

### §3.1.2 Function Testing

| Row | Content to write |
|-----|------------------|
| **Objective** | Exercise every committed FR end-to-end through the public API and UI — ingestion, inference, attribution, profiling, mutation, auditing — with valid and invalid input, verifying correct results, correct typed errors, and correct business rules. |
| **Technique** | Route-level black-box tests via `httpx.AsyncClient` against the FastAPI app for all 15 routers. Domain-level tests per engine. Reference the traceability matrix (§4.2) rather than prose-listing 17 FRs. Call out the high-value functional checks: safetensors-only enforcement and `UNSUPPORTED_ARCHITECTURE` within 60 s (FR1.1, FR1.3); concurrent ASR+SER+ADD dispatch on one clip (FR3.1); SER returning ≥6 categories with a full probability distribution (FR6.1–6.2); ADD binary bona-fide/synthetic with confidence (FR7.1); Grad-CAM genuinely gradient-weighted and **not** Integrated Gradients (FR8.2, FR9); fallback attributions carrying an explicit provenance flag (FR17.1); F0/RMS/log-mel computed and validated (FR10.1); mutations non-destructive and returning 16 kHz mono (FR12.1, FR12.3); per-cohort WER disparity (FR15.1); deletion-score audit (FR16.1). |
| **Oracles** | Deterministic for contracts — status codes, typed error codes, JSON schema shape, label sets. Tolerance-based for numeric outputs — F0 and RMS validated against a **Librosa/Praat reference implementation, which FR10.3 mandates**; WER within a delta. Metamorphic for the interpretability claims, because no ground-truth saliency map exists: masking top-K salient regions must reduce confidence more than masking random regions of equal size and count. Say plainly that a saliency map's *correctness* is not directly assertable and that faithfulness metrics are the substitute. |
| **Required Tools** | `pytest` 8.2, `pytest-asyncio`, `httpx` 0.27, `fakeredis`; Jest 29 + Testing Library for frontend logic; Playwright 1.63 for full-stack dataflow; FastAPI `/docs` (OpenAPI) for contract inspection; Captum 0.6 and Librosa 0.10 as reference implementations. |
| **Success Criteria** | Every committed FR traces to ≥1 executed test (100% FR coverage in the §4.2 matrix); every route has happy-path and invalid-input coverage; every inherited defect (FR9, FR17) has a dedicated regression test that fails against the baseline behaviour. |
| **Special Considerations** | Model downloads make cold tests slow and network-dependent — mark `slow`, mock the Hub where the model is not what is under test. Inference is non-deterministic: pin seeds and revisions (the SER checkpoint is pinned at `611e6db8`) or assert on tolerance, never on exact floats. Tests calling any orchestrator function need the `broker` fixture even when the domain call is mocked, because the wrapper itself touches `publish_progress`/`get_redis_connection`. |

---

### §3.1.3 User Interface Testing

| Row | Content to write |
|-----|------------------|
| **Objective** | Verify navigation, panel state, canvas interaction, playback synchronisation, and accessibility conformance across the workbench. Accountable to **SRS §3.2.3** (WCAG 2.1 AA target), **§3.9.1** (panel inventory), **FR8.4**, **FR10.2**, **FR11.2**, **FR12.2**. |
| **Technique** | Jest + Testing Library component tests for the interaction-heavy primitives — `WaveformViewer.test.tsx`, `XAIOverlayCanvas.test.tsx`, `SpectrogramGridSelector.test.tsx`, `PerturbationTools.test.tsx`, `ui-components.test.tsx`. Playwright `e2e/layout.spec.ts` for responsive layout across three engines. Manual keyboard-only traversal, screen-reader spot-check, and dark/light contrast audit. Verify time-synchronisation between playback, F0 contour, and attribution overlay (FR10.2). Verify alpha-blend transparency control and the perceptually uniform colour scale (FR8.4). Verify the Web Audio preview mutes/plays a region locally before dispatch (FR12.2). Verify a fallback-derived attribution is **visibly** distinguished (FR17.1) — a UI requirement, not only an API one. |
| **Oracles** | Automatable: DOM assertions, ARIA role/label presence, computed contrast ratios against the 4.5:1 bar, Playwright layout assertions and screenshot diffs. **Not automatable, and say so:** whether an explanation *reads* as interpretable, and whether progressive disclosure achieves the 30-minute first-counterfactual goal (§3.2.1) — these need human judgement, so a small structured usability walkthrough is the oracle, with the task-time table in §3.2.2 as the measured bar. |
| **Required Tools** | Jest 29 + jsdom, `@testing-library/react` 16 + `user-event` 14, `jest-dom` 6; Playwright 1.63 (Chromium, Firefox, WebKit); browser DevTools; an axe-style accessibility checker; a colour-contrast analyser; a screen reader (VoiceOver / NVDA). |
| **Success Criteria** | Every SRS §3.9.1-committed panel has ≥1 automated test; all three engines pass the layout suite; no WCAG AA contrast failure on text or heatmap legend; keyboard traversal reaches every interactive control in a logical order. |
| **Special Considerations** | Canvas and WebGL content is largely opaque to DOM assertions — test the *state* driving the canvas plus a visual diff, not the pixels. jsdom has no real Web Audio or Canvas 2D, so those are mocked in Jest and must be covered for real in Playwright. Plotly and WaveSurfer render asynchronously; assert on settled state. Dark mode has a known history of contrast regressions (LIT-234) — re-audit, do not assume. |

---

### §3.1.4 Performance Profiling

Anchor every number to the **SRS §3.4.1 table** — reproduce it as the requirement
baseline, then report measured-vs-target. This section is single-user timing;
§3.1.5 is concurrency.

| Row | Content to write |
|-----|------------------|
| **Objective** | Measure single-user response times and resource consumption for each timed operation under normal and worst-case single-user workload, and compare against the SRS §3.4.1 targets. |
| **Technique** | Instrument and time each operation: cached tensor retrieval (<10 ms), cached API response (<200 ms), cache-miss-to-enqueue (<50 ms), cold Whisper-base on 15 s audio (<3 s), multi-task ASR+SER+ADD (<8 s cold), IG/saliency attribution (<8 s), canvas mutation UI (<500 ms) and backend (<2 s), accent-bias profile (<30 s), faithfulness audit (<15 s), model download + hook registration (<60 s). Run each N times, report median and p95, not a single sample. Profile memory with `scripts/run_memory_profile.py` and `test_memory_profiling.py`; assert on **growth across iterations, not absolute RSS**. Profile frontend render and WS-update latency in DevTools. Track VRAM/RAM against the §3.4.3 budgets. |
| **Oracles** | Tolerance oracles against the §3.4.1 targets, with the measurement method stated (median of N, p95 of N, warm vs cold, GPU vs CPU). **State the confound honestly:** the SRS targets assume an NVIDIA T4; CI is CPU-only and dev machines vary, so an absolute pass/fail on a cold-inference target is not meaningful off the reference hardware. Report the hardware with every figure. Cache-hit targets *are* hardware-stable and so are the strongest oracles here. |
| **Required Tools** | `pytest` timing tests (`test_performance_load.py`, `test_memory_profiling.py`, `test_warmup_cache_contract.py`); `scripts/run_memory_profile.py`; `time.perf_counter`; `psutil`; `redis-cli --latency`; `nvidia-smi` for VRAM; Chrome DevTools Performance and Network panels; React Profiler; RQ queue-depth via `/health/workers`. |
| **Success Criteria** | Every row of SRS §3.4.1 has a measured figure with stated hardware and method; the hardware-stable targets (cache hit, enqueue, API response) are met; deviations on model-bound targets are explained rather than hidden. |
| **Special Considerations** | Measure on a quiet machine — background load invalidates results. Separate cold from warm explicitly; a cache hit makes almost any target trivially passable, so never report a warm number against a cold target. First call after worker start includes model load; exclude or report separately. Memory assertions must tolerate GC non-determinism (`gc.collect()` before measuring). |

---

### §3.1.5 Load Testing

Evidence lives on the `testing` branch (decision D4). If D4 is resolved the
other way, this section must say plainly that load testing was not executed.

| Row | Content to write |
|-----|------------------|
| **Objective** | Subject the API and worker fabric to increasing concurrent workload — normal, peak, and beyond expected maximum — to find the saturation point and confirm graceful behaviour rather than collapse. Accountable to **SRS §3.4.1**, **§3.4.3**, **§3.3.1**. |
| **Technique** | Locust (`Backend/loadtests/locustfile.py`) driving concurrent virtual users against the upload → enqueue → poll → result path. Ramp in stages (e.g. 1 → 10 → 50 → 100 users), holding each stage long enough for a stable reading. Exercise both instantaneous spikes and sustained peaks. Mix cache-hit-heavy and cache-miss-heavy profiles separately — they stress completely different subsystems (Redis read path vs GPU worker pool). Observe RQ queue depth per family, worker saturation under the concurrency-1 GPU pin, Redis memory against the 2 GB cap and LRU eviction behaviour, and WebSocket fan-out under many concurrent subscribers. |
| **Oracles** | Throughput and latency percentiles under each stage; error rate as a function of load. The decisive oracle is **behavioural, not numerical**: past saturation the system must *queue and degrade*, not corrupt state, drop jobs silently, or lose progress messages. A job enqueued must always be observable — either completing or failing with a typed error. Note that the GPU families are deliberately pinned to concurrency 1, so queueing under load is expected and correct, not a defect. |
| **Required Tools** | Locust; Redis `INFO memory` / `MONITOR`; `/health/workers` for queue depth; `rq info`; `psutil` / `nvidia-smi`; Playwright `dataflow` project for a concurrent full-stack path. |
| **Success Criteria** | Saturation point identified and reported; no data loss, no silent job drop, no cache corruption at or beyond it; errors under overload are typed and retryable-flagged per SRS §3.3.2; recovery to normal latency after load is removed. |
| **Special Considerations** | Load testing must run on a dedicated machine at a dedicated time. Real model inference is expensive, so a realistic 100-user cache-miss load may exceed the academic hardware budget — if so, **say what was actually run and what was extrapolated.** Do not report an extrapolation as a measurement. Redis LRU eviction under load will evict entries mid-test; that is correct behaviour and must not be read as a cache bug. |

---

### §3.1.6 Security and Access Control Testing

Map directly to **SR1–SR7** (SRS §3.4.2) and the inherited items in §4.5. This
section has unusually concrete requirements — use them.

| Row | Content to write |
|-----|------------------|
| **Objective** | Verify upload validation, model-deserialisation safety, session isolation, data-minimisation in keys and logs, and remediation of the inherited exposure points. |
| **Technique** | **SR1:** submit oversized (>100 MB), over-long (>15 min), wrong-MIME, magic-number-mismatched, zero-byte, and structurally malformed audio; confirm rejection *before* hashing. Include a polyglot file (valid WAV header, hostile payload). **SR2:** attempt to ingest a `.bin`/pickle checkpoint and confirm refusal before any deserialisation — this is arbitrary-code-execution prevention, the highest-severity check in the report. **SR4:** confirm uploads are purged on TTL. **SR5:** inspect cache keys for filenames, session ids, or user identifiers; inspect logs for audio, transcripts, or PII. **SR6:** confirm the inherited unauthenticated debug endpoint and wildcard CORS on file-serving routes are hardened, and that the cross-session dataset path is no longer authorised by a guessable session id alone — attempt cross-session access with a forged `sid` cookie. **SR7:** dependency vulnerability scan in CI. Path-traversal attempts against every file-path-accepting route (`file_path` is accepted by several). Files: `test_security.py`, `test_session_cookie.py`. |
| **Oracles** | Deterministic and self-verifying for most: rejection is an explicit typed error plus a status code, and cross-session access must return 403/404, never data. Negative-result caution belongs here — **a passing security test proves the tested attack failed, not that the system is secure.** Scanner output is an oracle only for *known* CVEs. State the residual risk rather than implying completeness. |
| **Required Tools** | `pytest` (`test_security.py`, `test_session_cookie.py`); `httpx` for forged requests; `pip-audit` / `npm audit` in CI (SR7); `curl` for raw header and CORS probing; crafted malformed-audio and polyglot fixtures; `safetensors` for format verification. |
| **Success Criteria** | Every SR1–SR7 clause has ≥1 executed test; every SRS §4.5 inherited exposure is either demonstrably remediated or explicitly recorded as outstanding with a Linear id; no high-severity dependency CVE unacknowledged at submission. |
| **Special Considerations** | AudioLIT has **no user authentication or role model** — it is session-cookie scoped, single-tenant, academic. The template's "test each user type's permissions" therefore does not apply; **say this explicitly** rather than leaving the row looking unaddressed. TLS (SR3) is a deployment concern, not testable on localhost — state that. Do not run intrusive scanning against any host you do not own. |

---

### §3.1.7 Failover and Recovery Testing

The template's DASD/power-cable framing is 1990s mainframe. Reframe to this
architecture and **say you are reframing**. Anchor to **SRS §3.3.1–§3.3.2**.

| Row | Content to write |
|-----|------------------|
| **Objective** | Simulate infrastructure and resource failures and verify graceful degradation, correct retry classification, and recovery to a known-good state without data loss or silent wrong answers. |
| **Technique** | Scenario matrix, each with a defined expected behaviour: (a) **Redis unreachable** — the project's standard pre-push check, `REDIS_URL="redis://127.0.0.1:1/0" pytest -q`; (b) **Redis killed mid-job**; (c) **worker process killed mid-inference** — job must be observable as failed or retried, never silently lost; (d) **GPU OOM / unavailable** — must fall back to CPU with a user-visible warning (SRS §3.3.1), not fail; (e) **one task family fails** — the other two must still return (the partial-failure-cascade fault model); (f) **Hugging Face Hub unreachable** during model download; (g) **corrupt cached value** — treated as a miss and recomputed (FR4.3); (h) **WebSocket dropped** — `useTaskStatus` must fall back to polling and reconnect (FR3.2); (i) **backend down with a warm cache** — cached results must still serve (SRS §3.3.1); (j) **transient fault** — exponential backoff, bounded attempts, then a durable failure record. |
| **Oracles** | For each scenario, a stated expected observable outcome, checked three ways: the API response (typed error, correct retryable flag), the UI state (a retry control appears **only** when retrying is meaningful — SRS §3.3.2), and the durable failure record. The strongest oracle is the **retryable/non-retryable classification** being correct, because it is what makes recovery automatic. Some scenarios (process kill, network partition) are inherently manual — mark them so and record the procedure so the result is reproducible. |
| **Required Tools** | `docker compose stop/start redis`; `REDIS_URL` pointed at an unreachable port; `kill -9` on worker PIDs; `rq info` for orphaned jobs; network-disable for Hub tests; `fakeredis` for deterministic orchestrator-failure unit tests; `test_task_orchestrator.py`, `test_fanout_orchestrator.py`, `test_queue.py`, `test_system_integration.py`. |
| **Success Criteria** | Every scenario in the matrix executed with a recorded outcome; no scenario produces a silently wrong result (the unacceptable failure mode); no in-flight job becomes unobservable; CPU fallback and partial-failure isolation both demonstrated. |
| **Special Considerations** | Reframed from the template's DASD/power-interruption model — **state the reframing and why**, so it reads as deliberate. There is no redundant infrastructure and no SLA (SRS §3.3): recovery is by resubmission, which is inexpensive by design, and that is an architectural decision, not an untested gap. Known hazard: `SimpleWorker(burst=True)` draining a dependency-gated aggregator on `fakeredis` hangs `pytest` intermittently — stress-run these tests (~15×) with a kill-timeout; `perl alarm` does not work because Python resets SIGALRM. |

---

### §3.1.8 Configuration Testing

| Row | Content to write |
|-----|------------------|
| **Objective** | Verify correct operation across the supported browser, OS, Python, Node, and hardware-accelerator configurations, and identify configuration-dependent behaviour. |
| **Technique** | Browsers: Playwright across Chromium, Firefox, WebKit (`e2e/layout.spec.ts`, the LIT-160 suite). Python: 3.10 (CI) and 3.11 (local dev) — **this divergence is itself a configuration risk and must be tested, not assumed.** Node 20. OS: macOS (dev), Ubuntu (CI); Windows if any team member develops there. Accelerator: **CPU-only (CI runners, CPU-only torch wheel) vs GPU** — the highest-value axis, since the CPU-fallback path (FR1.4) only executes in one of them. Redis: containerised vs native vs absent. Frontend: `VITE_API_BASE_URL` set vs default; dev server (`:8080`) vs built `dist` preview. Viewports: desktop through ~400 px mobile. |
| **Oracles** | Cross-configuration **differential** oracle: the same functional suite must produce the same pass/fail across configurations, and any divergence is itself the finding. CI is the continuous instance of this — every PR runs the Ubuntu/Python 3.10/Node 20/CPU-only configuration. Numeric outputs may legitimately differ slightly between CPU and GPU float paths; assert on tolerance, and **state the tolerance chosen and why**. |
| **Required Tools** | GitHub Actions (`.github/workflows/ci.yml`); Playwright's three engine projects; `docker compose`; local Python 3.10 and 3.11 environments; `nvidia-smi`; browser DevTools device emulation. |
| **Success Criteria** | The functional suite passes on every declared supported configuration; the CPU-fallback path is exercised in at least one; no browser-specific layout or Web Audio failure; any configuration-dependent numeric divergence is quantified and bounded. |
| **Special Considerations** | CI deliberately installs the **CPU-only torch wheel** (the default Linux wheel is the multi-GB CUDA build), so CI never exercises the GPU path — GPU coverage is manual and local, and that must be stated. CI currently has **no Redis service container**; before one is added, confirm the direct-name redis import issue (LIT-229) stays fixed, or unrelated tests will fail with `RuntimeError: Event loop is closed`. Playwright's layout suite deliberately runs backend-free so it stays fast; the full-stack `dataflow` project needs the whole stack live. |

---

### §4 Deliverables

Per decision D1, this is where executed evidence lands. Open with a bulleted
list of the artefacts, then the two subsections.

**Artefacts to list:**
- `pytest` run logs and summary (backend, ~567 test functions / 49 files)
- Jest run logs (frontend components)
- Playwright cross-browser report (`layout.spec.ts`) and dataflow report
- Locust load-test report (HTML + CSV)
- Performance profiling results vs SRS §3.4.1
- Memory profiling output (`run_memory_profile.py`)
- Security test results and dependency-scan output
- Failover scenario matrix with recorded outcomes
- FR→test traceability matrix
- GitHub Actions CI history as the continuous regression record
- Defect log (Linear LIT-ids) with severity and resolution status

#### §4.1 Test Evaluation Summaries

Form, content, and frequency of each summary — and, per the sample's example,
**actual captured output**. Include:
- A headline results table: suite, tests run, passed, failed, skipped, duration,
  date, commit SHA. Pin every figure to a **commit SHA** — a test result without
  one is unreproducible.
- Real terminal output from `pytest`, `npm test`, and `npx playwright test`
  (the sample pasted PHPUnit output; do the equivalent).
- Locust report screenshot; DevTools performance trace; `nvidia-smi` during
  inference; the Playwright HTML report.
- Frequency: per-PR (CI, automatic), per-milestone (full manual + GPU + load),
  and once at Phase 4 submission.

#### §4.2 Reporting on Test Coverage

- **The FR→test traceability matrix is the centrepiece.** Columns:
  `FR/SR id | requirement summary | technique (§3.1.x) | test file(s) | status`.
  Cover FR1–FR4, FR6–FR12, FR15–FR17 and SR1–SR7. **There is no FR5, FR13, or
  FR14** — a matrix inventing them is a visible error.
- Line/branch coverage figures if `pytest-cov` is added (**not currently in
  `requirements.txt`** — adding it is a small, high-value change before
  submission; decide and record).
- The generic per-run report fields the sample lists (date, test case, executed,
  pass, fail, pass %, comments).
- State the reporting cadence and the tooling that produces each figure.
- **Report coverage gaps as findings, not omissions** — an examiner trusts a
  report that names what it did not cover.

---

### §5 Risks, Dependencies, Assumptions, and Constraints

Three-column table (Risk | Mitigation Strategy | Contingency), as in the
template and sample. Add a **Likelihood / Impact** ranking, which the template
requests and the sample skipped. Seed rows — expand with anything current:

| Risk | Mitigation | Contingency |
|------|-----------|-------------|
| MongoDB tier (SRS §3.10) unimplemented, so §3.1.1 cannot cover it | Flag in Linear and in the report; scope §3.1.1 to Redis + corpora | Record as a known scope deviation with its Linear id |
| No GPU in CI — FR1.4 CPU-fallback and GPU perf targets untested continuously | Manual GPU runs at milestones on declared hardware | Report GPU figures as point-in-time with hardware stated |
| Load testing may exceed academic hardware budget at realistic concurrency | Stage the ramp; separate cache-hit from cache-miss profiles | Report what was measured vs extrapolated; never conflate |
| No ground truth for saliency correctness | Metamorphic + deletion-score faithfulness oracles | Report faithfulness metrics, not accuracy claims |
| Flaky `SimpleWorker(burst=True)` hangs on fakeredis | Stress-run ~15× with kill-timeout; call aggregators directly | Quarantine and file the flake rather than retry-masking it |
| Two individually-green PRs breaking only in combination | Merge latest `develop` and re-run the full suite before pushing | Revert-and-fix forward; the PR #10/#13 incident is the precedent |
| Corpora are large, licence-restricted, and slow to provision | Streaming/sub-sampling loaders; ~100 GB footprint bound | Sub-sample and declare the reduced corpus in the report |
| Model downloads make tests network-dependent | Pin revisions; mock the Hub where the model is not under test | Mark `slow`; exclude from the per-PR gate |
| Three concurrent developers landing work the same day | Re-check Linear + `gh pr list` before starting; merge `develop` before pushing | Re-run the full suite after any merge you did not make |

Also state **dependencies** (Redis 7, HF Hub, corpus licences, GPU access),
**assumptions** (single-tenant academic deployment, no authentication tier, no
SLA), and **constraints** (C2 VRAM budget, C3 safetensors-only, ~100 GB dataset
footprint, CPU-only CI).

---

### §6 References

Follow the template's instruction: tool name, URL, **and an access date**. Use
IEEE style consistently. Cover at minimum:

- **Testing tools:** pytest, pytest-asyncio, fakeredis, httpx, Jest, Testing
  Library, Playwright, Locust, pip-audit/npm audit.
- **System dependencies (versioned):** FastAPI 0.111, RQ 2.10, Redis 7,
  PyTorch ≥2.6, Transformers ≥4.30, Captum ≥0.6, Librosa ≥0.10, soundfile,
  React 18.3, Vite 5.4.
- **Methods and standards:** a metamorphic-testing reference (the oracle-problem
  literature) — this substantiates §3's oracle argument and is the single
  highest-value citation in the report; a deletion-score / attribution-
  faithfulness paper for FR16; WCAG 2.1 AA; the RUP test-plan template itself.
- **Project documents:** AudioLIT SRS v1.0, SAD v1.0, ECHO 1.0 baseline repo.

---

## 4. House style

- **Every claim carries evidence.** A technique row naming no file, command, or
  measurement is filler. Examiners can tell.
- **Cite real paths.** `Backend/tests/test_security.py`, not "the security tests".
- **Verify before citing.** Per `CLAUDE.md`: if you are about to cite an SRS/SAD
  section number, an FR id, a file path, or a class name you did not read
  yourself while writing, `grep` it first. Fabricated section numbers have
  already bitten this project once (LIT-228) and a report is a worse place for it
  than a ticket.
- **Pin results to commit SHAs and state the hardware.**
- **Name what you did not test, and why.** Every strong section above contains at
  least one honest limitation. Keep them — they are the difference between a
  report and a brochure.
- **Never mention Celery.** Never mention `app/services/`. Never mention
  torchaudio. All three are removed project-wide.
- **Tense:** §1–§3 plan voice ("shall/will be exercised"); §4 report voice
  ("was executed, N passed").

---

## 5. Suggested work split

Three writers, eight technique sections. Split by ownership of the code under
test, not alphabetically, so each writer already knows the evidence:

| Writer | Sections |
|--------|----------|
| — | §1, §2, §3 preamble, §5, §6 (whoever assembles; do this last, after §3 is written) |
| — | §3.1.1 Data/Redis integrity, §3.1.2 Function testing, §4.2 traceability matrix |
| — | §3.1.3 UI, §3.1.8 Configuration, §4.1 evaluation summaries |
| — | §3.1.4 Performance, §3.1.5 Load, §3.1.6 Security, §3.1.7 Failover |

**Order of work:** settle §0 decisions → build the §4.2 traceability matrix
first (it exposes coverage gaps while there is still time to close them) →
write §3.1.x → write §1/§2 → write §4/§5/§6 last, once results exist.

---

## 6. Pre-submission checklist

- [ ] §0 decisions D1–D5 resolved and recorded
- [ ] Section numbering fixed (3.1 and 4.2, not the template's 1.1)
- [ ] All blue-italic template guidance text deleted
- [ ] Revision history and title page completed
- [ ] Table of contents regenerated with correct page numbers
- [ ] All eight technique tables present with all six rows, none empty
- [ ] Traceability matrix covers FR1–FR4, FR6–FR12, FR15–FR17, SR1–SR7 — and invents no FR5/FR13/FR14
- [ ] Every result pinned to a commit SHA; every performance figure states its hardware
- [ ] No mention of Celery, `app/services/`, or torchaudio
- [ ] Every cited SRS/SAD section, FR id, and file path re-verified against the source
- [ ] MongoDB gap flagged in Linear, not only in the report
- [ ] References carry access dates
