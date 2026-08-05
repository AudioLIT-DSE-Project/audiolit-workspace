# AudioLIT — Issue Plan & Dependency Map

**What this is:** a local, scannable index of every committed Phase 2–4 Linear
issue — build order, current status, and who owns what — so any of the 3
developers (or a cold Claude Code session) can look here first to decide what
to pick up next, without opening 50 Linear tickets. **This file is not the
source of truth for scope or acceptance criteria** — that's Linear (each
issue's Tier-C stamp, see LIT-228). This file only answers "what can I start,
and what's it waiting on."

Keep this file's **Status** and **Blocked by** columns updated as work lands —
it will drift from Linear otherwise. The dependency edges here are also
encoded in Linear itself as `blockedBy`/`blocks` relations, so Linear's own
dependency graph matches this document.

> **AGREED SEQUENCING PLAN (2026-08-05) — the order we are actually working
> in; read before claiming an issue so parallel sessions don't collide:**
> 1. **LIT-150 re-add/fix lands first.** PR #21 (the revert) was *merged*
>    2026-08-05 and DELETED the orchestrator + its test from `develop`. The CI
>    failure it cited was a post-migration import break, not a logic bug —
>    **PR #22** (`fix/lit-150-post-migration-imports-ci`) re-introduces the
>    orchestrator with the imports fixed and its test restored (backend 149
>    passed / 2 skipped, frontend green). Land #22 through review; don't
>    re-revert LIT-150.
> 2. **Then complete LIT-123 (Ravindu, dataset core) and LIT-127 (Rahim, RQ
>    broker, Urgent) FIRST — before any downstream critical path.** These two
>    are the shared base; each goes through review + merge before the parallel
>    build-out starts, so the two critical paths begin from solid ground.
> 3. **Then work the LIT-127 critical path step by step** (LIT-127 → LIT-149
>    → orchestrator wiring → LIT-131/157), in parallel with the LIT-123 →
>    LIT-142 → LIT-128 → LIT-148 dataset/ADD path.
>
> The whole infra tier (LIT-207/211/225/226/227) is Done; PRs #12/#16/#17 all
> merged; #21 merged (removed LIT-150 — being re-added fixed by #22). **Fresh
> sessions:** LIT-150-fix (#22), LIT-123, LIT-127 are claimed/in-flight — take
> independent unblocked work instead (LIT-206/224 SER, LIT-126/130 XAI,
> LIT-222) and coordinate. Always run `gh pr list` + re-check Linear before
> starting; this note drifts.
>
> _Status column last reconciled against Linear: 2026-08-05._

---

## How to read the tiers

Each tier can start once everything in the tiers above it (that it's actually
blocked by — check the **Blocked by** column, not just tier position) is
merged. Issues within the same tier are parallelizable across the 3
developers. Assignees below reflect Linear at the time this doc was written —
Linear is authoritative for current assignment.

Status legend: 🟢 Todo (not started) · 🟡 In Progress · 🔵 In Review · ✅ Done

---

## Phase 2 — MVP (target: end of Phase 2)

### Tier 0 — Foundational infra (start immediately, no blockers within this project)

| ID | Title | FR | Assignee | Status | Blocked by | Notes |
|---|---|---|---|---|---|---|
| LIT-225 | Prototype RQ fan-out/fan-in | infra | Tharusha | ✅ Done | none | **Merged (PR #10).** Pattern documented in `docs/rq_fanout_pattern.md`. Found a pre-existing, out-of-scope bug while testing (health.py bypasses the fake_redis fixture) - flagged, not fixed here. |
| LIT-207 | Dynamic HF model ingestion (Model Registry) | FR1 | Rahim | ✅ Done | none | Also satisfies LIT-227's "model-loading reorganised" migration step — same work, don't duplicate. **Correction:** was marked Done via draft PR #14, but its DoD's "registers hooks... selectable layers" line wasn't actually met until PR #16 wired `HookManager` (from LIT-211/PR #12) into `ModelRegistry` (`LoadedModel.available_layers` / `.attach_hooks()`). Flagged in a Linear comment rather than silently reopened — genuinely Done now that #16 has merged. |
| LIT-210 | Model ID resolver / safetensors / cache | FR1 | Rahim | 🟢 | none | Sub-task of LIT-207; can run in parallel with it. |
| LIT-211 | Forward/attention hook registration | FR1 | Tharusha | ✅ Done | LIT-207 (loose — same PR is fine) | **Merged (PR #12).** Unlocks all attribution work (FR8/FR9/FR17) — LIT-126/130/222 now unblocked. |
| LIT-226 | Remove torchaudio, standardise on soundfile | infra | — | ✅ Done | none | **Merged (PR #9).** Unblocks LIT-165 (mutation engine), same `perturbation_service.py` file. |
| LIT-227 | Layered migration (restructure ECHO code) | infra | Tharusha | ✅ Done | none | **Merged (PR #16 completed slice 2).** Incremental per SAD §8.2 — infra → registry → explanation code → orchestration → new features, system kept working at each step. Coordinate with LIT-207 (same registry work). **PR #13 (slice 1, merged):** infra/domain/orchestration skeleton stood up, `settings`/`redis`/`session` moved into `infrastructure`, `upload.py`↔`inferences.py` route coupling removed, 5 duplicated `get_session_id` defs collapsed, `pertubation_service.py` renamed, unused LRP import removed. **Correction:** this was then marked Done in Linear even though the DoD wasn't fully met (found via repo audit) — `domain`/`orchestration` were still empty placeholders (everything stayed flat in `app/services/`), and the "unreachable model option" dead-code item was still present. **PR #16 (slice 2):** actually moves every service into its real domain/infrastructure/orchestration home per SAD §5.1/§5.2, fixes `Toolbar.tsx`'s unreachable `whisper-large` option; investigated "orphaned visualisation components" and found none (all 5 files reachable). **Still open, deliberately out of scope:** `queue_service.py` → real RQ / no synchronous inference on `/upload`'s request path — changes the HTTP contract, needs LIT-157 (frontend polling, not started); that's LIT-150's job (merged as a draft via PR #17/#20 without touching the routes — but the real-RQ swap itself is still open, needs LIT-157). |
| LIT-123 | Multi-task dataset ingestion core | FR2 | Ravindu | 🟢 **NEXT (claimed)** | none | Parent of 141/142/208/181 below. **Per the sequencing plan, one of the two "base first" issues — starts right after the LIT-150 fix PR merges; don't duplicate.** |
| LIT-125 | Librosa DSP extraction pipeline (STFT/pYIN/RMS) | FR10 | Ravindu | 🟢 | none | Fully independent — no model or async-fabric dependency. |
| LIT-145 | pYIN F0 tracking | FR10 | Ravindu | 🟢 | none | Sub-task of LIT-125. |
| LIT-146 | RMS energy estimation | FR10 | Ravindu | 🟢 | none | Sub-task of LIT-125. |

### Tier 1 — Dataset loaders (parallel with Tier 0, children of LIT-123)

| ID | Title | FR | Assignee | Status | Blocked by | Notes |
|---|---|---|---|---|---|---|
| LIT-141 | Common Voice / LibriSpeech ingestion | FR2 | Ravindu | 🟢 | LIT-123 (parent) | |
| LIT-142 | ASVspoof 2021 DF loader | FR2 | Ravindu | 🟢 | LIT-123 (parent) | **Blocks LIT-128** (FR7 needs this dataset). |
| LIT-208 | CREMA-D/RAVDESS demo subset | FR2 | Tharusha | 🟢 | LIT-123 (parent) | |
| LIT-181 | L2-ARCTIC ingestion | FR2 | Ravindu | 🟢 | LIT-123 (parent) | **Blocks LIT-168/182** (FR15 needs this). |

### Tier 2 — Async fabric build-out (blocked by the Tier-0 prototype)

| ID | Title | FR | Assignee | Status | Blocked by | Notes |
|---|---|---|---|---|---|---|
| LIT-127 | Deploy RQ broker | FR3 | Rahim | 🟢 **NEXT (claimed)** | **LIT-225** ✅ | Blocker LIT-225 is merged → **unblocked.** Per the sequencing plan, the second "base first" issue (starts right after the LIT-150 fix PR merges), then its critical path is worked step by step (→ LIT-149 → orchestrator wiring → LIT-131/157). Don't duplicate. |
| LIT-149 | RQ worker scaffolding | FR3 | Rahim | 🟢 | LIT-127 (parent) | |
| LIT-150 | ASR+SER+ADD orchestrator | FR3 | Rahim | 🔵 Re-add in review (PR #22) | LIT-127, LIT-225 | **Churny history:** draft merged (PR #17) → reverted (#19) → re-applied (#20) → **#21 merged 2026-08-05, DELETING it (service + test) from `develop`.** The CI failure #21 cited was a post-migration import break, not a logic bug. **PR #22 re-adds the orchestrator with imports fixed** (`..core`→`..infrastructure`, `.fanout`→`..orchestration`, `.model_loader`→`..domain`) + restored test; backend 149 passed / 2 skipped, frontend green. ⚠ Still a draft that landed ahead of its LIT-127 blocker; the synchronous `/upload` → real-RQ swap it needs is still open (LIT-157). **Don't build on `app/services/multitask_orchestrator_service.py` until #22 merges.** |

### Tier 3 — Model integration (blocked by the registry + relevant datasets)

| ID | Title | FR | Assignee | Status | Blocked by | Notes |
|---|---|---|---|---|---|---|
| LIT-206 | Integrate SER model | FR6 | Tharusha | 🟢 | **LIT-207** | |
| LIT-224 | Verify/select working SER checkpoint | FR6 | — | 🟢 | **LIT-207** | De-risks LIT-206 — the currently-referenced checkpoint may have an untrained classifier head; do this alongside/before LIT-206 lands. |
| LIT-128 | Integrate ADD classifier | FR7 | Tharusha | 🟢 | **LIT-207, LIT-142** | |
| LIT-151 | Binary deepfake detection head | FR7 | Tharusha | 🟢 | LIT-128 (parent) | |
| LIT-152 | Forensic feature map API routing | FR3/FR7 | Tharusha | 🟢 | LIT-128 (parent), **LIT-127** | Needs the async fabric to serialize fan-in results. |

### Tier 4 — Attribution / XAI (blocked by hook registration)

| ID | Title | FR | Assignee | Status | Blocked by | Notes |
|---|---|---|---|---|---|---|
| LIT-126 | Captum IG saliency core | FR9 | Rahim | 🟢 | **LIT-211** | |
| LIT-147 | IG temporal attribution | FR9 | Rahim | 🟢 | LIT-126 (parent) | |
| LIT-209 | Extend Captum to SER outputs | FR9 | Rahim | 🟢 | LIT-126 (parent), LIT-206 | Needs SER integrated first. |
| LIT-130 | LIME/SHAP spectrogram translators | FR8 | Rahim | 🟢 | **LIT-211** | |
| LIT-148 | Grad-CAM registration hooks | FR8 | Rahim | 🟢 | LIT-130 (parent), **LIT-128** | Grad-CAM targets the ADD classifier's final layer (SRS FR8.2) — needs LIT-128. |
| LIT-155 | Spectrogram patch/perturbation engine | FR8 | Rahim | 🟢 | LIT-130 (parent) | |
| LIT-156 | SHAP coordinate transformer (frontend) | FR8 | Rahim | 🟢 | LIT-130, LIT-155 | |
| LIT-222 | Faithful attention extraction (FR17 fix) | FR17 | — | 🟢 | **LIT-211** | Correctness fix — touches the same attention path as LIT-211. |

### Tier 5 — Mutation suite (blocked by torchaudio removal)

| ID | Title | FR | Assignee | Status | Blocked by | Notes |
|---|---|---|---|---|---|---|
| LIT-165 | Backend mutation engine | FR12 | Rahim | 🟢 | **LIT-226** | Same file (`pertubation_service.py`) as LIT-226 — sequence to avoid conflicting edits. |
| LIT-175 | Time-frequency slice masking | FR12 | Rahim | 🟢 | LIT-165 (parent) | |
| LIT-164 | Canvas selection controls (frontend+backend) | FR12 | Ravindu | 🟢 | **LIT-165** | |
| LIT-176 | Canvas mouse drag/bbox tracker | FR12 | Ravindu | 🟢 | LIT-164 (parent) | |
| LIT-177 | 2D spectrogram grid selector | FR12 | Ravindu | 🟢 | LIT-164 (parent) | |
| LIT-178 | Mutation trigger/state dispatcher | FR12 | Ravindu | 🟢 | LIT-164 (parent), LIT-165 | Needs the backend mutation endpoint contract. |

### Tier 6 — Frontend integration (blocked by the backend features it binds to)

| ID | Title | FR | Assignee | Status | Blocked by (from Linear) | Notes |
|---|---|---|---|---|---|---|
| LIT-131 | Connect UI to async API/XAI endpoints | FR3 | Tharusha | 🟢 | LIT-126, LIT-130, LIT-121, LIT-127 *(relation already set in Linear)* | |
| LIT-157 | WebSocket/polling handlers | FR3 | Rahim | 🟢 | LIT-131 (parent) | |
| LIT-158 | Frontend XAI overlay binding | FR8/FR9 | Ravindu | 🟢 | LIT-131 (parent) | |
| LIT-159 | Reactive multi-model analytics widgets | FR3 | Tharusha | 🟢 | LIT-131 (parent) | |

### Security & migration wrap-up (parallel, no strict feature blockers)

| ID | Title | Assignee | Status | Blocked by | Notes |
|---|---|---|---|---|---|
| LIT-223 | Remediate inherited security gaps | — | 🟢 | **LIT-227** (loose) | Touches `dataset_service.py`, which LIT-227 also restructures — sequence to avoid rework. |

---

## Phase 3 — Refinement, Bias/Faithfulness, Testing & Evaluation

### Tier 7 — Latent projection & accent bias (blocked by registry + L2-ARCTIC)

| ID | Title | FR | Assignee | Status | Blocked by | Notes |
|---|---|---|---|---|---|---|
| LIT-185 | Projection-space lasso handler | FR11 | Tharusha | 🟢 | **LIT-207** | Needs embeddings from a registry-loaded model. |
| LIT-167 | Lasso selection UI (committed part only) | FR11 | Tharusha | 🟢 | LIT-207 | Multi-model comparison part is stretch — do not build. |
| LIT-168 | Accent bias profiling scripts | FR15 | Ravindu | 🟢 | **LIT-181**, LIT-207 | Needs L2-ARCTIC + a working ASR path for WER. |
| LIT-182 | Group-wise WER diagnostic runner | FR15 | Ravindu | 🟢 | LIT-168 (parent) | |

### Tier 8 — Faithfulness auditing (blocked by at least one attribution method)

| ID | Title | FR | Assignee | Status | Blocked by | Notes |
|---|---|---|---|---|---|---|
| LIT-169 | Quantitative faithfulness checking | FR16 | Tharusha | 🟢 | **LIT-126** (needs ≥1 attribution method) | |
| LIT-183 | High-saliency masking engine | FR16 | Tharusha | 🟢 | LIT-169 (parent) | |
| LIT-184 | Downstream degradation scoring (reassigned from FR15 — see LIT-228 comment) | FR16 | Tharusha | 🟢 | LIT-169 (parent) | |
| LIT-212 | Deletion/insertion AUC metric (deletion part only) | FR16 | Tharusha | 🟢 | LIT-169 (parent) | Insertion/AUC part is stretch — do not build. |

### Tier 9 — Mid-evaluation & integration testing

| ID | Title | Assignee | Status | Blocked by (from Linear, corrected) | Notes |
|---|---|---|---|---|---|
| LIT-132 | Mid-eval integration testing | Tharusha | 🟢 Urgent | LIT-206, LIT-130, LIT-128, LIT-131, LIT-123, **LIT-207** (was LIT-124 — corrected, see below), LIT-126, LIT-127, LIT-122, LIT-148 *(LIT-154 removed as blocker — see below)* | |
| LIT-160 | Cross-browser E2E QA | Ravindu | 🟢 | LIT-132 (parent) | |
| LIT-161 | API stress/memory profiling | Rahim | 🟢 | LIT-132 (parent) | |
| LIT-162 | Code review & mock walkthrough prep | Tharusha | 🟢 Urgent | LIT-132 (parent) | |
| LIT-170 | Full software testing + DS evaluation | Ravindu | 🟢 | LIT-132, LIT-130, LIT-126 | |
| LIT-187 | Backend pytest + UI boundary testing | Ravindu | 🟢 | LIT-170 (parent) | |
| LIT-188 | WER/deletion-score metric computation | Tharusha | 🟢 | LIT-170 (parent) | |
| LIT-171 | Testing & evaluation document | Rahim | 🟢 | LIT-170 | |
| LIT-189 | Latency/FPS metric synthesis | Rahim | 🟢 | LIT-171 (parent) | |
| LIT-190 | Error analysis (accent/faithfulness) | Tharusha | 🟢 | LIT-171 (parent) | |
| LIT-172 | Mentor Meetup 3 prep & sign-off | Tharusha | 🟡 In Progress, Urgent | LIT-170 | |
| LIT-191 | Live evaluation walkthrough script | Tharusha | 🟢 | LIT-172 (parent) | |
| LIT-192 | Mentor Meetup 3 execution | Tharusha | 🟢 | LIT-172 (parent) | |

**Corrections applied to Linear (this pass):**
- **LIT-132** was blocked by **LIT-154**, a non-committed stretch sub-task (LIT-129 → LIT-153 "do not start until committed work done" → LIT-154, which lacked its own stretch banner). An optional diffusion-artifact task should never gate an urgent committed mid-eval deliverable — removed as a blocker; flagged LIT-154 for a stretch banner instead.
- **LIT-132** was blocked by **LIT-124**, one of the FR1 duplicate issues not being built (superseded by LIT-207/210/211) — replaced with LIT-207.

---

## Phase 4 — Final Submission (lightweight — mostly process/deliverable, not FR-mapped)

Not Tier-C stamped (per LIT-228 rule 2 — process/deliverable issues carry no FR). Listed here only for overall sequencing awareness; treat Linear as authoritative for these.

| ID | Title | Blocked by |
|---|---|---|
| LIT-194 | Final individual examiner demonstration | LIT-172, LIT-193 |
| LIT-200 | Sandbox staging & mock defense runs | LIT-194 (parent) |
| LIT-201 | Individual viva defense | LIT-194 (parent) |
| LIT-195 | GitHub landing page | — |
| LIT-202 | Landing page deployment | LIT-195 (parent) |
| LIT-196 | Package source, finalize report | LIT-194 |
| LIT-203 | Production containerization | LIT-196 (parent) |
| LIT-204 | Final IEEE-style report | LIT-196 (parent) |
| LIT-205 | Academic portal submission | LIT-196 (parent) |

---

## Non-committed / stretch — do not build without explicit sign-off

Per SRS §4.4 and LIT-228 rule 3. Listed so nobody accidentally picks these up thinking they're in the plan above.

| ID | Title | Gate |
|---|---|---|
| LIT-166 | Demographic corpus ingestion + SER fine-tuning | Stretch — sub-task LIT-179 also stretch |
| LIT-180 | SER downstream probability matrix optimization | Stretch — sub-task of LIT-166 |
| LIT-186 | Multi-model side-by-side comparison grid (dropped FR5) | Stretch |
| LIT-129 | ADDSegDiff diffusion artifact localization | Stretch — fallback is LIT-130 |
| LIT-153 | ADDSegDiff execution | Stretch — explicit "do not start until committed work + Phase-3 essentials done" |
| LIT-154 | Boundary mask exporter | Stretch by inheritance from LIT-129/153 (needs its own banner — not yet added) |
| LIT-213 | Multi-class generator fingerprinting | Stretch — committed counterpart is LIT-151 |
| LIT-124 / 143 / 144 | Early FR1 duplicates | Superseded by LIT-207/210/211 — recommend closing as duplicate |

---

## Critical path (Phase 2 core)

```mermaid
graph TD
    LIT225[LIT-225 RQ fan-in prototype] --> LIT127[LIT-127 RQ broker]
    LIT127 --> LIT149[LIT-149 worker scaffolding]
    LIT127 --> LIT150[LIT-150 orchestrator]
    LIT225 --> LIT150
    LIT207[LIT-207 Model Registry] --> LIT210[LIT-210 resolver/safetensors]
    LIT207 --> LIT211[LIT-211 hook registration]
    LIT207 --> LIT206[LIT-206 SER integration]
    LIT207 --> LIT128[LIT-128 ADD integration]
    LIT142[LIT-142 ASVspoof dataset] --> LIT128
    LIT211 --> LIT126[LIT-126 Integrated Gradients]
    LIT211 --> LIT130[LIT-130 Spectrogram attribution]
    LIT128 --> LIT148[LIT-148 Grad-CAM]
    LIT130 --> LIT148
    LIT211 --> LIT222[LIT-222 Faithful attention]
    LIT226[LIT-226 torchaudio removal] --> LIT165[LIT-165 Mutation engine]
    LIT165 --> LIT164[LIT-164 Canvas mutation UI]
    LIT127 --> LIT152[LIT-152 results serialization]
    LIT150 --> LIT131[LIT-131 Frontend API wiring]
    LIT126 --> LIT131
    LIT130 --> LIT131
    LIT131 --> LIT132[LIT-132 Mid-eval integration testing]
    LIT206 --> LIT132
    LIT128 --> LIT132
```

---

*Update this file as issues move through Todo → In Progress → Done. If a dependency here turns out to be wrong (or Linear's own relations drift from this doc), fix both — this file and the Linear relations should always agree.*
