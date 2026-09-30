# AudioLIT — Project Documents & Conventions

This directory holds the **authoritative planning documents** for AudioLIT, an
interpretability workbench for Automatic Speech Recognition (ASR), Speech Emotion
Recognition (SER), and Audio Deepfake Detection (ADD), extending the ECHO 1.0
baseline.

Anyone (human or agent) working in this repository should read this file first,
then the SAD and SRS, before making architectural or scope decisions.

---

## Documents in this directory

| File | What it is | Status |
|------|------------|--------|
| `SAD.md` | Software Architecture Document (v1.0) | **Final — authoritative** |
| `SRS.md` | Software Requirements Specification (v1.0) | **Final — authoritative** |
| `MONGODB_METADATA_TIER.md` | MongoDB metadata tier — operations, configuration, degradation (SRS §3.10 / SAD §9) | Living |
| `README.md` | This file — conventions and errata | Living |
| `handbook/` | **The AudioLIT Handbook** — a 17-chapter, from-first-principles account of the theory, the architecture and every module at code level, written so the system can be rebuilt from scratch with no prior ML background. Start at `handbook/README.md`. | Living — explanatory, **not** authoritative over SAD/SRS |

> Export both from Google Docs as Markdown and commit them here as `SAD.md` and
> `SRS.md`. If a figure is essential (e.g. the SAD migration or layered-view
> diagrams), export it as PNG into `docs/assets/` and reference it, since Docs'
> Markdown export does not embed images reliably.

---

## Authoritative source order

When two sources disagree, the one higher in this list wins:

1. **SAD** (`docs/SAD.md`) — architecture of record
2. **SRS** (`docs/SRS.md`) — committed requirements (FRs, NFRs, scope)
3. **Linear issue LIT-228** — the Claude Code convention / bootstrapping doc
   (repo layout, FR→issue map, SAD component map, per-FR acceptance criteria)
4. **Other Linear issues** — implementation work orders

If an existing Linear issue body or existing code conflicts with the SAD/SRS,
the SAD/SRS wins. Flag the conflict; do not silently conform to the stale source.

> Note on history: the SRS was finalized **before** the SAD. Several early Linear
> issues (including some marked Done) reflect pre-SAD assumptions and were later
> corrected. The errata below capture the cases where the finalized SAD/SRS
> superseded earlier decisions.

---

## Errata — decisions that supersede stale text in the source documents

These are known points where the documents (or early issues) contain outdated
statements. The **decision** column is authoritative.

| # | Topic | Stale text / assumption | Decision (authoritative) |
|---|-------|-------------------------|--------------------------|
| E1 | **Repository topology** | SAD §8.2 / Figure 11 describe **two repos** (Repository 1 - Frontend & Backend, Repository 2 - Machine Learning). | **Single monorepo: `audiolit-workspace`.** ECHO 1.0 is cloned and extended in place; frontend and backend live in one tree. The two-repo diagram is superseded. (Corrected citation: the real SAD has no §8.3 or Figure 21 - it tops out at §12/Figure 14. This erratum's own citation was never checked against the source and is exactly the kind of fabrication LIT-228 was corrected for; verified against `docs/SAD.md` directly on 2026-07-30.) |
| E2 | **Async task fabric** | SRS §3.6.1 / §3.9.3 / §3.10 mention **"Celery/RQ"** or a Celery broker. | **RQ + Redis only.** Celery is removed project-wide (lighter setup, simpler single-host operation). Never reintroduce Celery. The SAD already commits to RQ; the SRS text is the stale side. |
| E3 | **Audio I/O library** | ECHO baseline and some early issues use **torchaudio**. | **soundfile** is the standard for all audio load/save (SAD §3.3, tracked by LIT-226). torchaudio is on a discontinuation path and is removed. Never reintroduce it. |
| E4 | **FR numbering** | The **submitted SRS PDF** numbers the functional requirements FR1-FR14 with no gaps. `docs/SRS.md`, the reconciled SRS, uses FR1-FR4, FR6-FR12, FR15-FR17 and has **no FR5, FR13 or FR14**. Anything citing an FR id has to say which numbering it means. | **`docs/SRS.md`'s numbering is authoritative.** Every capability in the submitted SRS survives; only seven identifiers shifted. Mapping (verified against both documents on 2026-09-29, not carried over from an earlier note): <br>`FR1`-`FR4` unchanged. <br>submitted `FR5` Speech Emotion Recognition -> **`FR6`** <br>submitted `FR6` Audio Deepfake Detection -> **`FR7`** <br>submitted `FR7` Spectrogram-Adapted Attribution and Grad-CAM -> **`FR8`** <br>submitted `FR8` Integrated Gradients -> **`FR9`** <br>submitted `FR9` Faithful Attention Extraction -> **`FR17`** <br>`FR10`-`FR12` unchanged. <br>submitted `FR13` Accent Bias Profiling -> **`FR15`** <br>submitted `FR14` Attribution Faithfulness Auditing -> **`FR16`** <br>This is why the Test Plan Report cites FR15/FR16 for accent bias and faithfulness auditing while the submitted SRS calls them FR13/FR14 - the same two requirements, not extra scope. `FR5`, `FR13` and `FR14` are **unused identifiers** in the reconciled numbering: never write a requirement against them. Corroboration, not inference: the submitted SRS is **internally inconsistent** on this. Its §3.2 requirement list runs FR1-FR14, but its own §4.5 already cites "FR17" for the synthetic-attention fallback and §4.6 cites "FR7" and "FR16.1" - the reconciled ids. The renumbering happened while that document was being written, and its §3.2 list is the stale side. Concurrent multi-model side-by-side comparison is an **unnumbered** SRS §4.4 out-of-scope item (internal notes and `ISSUE_PLAN.md` LIT-186 call it "dropped FR5", which is shorthand for the vacated identifier, not a renumbering of the submitted FR5). |
| E5 | **Speech enhancement** | The Project Idea lists "Multi-Task & Speech Enhancement Evaluation" as a headline upgrade and the Proposal's executive summary promises the platform will "validate speech enhancement networks". | **Replaced by Audio Deepfake Detection (FR7).** The submitted SRS commits no enhancement requirement and mentions enhancement only in FR7's baseline note; it is also absent from SRS §4.4's out-of-scope list, so the substitution was never written down anywhere. Recording it here: ADD is the committed third task, enhancement is not delivered and is not a specification gap. Nothing in the tree, the SAD or the test plan depends on enhancement - do not start one on the strength of the Idea or Proposal text. |
| E6 | **The SRS's three TBDs** | Submitted SRS §4.6 leaves TBD-1 (SER checkpoint), TBD-2 (faithfulness K and method) and TBD-3 (deepfake detector suitability for ASVspoof 2021 DF) open. | **All three are resolved in code, with the measurements recorded at the decision site.** <br>**TBD-1**: `firdhokk/speech-emotion-recognition-with-facebook-wav2vec2-large-xlsr-53`, pinned at revision `611e6db8ee667aa07fe66596f9fc761e036ff5b9`. Chosen over the inherited ECHO classifier and four rejected alternatives; `superb/wav2vec2-base-superb-er` was rejected for shipping only 4 classes, failing FR6.1's six. Rationale in `Backend/app/domain/model_loader_service.py` above `_EMO_MODEL_ID`. <br>**TBD-2**: method `gradcam` (the route default), deletion-score swept over K = 10/20/30/50/70/100 % with 20 % as the single-point default (`evaluation_service.mask_top_k_features`). A sweep rather than one K, because a single point cannot distinguish a faithful map from a lucky one. <br>**TBD-3**: confirmed, and it changed the default. Measured on 200 labelled clips (100 genuine from common-voice/CREMA-D/RAVDESS, 100 spoof from ASVspoof 2021 DF): `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification` 88.5 % accuracy, 0 false alarms, separation +0.62; `MelodyMachine/Deepfake-audio-detection-V2` 38.5 %, separation **-0.23, i.e. anti-correlated with the truth** while reporting 0.9999 confidence. Gustking is now the default; both stay selectable. |

Add new errata here as they are discovered, rather than editing the source
documents mid-stream.

---

## Repository layout (target, per SAD)

Single monorepo, organized into the five logical layers from the SAD:

```
audiolit-workspace/
├── docs/                    # this directory (SAD, SRS, README)
├── backend/
│   └── app/
│       ├── api/             # FastAPI gateway (the "application layer"): routes, CORS,   (SAD §5.1, application layer)
│       │                    #   enqueue, WebSocket relay — deliberately contains no AI code
│       ├── orchestration/   # RQ/Redis per-model workers, Task Orchestrator fan-out/fan-in (SAD §5.1 orchestration layer; §6.1 worker design; §5.2 Task Orchestrator)
│       ├── domain/          # framework-free: Model Registry, Explanation Strategies       (SAD §5.1 domain layer; §5.2 component table)
│       │                    #   (IG/LIME/SHAP/Grad-CAM), Mutation Engine, Acoustic Profiler,
│       │                    #   Bias Profiler and Faithfulness Auditor
│       └── infrastructure/  # Cache Manager (Redis, fingerprint-keyed), MongoDB,           (SAD §5.1 infrastructure layer; §5.2 Cache Manager)
│                            #   dataset-reading tools, activity logging
└── frontend/
    └── src/                 # React 18 Workspace (shared interface state), HTML5 canvas,   (SAD §5.1 presentation layer; §3.3 Plotly)
                             #   Plotly projection, spectrogram overlays
```

> Note on SAD citations above: `docs/SAD.md` describes the five layers in prose in §5.1
> and lists components in a single flat table in §5.2 — it does **not** have numbered
> per-layer subsections (`§5.2.1`...`§5.2.5`), and its component names are plain
> (`Model Registry`, `Explanation Strategies`, `Cache Manager`, `Mutation Engine`,
> `Acoustic Profiler`, `Bias Profiler and Faithfulness Auditor`, `Task Orchestrator`,
> `Workspace`) rather than class-style names like `HookManager`/`CacheGateway`/`TensorCodec`.
> An earlier pass (LIT-228) cited fine-grained section numbers and class names that were
> never verified against the actual document; both LIT-228 and downstream Tier-C-stamped
> issues have been corrected to match the real SAD.md structure above.

> The ECHO 1.0 clone may still be in its inherited `Backend/` / `Frontend/`
> shape until the layered migration (LIT-227) completes. Verify the actual tree
> before assuming the structure above exists.

---

## Branch model

- **`main`** — production. Never receive a PR directly from a feature branch.
- **`develop`** — integration branch. All feature work branches off `develop`;
  PRs merge **into `develop`**.
- **`testing`** — dedicated test harness and evaluation branch. Incorporates all
  commits from `develop` and `main`, hosting full-stack end-to-end dataflow
  suites (Playwright `dataflow`), performance benchmarks & load testing (Locust),
  and diagnostic evaluation scripts. Hosting testing tooling in this branch
  ensures `develop` and `main` remain clean, lightweight, and uncluttered.
- **`develop` → `main`** only after a full audit.
- **One feature branch per Linear issue** (`feature/lit-xxx-...`), **one PR per
  issue**, referencing the LIT-id. CI (pytest + Jest, ruff/Black,
  ESLint/Prettier) green before merge.
- **Every PR requires at least one approving review from a different team
  member before merging** — this is mandatory, not optional, even when CI is
  green. With 3 people on this project, self-merging is exactly how scope
  drifts and mistakes go unnoticed. Opening a PR automatically moves the
  linked Linear issue to **In Review** (LIT-134's GitHub↔Linear automation)
  — that's expected and not something to "fix"; it reflects the PR waiting
  on a human reviewer, not on more work.
- **Claude Code sessions must not self-merge PRs.** Open the PR, verify CI is
  green (`gh pr checks <n>`, waiting for an actual terminal result — see
  below), then stop and hand off for review. Only merge if a human
  explicitly instructs it for that specific PR.

---

## Scope discipline

The SRS separates **committed** functional requirements from **non-committed
(stretch)** scope (SRS §4.4). Committed features must ship within their phase;
nothing non-committed may displace committed work.

**Non-committed / stretch (SRS §4.4) — do not build as committed scope:**
ADDSegDiff diffusion-based artefact localization, multi-class generator
fingerprinting, multi-model side-by-side comparison, per-demographic confusion
matrices, cross-lingual disparity, insertion-score / deletion-insertion AUC,
IoU-against-ground-truth-mask validation.

Stretch issues must carry a ⚠ STRETCH banner and a "do not start until committed
work is merged" gate. Never promote a stretch item to a committed FR.

> There is deliberately **no FR5, FR13, or FR14** in the reconciled SRS. Do not
> invent them.

---

## Committed functional requirements (quick index)

Full specifications live in `SRS.md`; this is a pointer index.

| FR | Capability |
|----|------------|
| FR1 | Dynamic Hugging Face model ingestion (supported-family registry) |
| FR2 | Benchmark dataset management |
| FR3 | Asynchronous multi-task inference (RQ) |
| FR4 | SHA-256 cache-by-hash |
| FR6 | Speech Emotion Recognition |
| FR7 | Audio Deepfake Detection (binary) |
| FR8 | Spectrogram attribution + Grad-CAM |
| FR9 | Integrated Gradients |
| FR10 | Acoustic wave profiling |
| FR11 | Latent projection (committed lasso/linking) |
| FR12 | Canvas-driven mutation |
| FR15 | Accent bias profiling |
| FR16 | Faithfulness auditing (deletion score) |
| FR17 | Faithful attention extraction |

---

*Keep this file current. When a decision changes the architecture or scope,
record it here as an erratum before propagating it into issues or code.*
