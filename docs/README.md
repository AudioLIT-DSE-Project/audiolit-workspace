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
| `README.md` | This file — conventions and errata | Living |

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
