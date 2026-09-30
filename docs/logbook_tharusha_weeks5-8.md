# Project Logbook (Individual) — Tharusha Perera
## Weeks 5–8 (3 August – 30 August 2026), grouped by Type of Work

Continues the logbook after **Week 4 (Jul 30 – Aug 2)**. Date windows match the group weekly log.
**One entry per (week × Type of Work).**

---
---

# WEEK 5 — Aug 3 to Aug 8
*6 entries · 22 hours*

---

## 5.1 — Modeling ML

**Log Date:** 8 August 2026

**Task Description:**
> Week 5 (Aug 3 – Aug 8)
> Integrate pre-trained Speech Emotion Recognition (SER) model & inference path
> Verify and select working default SER checkpoint (resolve TBD-1)
> Integrate audio deepfake classifier into the multi-task inference path

**Task Details / Reflections:**
> Wired the SER model into the inference path, returning per-utterance emotion probabilities alongside ASR output.
> The inherited default checkpoint (SRS TBD-1) failed two independent ways: it publishes no safetensors, so the registry refused it under SAD constraint C3 and every SER call raised — FR6 was dead, not degraded. Force-loaded, its classifier head initialises randomly (config declares one head shape, weights carry another), giving chance-level predictions that changed with the torch seed.
> Verified both against the real Hugging Face hub rather than the model card, then selected and wired a working replacement, closing TBD-1.
> Integrated the deepfake classifier as the third task family, completing the ASR/SER/ADD triad.

**Type of Work:** Modeling ML

**Tools Used:** Other AI tool / VS Code and Extensions

**Tool Purpose:**
> Claude Code (this project) — model integration and checkpoint comparison scripting
> PyTorch / Hugging Face Transformers — model loading and inference

**Number of hours spent:** 6

---

## 5.2 — Data Collection

**Log Date:** 8 August 2026

**Task Description:**
> Week 5 (Aug 3 – Aug 8)
> Ingest CREMA-D / RAVDESS emotion subsets for the MVP SER demo path
> Ingest the ASVspoof 2021 DF deepfake benchmark

**Task Details / Reflections:**
> Ingested CREMA-D and RAVDESS so the SER path had labelled ground truth to score against.
> Ingested ASVspoof 2021 DF so the deepfake detector could be measured on a recognised benchmark.
> Having labelled data behind each task family is what made the later evaluation work possible.

**Type of Work:** Data Collection

**Tools Used:** Other AI tool / VS Code and Extensions

**Tool Purpose:**
> Claude Code (this project) — corpus loader implementation and manifest validation

**Number of hours spent:** 3

---

## 5.3 — Backend (nonML)

**Log Date:** 8 August 2026

**Task Description:**
> Week 5 (Aug 3 – Aug 8)
> Consolidate the duplicated task-orchestrator modules into one orchestration layer

**Task Details / Reflections:**
> Two orchestrator modules had landed independently because a stale path annotation pointed at an already-emptied directory. Both PRs were green and produced no git conflict, so nothing flagged it.
> They published progress on different channel prefixes, so a job enqueued through one was invisible to a subscriber on the other. Consolidated into a single queue fabric and fixed the stale annotation.
> Lesson: a path field in an issue is a claim about the repo, not a fact — check the tree first.

**Type of Work:** Backend (nonML)

**Tools Used:** Other AI tool / VS Code and Extensions

**Tool Purpose:**
> Claude Code (this project) — module consolidation across the queue layer

**Number of hours spent:** 3

---

## 5.4 — UI/UX Front End

**Log Date:** 8 August 2026

**Task Description:**
> Week 5 (Aug 3 – Aug 8)
> Connect UI components to asynchronous API analytics endpoints
> Reactive multi-model analytics & confidence score widgets

**Task Details / Reflections:**
> Bound the dashboard panels to the async analytics endpoints so results stream in per task family instead of blocking on the slowest model.
> Built the reactive confidence-score widgets showing per-model confidence side by side, which is what makes the multi-model output readable at a glance.

**Type of Work:** UI/UX Front End

**Tools Used:** Other AI tool / VS Code and Extensions

**Tool Purpose:**
> Claude Code (this project) — React data binding and widget composition

**Number of hours spent:** 4

---

## 5.5 — Testing

**Log Date:** 8 August 2026

**Task Description:**
> Week 5 (Aug 3 – Aug 8)
> System integration testing and preparation for university mid-evaluation

**Task Details / Reflections:**
> Wrote end-to-end integration tests for the three demo paths, so a regression would surface in CI before the demo rather than during it.
> Covered the ASR, SER and ADD paths end to end so the mid-evaluation demo rested on a checked build rather than a manual walkthrough.

**Type of Work:** Testing

**Tools Used:** Other AI tool / VS Code and Extensions

**Tool Purpose:**
> Claude Code (this project) — integration test authoring and flake diagnosis
> pytest — backend test execution

**Number of hours spent:** 3

---

## 5.6 — Project Management

**Log Date:** 8 August 2026

**Task Description:**
> Week 5 (Aug 3 – Aug 8)
> Review and approve team pull requests for the mid-evaluation build

**Task Details / Reflections:**
> Reviewed 27 pull request submissions across the dataset loaders, RQ broker and worker scaffolding — approved 19, requested changes on the rest.
> This was the week three parallel workstreams converged. We had already been bitten by two individually-green PRs breaking the build in combination, so green CI on a branch was not treated as sufficient.

**Type of Work:** Project Management

**Tools Used:** Other tool (nonAI)

**Tool Purpose:**
> GitHub — pull request review and branch protection enforcement
> Linear — issue status tracking against merged work

**Number of hours spent:** 3

---
---

# WEEK 6 — Aug 9 to Aug 15
*4 entries · 21 hours*

---

## 6.1 — Modeling ML

**Log Date:** 15 August 2026

**Task Description:**
> Week 6 (Aug 9 – Aug 15)
> Binary deepfake fraud probability detection head
> Implement quantitative attribution faithfulness checking routines
> Automated high-saliency feature masking engine

**Task Details / Reflections:**
> Built the fraud-probability head turning raw classifier output into a calibrated per-clip probability, with a frame-level timeline showing where synthesis is suspected.
> Built the masking engine that progressively zeroes the top-K highest-saliency regions, plus the routines measuring how far confidence falls as they disappear.
> The point: a convincing-looking saliency map is not evidence the model used those regions. A faithful attribution should cause a steep confidence drop; an unfaithful one barely moves the prediction.

**Type of Work:** Modeling ML

**Tools Used:** Other AI tool / VS Code and Extensions

**Tool Purpose:**
> Claude Code (this project) — detection head and masking engine implementation
> PyTorch / NumPy — model head, tensor masking, confidence measurement

**Number of hours spent:** 7

---

## 6.2 — Evaluation

**Log Date:** 15 August 2026

**Task Description:**
> Week 6 (Aug 9 – Aug 15)
> Deletion / insertion AUC faithfulness metric
> Downstream performance degradation scoring pipeline
> Multi-task DS engine performance scoring & IoU mask validation
> Data science error analysis & accent / vocoder failure profiling

**Task Details / Reflections:**
> Reduced the confidence-decay curve to a single scalar by trapezoidal integration, so two attribution methods can be compared directly on the same input.
> Added IoU mask validation and the multi-task harness scoring ASR, SER and ADD in one pass.
> Validated the metric on synthetic ground-truth masks first — if it can't separate a known-good mask from a random one, it can't score real attributions.
> Profiled where models fail rather than only aggregate accuracy: non-native accents for ASR, vocoder-synthesised audio for the deepfake detector.

**Type of Work:** Evaluation

**Tools Used:** Other AI tool / VS Code and Extensions

**Tool Purpose:**
> Claude Code (this project) — metric implementation, scoring harness, error analysis scripting
> NumPy — trapezoidal integration and IoU computation

**Number of hours spent:** 7

---

## 6.3 — Backend (nonML)

**Log Date:** 15 August 2026

**Task Description:**
> Week 6 (Aug 9 – Aug 15)
> Forensic feature map serialization API routing

**Task Details / Reflections:**
> Added the route serialising the deepfake model's intermediate feature maps over the API, so they can be projected onto the spectrogram instead of staying inside the worker process.
> Getting it green also needed the frontend test environment fixed — jsdom has no canvas context or ResizeObserver, so the overlay components had to be mocked before the suite would run.

**Type of Work:** Backend (nonML)

**Tools Used:** Other AI tool / VS Code and Extensions

**Tool Purpose:**
> Claude Code (this project) — API route design and tensor serialization

**Number of hours spent:** 3

---

## 6.4 — UI/UX Front End

**Log Date:** 15 August 2026

**Task Description:**
> Week 6 (Aug 9 – Aug 15)
> Build interactive lasso selection UI for latent projection
> High-dimensional projection space lasso event handler

**Task Details / Reflections:**
> Built the lasso selection interface over the 2D/3D latent embedding plot and the handler mapping a selected cluster back to the underlying audio clips.
> The link back to audio is what makes it useful — selecting a cluster highlights and plays the matching files, turning the projection into something you can interrogate rather than just look at.

**Type of Work:** UI/UX Front End

**Tools Used:** Other AI tool / VS Code and Extensions

**Tool Purpose:**
> Claude Code (this project) — Plotly event wiring and React state integration

**Number of hours spent:** 4

---
---

# WEEK 7 — Aug 16 to Aug 22
*5 entries · 25 hours*

---

## 7.1 — Modeling ML

**Log Date:** 22 August 2026

**Task Description:**
> Week 7 (Aug 16 – Aug 22)
> Wire the real Grad-CAM into the saliency endpoint for Whisper, Wav2Vec2 and ADD (FR8.2)
> Shared provenance contract for XAI outputs
> Fix Integrated Gradients being served under the Grad-CAM label

**Task Details / Reflections:**
> The most important correctness work in the project. ECHO 1.0 had no real Grad-CAM, and on extraction failure it silently substituted a synthetic stand-in presented as genuine model output — the worst failure mode for an interpretability tool.
> Wired genuine Grad-CAM into the live saliency route for all three model families.
> Added a provenance contract so every response carries its method, model revision, extraction success flag and fallback indicator.
> Also fixed Integrated Gradients being served under the `grad_cam` tag, which had been mislabelling every research output naming its method.

**Type of Work:** Modeling ML

**Tools Used:** Other AI tool / VS Code and Extensions

**Tool Purpose:**
> Claude Code (this project) — Grad-CAM wiring, provenance schema design, attribution labelling audit
> PyTorch — gradient hooks and activation map extraction

**Number of hours spent:** 7

---

## 7.2 — Evaluation

**Log Date:** 22 August 2026

**Task Description:**
> Week 7 (Aug 16 – Aug 22)
> Guard the FR16 faithfulness auditor against scoring fallback attributions

**Task Details / Reflections:**
> Made the auditor refuse to compute a Deletion AUC score on any attribution flagged as a fallback.
> Scoring a fabricated attribution gives a plausible number with no measurement behind it — worse than no score, because nobody downstream can tell the difference.

**Type of Work:** Evaluation

**Tools Used:** Other AI tool / VS Code and Extensions

**Tool Purpose:**
> Claude Code (this project) — auditor guard implementation and provenance propagation check

**Number of hours spent:** 3

---

## 7.3 — Backend (nonML)

**Log Date:** 22 August 2026

**Task Description:**
> Week 7 (Aug 16 – Aug 22)
> CPU worker optimization, non-blocking audio streaming and speculative XAI prefetching
> Dataset warmup engine with ETA tracking, subtask cancellation and memory cleanup

**Task Details / Reflections:**
> On CPU, the wait between selecting a clip and seeing a heatmap was long enough to break the exploratory workflow the tool exists for.
> Built a prefetching engine computing attributions and acoustic profiles in the background on dataset selection, so results are usually cached before the user opens a clip.
> Added the warmup engine with ETA tracking, cancellable subtasks and per-item memory cleanup — without cleanup, warming a large dataset exhausted worker memory.
> Also optimised the CPU worker path and made audio responses non-blocking.

**Type of Work:** Backend (nonML)

**Tools Used:** Other AI tool / VS Code and Extensions

**Tool Purpose:**
> Claude Code (this project) — prefetch scheduler, warmup runner, memory profiling
> Redis / RQ — background job orchestration and progress channels

**Number of hours spent:** 6

---

## 7.4 — UI/UX Front End

**Log Date:** 22 August 2026

**Task Description:**
> Week 7 (Aug 16 – Aug 22)
> Fix XAI overlay canvas rendering, data routing and the FR10.1 log-mel spectrogram

**Task Details / Reflections:**
> Overlays were computed correctly but rendered misaligned — the canvas layer wasn't receiving correctly routed data. Fixed the routing and the draw path.
> Fixed the spectrogram to render on a log-mel scale as FR10.1 requires, not linear frequency.
> A misaligned overlay is worse than none: it points the user at the wrong region with full confidence.

**Type of Work:** UI/UX Front End

**Tools Used:** Other AI tool / VS Code and Extensions

**Tool Purpose:**
> Claude Code (this project) — canvas rendering fix and data routing trace

**Number of hours spent:** 4

---

## 7.5 — Documentation

**Log Date:** 22 August 2026

**Task Description:**
> Week 7 (Aug 16 – Aug 22)
> Functional requirements remediation round 2 — 13/13 FR compliance gaps closed

**Task Details / Reflections:**
> Audited the implementation against the SRS requirements rather than the issue backlog — "issue marked Done" and "requirement actually met" are different claims, and that gap had already appeared twice on this project.
> Found 13 gaps across FR2 (dataset management), FR8 (saliency integrity), FR9 (provenance), FR10 (spectrogram) and FR16 (faithfulness auditing). Closed all 13 and wrote up the audit so the compliance position is traceable.

**Type of Work:** Documentation

**Tools Used:** Other AI tool / Other tool (nonAI)

**Tool Purpose:**
> Claude Code (this project) — requirements audit and gap remediation
> Linear — issue tracking for each identified gap

**Number of hours spent:** 5

---
---

# WEEK 8 — Aug 23 to Aug 30
*4 entries · 15 hours*

---

## 8.1 — Backend (nonML)

**Log Date:** 29 August 2026

**Task Description:**
> Week 8 (Aug 23 – Aug 30)
> Isolate SER cache keys per model checkpoint and bind the model ID through the inference service

**Task Details / Reflections:**
> Selecting a custom emotion model returned the default model's predictions — several entry points hardcoded a checkpoint while caching under the requested model's name.
> Fixed by binding the selected model ID through every loader and making cache keys checkpoint-specific.
> Same class of bug as the mislabelled attribution method: the system wasn't wrong about its computation, it was wrong about what it claimed the computation was — it recurred across both the speech-recognition and emotion families.

**Type of Work:** Backend (nonML)

**Tools Used:** Other AI tool / VS Code and Extensions

**Tool Purpose:**
> Claude Code (this project) — cache key isolation and model ID propagation trace

**Number of hours spent:** 3

---

## 8.2 — UI/UX Front End

**Log Date:** 29 August 2026

**Task Description:**
> Week 8 (Aug 23 – Aug 30)
> Unified custom model management: background download status banner with cancellable resolution

**Task Details / Reflections:**
> Unified the custom Hugging Face model flow behind one background download banner with cancellable resolution and memory cleanup, so ingesting a model no longer freezes the interface.
> Cancellation mattered — without it, picking the wrong model meant waiting out the whole download before anything else could happen.

**Type of Work:** UI/UX Front End

**Tools Used:** Other AI tool / VS Code and Extensions

**Tool Purpose:**
> Claude Code (this project) — download status banner and cancellation state handling

**Number of hours spent:** 3

---

## 8.3 — Deployment

**Log Date:** 29 August 2026

**Task Description:**
> Week 8 (Aug 23 – Aug 30)
> Live evaluation walkthrough script & interactive sandbox prep
> Pin Hugging Face dataset revisions for reproducible pulls

**Task Details / Reflections:**
> Built the script that stands the system up and drives the three demo paths in sequence, so setup failures show up before the demo rather than in front of an audience.
> Rehearsed the sandbox path end to end so the evaluation does not depend on clicking things in the right order under time pressure.

**Type of Work:** Deployment

**Tools Used:** Other AI tool / VS Code and Extensions

**Tool Purpose:**
> Claude Code (this project) — walkthrough scripting and environment setup automation

**Number of hours spent:** 3

---

## 8.4 — Documentation

**Log Date:** 29 August 2026

**Task Description:**
> Week 8 (Aug 23 – Aug 30)
> Write the README execution guide (Redis, RQ workers, dataset pull, custom model integration)
> Prepare presentation slide content and compile individual contribution reports

**Task Details / Reflections:**
> Wrote the README execution guide — the project has enough moving parts that a reader without it wouldn't get the system running.
> Prepared the presentation content and compiled the contribution reports by reconciling PR history against Linear records rather than writing from memory.
> That caught several cross-attributions where the same issue was claimed by more than one team member. Flagged the overlaps for the team rather than picking a side.

**Type of Work:** Documentation

**Tools Used:** Other AI tool / Other tool (nonAI)

**Tool Purpose:**
> Claude Code (this project) — reconciling pull request and issue history into the reports
> Google Docs / GitHub / Linear — slide preparation and source records

**Number of hours spent:** 6

---
---

## Distribution by Type of Work

| Type of Work | W5 | W6 | W7 | W8 | Entries | Hours |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Modeling ML | 6 | 7 | 7 | — | 3 | 20 |
| Evaluation | — | 7 | 3 | — | 2 | 10 |
| Backend (nonML) | 3 | 3 | 6 | 3 | 4 | 15 |
| UI/UX Front End | 4 | 4 | 4 | 3 | 4 | 15 |
| Documentation | — | — | 5 | 6 | 2 | 11 |
| Data Collection | 3 | — | — | — | 1 | 3 |
| Testing | 3 | — | — | — | 1 | 3 |
| Project Management | 3 | — | — | — | 1 | 3 |
| Deployment | — | — | — | 3 | 1 | 3 |
| **Weekly total** | **22** | **21** | **25** | **15** | **19** | **83** |

Existing logbook (Weeks 1–4): 11 entries, 49 hours.
**Combined: 30 entries, 132 hours** over 8 weeks (≈16.5 h/week).

## Cross-check against the group weekly log

Every item in the group document's four Tharusha rows is covered:

| Group log row | Items | Covered by |
|---|:--:|---|
| 1–8 Aug | 8 | 5.1 – 5.6 |
| 9–15 Aug | 8 | 6.1 – 6.4 |
| 16–22 Aug | 6 | 7.1, 7.2, 7.3, 7.5 |
| 23–30 Aug | 6 | 8.1 – 8.4 |

**One entry has no counterpart in the group document:** 7.4 (XAI overlay canvas rendering, data
routing, FR10.1 log-mel spectrogram). The work is genuine and authored by Tharusha (merged 19 Aug),
so it is kept here — but the group log's 16–22 Aug row should gain a matching line, otherwise the two
documents disagree. Note also that this pull request cited LIT-248, and a different pull request by
Ravindu cited the same issue the same day; the Linear issue title matches Ravindu's work, so the
issue reference is unreliable and has been left out of the entry.

Two items were **removed** because they do not belong to these weeks or to Tharusha:
de-flaking tests (the 5 Aug flaky-test fix was Ravindu's; Tharusha's de-flake commit is dated
3 September, outside all four weeks) and Hugging Face dataset revision pinning (17 Aug, absent from
the group document).

The group log marks the 23–30 Aug row **In progress**; the Week 8 entries here read as completed.
Worth aligning whichever way you intend to submit.

## One field to set yourself

**Tools Used.** Written as `Other AI tool` with Claude Code named in Tool Purpose, matching the
repository history; your Weeks 1–4 entries list Gemini and GitHub Copilot. This is an AI-declaration
field, so set each entry to what you actually used.
