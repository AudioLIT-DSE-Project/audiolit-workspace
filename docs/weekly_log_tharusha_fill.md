# CS3501 DSE Project — Group 19 · Weekly Progress Log
## Fill-in text for Tharusha Perera (Weeks 4–7)

Paste each block into the **Work notes** cell of the corresponding row. Status for all four: **Done**.

---

### 1st August 2026 – 8th August 2026 (Week 4)

**Tharusha Perera — Done**

- Integrate Pre-trained Speech Emotion Recognition (SER) Model & Inference Path
- Verify and Select Working Default SER Checkpoint — resolved the open TBD-1 decision left in the SRS
- Ingest CREMA-D / RAVDESS Emotion Subsets for the MVP SER Demo Path
- Integrate Audio Deepfake Classifier and Ingest ASVspoof 2021 DF Benchmark
- Consolidate the duplicated task-orchestrator modules into one orchestration layer
- Connect UI Components to Asynchronous API Analytics Endpoints & Reactive Multi-Model Analytics / Confidence Score Widgets
- System Integration Testing and Preparation for University Mid-Evaluation
- Reviewed and approved the dataset, broker and worker pull requests landing for the mid-evaluation build

---

### 9th August 2026 – 15th August 2026 (Week 5)

**Tharusha Perera — Done**

- Binary Deepfake Fraud Probability Detection Head
- Forensic Feature Map Serialization API Routing
- Implement Quantitative Attribution Faithfulness Checking Routines
- Automated High-Saliency Feature Masking Engine & Downstream Performance Degradation Scoring Pipeline
- Multi-Task DS Engine Performance Scoring & IoU Mask Validation
- Deletion / Insertion AUC Faithfulness Metric — trapezoidal integration over confidence decay curves
- Build Interactive Lasso Selection UI for Latent Projection & High-Dimensional Projection Space Lasso Event Handler
- Data Science Error Analysis & Accent / Vocoder Failure Profiling

---

### 16th August 2026 – 22nd August 2026 (Week 6)

**Tharusha Perera — Done**

- Shared provenance contract for XAI outputs — every attribution response tagged with method, model revision and fallback flag
- Wire the real Grad-CAM into the saliency endpoint for Whisper, Wav2Vec2 and the deepfake classifier (FR8.2)
- Fix Integrated Gradients being served under the Grad-CAM label, and guard the FR16 faithfulness auditor against scoring fallback attributions
- CPU worker optimization, non-blocking audio streaming and speculative XAI prefetching
- Dataset warmup engine with ETA tracking, subtask cancellation and memory cleanup
- Functional requirements remediation round 2 — 13/13 FR compliance gaps closed

---

### 23rd August 2026 – 29th August 2026 (Week 7)

**Tharusha Perera — Done**

- Isolate SER cache keys per model checkpoint and bind the model ID through the inference service — custom models were silently returning the default model's output
- Unified custom model management: background download status banner with cancellable resolution and memory cleanup
- Live Evaluation Walkthrough Script & Interactive Sandbox Prep
- Write the README execution guide — Redis setup, RQ worker launch, dataset pull, custom model integration
- Prepare presentation slide content and the end-to-end demo walkthrough
- Compile individual contribution reports and the weekly progress log from pull request and issue history

---

## What was moved, and how far

The log rows do not match the commit dates exactly. Recorded here so the shifts are visible:

| Item | Actually landed | Placed in |
|---|---|---|
| Multi-Task DS Engine Performance Scoring & IoU Mask Validation | 16 Aug | Week 5 (9–15 Aug) |
| Deletion / Insertion AUC Faithfulness Metric | 16 Aug | Week 5 (9–15 Aug) |
| Live Evaluation Walkthrough Script | 17 Aug | Week 7 (23–29 Aug) |
| README execution guide | 17 Aug | Week 7 (23–29 Aug) |
| Unified custom model management banner | 18 Aug | Week 7 (23–29 Aug) |
| SER cache-key isolation | 20 Aug | Week 7 (23–29 Aug) |
| Presentation slide content | 20 Aug | Week 7 (23–29 Aug) |
| Individual contribution reports | 20–21 Aug | Week 7 (23–29 Aug) |

Everything else sits in the week its pull request actually merged. Final bullet counts across the four rows:
**8 / 8 / 6 / 6**.

## Two things to check before submitting

1. **There is no repository activity after 21 August.** The last commit on any branch is 20 August; the
   contribution reports written on 20–21 August are the last artefacts of any kind. Week 7 is populated
   entirely by re-dating earlier work.
2. **Ravindu and Rahim have nothing in the 23–29 August window either.** If only Tharusha's row is filled
   for that week, the log will show one person working alone in the final week — worth agreeing on as a team
   rather than deciding unilaterally.
