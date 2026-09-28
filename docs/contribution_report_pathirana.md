# Individual Contribution Report

**Project:** AudioLIT — An Interactive Multimodal Explainable-AI Workbench for Speech Recognition, Emotion Analytics, and Deepfake Detection (Group 19)
**Name:** Pathirana K.P.R.J.
**Student ID:** 230467R

| Section | Details |
|---|---|
| **Documentation** | • Drafted the functional requirements for the multi-task XAI overlays and the canvas-driven audio mutation feature.<br>• Produced the dataset-management compliance audit and kept the issue plan's status current after each round of merges. |
| **Dataset Ingestion** | • Built the multi-task dataset ingestion core — a single streaming loader interface covering the speech-recognition, emotion and deepfake corpora.<br>• Wrote the individual corpus loaders: Common Voice, LibriSpeech with silence validation, the L2-ARCTIC non-native reading corpus, ASVspoof 2021 DF, and ESD, completing the seven-corpus registry.<br>• Closed the dataset-management compliance gaps: per-file integrity validation, a bounded working footprint, and licence surfacing for the non-commercial corpora.<br>• Added custom-dataset ground-truth CSV upload with order-independent filename matching.<br>• Wired the real dataset registry into the live dataset routes, which had still been serving a hardcoded list. |
| **Acoustic Profiling** | • Implemented the pYIN probabilistic fundamental-frequency tracker, marking unvoiced frames rather than interpolating through them.<br>• Implemented RMS energy estimation and the localised amplitude contour.<br>• Assembled these into the combined Librosa DSP acoustic profile pipeline. |
| **Canvas & Visualisation** | • Developed the HTML5 canvas waveform visualisation suite and the downsampled waveform buffer rendering maths.<br>• Built the 2D spectrogram grid selector and the coordinate resolver that converts pixel selections into time and frequency units, so the backend never receives pixels.<br>• Implemented the drag and bounding-box tracker, the layered alpha-blended heatmap overlay infrastructure, and the frontend data binding for XAI overlays. |
| **Signal Mutation** | • Built the frontend mutation trigger and the asynchronous state dispatcher that sends a selected spectrogram region to the backend and tracks the resulting job. |
| **Explainability** | • Implemented the Grad-CAM attribution utility and the Integrated Gradients attribution core.<br>• Adapted LIME and SHAP to operate on 2D spectrogram patches rather than raw waveform samples. |
| **Bias & Evaluation** | • Built the accent-bias profiling core: cohort batching with per-group reservoir sampling and the per-sample word-error-rate primitive.<br>• Implemented the group-wise WER diagnostic runner that ranks accent cohorts into a disparity report. |
| **Frontend Architecture** | • Led the interface redesign — navigation, sidebar, workspace grid, explainability panels, status bar, dark mode and the overall visual identity.<br>• Built the task selector for speech-recognition / emotion / deepfake and the custom Hugging Face model interface.<br>• Consolidated the frontend inference pipeline onto the asynchronous multi-task job.<br>• Added the per-row regenerate control with cache bypass, and fixed the dataset table rendering raw JSON instead of the predicted label. |
| **API Surface** | • Exposed the model registry, acoustic profiler, and evaluation/bias services through API routes. |
| **Testing & CI** | • Wrote the automated backend test suites and the UI component boundary tests, and wired up the frontend test runner.<br>• Added backend route tests, which surfaced a genuine routing bug.<br>• Wrote the end-to-end system integration tests covering the demo paths.<br>• Sped up CI by caching package downloads and installing the CPU-only build of the deep-learning framework. |

### Challenges & Solutions

- **The continuous-integration pipeline hung intermittently** on the fan-out / fan-in orchestration tests. → Traced it to non-deterministic worker ordering and rewrote the tests to be deterministic.
- **CI runs were slow and heavy**, pulling multi-gigabyte GPU builds onto CPU-only runners. → Cached dependency downloads and pinned the CPU-only framework build.
- **The dataset registry existed but was not connected** to the routes the interface actually called, so new corpora never appeared. → Wired the registry through to the live routes.
- **A bad merge reintroduced regressions** that had already been fixed. → Reverted the merge and repaired the affected loader tests rather than patching over the symptoms.
- **Converting canvas pixels into signal units** correctly across zoom levels and sample rates required deriving the mel-scale mapping in both directions. → Implemented and verified the forward and inverse transforms so selections round-trip.
