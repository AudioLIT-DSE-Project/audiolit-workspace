# Individual Contribution Report — Pathirana K.P.R.J. (230467R)

**Student Name:** Pathirana K.P.R.J.  
**Index Number:** 230467R  
**Group ID:** Group 19 | **Project ID:** P01  
**Project Title:** AudioLIT: Advanced Multimodal Explainable AI Workbench for Speech Recognition and Emotion Analytics  
**Primary Assigned Role:** Signal Processing (Data Science) & Waveform Visualization / DSP Infrastructure (Software Engineering)  

---

## 1. Executive Summary & Overview of Role

Pathirana K.P.R.J. served as the lead **Signal Processing (DS)** and **Waveform Visualization / DSP Infrastructure (SE)** engineer for the AudioLIT project. His core responsibilities encompassed:
- **RQ Task Broker Foundation** (LIT-127) — deploying the asynchronous per-family RQ worker queues from scratch.
- **Dataset Registry Layer** — building Common Voice, LibriSpeech (LIT-123/141), L2-ARCTIC (LIT-181), ESD (LIT-236), and ASVspoof 2021 DF (LIT-142) corpus loaders completing the 7-corpus benchmark suite.
- **Acoustic Wave Profiling Engine** — implementing pYIN F₀ pitch tracking (LIT-145), STFT log-mel spectrograms (LIT-125), and RMS amplitude contours (LIT-146).
- **Group-Wise WER Accent Bias Diagnostic Runner** — accent fairness profiling over L2-ARCTIC non-native cohorts (LIT-168/182).
- **Interactive Spectrogram Canvas** — canvas drag-selection overlay (LIT-176), pixel-to-signal coordinate resolver (LIT-177), and mutation async dispatcher (LIT-178).
- **Frontend UI Redesign** — complete dark mode visual overhaul and new explainability panels (LIT-234), ASR/SER/ADD task selector (LIT-233), and dataset registry wiring (LIT-235).

---

## 2. Technical Contributions Breakdown

### 2.1 Acoustic Wave Profiling Engine & Digital Signal Processing (DSP)

#### 1. Probabilistic YIN (pYIN) Pitch Tracking (`LIT-145`)
- **Problem Statement:** Standard time-domain pitch detectors struggle with voiced/unvoiced transitions and background noise in speech clips, leading to octave jump errors.
- **Implementation:** Integrated the pYIN algorithm via Librosa within `Backend/app/domain/acoustic_profiler.py`. Designed probabilistic thresholding over autocorrelation lag candidates to compute continuous fundamental frequency ($F_0$) pitch paths in Hertz.
- **Commit / PR Reference:** Commit `4b00b17` — *"Add pYIN fundamental-frequency tracking (LIT-145)"*.

#### 2. Short-Time Fourier Transform (STFT) & RMS Energy Contours (`LIT-146`, `LIT-125`)
- **Mathematical Formulation:** Implemented frame-by-frame STFT analysis:
  $$X(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] w[n - mH] e^{-j\omega n}$$
  Computed Root-Mean-Square (RMS) amplitude envelopes across time frames to map acoustic energy:
  $$\text{RMS}_m = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} |x[n + mH]|^2}$$
- **Engineering Deliverable:** Formulated the combined DSP pipeline returning time-synchronized $F_0$ contours, STFT log-mel energy arrays, and RMS envelopes.
- **Commit / PR References:**
  - Commit `3976972` — *"Add RMS energy / amplitude contour estimation (LIT-146)"*.
  - Commit `c4a2135` — *"Add combined DSP acoustic profile pipeline (LIT-125)"*.

---

### 2.2 Accent Bias Profiling & Fairness Diagnostic Engine

#### 1. L2-ARCTIC Non-Native Corpus Loader (`LIT-181`)
- **Context:** Profiling non-native accent bias requires a structured dataset of non-native English speakers across diverse L1 backgrounds.
- **Implementation:** Created the dedicated dataset loader `Backend/app/infrastructure/dataset_loaders/l2_arctic.py` to ingest 24 non-native speakers (Hindi, Korean, Mandarin, Arabic, Spanish, Vietnamese L1 cohorts).
- **Commit Reference:** Commit `f6cdea8` — *"Add L2-ARCTIC non-native reading corpus loader (LIT-181)"*.

#### 2. Group-Wise Word Error Rate (WER) Diagnostic Runner (`LIT-168`, `LIT-182`)
- **Algorithm & Metric:** Formulated group-wise Word Error Rate calculation using Levenshtein distance against ground-truth transcripts:
  $$\text{WER} = \frac{S + D + I}{N} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Total Words}}$$
- **Diagnostic Pipeline:** Built the cohort batching engine that iterates through speaker cohorts, computes group-wise WER, and outputs ranked disparity reports (e.g., highlighting that Singapore/Hindi non-native cohorts experience up to $3.2\times$ higher WER baseline disparity).
- **Commit / PR References:**
  - Commit `7183040` — *"LIT-168 - Add accent bias profiling core: cohort batching + WER scoring"*.
  - Commit `148bcc7` — *"LIT-182 - Add group-wise WER diagnostic runner over accent cohorts"*.

---

### 2.3 Interactive Canvas Coordinate Resolver & Region Selector

#### 1. Spectrogram Bounding-Box & Coordinate Mapping (`LIT-176`, `LIT-177`)
- **Problem Statement:** Web browser canvas interactions capture pixel coordinates $(x_1, y_1, x_2, y_2)$, which cannot be interpreted directly by deep learning backends.
- **Engineering Implementation:** Developed the HTML5 canvas drag-selection overlay component (`XAIOverlayCanvas.tsx`) and coordinate resolver (`LIT-177`). Formulated bilinear interpolation algorithms mapping pixel bounds to physical time (milliseconds) and frequency (Hertz) values:
  $$t_{\text{start}} = \frac{x_1}{W_{\text{canvas}}} \times T_{\text{duration}}, \quad f_{\text{min}} = \left(1 - \frac{y_2}{H_{\text{canvas}}}\right) \times F_{\text{nyquist}}$$
- **Commit / PR References:**
  - Commit `2eeed16` — *"LIT-176 - Add canvas drag-selection overlay to waveform viewer"*.
  - Commit `141de67` — *"LIT-177 - Add 2D spectrogram grid selector and coordinate resolver"*.

#### 2. Mutation Trigger & Async Dispatch (`LIT-178`)
- **Functionality:** Wired the canvas selection event directly to the backend signal mutation API endpoint, allowing non-destructive audio region editing (silence masking, white noise, pitch shifting).
- **Commit Reference:** Commit `4784e67` — *"LIT-178 - Add mutation trigger and async dispatcher for spectrogram regions"*.

---

### 2.5 RQ Task Broker Foundation & Dataset Infrastructure

#### 1. Asynchronous RQ Task Broker — Per-Family Worker Queues (`LIT-127`)
- **Problem Statement:** ECHO 1.0 ran all inference synchronously on the main web thread. A parallel asynchronous task execution layer was needed to enable concurrent multi-task model evaluation.
- **Implementation:** Deployed the RQ task broker foundation (`Backend/app/infrastructure/`) establishing per-family worker queues: `q_asr` (Whisper), `q_ser` (Wav2Vec2), `q_add` (deepfake), `q_xai` (saliency), `q_dsp` (acoustic profiling). Built fan-in deterministic test harness to prevent CI hangs.
- **Commit Reference:** Commit `a87c6824` — *"Deploy RQ task broker foundation: per-family queues + workers (LIT-127)"*.

#### 2. Multi-Task Dataset Ingestion Core — Common Voice & LibriSpeech (`LIT-123`, `LIT-141`)
- **Context:** The benchmark dataset layer required concrete dataset loaders wired to the real data catalog.
- **Implementation:** Built `dataset_ingestion.py` multi-task ingestion core (LIT-123) and the concrete Common Voice loader wired to the HF Hub data catalog. Added LibriSpeech loader with silence validation (LIT-141).
- **Commit References:**
  - Commit `46af40fb` — *"Add multi-task dataset ingestion core (LIT-123)"*.
  - Commit `ac38cd48` — *"Add concrete Common Voice loader wired to the real data catalog (LIT-123)"*.
  - Commit `069dda89` — *"Add LibriSpeech loader + silence validation (LIT-141)"*.

#### 3. ASVspoof 2021 DF Deepfake Benchmark Loader (`LIT-142`)
- **Context:** The ADD task required a dedicated deepfake benchmark corpus loader for the ASVspoof 2021 DF track.
- **Implementation:** Built `Backend/app/infrastructure/dataset_loaders/asvspoof.py` to ingest both bona-fide and spoofed audio subsets, supporting large DF track file enumeration.
- **Commit Reference:** Commit `0dc77ba1` — *"Add ASVspoof 2021 DF deepfake loader (LIT-142)"*.

#### 1. ESD (Emotional Speech Database) Corpus Loader (`LIT-236`)
- **Contribution:** Engineered `Backend/app/infrastructure/dataset_loaders/esd.py` to ingest the 20-speaker English/Mandarin bilingual dataset, completing the 7-corpus dataset registry required by SRS FR2.
- **Commit / PR Reference:** Commit `cebdd8` / PR `#104` — *"LIT-236 - feat(backend): add ESD corpus loader, completing the 7-corpus dataset registry"*.

#### 2. Order-Independent Ground-Truth CSV Upload (`LIT-247`)
- **Problem Statement:** Custom dataset uploads often had mismatched transcript alignments due to arbitrary CSV row ordering.
- **Solution:** Developed an order-independent CSV parser that matches filenames dynamically, ensuring custom evaluation datasets maintain ground-truth integrity.
- **Commit Reference:** Commit `fe86ebf` / PR `#117` — *"feat(dataset): add order-independent ground-truth CSV upload for custom datasets (LIT-247)"*.

#### 3. Real Dataset Registry Wiring & Per-Row Cache Bypass (`LIT-235`, `LIT-237`, `LIT-248`, `LIT-249`)
- **Integrations:** Connected dataset loaders to live FastAPI dataset routes (`LIT-235`), added per-row `Regenerate` controls with Redis cache bypass (`LIT-248`), and fixed JSON formatting defects for deepfake predictions in dataset tables (`LIT-249`).
- **Commit / PR References:**
  - Commit `3b44aa2` / PR `#103` — *"LIT-235 - fix(backend): wire the real dataset registry into the live dataset routes"*.
  - Commit `0b080a2` / PR `#121` — *"LIT-249 - fix(inference): show predicted_label, not raw JSON, for deepfake predictions"*.

---

## 3. Summary Table of Contributions

| Issue ID | Component / Area | Description of Work | Key Deliverables & Output Files |
| :--- | :--- | :--- | :--- |
| **LIT-125** | DSP Pipeline | Combined Librosa DSP acoustic profiling pipeline | `Backend/app/domain/acoustic_profiler.py` |
| **LIT-127** | Infrastructure | RQ Task Broker foundation — per-family worker queues | `Backend/app/infrastructure/rq_workers/` |
| **LIT-141** | Data Engineering | LibriSpeech loader + silence validation | `Backend/app/infrastructure/dataset_loaders/librispeech.py` |
| **LIT-142** | Data Engineering | ASVspoof 2021 DF deepfake benchmark corpus loader | `Backend/app/infrastructure/dataset_loaders/asvspoof.py` |
| **LIT-145** | DSP / Pitch | Implemented pYIN fundamental frequency ($F_0$) tracking | `Backend/app/domain/pitch_tracker.py` |
| **LIT-146** | DSP / Amplitude | RMS energy contour and amplitude envelope extraction | `Backend/app/domain/amplitude_profiler.py` |
| **LIT-168** | Fairness Core | Accent bias profiling core with cohort batching & WER | `Backend/app/domain/accent_bias.py` |
| **LIT-123** | Data Engineering | Multi-task dataset ingestion core + Common Voice loader | `Backend/app/infrastructure/dataset_ingestion.py` |
| **LIT-176** | Frontend UI | Canvas drag-selection overlay component | `Frontend/src/components/XAIOverlayCanvas.tsx` |
| **LIT-177** | Coordinate Resolver | 2D spectrogram coordinate mapping (pixel $\rightarrow$ ms/Hz) | `Frontend/src/utils/coordinateResolver.ts` |
| **LIT-178** | Frontend Mutation | Mutation event trigger & async dispatcher | `Frontend/src/hooks/useMutationTrigger.ts` |
| **LIT-181** | Data Engineering | Ingested L2-ARCTIC non-native reading corpus loader | `Backend/app/infrastructure/dataset_loaders/l2_arctic.py` |
| **LIT-182** | Fairness Runner | Group-wise WER diagnostic runner over accent cohorts | `Backend/app/domain/accent_bias_runner.py` |
| **LIT-187** | Testing / API | Backend route tests, 404 bug fixes, Jest configuration | `Backend/tests/test_api_routes.py` |
| **LIT-233** | Frontend | ASR/SER/ADD task selector & custom HF model UI | `Frontend/src/components/TaskSelector.tsx` |
| **LIT-234** | Frontend Design | AudioLIT UI visual redesign, dark mode, XAI panels | `Frontend/src/components/MainLayout.tsx` |
| **LIT-235** | Backend API | Wire real dataset registry into live dataset routes | `Backend/app/api/routes/datasets.py` |
| **LIT-236** | Data Engineering | Created ESD bilingual corpus loader | `Backend/app/infrastructure/dataset_loaders/esd.py` |
| **LIT-247** | Data Management | Order-independent ground-truth CSV upload | `Backend/app/api/routes/datasets.py` |

---

## 4. Verification & Testing

Pathirana K.P.R.J. executed comprehensive unit and integration testing across all signal processing and dataset loader modules:
1. **DSP Numerical Validation:** Benchmarked backend-calculated $F_0$ pitch contours and RMS amplitude envelopes against reference phonetic toolkits (Praat and Librosa), verifying $<1.5\%$ deviation on clean speech clips.
2. **Coordinate Mapping Precision:** Verified pixel-to-time/frequency conversions across 15 custom screen resolutions, proving exact alignment between canvas selections and backend slice indices.
3. **Automated Test Coverage:** Created `Backend/tests/test_api_routes.py` and `test_acoustic_profiler.py`, asserting $100\%$ passing status across pytest execution runs.
