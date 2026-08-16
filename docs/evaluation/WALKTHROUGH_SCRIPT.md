# AudioLIT — Live Evaluation Walkthrough Script & Interactive Sandbox Prep

> **Document ID:** `docs/evaluation/WALKTHROUGH_SCRIPT.md`  
> **Ticket Link:** `LIT-191` (Parent: `LIT-172`, Mentor Meetup 3 Deliverables)  
> **Scope:** Supervisor Demonstration & Live System Sandbox Verification Protocol  

---

## 1. Executive Summary & Demonstration Objectives

This document provides the canonical step-by-step walkthrough script and staging environment prep checklist for the live supervisor demonstration of **AudioLIT (Audio Latent Inspection Tool)**. 

The demonstration highlights three core technical pillars:
1. **High-Throughput Audio Streaming & Hash-Based Tensor Caching** (RQ + Redis execution pipeline).
2. **Interactive Latent Space & Spectrogram Mutation Triggers** (2D coordinate resolution, visual bounding box selection, and lasso event handling).
3. **Multi-Task Data Science Diagnostics & Faithfulness Scoring** (Group-wise WER cohort profiling and deletion AUC trapezoidal integration metrics).

---

## 2. Staging Environment Setup & Verification Checklist

Before initiating the live supervisor walkthrough, execute the following staging verification protocol to ensure zero deployment issues during the live session.

### 2.1 Backend Environment Verification
- [x] **Python Environment & Dependencies:** Virtual environment active with `soundfile`, `librosa`, `torch`, `fastapi`, `redis`, and `rq`.
- [x] **Health Check Endpoint:** `GET http://localhost:8000/api/v1/health` returns `{"status": "ok", "redis": "connected"}`.
- [x] **Interactive Swagger Docs:** Accessible at `http://localhost:8000/docs`.

### 2.2 Frontend Staging Build & Test Verification
- [x] **Production Bundle Build:**
  ```bash
  cmd /c "npm run build --prefix Frontend"
  ```
  *(Verified: `vite build` completed cleanly in `Frontend/dist`)*.
- [x] **UI Component & Jest Test Suite:**
  ```bash
  cmd /c "npm run test --prefix Frontend -- --watchAll=false"
  ```

---

## 3. Step-by-Step Live Demonstration Script

### Step 1: Data Ingestion & Instant Hash-Based Cache Return
* **Presenter Narrative:**  
  *"We begin by demonstrating AudioLIT's high-performance data ingestion. When an audio file is uploaded, the system computes an immutable SHA-256 payload hash. If the feature maps have been previously extracted, Redis returns the cached tensor representation instantly without re-running heavy PyTorch model inference."*
* **Interactive Actions:**
  1. Open AudioLIT Web Interface (`http://localhost:5173`).
  2. Upload sample audio file `arctic_a0001.wav` from the L2-ARCTIC cohort.
  3. Observe instant payload SHA-256 hash generation (`sha256:e3b0c442...`).
  4. Verify latency indicator: **Cache Hit (< 15ms)** vs. **RQ Worker Async Job Dispatch**.

---

### Step 2: Interactive 2D Waveform & Spectrogram Coordinate Mutations
* **Presenter Narrative:**  
  *"Next, we demonstrate our real-time interactive selection tools. Users can inspect localized audio anomalies across both time and frequency axes simultaneously."*
* **Interactive Actions:**
  1. **Waveform Pointer Selection:** Drag a visual bounding box over the time-domain waveform (`[0.42s - 0.88s]`). Observe the reactive state update triggering instant region-scoped RMS energy contour mapping.
  2. **2D Spectrogram Coordinate Resolver:** Click on a high-energy formant region in the 2D STFT spectrogram viewer (`1250 Hz, t = 0.65s`).
  3. **Mutation Event Trigger:** Trigger an asynchronous spectrogram region mutation (masking high-saliency spectral bands). Observe live mutation state dispatching to the underlying model diagnostic head.

---

### Step 3: Latent Space Projection & Interactive Lasso Selection
* **Presenter Narrative:**  
  *"Here we view the high-dimensional latent space projection of audio embeddings. AudioLIT provides an interactive Lasso selection tool to group audio clips by task attributes."*
* **Interactive Actions:**
  1. Navigate to **Embedding Projection Panel**.
  2. Select **Lasso Selection Tool** and draw a closed polygon loop around an outlier cluster in the UMAP/t-SNE projection.
  3. Observe automatic color-coding toggle between **Emotion**, **Deepfake Spoof/Bona-fide**, and **Speaker Accent Region**.
  4. Click the audio play button on any lasso-selected clip to trigger localized audio playback.

---

### Step 4: Multi-Task DS Engine Performance & Attribution Faithfulness Scoring
* **Presenter Narrative:**  
  *"Finally, we showcase quantitative XAI faithfulness auditing and ASR accent-bias profiling."*
* **Interactive Actions:**
  1. Open **DS Diagnostic & Faithfulness Panel**.
  2. Review the **Group-Wise Word Error Rate (WER)** breakdown across L2-ARCTIC demographic cohorts (Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese).
  3. Inspect the **Deletion AUC Faithfulness Metric**: Observe the trapezoidal numerical integration curve tracking confidence degradation as high-saliency acoustic features are iteratively deleted.

---

## 4. Technical Architecture Reference & Verification Summary

```
+-----------------------------------------------------------------------+
|                       AUDIO-LIT STAGING RUNTIME                       |
+-----------------------------------------------------------------------+
|                                                                       |
|  [ Frontend React + Vite ] <---> [ FastAPI Async Middleware Layer ]   |
|          |                                   |                        |
|          v                                   v                        |
|  (Interactive Lasso UI,              (SHA-256 Tensor Hash Cache,       |
|   2D Spectrogram Mutations)           RQ Worker Inference Dispatch)   |
|                                              |                        |
|                                              v                        |
|                                 [ DS Evaluation Service ]             |
|                                 - Group-Wise WER Profiler             |
|                                 - Deletion AUC Faithfulness Head      |
|                                                                       |
+-----------------------------------------------------------------------+
```

### Verification Command & Output
```bash
cmd /c "npm run build --prefix Frontend"
```
**Output:**
```text
vite v6.2.0 building for production...
✓ 1876 modules transformed.
dist/index.html                                                  0.46 kB
dist/assets/index-DYV3oBsc.css                                  80.60 kB
dist/assets/index-6fLgA_rW.js                                1,489.17 kB
✓ built in 14.54s
```
