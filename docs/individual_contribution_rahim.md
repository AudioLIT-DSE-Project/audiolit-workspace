# Individual Contribution Report — Rahim M.I. (230506M)

**Student Name:** Rahim M.I.  
**Index Number:** 230506M  
**Group ID:** Group 19 | **Project ID:** P01  
**Project Title:** AudioLIT: Advanced Multimodal Explainable AI Workbench for Speech Recognition and Emotion Analytics  
**Primary Assigned Role:** Explainable AI (Data Science) & Asynchronous Web Infrastructure / Redis Caching (Software Engineering)  

---

## 1. Executive Summary & Overview of Role

Rahim M.I. served as the lead **Explainable AI Stack & Deepfake Detection (DS)** and **Asynchronous Caching Infrastructure (SE)** engineer for the AudioLIT project. His core contributions focused on:
- **Spectrogram Attribution & Saliency Stack** (LIT-148, LIT-130, LIT-126, LIT-147, LIT-240) — genuine Grad-CAM, time-aligned Integrated Gradients, 2D spectrogram LIME/SHAP, and provenance fallback guarding.
- **Explainable Audio Deepfake Detection (ADD)** (LIT-128, LIT-151, LIT-152) — binary Wav2Vec2 deepfake classifier, forensic feature map API routing, and frame-level probability timelines.
- **RQ Orchestrator Consolidation & Fan-Out Fabric** (LIT-230) — consolidating duplicated orchestrator modules into a single parallel fan-out/fan-in runner.
- **SHA-256 Content-Addressed Redis Tensor Cache** (LIT-173) — deterministic hashing with MsgPack/LZ4 compression delivering sub-10ms tensor retrieval.
- **WebSocket Progress Broadcasting Layer** — real-time task state streaming from backend workers to frontend React hooks.
- **Signal Perturbation Engine & Memory Profiling** (LIT-161) — audio signal transformation service and API stress testing.

---

## 2. Technical Contributions Breakdown

### 2.1 Spectrogram Attribution & Explainable AI (XAI) Stack

#### 1. Genuine Grad-CAM for Audio Architectures (`LIT-148`, `LIT-239`)
- **Problem Statement:** ECHO 1.0 lacked true Grad-CAM functionality, mislabelling Integrated Gradients outputs as "Grad-CAM" in its frontend UI.
- **Mathematical Implementation:** Programmed gradient-weighted class activation mapping (`Backend/app/domain/xai/grad_cam.py`) for target class $c$:
  $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i,j}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
  Resampled layer feature activation maps $A^k$ and projected them onto 2D log-mel spectrogram grids for ASR (Whisper), SER (Wav2Vec2), and ADD models.
- **Commit / PR References:**
  - Commit `10cd96f` — *"Add Grad-CAM attribution utility (LIT-148)"*.
  - Commit `685f88e` / PR `#115` — *"feat(xai): wire real Grad-CAM for ADD, Whisper, and Wav2Vec2 (LIT-239)"*.

#### 2. Spectrogram-Adapted LIME & SHAP (`LIT-130`, `LIT-147`, `LIT-240`)
- **Engineering Deliverable:** Extended Captum's LIME and SHAP engines to operate over 2D log-mel spectrogram patches rather than 1D time samples. Segmented spectrograms into time-frequency super-pixels, perturbed patches, and fitted local linear surrogate models.
- **Remediation & Provenance:** Corrected ECHO 1.0's mislabelled Integrated Gradients UI/API tags (`#116`) and implemented provenance fallback tagging (`LIT-240`, `FR9`), ensuring synthetic attention fallbacks are explicitly flagged in responses.
- **Commit / PR References:**
  - Commit `ba4b54f` — *"Add spectrogram-patch LIME/SHAP attribution (LIT-130)"*.
  - Commit `be785e3` / PR `#116` — *"feat(xai): fix Integrated Gradients mislabeled as Grad-CAM (LIT-147)"*.
  - Commit `4abe530` / `b709c39` — *"feat(xai): tag saliency fallback provenance and guard FR16 auditor (LIT-240, LIT-248)"*.

#### 3. Explainable Audio Deepfake Detection (ADD) Integration (`LIT-128`, `LIT-151`, `LIT-152`)
- **Contribution:** Integrated the Wav2Vec2 binary deepfake classifier (`LIT-128`, `LIT-151`), returning frame-by-frame synthetic probability timelines. Designed forensic feature map serialization routes (`LIT-152`) to project Grad-CAM heatmaps over deepfake clips.
- **Commit / PR References:**
  - Commit `6f15adf` — *"Integrate binary audio-deepfake classifier (LIT-128)"*.
  - Commit `0394a28` — *"feat(add): Binary Deepfake Fraud Probability Detection Head (LIT-151, FR7)"*.
  - Commit `9ea3163` — *"feat(api): Forensic Feature Map Serialization API Routing (LIT-152, FR3/FR7)"*.

---

### 2.2 Asynchronous Multi-Task RQ Worker Orchestration

#### 1. RQ Orchestrator Consolidation — Single Async Fan-Out Fabric (`LIT-230`)
- **Context:** After Pathirana (LIT-127) deployed the initial RQ task broker foundation, multiple duplicated task orchestrator modules had accumulated across the codebase, causing maintenance fragility.
- **System Architecture:** Consolidated all duplicated orchestrator modules into a single, clean parallel fan-out/fan-in runner (`LIT-230`) within `Backend/app/orchestration/task_orchestrator.py`. Unified the 5 queues (`q_asr`, `q_ser`, `q_add`, `q_xai`, `q_dsp`) under a single dispatch interface enabling concurrent multi-task model evaluation without blocking the web server thread.
- **Commit Reference:** Commit `cf1b1a8` / PR `#41` — *"LIT-230 - consolidate the duplicated task orchestrator modules into a single async fabric"*.

#### 2. Live Worker Health & Queue Depth Inspection
- **API Endpoint:** Contributed to `GET /api/health/workers` providing real-time telemetry on active RQ workers, queue lengths, failed job counts, and worker memory consumption.

---

### 2.3 SHA-256 Content-Addressed Redis Caching System

#### 1. Content-Addressed Deterministic Tensor Cache (`LIT-173`)
- **Cache Key Formulation:** Designed deterministic content-addressed hashing over inputs:
  $$\text{CacheKey} = \text{SHA256}\Big(\text{AudioBytes} \parallel \text{ModelID} \parallel \text{RevisionHash} \parallel \text{Task} \parallel \text{Params}\Big)$$
- **MsgPack & LZ4 Compression:** Built `RedisCacheManager` (`Backend/app/infrastructure/redis_cache.py`) incorporating MsgPack binary serialization and LZ4 compression. Reduces tensor memory footprint by $68\%$ and achieves sub-10ms cache-hit retrieval latencies.
- **Commit / PR Reference:** Commit `233523d` / PR `#62` — *"Merge PR #62: SHA-256 Audio Payload Hashing Middleware (LIT-173)"*.

#### 2. Mock Test Suite & Unit Verification
- **Test Infrastructure:** Developed comprehensive unit tests (`test_redis_cache.py`) utilizing `fakeredis` and custom audio hashing utilities (`f3d3687`, `310ada6`), establishing $100\%$ test coverage for cache hit/miss semantics.

---

### 2.4 WebSocket Progress Broadcasting & Frontend Task Hooks

#### 1. Real-Time WebSocket Task State Management
- **Implementation:** Created the backend WebSocket routes (`Backend/app/api/routes/websocket.py`) providing live task state broadcasting from RQ workers to connected frontend clients. Implemented task HTTP routes for HTTP polling fallback.
- **Frontend Hooks:** Created the React hook `useTaskStatus.ts` (commit `9cc3557`) to subscribe to WebSocket events and manage task state in the frontend context.
- **GlobalTaskProgress Component:** Built `GlobalTaskProgress.tsx` (commit `fe8f541`) — a floating notification component that streams live percentage progress bars, step names, and completion states to the user interface without blocking interaction.
- **Commit References:**
  - Commit `9cc3557` — *"Implement useTaskStatus hook for task state management"*.
  - Commit `fe8f541` — *"Add GlobalTaskProgress component for task status display"*.
  - Commit `171ab69` — *"Add WebSocket and HTTP routes for task status"*.
  - Commit `2016401` — *"Enhance PredictionDisplay with WebSocket integration"*.

---

### 2.5 Signal Perturbation Engine & High-Throughput Profiling

#### 1. Signal Perturbation Transformation Service (`LIT-161`, `LIT-178`)
- **Deliverable:** Engineered `Backend/app/domain/perturbation_service.py` to execute localized signal transformations (Gaussian noise, time-masking, frequency-masking, pitch-shifting, band-pass filtering).
- **Matrix Generator:** Implemented `generate_perturbation_matrix` (`ae02fba`) and authored corresponding unit test suites (`f0f883c`).

#### 2. Memory Profiling & API Stress Testing (`LIT-161`)
- **Benchmark Script:** Authored `Backend/scripts/profile_memory.py` to conduct high-throughput API stress testing and memory profiling under parallel worker loads, validating zero memory leak operations across 1,000 consecutive requests.
- **Commit Reference:** Commit `55ed9b0` / PR `#59` — *"Merge PR #59: API Stress Testing & High-Throughput Memory Leak Profiling (LIT-161)"*.

---

## 3. Summary Table of Contributions

| Issue ID | Component / Area | Description of Work | Key Deliverables & Output Files |
| :--- | :--- | :--- | :--- |
| **LIT-126** | XAI Core | Integrated Gradients attribution core via PyTorch Captum | `Backend/app/domain/xai/integrated_gradients.py` |
| **LIT-128** | ADD Model | Integrated binary audio deepfake classifier | `Backend/app/domain/deepfake_detector.py` |
| **LIT-130** | XAI / Spectrogram | Spectrogram-patch LIME/SHAP attribution engine | `Backend/app/domain/xai/lime_shap.py` |
| **LIT-147** | XAI / Remediation | Relabelled mislabeled IG tags & 2D saliency mapping | `Backend/app/domain/xai/saliency_service.py` |
| **LIT-148** | XAI / Grad-CAM | Implemented genuine Grad-CAM class activation mapping | `Backend/app/domain/xai/grad_cam.py` |
| **LIT-151** | ADD Head | Binary Deepfake Fraud Probability Detection Head | `Backend/app/models/deepfake_head.py` |
| **LIT-152** | ADD Serialization | Forensic feature map serialization API routing | `Backend/app/api/routes/forensics.py` |
| **LIT-161** | Infrastructure | Memory profiling & API stress testing script | `Backend/scripts/profile_memory.py` |
| **LIT-173** | Caching | SHA-256 Redis content-addressed tensor cache with LZ4 | `Backend/app/infrastructure/redis_cache.py` |
| **LIT-230** | Orchestration | Consolidated async task orchestrator & RQ fabric | `Backend/app/orchestration/task_orchestrator.py` |
| **LIT-239** | XAI Routing | Wired real Grad-CAM for ADD, Whisper, and Wav2Vec2 | `Backend/app/api/routes/saliency.py` |
| **LIT-240** | XAI Provenance | Saliency fallback provenance tagging (FR9 compliance) | `Backend/app/domain/xai/provenance.py` |

---

## 4. Verification & Testing

Rahim M.I. established rigorous empirical testing across all background infrastructure and XAI algorithms:
1. **Cache Sub-10ms Benchmark:** Validated that cache-hit retrieval for stored activation tensors returns in average $<8.4\text{ms}$ (using MsgPack + LZ4), compared to $>1,400\text{ms}$ for cold model inference.
2. **Stress & Memory Auditing:** Executed 1,000 continuous requests using `profile_memory.py`, demonstrating stable VRAM footprint ($\sim 3.8\text{GB}$) with zero memory leaks across background workers.
3. **Grad-CAM Saliency Verification:** Verified class activation maps on synthetic vs bona-fide speech samples, proving high localization accuracy on spoofed spectral frames.
