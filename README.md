# AudioLIT — Interpretability Workbench for Speech & Audio ML

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https.python.org)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/frontend-React%2018%20%2B%20Vite-61dafb.svg)](https://reactjs.org)
[![Redis Queue](https://img.shields.io/badge/task%20fabric-RQ%20%2B%20Redis-red.svg)](https://python-rq.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AudioLIT** is an open-source, multi-task interpretability workbench for Automatic Speech Recognition (**ASR**), Speech Emotion Recognition (**SER**), and Audio Deepfake Detection (**ADD**). It extends the **ECHO 1.0** baseline (`AudioLIT-DSE-Project/ECHO`) into a production-grade explanatory environment equipped with:

- **Asynchronous Task Architecture**: Non-blocking RQ (Redis Queue) background execution for heavy PyTorch XAI workloads.
- **Explainable AI (XAI)**: Grad-CAM attribution heatmaps, Integrated Gradients, and Layer-Head Attention Pair Extraction.
- **Dataset Management & Warmup**: Automated dataset pre-caching engine for sub-millisecond XAI visualization.
- **Model Ingestion Registry**: Hugging Face `.safetensors` model resolution for Whisper and Wav2Vec2 architectures.
- **Forensic Audio Diagnostics**: Acoustic wave profiling, latent space projection, canvas-driven audio perturbations, and quantitative deletion-based faithfulness auditing.

---

## 📂 Repository Layout

```
audiolit-workspace/
├── docs/        # Authoritative project planning documents (SAD, SRS, Errata)
├── Backend/     # FastAPI backend service, RQ worker fabric, & domain ML engines
│   ├── app/
│   │   ├── api/             # FastAPI gateway & HTTP routes (predictions, saliency, datasets)
│   │   ├── orchestration/   # RQ background workers, Task Orchestrator & warmup runner
│   │   ├── domain/          # Model loader, XAI engines (Grad-CAM, SHAP), & acoustic profiler
│   │   ├── infrastructure/  # Redis cache manager, dataset loaders, & settings
│   │   └── core/            # Redis cache manager & app configuration
│   └── data/                # Benchmark datasets (Common Voice, RAVDESS, CREMA-D)
└── Frontend/    # React 18 + Vite web interface with canvas XAI overlays & Plotly projections
```

For authoritative architectural details, refer to [`docs/README.md`](docs/README.md), [`docs/SAD.md`](docs/SAD.md), and [`docs/SRS.md`](docs/SRS.md).

---

## 🛠️ System Prerequisites

Before running AudioLIT locally, ensure your system has:
- **Python**: `3.11` or higher
- **Node.js**: `18.0` or higher (with `npm`)
- **Redis Server**: `7.0+` (running via Docker or local installation on port `6379`)
- **FFmpeg**: Required for audio decoding & resampling

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/AudioLIT-DSE-Project/audiolit-workspace.git
cd audiolit-workspace
```

---

### Step 2: Start the Redis Infrastructure Broker
AudioLIT requires a running Redis instance for background RQ task queues, session tracking, and prediction/XAI result caching.

**Option A: Using Docker (Recommended)**
```bash
cd Backend
docker compose up -d
```

**Option B: Using Local Redis**
Ensure Redis is running on port `6379`:
```bash
redis-server --port 6379
```

---

### Step 3: Setup & Launch the Backend FastAPI Service

1. Create and activate a Python virtual environment:
   ```bash
   cd Backend
   python -m venv .venv
   
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the FastAPI server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   The backend API will be available at `http://localhost:8000`. You can inspect the OpenAPI documentation at `http://localhost:8000/docs`.

---

### Step 4: Start the Background RQ Workers

AudioLIT delegates heavy PyTorch inference and XAI attribution tasks to background workers so the API remains responsive.

#### Understanding Worker Task Queues
There are 5 specialized task queues:
- **`asr`**: Speech-to-Text transcription & accent bias profiling
- **`ser`**: Speech Emotion Recognition
- **`add`**: Audio Deepfake Detection
- **`xai`**: Grad-CAM saliency heatmaps, Integrated Gradients, & attention weights
- **`mutation`**: Audio perturbation & downstream faithfulness auditing

#### Option 1: Run All Worker Queues in One Terminal (Recommended)
Launch a unified multi-worker process listening across all queues:
```bash
# From the Backend/ directory with active .venv:
python -m app.orchestration.worker all
```
*Note: The unified launcher automatically purges stale Redis locks upon startup to prevent execution deadlocks.*

#### Option 2: Run Dedicated Family Workers in Separate Terminals
For distributed setups or fine-grained resource control:
```bash
# Terminal 1: ASR Worker
python -m app.orchestration.worker asr

# Terminal 2: SER Worker
python -m app.orchestration.worker ser

# Terminal 3: ADD Worker
python -m app.orchestration.worker add

# Terminal 4: XAI Worker (Saliency & Attention)
python -m app.orchestration.worker xai

# Terminal 5: Mutation Worker (Perturbation & Faithfulness)
python -m app.orchestration.worker mutation
```

#### Worker Health & Monitoring
Check worker status and active queue depth via HTTP:
```bash
curl http://localhost:8000/health/workers
```

---

### Step 5: Setup & Launch the Frontend Web UI

1. Open a new terminal and navigate to `Frontend`:
   ```bash
   cd Frontend
   ```

2. Install frontend dependencies:
   ```bash
   npm install
   ```

3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   Open your browser and navigate to `http://localhost:5173`.

---

## ⚡ Dataset Pre-warming & Cache Performance

To achieve sub-second XAI visualization on CPU environments, AudioLIT features an automated **Dataset Warmup Engine**:

1. Click **"Warmup Dataset"** in the top navigation bar of the Web UI.
2. Select a dataset (`Common Voice`, `RAVDESS`, or `CREMA-D`) and specify a sample range (e.g. 100 samples).
3. The background worker pre-populates Redis with predictions, acoustic profiles, and Grad-CAM saliency heatmaps.
4. Active progress is displayed in real-time with step badges (`Inference`, `Acoustic`, `Saliency`).

---

## 🤖 Custom Hugging Face Model Integration

AudioLIT supports loading custom fine-tuned models from Hugging Face Hub:

1. **Architecture Requirements**: Must belong to the `whisper` (ASR) or `wav2vec2` (SER/ADD) model families.
2. **Security Standard**: Weight files **must** be formatted as `.safetensors`. PyTorch pickle checkpoints (`.bin`/`.pkl`) are rejected for security compliance.
3. **Usage in UI**: Select **"Custom HF Model"** in the model dropdown and enter any valid Hugging Face repository ID (e.g. `openai/whisper-tiny` or `distil-whisper/distil-small.en`).

To programmatically resolve a model via API:
```bash
curl -X POST "http://localhost:8000/models/resolve" \
     -H "Content-Type: application/json" \
     -d '{"model_id": "distil-whisper/distil-small.en", "revision": "main"}'
```

---

## 📜 License

MIT — see [LICENSE](LICENSE). Incorporates code from ECHO 1.0 (`AudioLIT-DSE-Project/ECHO`, MIT-licensed, originally by Anas Hussaindeen, Chandupa Ambepitiya, and Dewmike Amarasinghe).
