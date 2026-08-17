# audiolit-workspace

Core monorepo for **AudioLIT** — an interpretability workbench for Automatic Speech
Recognition (ASR), Speech Emotion Recognition (SER), and Audio Deepfake Detection
(ADD), extending the **ECHO 1.0** baseline (`AudioLIT-DSE-Project/ECHO`) in place.

Start with [`docs/README.md`](docs/README.md) — it indexes the authoritative
planning documents (SAD, SRS), the repo's branch model, and known errata —
before making architectural or scope decisions here.

## Layout

```
audiolit-workspace/
├── docs/        # SAD, SRS, conventions (read this first)
├── Backend/     # FastAPI service, XAI domain engines, RQ worker orchestration
└── Frontend/    # React 18 + Vite presentation layer & visualization canvas
```

---

## Quick Start & Local Execution

### 1. Dataset Remote Restore (First-time clone)
Authenticate with Hugging Face and pull the byte-identical benchmark audio corpora pinned in `datasets.lock`:
```bash
pip install huggingface_hub
hf auth login
python .antigravity/bin/ag.py data pull
```

---

### 2. Start Redis Service
AudioLIT requires Redis for task queuing (RQ) and sub-10ms prediction caching:
```bash
# Option A: Run via Docker container
docker run -d -p 6379:6379 redis:latest

# Option B: Run via Docker Compose (from Backend/)
cd Backend
docker compose up -d
```

---

### 3. Start Backend FastAPI Server
Set up the Python virtual environment and launch Uvicorn:

**Linux / macOS:**
```bash
cd Backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Windows (PowerShell / Command Prompt):**
```powershell
cd Backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
.venv\Scripts\uvicorn app.main:app --reload --port 8000
```

---

### 4. Start Asynchronous Task Workers
Launch the parallel worker processes for `saliency`, `acoustic`, `evaluation`, and `analysis` queues in a separate terminal:

**Linux / macOS:**
```bash
cd Backend
source .venv/bin/activate
python -m app.orchestration.worker all
```

**Windows:**
```powershell
cd Backend
.venv\Scripts\python -m app.orchestration.worker all
```

---

### 5. Start Frontend UI Development Server
Launch the React + Vite frontend:

```bash
cd Frontend
npm install
npm run dev
```

Open your browser and navigate to: **`http://localhost:5173`** (or `http://localhost:8080`).

---

## Testing Workflow

1. **Task Compatibility Guards:** Select `Whisper Base (ASR)` or `Wav2Vec2 (SER)` in the top toolbar. Notice dataset options dynamically lock to valid task family corpora (`Common Voice` for ASR, `RAVDESS` for SER).
2. **On-Demand Inference:** Click the **Get Inferences** (`Play` icon) button in the dataset panel to start batch processing. Click **Stop** at any time to halt processing and abort pending requests.
3. **Spectrogram & Attribution Canvas:** Click any row in the dataset table to inspect STFT spectrograms, $F_0$ pitch contours, attention heatmaps, and ground truth labels.

---

## License

MIT — see [LICENSE](LICENSE). Incorporates code from ECHO 1.0
(`AudioLIT-DSE-Project/ECHO`, MIT-licensed, originally by Anas Hussaindeen,
Chandupa Ambepitiya, and Dewmike Amarasinghe).
