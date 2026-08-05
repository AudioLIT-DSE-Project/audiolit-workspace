# audiolit-workspace

Core monorepo for **AudioLIT** — an interpretability workbench for Automatic Speech
Recognition (ASR), Speech Emotion Recognition (SER), and Audio Deepfake Detection
(ADD), extending the **ECHO 1.0** baseline (`AudioLIT-DSE-Project/ECHO`, itself a
fork of `AnasSAV/ECHO`) in place.

Start with [`docs/README.md`](docs/README.md) — it indexes the authoritative
planning documents (SAD, SRS), the repo's branch model, and known errata —
before making architectural or scope decisions here.

## Layout

```
audiolit-workspace/
├── docs/        # SAD, SRS, conventions (read this first)
├── Backend/     # FastAPI service — ECHO 1.0 backend, extended
└── Frontend/    # React 18 + Vite workspace — ECHO 1.0 frontend, extended
```

The five-layer target architecture (`backend/app/{api,orchestration,domain,infrastructure}`,
lower-case `frontend/src/`) described in the SAD has not been migrated to yet —
see LIT-227. `Backend/`/`Frontend/` currently reflect ECHO 1.0's own structure
(`app/{api,core,services}`).

## Running locally

**Backend**
```bash
cd Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d   # Redis
uvicorn app.main:app --reload
```

**Frontend**
```bash
cd Frontend
npm install
npm run dev
```

## License

MIT — see [LICENSE](LICENSE). Incorporates code from ECHO 1.0
(`AudioLIT-DSE-Project/ECHO`, MIT-licensed, originally by Anas Hussaindeen,
Chandupa Ambepitiya, and Dewmike Amarasinghe).
