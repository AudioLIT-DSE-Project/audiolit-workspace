# Backend runtime for local development.
#
# Exists because Windows Smart App Control blocks llvmlite.dll (numba's unsigned
# native binary), which makes numba/shap/umap — and therefore `import app.main`,
# via model_loader_service's module-level `import umap` — unimportable in a host
# venv. A Linux container has no such policy.
#
# Python 3.10 deliberately matches .github/workflows/ci.yml, so a local run and
# a CI run differ in as few variables as possible.
FROM python:3.10-slim

# libsndfile1: soundfile's native backend (the project's only sanctioned audio
# I/O path since LIT-226 removed torchaudio).
# ffmpeg: librosa's audioread fallback, needed for the Common Voice .mp3 clips —
# soundfile alone cannot decode them.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first, exactly as CI does (ci.yml:54). The default Linux torch
# wheel is the multi-GB CUDA build; installing the CPU wheel here means
# requirements' `torch>=2.6.0` is already satisfied and won't re-fetch it.
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir torch \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple

# Copied separately from the source so a code edit doesn't invalidate this layer.
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# Model weights land in the bind-mounted tree rather than the image or the
# user profile, so deleting the repo removes them too.
ENV HF_HOME=/app/.hf-cache
ENV PYTHONUNBUFFERED=1

# Source arrives via bind mount in docker-compose.yml, not COPY, so edits are
# live without a rebuild.
CMD ["pytest"]
