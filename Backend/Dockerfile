# LIT-203 — production backend image.
#
# Build context is the repo root (docker-compose.yml sets context: . so both
# images share one .dockerignore). Multi-stage: a venv is built in a slim
# builder and copied whole into a runtime image that carries only the system
# libs the audio stack needs (soundfile -> libsndfile1, torch -> libgomp1).
# No test or load-test tooling ever lands in the image (LIT-203).

FROM python:3.11-slim AS builder
WORKDIR /build
# CPU-only torch, exactly like ci.yml: the default Linux wheel is the
# multi-GB CUDA build; this is several GB smaller. Installed into the venv
# first so requirements.txt's torch>=2.6.0 is already satisfied and the -r
# install never re-fetches it.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
COPY Backend/requirements.txt /build/requirements.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r /build/requirements.txt

FROM python:3.11-slim AS runtime
# libsndfile1: soundfile's native library (LIT-226 keeps audio I/O on
# soundfile; torchaudio is intentionally absent). libgomp1: torch OpenMP.
# apt-get upgrade hardens the base tag against freshly-published CRITICALs
# (Trivy gate, LIT-203) without a base-image rebuild.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libgomp1 \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY Backend/app /app/app

# Non-root runtime. uploads/ and the HF cache volume (HF_HOME=/models) sit in
# the image as owned by this user so named volumes initialise writable.
RUN useradd --create-home --uid 1000 audiolit \
    && mkdir -p /app/uploads /models \
    && chown -R audiolit:audiolit /app /models

USER audiolit

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]