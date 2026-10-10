# AudioLIT

An interpretability workbench for speech models. Upload or record a clip, run
speech recognition, emotion recognition and deepfake detection on it, and see
**why** each model answered the way it did — attribution heatmaps over the
spectrogram, pitch and loudness contours, counterfactual edits, and an audit of
whether the explanations are honest.

Runs entirely on your own machines. No audio leaves your server.

---

## What it does

| Capability | What you get |
|---|---|
| **Speech recognition (ASR)** | Transcription with Whisper, word-level timings |
| **Emotion recognition (SER)** | Seven emotion classes with a full probability distribution |
| **Deepfake detection (ADD)** | Bona-fide vs synthetic, plus a per-second confidence timeline showing *where* in a clip the synthesis is |
| **Attribution overlays** | Four methods — Grad-CAM, Integrated Gradients, LIME, SHAP — drawn over the spectrogram |
| **Acoustic profiling** | Pitch (F0) and loudness (RMS) contours on a shared time axis with the attribution |
| **Counterfactual editing** | Mute, filter, add noise, shift pitch or stretch time on a selected region, and see how the prediction moves |
| **Latent projection** | PCA, t-SNE and UMAP views of the model's internal space, colourable by label |
| **Accent-bias profiling** | Word error rate per accent cohort, with the disparity between best and worst |
| **Faithfulness auditing** | Measures whether an explanation is honest by masking what it highlights and re-running the model |
| **Live recording** | Record in the browser; it is converted to 16 kHz mono WAV before upload |
| **Custom models** | Any Hugging Face Whisper or Wav2Vec2 checkpoint, loaded safely and version-pinned |

Every explanation is labelled `measured`, `fallback` or `unavailable`, so you can
always tell a real attribution from a stand-in. That distinction is the point of
the product — see [DEPLOYMENT.md](DEPLOYMENT.md#reading-the-provenance-label).

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Linux, macOS, or Windows 10/11 | Linux |
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB |
| Disk | 20 GB free | 120 GB if you provision the benchmark corpora |
| GPU | none — CPU works | NVIDIA with 6+ GB VRAM and the NVIDIA Container Toolkit |
| Software | **Docker** 20.10+ and **Docker Compose** v2 | same |
| Network | outbound HTTPS to `huggingface.co` on first run, to download model weights | same |

Docker is the only prerequisite. Python and Node are not needed on the host —
they are inside the images.

**Without a GPU everything works but model operations are slower.** Expect
2–4 seconds for a transcription and 20–40 seconds for an attribution on CPU,
against roughly 1 second and 8 seconds on a mid-range GPU. Cached results return
in milliseconds either way.

---

## Install

```bash
git clone https://github.com/AudioLIT-DSE-Project/audiolit-workspace.git audiolit
cd audiolit
docker compose up --build -d
```

First run downloads about 1–2 GB of model weights and takes several minutes.
They are cached in a Docker volume, so later starts are fast.

Open **<http://127.0.0.1:8080>**

> **Use `127.0.0.1`, not `localhost`.** The session cookie is `SameSite=Lax`, and
> the browser treats `localhost:8080 → 127.0.0.1:8000` as cross-site, so the
> cookie is dropped and your session silently resets. Ports do not change the
> site; host names do. This is the single most common setup problem.

### With a GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

Requires the NVIDIA Container Toolkit on the host. Only the worker containers
get the GPU; the gateway, Redis and MongoDB do not need it.

---

## Verify

Three checks, in order. If all three pass the deployment is sound.

**1. Every service is healthy.**

```bash
docker compose ps
```

All five — `redis`, `mongo`, `api`, `worker`, `web` — should read `healthy` or
`running`. The `worker` healthcheck deliberately requires **all five worker
families** to be registered, so a partially crashed fleet reads as unhealthy
rather than green.

**2. The API answers and reports its dependencies.**

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/health/workers
```

`/health` reports whether the queue broker is reachable. `/health/workers`
lists the registered worker families; you should see `asr`, `ser`, `add`, `xai`
and `mutation`.

**3. A clip goes in and a prediction with an explanation comes out.**

In the browser: click **Upload / Record**, record a few seconds of speech or
choose a `.wav`, then run an analysis. You should get a transcript, an emotion
distribution, a deepfake verdict, and an attribution heatmap labelled
`measured`.

If the heatmap says `fallback`, that is the system being honest rather than
broken — read [why](DEPLOYMENT.md#reading-the-provenance-label).

### Run the test suite

The suites ship with the release so you can verify your own deployment:

```bash
# Backend
docker compose exec api python -m pytest -q

# API contract, from an independent client
npx newman run Backend/apitests/AudioLIT.postman_collection.json \
  --env-var baseUrl=http://127.0.0.1:8000
```

---

## Everyday use

| Action | Where |
|---|---|
| Upload a file or record live | **Upload / Record** in the toolbar |
| Pick a model | model selector in the toolbar; paste any Hugging Face id for a custom one |
| Choose which tasks run | ASR / SER / ADD toggles in the toolbar |
| Switch attribution method | the method tabs above the overlay |
| Adjust overlay transparency | the opacity slider on the overlay |
| Edit a region and re-run | **Perturbation** panel; drag a box on the spectrogram |
| Browse a corpus | **Dataset** panel |
| See the model's internal space | **Embedding** panel |

Uploaded audio is deleted automatically after 24 hours by default. Nothing is
sent anywhere outside your deployment except the one-time model download from
Hugging Face.

---

## Stop, start, update

```bash
docker compose stop            # stop, keep data
docker compose start           # start again
docker compose down            # stop and remove containers (volumes survive)
docker compose down -v         # also delete cached models and stored metadata
docker compose logs -f api     # follow the gateway log
```

To update:

```bash
git pull
docker compose up --build -d
```

---

## Further reading

- **[DEPLOYMENT.md](DEPLOYMENT.md)** — configuration reference, the datasets,
  security notes, operations, troubleshooting, and how to read a provenance
  label.
- `docker-compose.yml` — every service, port and volume, with the reasoning for
  each setting in comments.

## Licence

See [LICENSE](LICENSE). Note that several of the optional benchmark corpora are
**research-use only**; the application displays a licence notice when you load
one. Check each corpus's own terms before using it for anything commercial.
