# AudioLIT — Deployment and Operations

Companion to [README.md](README.md), which covers installation. This document is
the reference: what each service does, every setting, how to operate it, and what
to do when something is wrong.

---

## 1. What runs

```
                          ┌──────────────┐
  browser ──── :8080 ───► │  web (nginx) │   the built workbench
                          └──────────────┘
      │
      └────── :8000 ─────►┌──────────────┐
             HTTP + WS    │     api      │   FastAPI gateway.
                          └──────┬───────┘   Never runs a model itself.
                                 │ enqueues
                          ┌──────▼───────┐
                          │    redis     │   job queues, result cache, progress
                          │    :6379     │   channels, sessions
                          └──────┬───────┘
                                 │ dequeues
                          ┌──────▼───────┐
                          │   worker     │   five families: asr, ser, add,
                          └──────────────┘   xai, mutation. Holds the models.
                          ┌──────────────┐
                          │    mongo     │   durable metadata. Optional.
                          └──────────────┘
```

| Service | Image | Host port | Role |
|---|---|---|---|
| `web` | built from `Frontend/Dockerfile` | **8080** | nginx serving the built app |
| `api` | built from `Backend/Dockerfile` | **8000** | HTTP + WebSocket gateway |
| `worker` | same image, different command | none | runs all five worker families |
| `redis` | `redis:7-alpine` | 6379 | queue broker, cache, pub/sub |
| `mongo` | `mongo:6` | **none published** | durable metadata records |

### Why heavy work is not on the request path

A model load takes about 5 seconds and an attribution can take tens. The gateway
therefore never loads a model: it validates the request, checks the cache, and
otherwise enqueues a job and returns a job id immediately. The browser follows
the job over a WebSocket, falling back to HTTP polling if the socket cannot be
established.

This is why the gateway stays responsive under load, and why you should not be
alarmed by a request that returns in 40 ms and *then* takes 30 seconds to
produce a result.

### Why there are five worker families

One queue and one process per family, so a process only ever holds its own
models. GPU-bound families run one job at a time, because two jobs on one GPU do
not run twice as fast — they contend for memory and one eventually fails to
allocate. The CPU-only mutation family may run two.

Consequence for capacity planning: **ASR, SER and ADD run in parallel with each
other**, but two transcriptions queue behind one another.

---

## 2. Configuration

All settings are environment variables. Set them in `docker-compose.yml` under
the relevant service, or in a `.env` file beside it.

### Core

| Variable | Default | Notes |
|---|---|---|
| `REDIS_URL` | `redis://redis:6379/0` | **Required.** The queue broker. |
| `MONGO_URL` | *(empty)* | Empty **disables** the durable metadata tier. See §4. |
| `MONGO_DB_NAME` | `audiolit` | |
| `HF_HOME` | `/models` | Where model weights are cached. Backed by a volume. |
| `ALLOWED_ORIGINS` | localhost/127.0.0.1 on any port | Comma-separated. Set this if you serve from a real host name. |
| `LOG_FORMAT` | `json` | `text` for human-readable local logs. |

### Uploads

| Variable | Default | Notes |
|---|---|---|
| `AUDIOLIT_MAX_UPLOAD_BYTES` | `104857600` (100 MB) | Enforced while streaming, not after. |
| `AUDIOLIT_MAX_UPLOAD_SECONDS` | `900` (15 min) | **`0` disables the cap.** A long quiet clip easily passes the size cap, so this is a separate limit. |
| `AUDIOLIT_UPLOAD_RETENTION_SECONDS` | `86400` (24 h) | **`0` keeps every upload forever.** Swept on each upload and at startup. |

### Sessions and security

| Variable | Default | Notes |
|---|---|---|
| `SESSION_TTL_SECONDS` | `86400` | |
| `COOKIE_SECURE` | `false` | **Set to `true` when serving over HTTPS.** |
| `COOKIE_SAMESITE` | `lax` | Use `none` for a cross-site HTTPS deployment. |
| `COOKIE_DOMAIN` | *(unset)* | |
| `DEBUG_ENABLED` | `false` | **Leave off in production.** Enables a diagnostic endpoint. |

### Datasets

| Variable | Default | Notes |
|---|---|---|
| `DATASET_FOOTPRINT_LIMIT_GB` | `100.0` | Warns in the log when exceeded; does not refuse to start. |
| `DATASET_METADATA_ROW_CAP` | `2000` | Caps rows returned per metadata request. |

### Frontend

`VITE_API_BASE_URL` is a **build argument**, not a runtime variable — it is
compiled into the bundle. Changing it requires a rebuild:

```yaml
web:
  build:
    args:
      VITE_API_BASE_URL: http://127.0.0.1:8000
```

> **Keep the host name identical between `web` and `api`.** The session cookie is
> `SameSite=Lax`, so `localhost:8080 → 127.0.0.1:8000` is cross-site and the
> cookie is dropped. Use `127.0.0.1` for both, or a real host name for both.

---

## 3. Serving over HTTPS

The images terminate plain HTTP. **TLS is yours to add**, normally with a reverse
proxy (nginx, Caddy, Traefik) in front of `web` and `api`.

When you do:

1. Set `COOKIE_SECURE=true`.
2. Set `ALLOWED_ORIGINS` to your HTTPS origin.
3. Rebuild `web` with `VITE_API_BASE_URL` set to the HTTPS API URL. The
   WebSocket URL is derived from it, so `https://` correctly becomes `wss://`.
4. Ensure your proxy forwards WebSocket upgrade headers on `/api/ws/`. Without
   this the UI falls back to polling — it still works, but progress updates
   arrive every two seconds instead of instantly.

---

## 4. The MongoDB tier is optional, and fails silently by design

`MONGO_URL` is **empty by default**, which disables the tier. When it is
disabled — or when Mongo is unreachable — every write is a logged no-op and every
read returns empty. The application keeps working.

Four collections when enabled:

| Collection | Holds | Retention |
|---|---|---|
| `models` | model id, pinned revision, weight digest | permanent |
| `audio_samples` | paths and metadata — **never audio bytes** | permanent |
| `analysis_results` | prediction records | 24 h (`MONGO_ANALYSIS_TTL_HOURS`) |
| `bias_reports` | accent-bias reports | permanent |

**The debugging consequence matters.** A wrong prediction is *never* explained by
Mongo being down, because when Mongo is down the system works and simply records
nothing. Conversely, if you expected records and find none, check `MONGO_URL`
first.

The compose `mongo` service publishes **no host port**. If you run the API
natively against a containerised Mongo you must publish one:

```bash
docker run -d --name audiolit-mongo -p 27017:27017 -v audiolit-mongo-data:/data/db mongo:6
export MONGO_URL=mongodb://127.0.0.1:27017
```

Verify it is live:

```bash
docker compose exec mongo mongosh --quiet audiolit --eval "db.getCollectionNames()"
```

You should see all four collections.

---

## 5. Redis settings that are not defaults for a reason

```yaml
command: ["redis-server", "--maxmemory", "2gb", "--maxmemory-policy", "volatile-lru"]
```

**2 GB, not less.** Attribution payloads carry a full spectrogram plus an
attribution matrix, so a handful fill a small cache. When Redis fills, writes
fail — and because the worker fleet registers and heartbeats through the same
Redis, that kills workers rather than merely missing a cache entry.

**`volatile-lru`, not `allkeys-lru`.** This Redis is both the cache and the queue
broker. `allkeys-lru` evicts *any* key under pressure, including RQ's queue lists
and the per-family worker locks — queued jobs vanish and a live worker loses its
lock. `volatile-*` only evicts keys that carry a TTL: every cache, progress and
session key does; RQ's queues and registries do not.

Persistence is off deliberately. Every cache entry is recomputable, so losing the
cache costs time, not data.

---

## 6. Benchmark datasets (optional)

The workbench runs fine with no corpora — upload or record your own audio. The
datasets are needed only for batch evaluation and accent-bias profiling.

Place them under `Backend/data/`, which is mounted into both `api` and `worker`.

| Corpus | Task | Licence |
|---|---|---|
| Mozilla Common Voice | ASR | CC0-1.0 |
| LibriSpeech | ASR | CC-BY-4.0 |
| CREMA-D | emotion | ODbL, research use |
| RAVDESS | emotion | CC-BY-NC-SA-4.0 — **non-commercial** |
| ESD | emotion | research only — **non-commercial** |
| L2-ARCTIC | ASR, accent bias | research only — **non-commercial** |
| ASVspoof 2021 DF | deepfake | research only — **non-commercial** |

The application logs a licence notice and shows one in the UI when you load a
non-commercial corpus. **Check each corpus's own terms before any commercial
use.** Accent-bias profiling specifically needs L2-ARCTIC, because all its
speakers read identical prompts — that is what makes a word-error-rate difference
between accents attributable to accent rather than to text difficulty.

> The mount must be `./Backend/data:/app/data`. The loaders resolve the corpus
> root from their own module path, which is `/app` inside the image, so a mount
> at `/data` is invisible to them.

---

## 7. Reading the provenance label

Every attribution result carries one of three labels, and this is the most
important thing to understand about the product.

| Label | Meaning |
|---|---|
| **`measured`** | The model produced this on this input. Trust it. |
| **`fallback`** | A stand-in, **not model output**. A reason is always given. |
| **`unavailable`** | Nothing could be produced. |

`fallback` is the system telling you the truth, not failing. An attribution can
collapse for legitimate reasons — a method that cannot be computed for a given
architecture, an out-of-memory retry that gave up — and in that situation the
overlay shows a signal-energy map instead. **An energy map looks exactly like an
attribution and means something completely different**: it shows where the sound
is loud, not what the model used.

Without the label you could not tell those apart. With it, you can.

If you see `fallback` frequently for one model and method:

1. Read the `provenance_reason` field — it names the cause.
2. Try another method. All four have different failure conditions, and their
   disagreement is itself informative.
3. Check `docker compose logs worker` for out-of-memory retries.

**Never draw a conclusion from a `fallback` overlay.** The faithfulness auditor
refuses to score one, for the same reason.

---

## 8. Operations

### Health

```bash
curl -s http://127.0.0.1:8000/health          # broker reachability
curl -s http://127.0.0.1:8000/health/workers  # registered worker families
curl -s http://127.0.0.1:8000/metrics         # task and cache counters
```

`/health/workers` must list all five families. A missing family means that
family's queue is unserved and its jobs will sit forever.

### Logs

```bash
docker compose logs -f api
docker compose logs -f worker
docker compose logs --tail 200 worker | grep -i error
```

Logs are single-line JSON by default (`LOG_FORMAT=json`). They deliberately
contain **no filenames, session ids or transcripts** — only a job id, family,
queue, worker pid and model id.

### Backup

Two things are worth backing up. Neither is audio.

```bash
# Durable metadata (analysis records, model provenance, bias reports)
docker compose exec mongo mongodump --archive=/tmp/audiolit.archive --db audiolit
docker compose cp mongo:/tmp/audiolit.archive ./audiolit-$(date +%F).archive
```

`uploads/` holds transient audio that is purged on a TTL — do not treat it as
storage. Redis needs no backup; it is entirely recomputable.

### Capacity

| Resource | Guidance |
|---|---|
| RAM | ~2–3 GB per loaded model family. Four families plus overhead: 8 GB minimum. |
| VRAM | Everything running at once targets 3–5 GB. Exceeding it falls back to CPU **and says so** in the result. |
| Disk | 1–2 GB per model in the cache volume; up to ~100 GB if you provision all corpora. |
| Concurrency | Model families run in parallel with each other, one job deep each. Beyond a handful of simultaneous users the queue is the bottleneck, not the CPU. |

---

## 9. Troubleshooting

### Everything looks up but the UI cannot reach the API

Check `VITE_API_BASE_URL` was set at **build** time, not runtime:

```bash
docker compose exec web grep -ro "127.0.0.1:8000" /usr/share/nginx/html | head -1
```

No match means the bundle was built with a different address. Rebuild:
`docker compose up --build -d web`.

### My session resets on every page load

Host-name mismatch. Use `127.0.0.1` for both the UI and the API, or a real host
name for both. `localhost` and `127.0.0.1` are cross-site for a `SameSite=Lax`
cookie.

### A job never finishes

```bash
curl -s http://127.0.0.1:8000/health/workers
```

If a family is missing, its worker died. Check `docker compose logs worker` and
restart: `docker compose restart worker`.

A restarted worker waits up to 150 seconds for the previous holder's GPU lock to
lapse — that pause is expected, not a hang.

### Every attribution says `fallback`

Expected for some model-and-method combinations; see §7. If it happens for
*everything*, check the worker log for out-of-memory retries and try a smaller
model (`whisper-tiny` or `whisper-base`).

### Transcription is nonsense, or in the wrong language

Almost always a sample-rate problem in a custom model, or a checkpoint whose
head does not match what the loader expects. Try a built-in model on the same
clip: if that is correct, the custom checkpoint is the cause.

### An upload is rejected

| Status | Cause |
|---|---|
| `400` | disallowed file extension |
| `413` | over the size cap, **or** over the duration cap |
| `422` | the bytes do not decode as audio — corrupt, or not really audio |

The message names which. A `413` on a small file means it was the *duration*
cap.

### Out of disk

```bash
docker system df
docker compose down && docker volume rm audiolit-workspace_model-cache
```

Removing the model cache forces a re-download on next start but frees the most
space.

### Workers die under load

Redis memory. Confirm the cap is 2 GB and the policy `volatile-lru` (§5).

---

## 10. Security notes

**What the deployment does:**

- Uploads are validated by MIME type, extension, size, duration, and by actually
  decoding the bytes. Filenames are always regenerated as UUIDs, so an uploaded
  name is never used as a path.
- Model weights are accepted in `safetensors` format **only** — never pickle,
  which executes code on load — and the check happens before any weight is
  downloaded.
- Model versions are pinned to a commit hash, never a branch.
- CORS is restricted to an allow-list, never a wildcard.
- Uploaded audio is purged on a TTL.
- Logs contain no filenames, session ids or transcripts.
- The diagnostic endpoint is disabled by default.

**What is yours to do:**

- **TLS.** The images serve plain HTTP; put a reverse proxy in front (§3).
- **Authentication.** There is no user authentication. Anyone who can reach port
  8080 can use the workbench. Put it behind your own access control if it is not
  on a trusted network.
- **Network exposure.** Redis on 6379 is published for convenience. Remove that
  port mapping if the host is not isolated.
- **Dataset licences.** Several corpora are non-commercial (§6).

### Keeping dependencies current

```bash
docker compose exec api pip-audit
cd Frontend && npm audit --omit=dev
```

Both should report clean. CI runs them on every build.

---

## 11. Verifying a deployment you did not build

The test suites ship with the release precisely so that you can check your own
installation rather than take it on trust.

```bash
# Backend suite, inside the running container
docker compose exec api python -m pytest -q

# API contract from an independent HTTP client
npx newman run Backend/apitests/AudioLIT.postman_collection.json \
  --env-var baseUrl=http://127.0.0.1:8000

# Browser layout and accessibility (needs Node 20 on the host)
cd Frontend && npm ci && npm run test:e2e
```

A green backend suite plus a green newman run is strong evidence the deployment
is sound. The one thing they cannot verify is model *quality* on your audio —
for that, run a clip you know the answer to and read the result.
