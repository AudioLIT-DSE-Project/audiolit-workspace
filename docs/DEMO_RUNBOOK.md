# AudioLIT Demo Runbook

**Issue:** LIT-262 · **Audience:** LIT-197 (script), LIT-198 (screencast), LIT-200/201 (viva)
**Purpose:** run the live demo from a warm cache so no step stalls on CPU inference.

Everything below was verified against the repository on 2026-09-18. Dataset keys,
endpoint paths and payload fields are quoted from the code, not from memory.

---

## 0. Read this first — two constraints that will bite

**The batch-warmup endpoint warms a whole dataset, not a clip list.**
`POST /inference/batch-warmup` takes `dataset: str` (`inference.py:82`) and
`run_batch_dataset_warmup_task` calls `load_metadata(dataset)` over every row
(`task_orchestrator.py:763`). Its own docstring describes "multi-hour runs."
Warming `asvspoof-2021` means all 250 clips, not the 3 you are demoing. There is
no per-file filter. LIT-262 puts new warmup code out of scope, so §4 below uses
the supported path instead, and this gap is filed as a follow-up (§8).

**Never `FLUSHALL` the Redis on 6379.** It also holds another project's
`cv:demo:v1:*` keys. To reset for a cold rehearsal, delete only the `audiolit:*`
namespace (§5) or use a separate container on another port.

---

## 1. Demo clip list

All paths are relative to `Backend/data/`. Dataset keys are the registered
corpus names from `dataset_ingestion.CORPUS_REGISTRY` — use these exact strings
in API calls (`common-voice`, `crema-d`, `ravdess`, `esd`, `l2-arctic`,
`asvspoof-2021`). `librispeech` is registered but **not provisioned locally**.

### 1.1 Audio deepfake detection (FR7)

Dataset key `asvspoof-2021`, files under `asvspoof2021_df/`.

| Order | Sample ID | Label | Attack family | Why this clip |
|-------|-----------|-------|---------------|----------------|
| 1 | `DF_E_2518702` | bonafide | — (nocodec) | Clean genuine speech; establishes the baseline before any spoof. |
| 2 | `DF_E_2260389` | spoof | `traditional_vocoder` (A19) | The majority attack family in this subset (99/250); the expected-case detection. |
| 3 | `DF_E_2317763` | spoof | `neural_vocoder_autoregressive` | A harder, different attack family — shows the model is not keyed to one artefact type. |

Two further families are available if the ADD segment needs more range:
`DF_E_2712909` (`neural_vocoder_nonautoregressive`) and `DF_E_2191306`
(`waveform_concatenation`).

> The issue asks for one clip the model gets **wrong**, because a confident
> mistake makes the XAI story stronger. That clip cannot be chosen from
> metadata — it requires running the detector over the subset and reading the
> predictions. **Still to be selected during the rehearsal (§5).**

### 1.2 Canvas mutation / perturbation (FR6)

Dataset key `common-voice`, files under `common_voice_valid_dev/`.

| Order | File | Reference transcript |
|-------|------|----------------------|
| 4 | `cv-valid-dev/sample-000775.mp3` | "it must have fallen while i was sitting over there" |

Chosen because Whisper transcribes it cleanly at baseline, so a perturbation
that degrades it is unambiguous on screen rather than arguable. Demonstrate with
time masking or added Gaussian noise via `perturbation_service`; both visibly
change the transcript at moderate settings.

Backup with a shorter utterance if timing is tight:
`cv-valid-dev/sample-000179.mp3` — "worked hard just to have food and water like
the sheep".

### 1.3 Accent bias (FR15)

Dataset key `l2-arctic`, files under `l2arctic/`. The diagnostic uses six L1
cohorts at 20 samples each (120 total). Demo the **cohort comparison**, not a
single clip — the ranked table is the result.

Regenerate the figures live or beforehand with:

```bash
cd Backend && python scripts/evaluate_models.py --asr-model-id openai/whisper-base --skip-faithfulness --output ../docs/evaluation/results
```

Ground-truth transcripts must exist first, or every sample is silently skipped:

```bash
cd Backend && python -m scripts.prepare_l2arctic_transcripts
```

`Backend/data/` is gitignored, so this is re-run per machine.

> **Do not quote the mean WER without the median.** The measured run on this
> machine shows cohort means heavily skewed by a small number of clips where the
> ASR hallucinates (a sample's WER exceeds 1.0 when insertions outnumber
> reference words). See §8 — this is an open discrepancy against
> `TESTING_AND_EVALUATION.md` §6 and should not be presented as settled until
> it is resolved.

### 1.4 Speech emotion recognition (FR3)

| Order | Dataset key | File | Emotion |
|-------|-------------|------|---------|
| 5 | `crema-d` | `1010_DFA_FEA_XX.wav` | Fear — acted, high separability |
| 6 | `esd` | `0017_001346.wav` | Sad — "You woke me up!", text/emotion mismatch makes the point that SER is acoustic, not lexical |
| 7 | `ravdess` | `03-01-01-01-01-01-16.wav` | Neutral — the contrast case |

Clip 6 is the strongest demo: the words are an exclamation, the delivery is sad,
so the model cannot be getting it right from the transcript.

### 1.5 Faithfulness (FR16)

Use whichever clip from §1.4 produces the cleanest deletion curve. **To be
selected during the rehearsal** — it depends on the attribution actually
produced, and picking it from metadata would be guessing.

---

## 2. Start the stack

Use `127.0.0.1` everywhere, **not** `localhost`. On Windows `localhost` resolves
to `::1` (IPv6) first, while uvicorn's `--host 0.0.0.0` binds IPv4 only, so
`localhost:8000` intermittently fails to connect. `--host ::` does not fix this —
on Windows it binds IPv6 only.

Keeping the page and the API on the same host string also matters: the session
cookie is `SameSite=Lax`, and a page on `localhost:8080` calling an API on
`127.0.0.1:8000` is cross-site, so the cookie is dropped and custom datasets fail
with `session_id is required for custom datasets`.

```bash
# 1. Data tier — Redis + MongoDB (from Backend/)
docker compose up -d redis mongo
```

> **MongoDB is required as of LIT-255/256/257/258.** Two traps, both verified
> on 2026-09-19: the compose `mongo` service publishes **no host port**, and
> `MONGO_URL` defaults to `""`, which disables the tier. If you run the API
> natively (the steps below) rather than in a container, compose's Mongo is
> unreachable *and* the app starts anyway with the tier silently off, because
> LIT-258 degrades gracefully. For the hybrid workflow use:

```bash
docker run -d --name audiolit-mongo -p 27017:27017 -v audiolit-mongo-data:/data/db mongo:6
```

> then export `MONGO_URL=mongodb://127.0.0.1:27017` before starting the API and
> workers. Verify the tier is actually live — it should list four collections
> (`models`, `audio_samples`, `analysis_results`, `bias_reports`):

```bash
docker exec audiolit-mongo mongosh --quiet audiolit --eval "db.getCollectionNames()"
```

> **Upload limits and retention.** The API enforces SR1's caps and SR4's
> retention window with defaults that need no configuration: 100 MB, 15 minutes,
> and uploaded audio purged after 24 hours (on each upload and at startup).
> Override any of them for a demo with a long clip, or to keep the files:

```bash
export AUDIOLIT_MAX_UPLOAD_BYTES=209715200          # 200 MB
export AUDIOLIT_MAX_UPLOAD_SECONDS=1800             # 30 min; 0 disables the cap
export AUDIOLIT_UPLOAD_RETENTION_SECONDS=0          # 0 keeps every upload
```

```bash
# 2. API (from Backend/)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```bash
# 3. Workers — separate terminal, from Backend/
python -m app.orchestration.worker all
```

```bash
# 4. Frontend — from Frontend/
npm run dev
```

Confirm before continuing:

```bash
curl http://127.0.0.1:8000/health && curl http://127.0.0.1:8000/health/workers
```

> **This section changes when PR #145 (LIT-203) merges.** That PR containerises
> the full stack and adds a root-level `docker-compose.yml` plus
> `Frontend/Dockerfile` and `nginx.conf`, while keeping `Backend/docker-compose.yml`.
> After it lands, the whole stack comes up with one compose command and this
> section must be rewritten. Check whether #145 is merged before following it.

---

## 3. Warm the cache — the supported path

The demo touches roughly a dozen clips. Warming them is just requesting them
once: the cache is content-addressed (FR4), so the entry a warm-up writes is the
entry the demo click reads.

Walk the clip list once, in demo order, through the normal UI, before the
audience is watching. That is the whole procedure. It is not elegant, but it
uses only shipped behaviour and it warms exactly the clips being demoed.

Alternatively, drive the same endpoints with `curl` for each clip in §1.

## 4. Warm a whole dataset — only if you need it

If the demo will browse freely through a corpus rather than follow the script:

```bash
curl -X POST http://127.0.0.1:8000/inference/batch-warmup \
  -H "Content-Type: application/json" \
  -d '{"dataset":"asvspoof-2021","model":"whisper-base","tasks":["asr","ser","acoustic"],"cooldown_ms":100}'
```

Returns `{"job_id": "warmup_<hex>", "status": "running"}`. `cooldown_ms` throttles
between files to keep a laptop CPU from overheating on a long run — do not set it
to 0 on battery.

**Budget hours, not minutes**, and start it the night before. 250 files × 3 tasks
at CPU speed is not a pre-demo activity. Add `"saliency"` to `tasks` only if the
demo needs pre-warmed attributions; it is by far the slowest task.

---

## 5. Cold rehearsal

Reset **only** the AudioLIT namespace — re-read the warning in §0:

```bash
docker compose exec redis redis-cli --scan --pattern 'audiolit:*' | xargs -r docker compose exec -T redis redis-cli DEL
```

Then: start the stack (§2), warm the clips (§3), and click every step in demo
order confirming each responds in roughly a second or less. Watch
`GET /health/workers` — queues should sit idle during a warm run, because a
cache hit never enqueues anything. If a queue goes busy, that step missed cache.

### Rehearsal record

| Item | Value |
|------|-------|
| Date | _not yet run_ |
| Machine | _to fill_ |
| Cold → warm duration (§3 walk) | _to fill_ |
| Slowest warm step | _to fill_ |
| Every step under ~1 s? | _to fill_ |

> **This table is deliberately empty.** The rehearsal has not been run and timing
> it while another job saturates the CPU would produce a number that measures
> machine load rather than the system — a mistake already made once on this
> project, where a claimed 1.8× slowdown turned out to be background load and a
> "13 s" acoustic profile was a cold-start artefact that measures 0.1 s warm.
> Run it on an otherwise idle machine and fill this in.

---

## 6. If a worker dies mid-demo

Restart it:

```bash
cd Backend && python -m app.orchestration.worker all
```

An immediate restart is safe. `_cleanup_stale_worker_locks` purges the stale
per-family lock on startup, so a family is not left permanently blocked by a
worker that exited uncleanly. That purge was broken until recently — it built
its scan key from a hardcoded literal whose spelling did not match
`WORKER_LOCK_PREFIX`, so it matched nothing and never purged anything (defect
D09 in `docs/evaluation/DEFECT_LOG.md`). It is fixed, but **has no guarding
test**, so verify the restart works during the rehearsal rather than trusting it
live.

If the queue is still stuck after a restart, the demo fallback is to move to the
next clip — every warmed clip answers from cache without a worker.

---

## 7. Demo order summary

1. Deepfake: bona fide → traditional vocoder → neural vocoder (§1.1)
2. Canvas mutation on the Common Voice clip (§1.2)
3. Accent-bias cohort table (§1.3)
4. SER across CREMA-D / ESD / RAVDESS, leading with the ESD mismatch clip (§1.4)
5. Faithfulness deletion curve (§1.5)

---

## 8. Open items

1. **Per-clip warm-up is unsupported.** `batch-warmup` is whole-dataset only, so
   there is no way to warm the ~12 demo clips through it. §3 works around this
   with the normal request path. Worth filing as its own issue — an optional
   `files: list[str]` on `BatchWarmupRequest` would make a scripted, timeable
   pre-demo warm-up possible. Not done here: LIT-262 puts new warmup code out of
   scope.
2. **The §6 accent-bias figures do not reproduce, and the cause is a confirmed
   defect.** A measured run of `evaluate_models.py` against `openai/whisper-base`
   over all 120 L2-ARCTIC samples gives mean WER **0.6019** and discrepancy index
   **1.1676**, against the **0.1353** / **0.0670** quoted in
   `TESTING_AND_EVALUATION.md` §6.

   Root cause, diagnosed and verified: the profiler never forces the decode
   language, so Whisper transcribes heavily accented English into the speaker's
   L1 and then loops. Two of 120 samples come back as Vietnamese and Arabic with
   WER 22.30 and 17.80; forcing `language="en"` drops them to 0.20 and 0.30.
   Full detail is defect **D13** in `docs/evaluation/DEFECT_LOG.md`.

   **No fix is applied** — it changes LIT-170's code and moves published figures,
   so it needs its own issue. Until then, do not present the mean WER as an
   accent-bias result in the demo; the ranking is distorted by one clip per
   cohort. The cohort medians are sound.
3. **LIT-262's own text repeats a corrected measurement.** It cites the acoustic
   profile at "~13 s", which was a cold-start plus client artefact; warm it is
   0.1 s. The stated risk of dead air is still real for genuinely cold
   attribution, but that specific figure should not be quoted.
4. **The ASVspoof metadata has a duplicated class label** — `traditional_vocoder`
   appears as both 99 rows and 1 row, i.e. one row differs by whitespace or case.
   Harmless for the demo, cosmetic in any grouped count.
