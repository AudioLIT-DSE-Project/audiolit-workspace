# IEEE Report §4 — System Architecture, Infrastructure and Data Tier

LIT-265 draft — source for the shared Overleaf project. Deliverable enters the
repo through the LIT-204 (IEEE compilation) PR; this file is the authoring
draft. Target 1,200–1,500 words, one original figure, one table.

---

## IV. System Architecture, Infrastructure and Data Tier

### A. Architecture Overview

AudioLIT is organised into five layers that mirror the way a single
inference actually flows: **api** (HTTP and WebSocket surface), **domain**
(model registry, explanation strategies, acoustic profiler, mutation engine,
bias profiler, faithfulness auditor), **orchestration** (one RQ task fabric),
**infrastructure** (settings, Redis, MongoDB, sessions, cache keys), and the
**workspace** frontend (React client). The layering exists to enforce one
property above all: *the gateway never loads a model*. A FastAPI request only
validates input, enqueues work, and returns progress handles; every PyTorch
tensor is produced on a worker. The domain layer is free of framework
dependencies so it can be tested without a broker, a Redis server, or a GPU
[SRS §3.5.1].

[Figure 1 here — original block diagram: the five layers stacked, with a
request descending api → orchestration → Redis; worker families fanning out
to the domain services; results returning through the same cache. Draw this
figure yourself; do not reproduce the SAD diagram.]

### B. Asynchronous Fabric

The orchestration layer is **RQ + Redis only**; Celery, inherited with the
ECHO baseline, was removed project-wide and must not return [SAD RQ decision,
erratum E2]. The fabric provisions **one queue per model family** — ASR, SER,
deepfake, attribution, and mutation — where the four model families are
`gpu_bound` with concurrency pinned to one so no two model jobs contend for
VRAM at once (SAD constraint C2); the CPU-only mutation family scales out.
A task is enqueued per family, and multi-task requests are fanned out and
fanned back in through an aggregator service that publishes progress on each
stage. The logic for "whose turn is it" lives entirely in Redis keys shared
by workers, so a restarted worker cannot double-run a job.

### C. Caching

Two cache systems coexist, and the report states this honestly.

The SRS FR4 scheme is a content-addressed manager (`RedisCacheManager`): a
key of `SHA-256(audio bytes ‖ model id ‖ task ‖ canonicalised params)`, values
serialised with msgpack and lz4-compressed when over 1 MB, a `SET NX`-based
**dedup lock** so concurrent misses compute once, and a 24 h TTL under a
2 GB LRU cap with persistence disabled — every entry is recomputable from
audio + model + task + params [SRS §3.4.3]. Its primary consumer is the
saliency/results path.

The cached values that serve the *requests* most users hit still come out of
the inherited MD5-of-path key scheme in `app/infrastructure/cache_keys.py`.
That scheme keys on where a file sits, not what it contains, so identical
audio at two paths caches twice and an in-place edit can serve a stale entry.
This is a known, tracked gap: the module now also streams a SHA-256 of the
audio bytes and writes content-addressed spellings first, keeping the
location-derived keys as fallback during the transition, but the FR4 key is
not yet the single key scheme on the hot path. (At the viva: point at
`RedisCacheManager._generate_key` and at `cache_keys.both_hashes`.)

### D. Metadata Tier

MongoDB stores **metadata only** — file-path references and analysis
records, never audio bytes or personal profiles [SRS SR4, SAD C4]. Ordinary
analysis records carry a 24 h TTL; bias reports are retained permanently.
The tier degrades gracefully under failure: writes become recorded no-ops
when MongoDB is unreachable for the server-selection timeout, so the analysis
workflow keeps running rather than crashing into a hard dependency on a
secondary store.

### E. Reliability and Security

Three real worker failure modes were found and fixed during integration
testing (LIT-254), each worth its line in the final report: a blocking Redis
read that could outlive RQ's own worker timeout and strand a job; a
lock-prefix mismatch where one worker's lock key did not match the family
fabric another worker used, letting duplicate work run concurrently; and a
Redis OOM kill at the inherited 256 MB limit, which the 2 GB cap in the
deployment configuration now prevents.

On the security track: SR6 items (an unauthenticated debug/session endpoint
and permissive CORS inherited from ECHO) are fixed in the LIT-223 changes,
which were under review at the close of writing — the debug route is removed
on that branch and the CORS allowlist tightened. SR7 scanning is
two-parts: the container images are vulnerability-scanned at CRITICAL
severity in CI on every build (LIT-203, active), while dependency-level
scanning of the Python and JavaScript lockfiles is scheduled (LIT-260).

### F. Deployment

The full stack ships as five containers in one Compose project [SAD §7]:
the web interface (nginx, port 8080), the FastAPI gateway (port 8000), the
RQ worker fleet, Redis, and MongoDB. Workers and gateway share the model
cache and dataset volumes; nothing model-shaped is baked into an image. The
backend image installs only CPU-flavoured torch by default, so `docker
compose up` works on a machine with no GPU; a Compose override enables an
NVIDIA GPU when present. Images run as a non-root user, expose a `/health`
check, and carry no test or load-test tooling.

### G. Performance Results

Table I reports the SRS §3.4.1 engineering targets against measured
p95 values drawn from the test-and-evaluation ledger [T&E §5]. All model
numbers are CPU measurements unless marked; the GPU column is included where
a T4 was available. Cache-hit retrieval is measured directly (the cache-read
SLA is enforced by test — a cached tensor read must complete under 10 ms).

| Operation | SRS target | Measured p95 (this build, CPU unless noted) |
|---|---|---|
| Cached (repeat) tensor retrieval | < 10 ms | [fill from T&E §5] |
| API response, cached request | < 200 ms | [fill from T&E §5] |
| Cache miss → task enqueue | < 50 ms | [fill from T&E §5] |
| Cold ASR, Whisper-base, 15 s audio | < 3 s (GPU) | [fill from T&E §5] |
| Multi-task ASR+SER+ADD | < 8 s cold | [fill from T&E §5] |
| IG/saliency attribution, 15 s clip | < 8 s | [fill from T&E §5] |
| Canvas mutation, UI response | < 500 ms | [fill from T&E §5] |
| Accent bias profiling | < 30 s | [fill from T&E §5] |

All timing targets were validated on CPU hardware throughout development; the
GPU figures are targets inherited from the SRS and are expected to improve on
a T4 or better. Nothing in the measured set required more than the 2 GB Redis
cap or the ~100 GB dataset working footprint.

---

## Notes for the integrator (LIT-204)

- Keep the honesty clauses: the FR4-vs-MD5 cache gap (§C), the debug-route
  removal status (§E), and the dependency-scan TODO (§E) must be phrased as
  "as of writing", and re-checked on the merge commit date as the issue body
  instructs.
- Figure 1 must be drawn fresh (original), not a frame of the SAD diagram.
- [MEASURED p95] cells are to be filled from the T&E document (§5, LIT-189)
  before submission; do not invent numbers.