# Viva Prep — Rahim M.I. (230506M) — Code Ownership Map, Walkthrough & Question Bank

**Issue:** LIT-263 · **SRS:** §4.3 (FR1–FR4, FR6–FR12, FR15–FR17) · **SAD:** §5.2, §6.1, §6.2
**Ground rule:** every claim below was checked against `git log`/`git blame` on `origin/develop`. A claim the examiner can disprove with one `git log` command is worse than a smaller honest claim — so this doc only claims what git proves, and section 3 says explicitly what it does **not** prove.

---

## 1. One-page ownership map (what I actually wrote)

| # | Module / deliverable | Files | Evidence (commit → PR) | What to say about it |
|---|----|------|------------------------|----------------------|
| 1 | **SHA-256 content-addressed tensor cache** (FR4) | `Backend/app/core/redis.py` — `RedisCacheManager`, `generate_audio_hash_from_bytes`, `get_audio_hash_dependency`, `cached_inference` | `cbb3a82`, `f3d3687`, `5a5537e`, `0783fb2`; tests `5e1bb5e`, `310ada6`, `1e484f4`, `0d72cb7`, `ebcd6a1` (PRs #60/#62/#70/#71) | I built the content-addressed **tensor** cache on top of the ECHO 1.0 baseline, which already cached *predictions* by MD5-of-path (`AnasSAV 522fc69`). My layer is the deterministic cache-by-hash store. See walkthrough A. |
| 2 | **Perturbation engine — audio mutations** (FR12) | `Backend/app/domain/perturbation_service.py` — `_load_waveform`, `add_gaussian_noise`, `apply_time_masking`, `apply_frequency_masking`, `apply_2d_time_freq_mask`, `apply_band_pass_filter`, `apply_pitch_shift`, `apply_time_stretch`, `apply_perturbations`, `perturb_and_save`; `Backend/app/domain/saliency_service.py` — `generate_perturbation_matrix`, 2D saliency mapping | `8f122cf`, `1e703bb` (perturbation service), `ae02fba`, `c4e9aa9` (saliency_service) | The perturbation *primitives* and the positional-parameter design are mine. The FR16 faithfulness scoring (`HighSaliencyMaskingEngine`, `compute_deletion_score`, `evaluate_downstream_degradation`) in the same file is **Tharusha's** — do not claim it. See walkthrough B. |
| 3 | **Memory / API stress profiling** | `Backend/scripts/run_memory_profile.py`, `Backend/tests/test_memory_profiling.py` | `cc59c38`, `aa7bf9b` | Evidence for the "controlled memory" performance story (SAD §11.4) and for CI stability of the inference path. |
| 4 | **MongoDB metadata tier — Tier 1** (SRS §3.10, SAD §9) | `Backend/app/infrastructure/metadata_store.py` + provision + tests | `9a08aeb` (author `Abdur Rahim`), merged via **PR #135** | The store module, lazy client, schema validators, index/TTL provisioning, graceful degradation. Tier 2/3 (metrics collection etc.) were merged separately by other members. See walkthrough C. |
| 5 | **Latency/FPS metrics synthesis** (SR6/NFR) | `Backend/app/api/routes/metrics.py`, `Backend/app/infrastructure/metrics_synthesis.py` | `4914c25` (author `Abdur Rahim`) | Process-local collector scoring observed latencies against the SRS §3.4.1 budgets; `/metrics/synthesis`, `/metrics/samples/latency`, `/metrics/samples/frame`. |
| 6 | **Testing & Evaluation document** (LIT-171) | `docs/evaluation/` | `dcf181b` (author `Abdur Rahim`), merged via **PR #134** | The evaluation plan: how each FR/NFR will be demonstrated and measured. |
| 7 | Frontend touch-up | `Frontend/src/components/visualization/SaliencyVisualization.tsx` | `bfa5ad0` | Small refactor for clarity; file has heavy ECHO 1.0 history (`AnasSAV`, `DewmikeAmarasinghe`) — present as a minor contribution, not ownership. |

**Identity note (important for the examiner's git check):** my commits on `develop` appear under **two** identities:
- `Rahim Iqbal <mirahim2003@gmail.com>` — the 2025-term work (cache, perturbation, profiling);
- `Abdur Rahim <raheemabdur203@gmail.com>` — the 2026-term MongoDB / metrics / docs commits.

If the examiner greps by name, show both. ECHO 1.0 baseline authors to be clear about: `AnasSAV`, `Chand2103`, `DewmikeAmarasinghe`.

### 2. What the issue body implies but I did NOT write — the documented split

The draft "your modules" list in LIT-263 overlaps with teammates' work. If asked about any of these, redirect precisely:

| Topic | Actually by | Commits/PRs |
|-------|-------------|-------------|
| Grad-CAM / Integrated Gradients / LIME-SHAP attribution **cores** (LIT-148/126/130) | Ravindu Pathirana | `10cd96f`, `b62c74d`, `ba4b54f` |
| XAI wiring/fixes (LIT-239/147/248/254/240) | tharusha perera | `685f88e`, `be785e3`, `b709c39`, `5aab4b0`, `4abe530` |
| Binary deepfake detection + forensic feature map (LIT-128/151/152) | Ravindu (`6f15adf`, `0dc77ba`), tharusha (`0394a28`, `9ea3163`) | — |
| FR16 faithfulness auditor engine (LIT-183/184) | tharusha perera | `87ce84a`, `32d0f31` |
| FR4.1 cache-key families + path/content hashes (`cache_keys.py`) | tharusha perera | `d40fa95`, `6ea3c99`, `b709c39`, `0a043ec` |
| RQ task orchestrator consolidation (LIT-230, `task_orchestrator.py`) | tharusha + Ravindu | `0a043ec`, `9a803f6`, `cb688c4`, … |

Before the mock viva (LIT-200), confirm with Ravindu and Tharusha who presents what on shared files (`saliency_service.py`, `perturbation_service.py`, `task_orchestrator.py`).

Careful with the orchestrator: the old `app/services/queue_service.py` (which some old doc references) was deleted as part of LIT-230 — the orchestrator now lives in `app/orchestration/task_orchestrator.py`. If you built on `queue_service.py` you would wrongly claim LIT-230; you did not.

---

## 3. ~10-minute walkthrough plan (what to show while screen-sharing)

**A. Cache (2 min)** — `app/core/redis.py`. Walk the key: `audiolit:tensor:` + SHA-256 over
`audio_bytes ‖ model_id ‖ task ‖ canonical_params_json` where params are sorted, compact
(no whitespace) and carry an injected `_cache_schema_version`. Emphasise: deterministic, so
identical requests reuse one entry; streaming 8192-byte chunked hashing in
`generate_audio_hash_from_bytes` / the FastAPI `get_audio_hash_dependency` (hash computed
*before* the payload reaches the queue, `seek(0)` after, so the file is still consumable);
msgpack with a NumPy encoder/decoder; lz4-frame compression when packed `> 1MB` with a
`LZ4:`/`RAW:` prefix; 24 h TTL via `AUDIOLIT_CACHE_TTL`; dedup `SET NX EX 60` lock so two
workers never run the same cold inference; corrupt entries are deleted and re-run as misses.

**B. Perturbation engine (2.5 min)** — `app/domain/perturbation_service.py`. `_load_waveform`
(soundfile → float32 `always_2d` → `(channels, samples)`, downmix to mono, librosa-resample
to 16 kHz per SRS FR12.3); then the mutation set (noise; time masking; frequency masking via
`torch.fft`; 2D time–frequency masking via librosa STFT `n_fft=2048/hop=512`, outer-product
mask, `istft`, length pad/trim; Butterworth band-pass `order=5 sos`; pitch shift clamped to
`±6` semitones; time stretch). `apply_perturbations` returns `(waveform, per-op records)` with
`status ∈ {applied, failed, unsupported}`; `perturb_and_save` writes a UUID-named non-destructive
file and returns WAV preview bytes.

**C. MongoDB tier (2 min)** — `app/infrastructure/metadata_store.py`. Two stores, two
lifetimes (SAD §9): Redis = ephemeral cache/queue; Mongo = durable metadata. Metadata only,
never audio bytes (C4/SR4). Four collections with `$jsonSchema` validators; unique id-indexes
+ compound `(sample_id, model_id)` / `(model_id, cohort)`; TTL on `analysis_results` only
(24 h — bias reports are permanent). Lazy `MongoClient` on first use (import never requires a
server) and graceful degradation: unreachable Mongo ⇒ writes log-and-skip, reads return empty,
`available` pings with `serverSelectionTimeoutMS` bound (SRS §3.3.1).

**D. Metrics + profiling (1 min)** — `metrics_synthesis.py` / `metrics.py`: collector scored
against SRS §3.4.1 budgets; `/metrics/synthesis` report; POST sample endpoints for E2E.
`scripts/run_memory_profile.py` shows the CPU/GPU memory envelope used to motivate the cache.

Final 2.5 min: answer questions from the bank below.

---

## 4. Question bank with answers (credible, sourced from the code)

1. **Why SHA-256 instead of the old MD5-of-path scheme?** — MD5-of-path keys the cache on *where a file sits*; the same audio at two paths caches twice and any file edit serving stale results. My `RedisCacheManager` keys on content: SHA-256 over audio bytes + model + task + canonical params, so identical requests hit one entry across sessions and DACs.
2. **How does the audio hash avoid loading the whole file?** — both `generate_audio_hash_from_bytes` and the FastAPI `get_audio_hash_dependency` stream in 8192-byte chunks and update the digest incrementally; the file handle is seeked back to 0 so downstream loading still works.
3. **What makes two requests hash to the same cache key?** — same audio bytes, model id, task name, and *canonical* params. Params are made canonical by `sort_keys=True, separators=(',',':')` plus an injected `_cache_schema_version`, so `{"noise":0.1,"a":1}` and `{"a":1,"noise":0.1}` collide as intended.
4. **How do you stop duplicate cold runs when many workers miss at once?** — a lock key `{cache_key}:lock` is acquired with `SET NX EX 60`. The loser sleeps 1 s, re-reads the cache, and falls back to executing anyway (with a warning) if the winner still hasn't stored — an availability-over-efficiency trade.
5. **What exactly is stored in Redis?** — msgpack-packed values (`use_bin_type=True`) with a custom NumPy encoder carrying `dtype`/`shape`/raw bytes; payloads over 1 MB are lz4-frame compressed. Values are prefixed `LZ4:` or `RAW:` so deserialisation is unambiguous.
6. **TTL and eviction?** — 24 h default (`AUDIOLIT_CACHE_TTL`), set with Redis `EX`, so entries are LRU-evicted by Redis's own maxmemory policy.
7. **What happens on a corrupt cache entry?** — `get` catches the deserialisation error, logs, `delete`s the key and returns None — a self-healing miss rather than a 500.
8. **How do you load an audio file for perturbation and what constraints do you enforce?** — soundfile `dtype="float32", always_2d=True`, transposed to PyTorch `(channels, samples)` orientation; stereo downmixed to mono; anything not at 16 kHz resampled with librosa — the SRS FR12.3 16 kHz-mono contract for downstream inference.
9. **Enumerate the perturbations and the DSP behind each.** — Gaussian noise (`randn * level`); time masking (zero a `[start%, end%]` index window); frequency masking (zero bins in `[f_low, f_high]` of the torch FFT and IFFT back); 2D time–frequency mask (librosa STFT, boolean regions via `np.outer`, zeroed, `istft`, length padded/trimmed); band-pass (`scipy.signal.butter` order 5 in SOS form); pitch shift (librosa, clamped ±6 semitones, 30 s cap); time stretch (librosa). Unknown/edge cases return `status: "unsupported"/"failed"` per record, never a hard crash.
10. **How does the route know the perturbation succeeded?** — `apply_perturbations` returns one record per op `{"type","params","status","error?"}` with `status ∈ applied|failed|unsupported`; `perturb_and_save` also returns WAV preview bytes (`export_to_wav_bytes`, PCM_16) and computed `duration_ms`.
11. **Why does `perturb_and_save` never overwrite the original?** — it writes to `output_dir/{stem}_perturbed_{8-hex-uuid}.wav`, and it resolves dataset-relative paths through `resolve_file(dataset, file_path, session_id)` so cross-session access is denied.
12. **What is stored in MongoDB vs Redis?** — Mongo carries only durable metadata: models (id/name/architecture/revision/weight_digest/hf id), audio sample records (file-path reference, never bytes), analysis results (prediction + `redis_tensor_key`, never the tensor), and bias reports. Redis stays the temporary cache + job queue (SAD §9).
13. **Why do analysis records expire but bias reports do not?** — a TTL index on `analysis_results.created_at` (24 h, `MONGO_ANALYSIS_TTL_HOURS`) matches the recomputable-cache lifetime; `bias_reports` intentionally has no TTL index so audit artefacts survive.
14. **What happens to the app if MongoDB is down?** — nothing fatal: `_try_preflight` pings first; writes become logged no-ops, reads return empty/None, `available` reports state for health checks. Mongo is not a request-path dependency (SRS §3.3.1).
15. **Why a lazy client?** — constructing `MetadataStore` must never require a live server (import-time safety); the `MongoClient` is built on first use, and tests inject `mongomock` clients to bypass Mongo entirely.
16. **Your settings — how are indexes provisioned?** — `ensure_schema` creates the four collections with `$jsonSchema` validators (falling back to plain create under mongomock), unique id-indexes, the compound `(sample_id, model_id)` and `(model_id, cohort)` indexes, and the TTL index — all one-time and idempotent.
17. **What is LIT-189 exactly?** — an in-process latency/FPS collector (`metrics_synthesis.collector`) that scores observed latencies per SRS §3.4.1 operation and canvas frame renders against budget; `GET /metrics/synthesis` returns the scored report and the two `POST /metrics/samples/*` endpoints let CI/E2E pump real measurements in.
18. **What did the memory profiling script show, and why does it matter?** — `scripts/run_memory_profile.py` measured CPU/GPU memory across the inference + cache path; together with the 24 h TTL it supports the SAD §11 performance story of "fast repeated requests + controlled memory", which is exactly the design goal the LRU Redis cache (SAD §11.4) exists to meet.
19. **What is in the LIT-171 evaluation document?** — the testing & evaluation plan mapping each FR/NFR to a concrete demonstration/measurement, merged via PR #134.
20. **Which big features did you NOT contribute?** — the XAI attribution cores (Grad-CAM/IG/LIME-SHAP — Ravindu), the deepfake/forensic stack (Ravindu + Tharusha), FR4.1 cache-key families and FR16 faithfulness auditing (Tharusha). I can explain how my cache/perturbation code *feeds* those, but ownership of those modules is theirs.

### 5. Out of scope — do not present these as delivered (SRS §4.4)

If asked, say "documented future work":
multi-model side-by-side comparison (FR5, demoted to stretch), ADDSegDiff, fingerprinting,
per-demographic SER/ESD, insertion/infidelity/IoU metrics, bias-report export. The (committed)
deletion/degradation-style faithfulness work we do have lives in `perturbation_service.py` and
is Tharusha's.