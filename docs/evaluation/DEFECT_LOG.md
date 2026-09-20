# Defect Log & Remediation Record

**Issue:** LIT-253 · **Companion to:** `docs/evaluation/TESTING_AND_EVALUATION.md`
**Scope:** defects found and remediated on the ASR / SER / ADD inference and
explanation paths during Phase 3 verification. Last updated 2026-09-18.

Every row below was verified against the repository: the commit hash resolves on
`develop`, the named test exists in the named file, and the "no guarding test"
section is the result of an explicit search, not an omission.

---

## 1. How these defects were found

Three sources, in descending order of yield:

1. **End-to-end replay** — running one audio clip through the same model twice
   and diffing the two responses. This is what surfaced D04, D05 and D07; a
   unit test that calls the function directly cannot see any of them.
2. **Load testing** (`Backend/loadtests/locustfile.py`) — sustained enqueue
   traffic, which is the only thing that made D08 and D09 reproducible.
3. **Reading the fix, not the symptom** — D06 was found by a test written for
   D07 that then failed for an unrelated reason.

The recurring shape is worth naming: **a contract asserted on one side and
never checked on the other** — cache key vs. payload shape, function signature
vs. call site, response model vs. service, lock-key spelling vs. constant.
None of these produce an exception; they produce a plausible wrong answer.

---

## 2. Defect table

| ID | Symptom | Root cause | Fix | Guarding test | Commit / PR | Severity |
|----|---------|------------|-----|---------------|-------------|----------|
| **D01** | Whisper Grad-CAM heat-map rendered flat/empty for every clip. | Attribution target was total encoder energy — a non-class-discriminative scalar. The pre-ReLU map came out uniformly negative, so the ReLU zeroed all of it. | `_build_whisper_gradcam_target()` scores the decoded transcript's token logits; the energy target is retained as an explicit, labelled fallback. | `test_the_whisper_target_is_the_transcript_not_encoder_energy`, `test_a_class_style_target_yields_a_varying_map`, `test_whisper_gradcam_target_scores_the_transcript` (`test_inference_consistency.py`) | `6e51cbd` / #127 | High |
| **D02** | LIME attributions were all-zero, so the panel showed no explanation at all. | Captum's default surrogate is `SkLearnLasso(alpha=0.01)`; on standardised mel features L1 drove every coefficient to exactly zero. | `_lime_surrogate()` returns `SkLearnRidge(alpha=0.01)` — L2 shrinks but does not annihilate. | `test_surrogate_is_not_the_default_lasso`, `test_surrogate_recovers_a_known_linear_signal` (`test_inference_consistency.py`) | `6e51cbd` / #127 | High |
| **D03** | LIME output was high-frequency speckle with no interpretable time structure. | Perturbation operated on individual mel cells, so no coherent region was ever occluded. | `_time_band_feature_mask()` groups frames into ≤32 contiguous time bands, passed as `feature_mask` at all three LIME call sites (Whisper, SER, ADD). | `test_time_band_mask_groups_frames_not_cells`, `test_time_band_mask_handles_a_2d_waveform_input`, `test_bands_are_capped_by_available_frames` (`test_inference_consistency.py`) | `6e51cbd` / #127 (Whisper, SER); `5aab4b0` / #132 (ADD) | Medium |
| **D04** | SER attention panel displayed a confident attention pattern for checkpoints that expose no attention at all. | `model_loader_service` silently substituted a *different* model (`jonatasgrosman/wav2vec2-large-xlsr-53-english`, and an "enhanced" `facebook/wav2vec2-base-960h`) and returned its attention as the selected model's. | Both substitutions removed. The loader now returns real attention or none, with an explicit `attention_is_fallback` flag and FR17 provenance. | `test_real_attention_reaches_the_caller`, `test_a_synthesised_pattern_is_never_reported_as_measured`, `test_no_other_checkpoint_is_loaded_to_fill_the_gap` (`test_inference_consistency.py`) | `bbd3aaa` / #127 | **Critical** |
| **D05** | Word-level timestamps disagreed with the transcript shown above them. | `transcribe_whisper_with_timestamps` ran a second decode; its word labels could legitimately diverge from the canonical transcript, and the divergence was hidden. | Segments are relabelled from the canonical transcript; a word-count mismatch sets `word_labels_diverged` rather than being force-aligned. | `test_segments_are_relabelled_from_the_canonical_transcript`, `test_mismatched_word_counts_are_flagged_not_forced`, `test_agreeing_decodes_are_left_alone` (`test_inference_consistency.py`) | `bbd3aaa` / #127 | High |
| **D06** | A custom SER checkpoint was fed through Whisper's embedding extractor, yielding meaningless embeddings. | Model family was resolved by substring match on the model id. `myorg/custom-ser` matches neither `"whisper"` nor `"wav2vec"`, so it fell through to the Whisper branch by default. | `_embedding_family()` resolves family via the model registry; an unknown id raises instead of defaulting. | `test_custom_ser_checkpoint_reaches_the_ser_extractor`, `test_add_model_does_not_get_the_ser_extractor`, `test_family_alias_is_not_passed_as_a_hub_id` (`test_inference_consistency.py`); `test_a_broken_custom_model_raises_rather_than_using_whisper_base` (`test_custom_model_fidelity.py`) | `f2a1540` / #127 | High |
| **D07** | Selecting a second SER checkpoint returned the first checkpoint's prediction. | The SER cache key omitted the model identifier, so all checkpoints collided on one key. | Cache keys are namespaced per checkpoint; the default model keeps its historical spelling so existing entries stay valid. | `test_ser_cache_keys_differ_per_model`, `test_default_keeps_its_historical_spelling`, `test_a_second_model_is_not_answered_by_the_first` (`test_custom_model_fidelity.py`); `test_a_second_model_does_not_overwrite_the_first` (`test_inference_consistency.py`) | `d40fa95` / #124 | **Critical** |
| **D08** | RQ workers exited silently mid-job under sustained load; jobs stayed queued forever with no error logged. | The worker reused the request-path Redis connection, which sets a fail-fast `socket_timeout`. The worker's blocking dequeue (`worker_ttl 420` ⇒ `dequeue_timeout 405 s`) outlasts that timeout, so the read raised and the worker died. | `get_worker_redis_connection()` returns a separate connection with `socket_timeout=None` and `socket_keepalive=True`; the request path keeps its fail-fast timeout. | `test_worker_read_timeout_outlasts_the_blocking_dequeue`, `test_request_path_connection_keeps_its_fail_fast_timeout` (`test_task_orchestrator.py`) | `5aab4b0` / #132 | **Critical** |
| **D09** | After an unclean worker exit, that model family stayed permanently blocked — every later job for it hung. | `_cleanup_stale_worker_locks` built its scan pattern from a hardcoded literal `audiolit:worker_lock:`, while `WORKER_LOCK_PREFIX` spells it with a hyphen. The scan matched nothing, so no stale lock was ever purged. | The key is derived from `WORKER_LOCK_PREFIX` instead of being written out again. | **None** — see §3 | `5aab4b0` / #132 | High |
| **D10** | The ADD saliency panel offered only a subset of the XAI methods that actually work for it. | A UI gate added in `bfc71dd` filtered methods by declared model support; the support table was narrower than reality, so working methods were hidden. | Gate removed; the four-method array restored. The gate returned once via a `main`→branch merge and was removed a second time. | **None** — see §3 | `5aab4b0` / #132, reverting `bfc71dd` / #127 | Medium |
| **D11** | Redis evicted cache entries during ordinary dataset warm-up, so warmed results were gone before they were read. | `docker-compose.yml` capped Redis at `--maxmemory 256mb`; SRS §3.4.3 specifies 2 GB. | `--maxmemory 2gb`. | **None** — see §3 | `5aab4b0` / #132 | Medium |
| **D12** | A failed attribution was rendered identically to a successful one — the user could not tell a real explanation from a fallback. | No provenance was attached to attribution results. | FR17 provenance contract: every attribution carries `measured` / `fallback` / `unavailable`, and the UI renders the three states distinctly. | `test_a_synthesised_pattern_is_never_reported_as_measured`, `test_prediction_only_calls_carry_no_provenance_noise` (`test_inference_consistency.py`); `test_result_carries_the_flag` (`test_custom_model_fidelity.py`) | `4abe530` / #118 | **Critical** |

All backend test files cited above live in `Backend/tests/`.

---

## 3. Defects with no guarding test

These three are fixed in code but **nothing would catch a regression**. Listed
explicitly rather than left as a silent gap; each is a candidate follow-up
issue.

| ID | Why there is no test | What a test would need |
|----|----------------------|------------------------|
| **D09** — worker lock prefix | Verified by search: no test in `Backend/tests/` references `_cleanup_stale_worker_locks`. The bug is a string-literal mismatch, invisible to any test that imports the constant rather than re-typing it. | Write a stale lock using `WORKER_LOCK_PREFIX`, run the cleanup, assert the key is gone. The test must build the key from the constant, or it reproduces the bug it is meant to catch. |
| **D10** — ADD XAI method gate | Frontend; no test references the XAI method list or `PredictionPanel`'s method array. | A component test asserting all four methods render for an ADD model — or better, a Playwright assertion in `Frontend/e2e/dataflow.spec.ts`, since this defect regressed once already via a merge. |
| **D11** — Redis memory cap | Verified by search: `maxmemory` appears in no test file. It is deployment configuration, not application code, so no unit test observes it. | A config-lint check asserting `docker-compose.yml`'s cap matches the SRS §3.4.3 figure. |

---

## 4. Open defect — confirmed, not yet fixed

**D13 — the accent-bias diagnostic transcribes accented English as another
language.** Found on 2026-09-18 while building the LIT-252 evaluation runner.
Listed separately from the table above because it is diagnosed and its fix is
verified, but **no fix has been applied** — the change lives in LIT-170's code
and would move the published FR15 figures, which is a scope decision rather than
a bug fix to slip into an unrelated branch.

| Field | Detail |
|-------|--------|
| **Symptom** | Cohort mean WER is wildly inflated: Vietnamese 1.3328 against a median of 0.1742, Arabic 1.1285 against 0.1603. A WER above 1.0 means insertions outnumbered the reference words. |
| **Root cause** | `make_whisper_transcriber` (`app/domain/accent_bias_profiler.py:151`) builds the HF pipeline with no `language` argument and calls it with no `generate_kwargs`, so Whisper runs language ID per utterance. On heavily accented English it selects the speaker's L1 and transcribes into that language, then the decoder enters a repetition loop. |
| **Evidence** | 2 of 120 samples exceed WER 1.0. `TLV-arctic_b0251` (reference "They must have been swept away by the chaotic currents") returns Vietnamese text repeating "xe bánh" ~20 times, WER **22.30**. `ABA-arctic_b0066` returns Arabic text repeating "أن أرغب", WER **17.80**. Two further samples at WER 1.00 come back in Indonesian/Malay. |
| **Verified fix** | Passing `generate_kwargs={"language": "en", "task": "transcribe"}` collapses both: `TLV-arctic_b0251` 22.30 → **0.20**, `ABA-arctic_b0066` 17.80 → **0.30**. Measured directly, not inferred. |
| **Impact** | Two samples (1.7 % of the corpus) move the overall mean WER from 0.1617 to 0.4931. The bias ranking itself is affected: Vietnamese and Arabic rank worst almost entirely because of one clip each. |
| **Severity** | **High** — the diagnostic is reported as an accent-bias measurement, and for two cohorts it is substantially measuring language misdetection instead. |
| **Guarding test** | None. |

This also bears on the figures in `TESTING_AND_EVALUATION.md` §6 (mean WER 0.1353,
Δ 0.0670), which a current run does not reproduce — see `docs/DEMO_RUNBOOK.md`
§8.2. Excluding the two outliers gives 0.1617 / 0.1542, closer to §6 but still
not equal, so the language defect explains part of the gap and does not fully
close it. Both belong in one follow-up issue.

---

## 5. Verification-method note

Three findings during this work were **wrong on first measurement** and were
corrected before they reached a fix:

- An enqueue latency reported at 2100 ms (42× budget) was an artefact of
  `urllib` splitting headers and body across packets; `requests` measured
  23.6 ms. Caught because the load test said 41 ms and the contradiction was
  chased rather than averaged away.
- Acoustic profiling "taking 13 s" was a cold-start run; warm it is 0.1 s.
- A claimed ~1.8× saliency slowdown disappeared once the untouched control path
  was measured and had moved by the same factor — it was machine load.

This is recorded because it bears on how the rest of the table should be read:
no single timing observation was treated as evidence anywhere above.

---

## 6. Lessons

The defects that mattered most were not crashes. D04, D05, D06 and D07 all
returned a confident, well-formed, wrong answer, and D04 actively fabricated
evidence — it showed one model's attention as another's. In an interpretability
tool that is the worst available failure mode, because the output's whole
purpose is to be trusted as measurement. Unit tests calling functions directly
could not see any of them; each needed either a second call to compare against
or a real end-to-end path. Two structural conclusions follow: **never substitute
a model silently** — return nothing and label it, which is what the FR17
provenance contract now enforces; and **derive keys, never re-type them**, since
D07 and D09 are the same bug wearing different clothes. The three rows in §3 are
the honest remaining debt.
