# Chapter 9 — XAI implementation: the saliency service

Chapter 2 covered the theory of the four attribution methods. This chapter
covers the 1427 lines that turn a model, a clip and a method name into
something a browser can draw.

The pipeline has more stages than you would expect, and each one exists because
of a specific problem.

---

## 9.1 Entry point and dispatch

```python
def generate_saliency(audio_file_path: str, model: str, method: str = "gradcam",
                      existing_prediction: Dict = None) -> Dict:
    model_type = detect_model_type(model)

    # Lock keyed by the actual shared nn.Module identity each branch below
    # will mutate, not by the raw `model` string - whisper aliases
    # ("whisper-base" / "openai/whisper-base") must serialize against each
    # other since they resolve to the same cached instance, and every
    # wav2vec2/SER call shares the one emo_model singleton regardless of
    # `model`. Shared with inference_service.run_inference via
    # model_lock_key() - see its docstring for why plain inference must
    # serialize against this too, not just saliency-vs-saliency.
    lock_key = model_lock_key(model)
    if lock_key is None:
        raise ValueError(f"Unsupported model: {model}")

    with lock_for_model(lock_key):
        if method == "gradcam" and model_type == "add":
            return generate_add_gradcam_saliency(audio_file_path, model)

        if model_type == "whisper":
            return generate_whisper_saliency(audio_file_path, model, method, existing_prediction)
        elif model_type == "wav2vec2":
            ser_model_id = None if model == "wav2vec2" else model
            return generate_wav2vec2_saliency(
                audio_file_path, method, existing_prediction, model_id=ser_model_id
            )
        elif model_type == "add":
            return generate_add_saliency(audio_file_path, model, method)
        else:
            raise ValueError(f"Unsupported model: {model}")
```

**The lock key is the model *instance*, not the model *name*.** That
distinction is the whole reason `model_lock_key` exists as a separate function.
`"whisper-base"` and `"openai/whisper-base"` are two names for one cached
`nn.Module`; two requests using different names must still serialise against
each other. And every SER request shares one `emo_model` singleton no matter
what `model` string arrived, so they all map to the single key
`"wav2vec2:ser-emotion"`.

Locking on the raw string would let two spellings of the same model run
concurrently against one shared module — which is exactly the corruption §2.10
describes.

**`ser_model_id = None if model == "wav2vec2" else model`** translates the
generic built-in name to "use the default", and passes anything else through as
a checkpoint id. The `None` is meaningful: it means "default", not "missing".

The `method == "gradcam" and model_type == "add"` special case comes first
because the deepfake classifier has a dedicated Grad-CAM path that operates on
a 2-D spectrogram rather than a 1-D timeline.

---

## 9.2 Stage 1 — audio and prediction

```python
    if existing_prediction and "chunks" in existing_prediction:
        data = existing_prediction
        audio = data["audio"]
        chunks = data["chunks"]
        logger.info(f"Using existing prediction with {len(chunks)} chunks")
    else:
        logger.info("Transcribing audio with timestamps for saliency analysis")
        data = transcribe_whisper_with_timestamps(audio_file_path, model_size)
        audio = data["audio"]
        chunks = data["chunks"]
```

Attribution over ASR needs the *words and their timings* — an attribution map
is only interpretable if you can say which word each region belongs to.

`existing_prediction` lets the route pass a cached prediction in, avoiding a
second transcription. The route does exactly that:

```python
prediction_cache_key = f"{request.model}_{file_content_hash}"
existing_prediction = await get_result(request.model, prediction_cache_key)
```

This is a real optimisation: transcription is ~2 seconds, and the saliency panel
opens on a clip the transcript panel has usually already processed.

---

## 9.3 Stage 2 — duration capping

```python
    if isinstance(audio, (list, tuple)):
        audio = np.asarray(audio)
    if hasattr(audio, "shape") and audio is not None:
        max_seconds = MAX_SALIENCY_SECONDS_SHAP if method == "shap" else MAX_SALIENCY_SECONDS
        max_len = int(max_seconds * 16000)
        if len(audio) > max_len:
            audio = audio[:max_len]
            # Keep only chunks inside the window
            chunks = [c for c in chunks if c.get("timestamp", [0, 0])[0] < max_seconds]
```

12 seconds, or 6 for SHAP. Attribution cost grows with input length, and SHAP's
sampling multiplies it.

**Truncating the audio without truncating the chunks would be a bug.** Chunks
beyond the window refer to audio that is no longer there, so the segment loop
later would map them onto out-of-range frames. The filter keeps them
consistent.

The `isinstance(audio, (list, tuple))` coercion handles a cached prediction
that round-tripped through JSON, where a NumPy array became a list. A real
consequence of caching: what you store is not exactly what you get back.

---

## 9.4 Stage 3 — features, and the model that must match

```python
    # Attribute over the checkpoint the user selected, not whisper-base -
    # a heatmap from a different model than the prediction it explains is
    # worse than no heatmap (FR1, FR8).
    processor, model = get_whisper_base_models(resolve_whisper_model_id(model_size))

    device = next(model.parameters()).device
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    input_features.requires_grad_(True)
```

The substitution rule again, and here the consequence is sharpest: a heatmap
explaining *model A's* prediction, computed from *model B's* gradients, is not
a degraded explanation. It is a picture of a different model. Worse than no
heatmap.

Three lines of PyTorch idiom worth knowing:

- **`next(model.parameters()).device`** — ask the model where it lives rather
  than assuming. The model may have been moved to CPU by the VRAM fallback
  (§5.7), and inputs must be on the same device as the weights or you get a
  device-mismatch error.
- **`.to(device)`** on the inputs, for the same reason.
- **`requires_grad_(True)`** on the *input*. This is the line that makes
  attribution possible: normally only parameters track gradients. Asking for
  the input's gradient is what turns training machinery into an explanation
  (§1.4).

And the fallback target:

```python
    def model_forward(inputs):
        # Reduce to a scalar per batch: energy of encoder activations
        enc = model.encoder(inputs).last_hidden_state  # [B, T, H]
        return enc.pow(2).mean(dim=(1, 2))             # [B]
```

Attribution methods need a scalar per batch item. This one is encoder energy —
acceptable for IG, LIME and SHAP (which do not require class-discriminativity
the way Grad-CAM's ReLU does), and the reason Grad-CAM needed a different
target entirely (§2.3).

---

## 9.5 Stage 4 — method dispatch

Covered in Chapter 2. Summarised with the one implementation detail each:

**Grad-CAM** — builds a transcript-score target, computes the CAM over the last
conv layer, then has to get from the conv layer's resolution to the input's:

```python
            cam_tensor = torch.from_numpy(cam_np).to(device=input_features.device,
                                                     dtype=input_features.dtype)
            if cam_tensor.dim() == 1:
                cam_tensor = cam_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, T_conv]
                interp = torch.nn.functional.interpolate(
                    cam_tensor, size=input_features.shape[-1],
                    mode="linear", align_corners=False
                ).squeeze(0)  # [1, T_input]
                attributions = interp.repeat(input_features.shape[-2], 1)  # [80, T_input]
            else:
                attributions = cam_tensor
```

The conv layer's output is shorter than the input (stride downsampling), so the
CAM is linearly interpolated back up. `unsqueeze` twice because `interpolate`
wants `[batch, channels, length]`. `repeat` broadcasts the 1-D time map across
all 80 mel bins — Grad-CAM at this layer gives a *time* attribution, not a
time-frequency one, and copying it across frequency is honest about that:
every frequency at a given time gets the same value, so the visualisation shows
vertical stripes rather than implying frequency resolution it does not have.

**Integrated Gradients** — step count adapts to hardware, with an OOM retry:

```python
        try:
            ig = IntegratedGradients(model_forward)
            attributions = ig.attribute(input_features, n_steps=n_steps,
                                        internal_batch_size=internal_batch_size)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning("First attempt failed, trying with even lower memory settings...")
                n_steps = 8
                ...
            else:
                raise
```

Note `else: raise` — a non-OOM `RuntimeError` is not silently retried.

**LIME** — the time-band feature mask and the Ridge surrogate (§2.4).

**SHAP** — `GradientShap` with a zero baseline and an OOM retry at fewer
samples, degrading to `attributions = None` with a recorded reason rather than
raising.

**Unknown method** — raises:

```python
    else:
        raise ValueError(f"Unsupported saliency method '{method}'. Supported methods: 'gradcam', 'integrated_gradients', 'lime', 'shap'")
```

Never falls back to another method. A LIME-labelled Grad-CAM map is the
category of defect this project treats as most serious.

---

## 9.6 Stage 5 — reduce to a timeline

Attributions come back shaped like the input: `[80, 3000]` for Whisper. The
frontend needs one value per time frame.

```python
    if attributions is not None:
        saliency_np = attributions.detach().cpu().numpy().squeeze()
        if saliency_np.ndim == 2:
            if saliency_np.shape[0] in (64, 80, 128):
                agg = np.mean(np.abs(saliency_np), axis=0)
            else:
                agg = np.mean(np.abs(saliency_np), axis=1)
        elif saliency_np.ndim == 1:
            agg = np.abs(saliency_np)
        else:
            while saliency_np.ndim > 1:
                saliency_np = saliency_np.mean(axis=0)
            agg = np.abs(saliency_np)
        max_abs = float(np.max(agg)) if agg.size > 0 else 0.0
        saliency_scores = (agg / max_abs) if max_abs > 0 else np.zeros_like(agg)
    else:
        saliency_scores = np.array([])
```

Three things happening:

**`np.abs` before averaging.** A negative attribution means "this pushed the
score down" — still influential. Averaging signed values would let +0.5 and
−0.5 cancel to zero, reporting "unimportant" for a region that mattered twice
over. Magnitude is what a heatmap shows.

**Axis detection by shape.** `shape[0] in (64, 80, 128)` guesses that a
first dimension of 64, 80 or 128 is the mel axis, so it averages over axis 0 to
leave time. Otherwise it assumes time is first and averages axis 1.

This is a heuristic and it is worth being honest about it: a clip whose *time*
dimension happened to be exactly 80 frames would be reduced along the wrong
axis. At 16 kHz with Whisper's 10 ms hop, 80 frames is 0.8 seconds — possible.
A more robust implementation would carry the axis meaning explicitly rather
than inferring it from a magic-number set. It is a latent bug, documented here
because you should not copy it.

**Normalise to [0, 1]** by dividing by the maximum, with a zero guard.

---

## 9.7 Stage 6 — the fallback and its provenance

```python
    use_energy_fallback = (
        saliency_scores.size == 0 or
        (np.max(saliency_scores) - np.min(saliency_scores) if saliency_scores.size > 0 else 0.0) < 1e-6
    )
    fallback_reason = shap_fallback_reason
    if use_energy_fallback:
        if fallback_reason is None:
            fallback_reason = "attribution was empty or constant; showing encoder energy, not attribution"
        logger.info(f"Using Whisper energy-map fallback for saliency ({fallback_reason})")
        with torch.no_grad():
            enc = model.encoder(input_features).last_hidden_state
            energy = enc.abs().mean(dim=2).squeeze(0).detach().cpu().numpy()
        if energy.size > 0:
            e_min, e_ptp = float(np.min(energy)), float(np.ptp(energy))
            saliency_scores = (energy - e_min) / (e_ptp + 1e-9)
        else:
            saliency_scores = np.zeros(1, dtype=np.float32)
```

**The guard is "empty or constant".** A constant map carries no information —
every region equally important is the same as no explanation — so it is treated
as a failure.

This guard is what fired on *every* Whisper request for both Grad-CAM and LIME
before those bugs were fixed (§2.3, §2.4). It was working correctly. It was
correctly detecting that the attribution had collapsed. The failure was that
nobody noticed the fallback was always firing, because the fallback produced a
plausible picture.

**Two lessons.** A guard that fires constantly is telling you something, and
guards need monitoring. And the reason string matters precisely because it is
carried into the response:

```python
    prov = Provenance.FALLBACK if (use_energy_fallback or shap_fallback_reason is not None) else Provenance.MEASURED
    res_reason = fallback_reason if prov == Provenance.FALLBACK else None
```

An energy map is **not an attribution**, and the response says so. A user
looking at the panel can see "fallback" and the reason. Had that field existed
earlier, the Grad-CAM bug would have been visible from the UI from day one.

---

## 9.8 Stage 7 — smoothing for display

```python
    series = saliency_scores.astype(np.float32)
    if series.size > 0:
        win = max(3, int(series.size / 64))
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win, dtype=np.float32) / float(win)
        series = np.convolve(series, kernel, mode="same")
        p95 = float(np.percentile(series, 95))
        if p95 > 0:
            series = np.clip(series, 0, p95)
        smin, smax = float(np.min(series)), float(np.max(series))
        series = (series - smin) / (smax - smin + 1e-9)
```

Three display transforms, each answering a specific visual problem:

**Moving-average smoothing.** Gradient maps are speckled (§2.3). `np.convolve`
with a uniform kernel is a box filter. The window is 1/64th of the series, so
it scales with length — a fixed window would over-smooth short clips and
under-smooth long ones. Forced odd so the window is symmetric about each
sample; an even window shifts the output by half a sample, which over a whole
series is a visible time offset.

**95th-percentile clipping.** One extreme outlier compresses everything else
into the bottom of the colour range, so the heatmap looks uniformly dark with
one bright dot. Clipping at p95 sacrifices the exact value of the top 5% to
make the other 95% readable.

**Renormalise** after clipping, so the range is [0, 1] again.

> **These are display transforms, and they are not reversible.** `series` is
> for drawing. The unsmoothed `saliency_scores` is what the segment loop below
> uses for numbers, and what a faithfulness audit would consume. Mixing them up
> would mean auditing a smoothed, clipped version of the attribution rather
> than the attribution.

---

## 9.9 Stage 8 — mapping to words

```python
    T = len(saliency_scores)
    fps = (T / total_duration) if total_duration > 0 else 1.0

    if chunks and total_duration > 0:
        for chunk in chunks:
            start_time = chunk.get("timestamp", [0, 0])[0]
            end_time = chunk.get("timestamp", [0, 0])[1]
            word = chunk.get("text", "")

            # Skip invalid chunks
            if end_time <= start_time or start_time < 0 or end_time > total_duration:
                continue

            start_frame = max(0, min(T - 1, int(start_time * fps)))
            end_frame = max(start_frame + 1, min(T, int(end_time * fps)))

            if end_frame > start_frame:
                segment_saliency = float(np.mean(saliency_scores[start_frame:end_frame]))
                segments.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "word": word.strip(),
                    "saliency": segment_saliency,
                    "intensity": float(abs(segment_saliency))
                })
        segments.sort(key=lambda x: x["start_time"])
```

`fps` is attribution frames per second — derived, not assumed, because the
attribution length depends on the model's downsampling.

**The index clamping is doing real work.** `max(0, min(T-1, ...))` on the start
and `max(start+1, min(T, ...))` on the end guarantee three properties: both
indices are in range, and `end > start` always. Without the last one, a word
whose timestamps round to the same frame would produce an empty slice, and
`np.mean([])` is `nan` — which serialises to invalid JSON and breaks the whole
response.

**Chunk validation** rejects timestamps that are inverted, negative, or past
the end. Whisper's word timestamps are approximate and occasionally wrong; one
bad chunk must not poison the list.

**Sorting by start time** because the frontend renders in order.

### When there are no words

```python
    if len(segments) == 0 and T > 0 and total_duration > 0:
        logger.info("No word-level segments found, creating uniform time-based segments")
        num_segments = max(8, min(32, int(total_duration * 2)))
        for i in range(num_segments):
            start_time = (i / num_segments) * total_duration
            end_time = ((i + 1) / num_segments) * total_duration
            ...
            segments.append({..., "word": f"segment_{i+1}", ...})
```

SER and ADD have no words at all, and ASR can fail to produce timestamps.
Uniform segments give the UI something to render. `max(8, min(32, duration*2))`
targets roughly half-second segments, bounded to 8–32 so a 1-second clip still
gets a usable number and a 60-second clip does not get 120.

The label `segment_3` is deliberately not word-like. Nothing pretends these are
words.

### Intensity normalisation

```python
    if len(segments) > 0:
        raw_saliencies = [s.get("saliency", 0.0) for s in segments]
        abs_vals = np.abs(raw_saliencies)
        max_abs = float(np.max(abs_vals)) if len(abs_vals) > 0 else 0.0
        if max_abs > 1e-9:
            for i, segment in enumerate(segments):
                segment["intensity"] = float(abs_vals[i] / max_abs)
        else:
            # Fallback: use relative ranking if all values are very small
            sorted_indices = np.argsort(-abs_vals)
            for rank, idx in enumerate(sorted_indices):
                segments[idx]["intensity"] = float(1.0 - (rank / len(segments)) * 0.9)

        # Ensure minimum visibility for all segments
        for segment in segments:
            segment["intensity"] = max(0.1, segment["intensity"])
```

`saliency` keeps the raw value; `intensity` is a display-normalised copy. Two
fields because they answer different questions — "what was the attribution" and
"how bright should this be".

The rank-based fallback handles a series of uniformly tiny values, where
dividing by a near-zero maximum amplifies floating-point noise into a random
pattern. Ranking preserves the ordering without inventing magnitudes.

**The `max(0.1, ...)` floor is a UI decision with a real cost.** It guarantees
every segment is at least faintly visible, so a word does not disappear. But it
also means a genuinely zero-attribution region renders at 10% rather than 0% —
the visualisation cannot distinguish "unimportant" from "slightly important".
For a tool whose purpose is honest measurement that is a compromise worth
knowing about. The raw `saliency` field is unaffected.

---

## 9.10 Stage 9 — the 2-D matrix

```python
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=2048,
                                               hop_length=512, n_mels=128)
    log_mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    log_mel_spect_norm = (log_mel_spect - log_mel_spect.min()) / (log_mel_spect.max() - log_mel_spect.min() + 1e-9)

    n_frames = log_mel_spect_norm.shape[1]
    if series.size > 0:
        attr_resampled = np.interp(
            np.linspace(0, len(series) - 1, n_frames),
            np.arange(len(series)),
            np.abs(series)
        )
        mel_saliency = np.tile(attr_resampled, (128, 1))
    else:
        mel_saliency = np.zeros_like(log_mel_spect_norm)
```

Two arrays for the frontend: the spectrogram to draw underneath, and the
attribution to draw on top.

**The display spectrogram is not the model's input.** Whisper uses 80 mel bins
at a 10 ms hop; this is 128 bins at a 32 ms hop. The display version is chosen
to look good, the model's to match its training. They are different objects
with different parameters, and that is fine as long as the *time axis* is
consistent — which is what the resampling ensures.

**`np.interp` resamples the attribution to the display frame count.** Linear
interpolation onto `n_frames` positions.

**`np.tile(..., (128, 1))` copies the 1-D series across all 128 mel bins.** The
attribution has no frequency resolution (§9.6), so every frequency at a given
time gets the same value. The overlay therefore shows vertical bands. That is
the honest rendering of a time-only attribution — it does not imply frequency
information that was never computed.

---

## 9.11 The response contract

```python
    return {
        "model": resolve_whisper_model_id(model_size),
        "method": method,
        "segments": segments,
        "total_duration": total_duration,
        "series": series.tolist(),
        "base_spectrogram": log_mel_spect_norm.tolist(),
        "saliency_matrix": mel_saliency.tolist(),
        **provenance_fields(prov, reason=res_reason)
    }
```

`model` is the **resolved** id, not what the user typed — so the response says
exactly which checkpoint produced it.

And the route's response model, whose docstring is a lesson in itself:

```python
class SaliencyResponse(BaseModel):
    """Mirrors exactly what `generate_saliency` returns.

    Every field below appears in all six return paths of
    `app/domain/saliency_service.py` (verified against the source, not assumed).
    Declaring a field the service never sets makes pydantic reject a perfectly
    good result with a 422 - which is what happened when this model was
    rewritten to `success`/`max_val`/`target_class`/`duration_s`/`sample_rate`.
    """
    model: str
    method: str
    segments: list
    total_duration: float
    series: Optional[list] = None
    base_spectrogram: Optional[list] = None
    saliency_matrix: Optional[list] = None
    emotion: Optional[str] = None
    predicted_label: Optional[str] = None
    # LIT-238 contract: a string enum value, not a dict.
    provenance: Optional[str] = None
    provenance_reason: Optional[str] = None
```

Someone rewrote this model with fields that sounded right. Pydantic validates
responses as well as requests, so a correct result missing a declared required
field becomes a **422 error**. The service worked; the contract rejected it.

*"verified against the source, not assumed"* is the fix and the discipline.
`emotion` and `predicted_label` are optional because only the SER and ADD paths
set them — six return paths, and the model is the union of what they actually
produce.

---

## 9.12 Running it off the request thread

```python
        result = await asyncio.to_thread(
            generate_saliency, str(resolved_path), request.model, request.method, existing_prediction
        )
```

`generate_saliency` is synchronous and takes seconds. Calling it directly in an
`async def` would block the event loop for the whole duration — every other
request in the process, stalled (§7.8).

`asyncio.to_thread` runs it on a threadpool thread. But that is precisely what
creates the concurrency hazard: **real OS threads, running attribution against
a shared model**, which is why `lock_for_model` exists. The two facts are
connected — the fix for blocking the event loop is what created the need for
the lock.

---

## 9.13 Complete trace

Grad-CAM on a 20-second clip with `openai/whisper-base`:

1. Route resolves the reference to a path, builds a cache key including the
   method, checks the cache. Miss.
2. `asyncio.to_thread(generate_saliency, ...)`.
3. `detect_model_type` → `"whisper"`; `model_lock_key` →
   `"whisper:openai/whisper-base"`; lock acquired.
4. `existing_prediction` present from the transcript panel — audio and chunks
   reused, no second transcription.
5. Audio truncated 20 s → 12 s; chunks beyond 12 s dropped.
6. Processor → `[1, 80, 3000]`; moved to device; `requires_grad_(True)`.
7. `_build_whisper_gradcam_target` loads `WhisperForConditionalGeneration`,
   generates the token sequence, wraps it in `_WhisperTranscriptScore`, finds
   the last conv layer of *that* instance's encoder.
8. `compute_grad_cam` — forward hook, backward hook, zero grads, forward,
   backward from the transcript score, channel weights, ReLU'd weighted sum,
   normalise, hooks removed in `finally`.
9. CAM is 1-D over conv time → interpolated to 3000 → tiled to `[80, 3000]`.
10. `mean(|·|)` over axis 0 → 3000 values → divided by max.
11. Range is 0.9, well above 1e-6 → no fallback → `Provenance.MEASURED`.
12. Box-smoothed (window 47, forced odd), clipped at p95, renormalised.
13. `fps = 3000 / 12 = 250`. Each word chunk mapped to a frame range and
    averaged; segments sorted; intensities normalised with a 0.1 floor.
14. 128-bin display spectrogram; `series` interpolated to its frame count and
    tiled across 128 bins.
15. Dict returned; lock released.
16. Route caches it for 6 hours and validates it against `SaliencyResponse`.
17. The browser layers the spectrogram, the heatmap, the pitch contour and the
    playhead on stacked canvases (Chapter 15).

---

## 9.14 Summary

- Dispatch locks on the shared `nn.Module` identity, not the model string, and
  shares that key function with plain inference.
- An existing prediction can be passed in to skip a second transcription.
- Truncating audio must truncate the chunk list with it.
- Inputs go on the model's device, discovered from
  `next(model.parameters()).device`, and get `requires_grad_(True)`.
- Unknown methods raise; nothing silently becomes another method.
- Reduce with `mean(|·|)` — signed averaging cancels genuine influence. The
  axis detection by magic shape numbers is a latent bug; carry the axis
  meaning explicitly instead.
- The "empty or constant" fallback guard fired on every Whisper request for
  two separate bugs and nobody noticed, because the fallback looked plausible.
  Guards need monitoring, and provenance makes them visible.
- Smoothing, percentile clipping and the intensity floor are irreversible
  *display* transforms; keep the raw values separate for anything that measures.
- Segment index clamping guarantees `end > start`, because `np.mean([])` is
  `nan` and `nan` is invalid JSON.
- The attribution has no frequency resolution, so tiling it across mel bins is
  the honest rendering — vertical bands, not fake 2-D detail.
- The display spectrogram deliberately differs from the model's input; only the
  time axis must agree.
- Pydantic validates responses: a field the service never sets turns a correct
  result into a 422. Verify the contract against the source.
- `asyncio.to_thread` keeps the event loop free and is exactly why the
  per-model lock is necessary.

Next: [Chapter 10 — The acoustic profiler](10-acoustic-profiler.md).
