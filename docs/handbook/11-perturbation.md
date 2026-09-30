# Chapter 11 — The mutation engine: perturbation and counterfactuals

`Backend/app/domain/perturbation_service.py`, 657 lines. This is the module
that lets a user ask **"what if?"** — mute this half-second, add noise, shift
the pitch, and see what the model says then.

It is also the module that powers the faithfulness audit, because masking the
most salient regions is a perturbation like any other.

---

## 11.1 Why counterfactuals are the strongest kind of explanation

An attribution map is a *claim* about what the model used. A counterfactual is
a *demonstration*.

If the heatmap says the model relied on 1.2–1.5 seconds, mute that interval. If
the prediction changes, the claim is supported. If it does not, the claim was
wrong. No theory, no axioms — you changed the input and observed the output.

This is why the perturbation engine and the faithfulness auditor are the same
code: a faithfulness audit is a systematic counterfactual experiment.

---

## 11.2 Load and save: enforcing the contract at the boundary

```python
def _load_waveform(path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio via soundfile, returning the (channels, time) float32 tensor.
    Enforces 16kHz mono constraint for downstream inference endpoints (SRS FR12.3).
    """
    try:
        # always_2d=True ensures (samples, channels) shape even for mono
        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)

        # Transpose to (channels, samples) to match PyTorch's expected orientation
        waveform = torch.from_numpy(data.T.copy())

        # Enforce 16kHz mono constraint
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # Downmix to mono

        if sample_rate != 16000:
            audio_np = waveform[0].numpy()
            audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)
            waveform = torch.from_numpy(audio_np).unsqueeze(0)
            sample_rate = 16000

        return waveform, sample_rate
    except Exception as e:
        logger.error(f"Failed to load audio file {path}: {e}")
        raise
```

### Two different array conventions, and the `.copy()`

soundfile returns `(samples, channels)`. PyTorch audio convention is
`(channels, samples)`. So `data.T` transposes — and `.copy()` is **necessary,
not defensive**.

`numpy.T` returns a *view* with the strides reversed, not a new array. That
view is non-contiguous in memory. `torch.from_numpy` on a non-contiguous array
either raises or produces a tensor whose later operations fail confusingly.
`.copy()` materialises a contiguous array.

This is a genuine trap. It works for some operations and fails for others, so
it can pass testing and fail in production.

### The conversion, in one place

Downmix to mono, resample to 16 kHz — the input contract from §1.1, enforced
at the boundary where audio enters the module. Every function downstream can
then assume `(1, n_samples)` at 16 kHz.

**Putting the normalisation at the boundary rather than in every function** is
what makes the rest of the module simple. There is one place to get it right.

`keepdim=True` on the mean preserves the channel axis, so mono stays `(1, N)`
rather than collapsing to `(N,)`. Consistent shape means no `dim() == 1`
special-casing in the ten functions that follow.

```python
def export_to_wav_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
    """Exports tensor to in-memory WAV bytes for Web Audio preview (SRS FR12.2)."""
    buf = io.BytesIO()
    data = waveform.detach().cpu().numpy().T
    sf.write(buf, data, sample_rate, format='WAV', subtype='PCM_16')
    buf.seek(0)
    return buf.read()
```

`io.BytesIO` is an in-memory file. Writing to it avoids a temporary file for
audio that only needs to reach the browser once. `seek(0)` rewinds before
reading — a classic omission that returns empty bytes.

`.detach().cpu()` covers a tensor that is on a GPU or attached to an autograd
graph. Neither is expected here, but both are cheap to guard and produce
confusing errors when missed.

---

## 11.3 The seven transformations

### Gaussian noise

```python
def add_gaussian_noise(waveform: torch.Tensor, noise_level: float = 0.005) -> torch.Tensor:
    """Add Gaussian noise to the waveform."""
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise
```

`torch.randn_like` draws from a standard normal distribution with the same
shape, dtype and device. Scaled by `noise_level`, so 0.005 is the standard
deviation relative to a full-scale range of ±1.

Tests robustness: a model that changes its answer under inaudible noise is
brittle. (This is also the basic mechanism behind adversarial examples, where
the noise is chosen adversarially rather than randomly.)

### Time masking

```python
def apply_time_masking(waveform, mask_start_percent, mask_end_percent) -> torch.Tensor:
    """Apply time masking to a portion of the waveform."""
    channels, length = waveform.shape
    start_idx = int(length * mask_start_percent / 100)
    end_idx = int(length * mask_end_percent / 100)
    masked_waveform = waveform.clone()
    masked_waveform[:, start_idx:end_idx] = 0
    return masked_waveform
```

Silences an interval. Parameters are **percentages**, not samples or seconds,
so the UI can express "mute the middle third" without knowing the duration.

`.clone()` — every function clones. The non-destructive contract (§11.5).

### Frequency masking

```python
def apply_frequency_masking(waveform, sample_rate, mask_freq_start, mask_freq_end) -> torch.Tensor:
    """Apply frequency masking to the waveform."""
    fft = torch.fft.fft(waveform, dim=-1)
    freqs = torch.fft.fftfreq(waveform.shape[-1], 1/sample_rate)
    freq_mask = (freqs >= mask_freq_start) & (freqs <= mask_freq_end)
    fft[:, freq_mask] = 0
    return torch.fft.ifft(fft, dim=-1).real
```

The first genuinely interesting one, and the clearest demonstration of §1.2 in
code:

1. **FFT** the whole signal → complex frequency-domain representation.
2. **`fftfreq`** gives the frequency each bin corresponds to.
3. **Boolean mask** selects bins in the target range.
4. **Zero them.**
5. **Inverse FFT** back to a waveform.
6. **`.real`** discards the imaginary residue.

Step 6 needs explaining. Mathematically, the inverse FFT of a
conjugate-symmetric spectrum is purely real. Floating-point arithmetic leaves a
tiny imaginary component, and zeroing an asymmetric set of bins breaks the
symmetry slightly. Taking `.real` is correct — but note that this
implementation zeroes only the *positive* frequencies matching the range, not
their negative counterparts, which is an asymmetry. The practical effect is
small and the audible result is as intended, but a stricter implementation
would mask the conjugate bins too.

This operates on the **whole signal at once**, so the masking is global in
time. To mask a frequency band only during part of the clip, you need the next
function.

### 2-D time-frequency masking

```python
def apply_2d_time_freq_mask(waveform, sample_rate, params) -> torch.Tensor:
    """NumPy-Driven 2D Spectrogram Slice Masking (mute regions)."""
    t_start_ms = params.get("t_start_ms", 0)
    t_end_ms = params.get("t_end_ms", (waveform.shape[-1] / sample_rate) * 1000)
    f_low_hz = params.get("f_low_hz", 0)
    f_high_hz = params.get("f_high_hz", sample_rate / 2)

    audio_np = waveform[0].numpy() if waveform.dim() > 1 else waveform.numpy()

    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(audio_np, n_fft=n_fft, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sample_rate, hop_length=hop_length)

    freq_mask = (freqs >= f_low_hz) & (freqs <= f_high_hz)
    time_mask = (times * 1000 >= t_start_ms) & (times * 1000 <= t_end_ms)

    # Outer product creates the 2D mute region
    mask_2d = np.outer(freq_mask, time_mask)
    stft[mask_2d] = 0.0

    y_masked = librosa.istft(stft, hop_length=hop_length)

    if len(y_masked) < len(audio_np):
        y_masked = np.pad(y_masked, (0, len(audio_np) - len(y_masked)))
    else:
        y_masked = y_masked[:len(audio_np)]

    return torch.from_numpy(y_masked).unsqueeze(0)
```

**This is the function behind the canvas region selector.** A user drags a
rectangle on the spectrogram; this mutes exactly that rectangle.

`np.outer(freq_mask, time_mask)` is elegant: the outer product of two boolean
vectors is a 2-D boolean matrix that is `True` exactly where both are `True` —
the rectangle. One line instead of nested loops.

**The length reconciliation at the end is not optional.** `istft` does not
generally return exactly as many samples as the input, because of the framing
and windowing. Off by a few samples, and any downstream code that assumes the
perturbed clip is the same length as the original breaks — or worse, silently
misaligns an attribution against it. Padding or trimming to the original length
guarantees the contract.

**The defaults are full-range**, so an omitted parameter means "all of it"
rather than "none of it" — a missing `f_high_hz` masks up to Nyquist rather
than masking nothing.

> **Phase is preserved.** `librosa.stft` returns complex values; zeroing some
> and inverse-transforming keeps the phase of everything else. A naive
> implementation using only magnitudes and then reconstructing with
> Griffin-Lim would introduce audible artefacts that the model might react to —
> so you would be testing your reconstruction, not your mask.

### Band-pass filter

```python
def apply_band_pass_filter(waveform, sample_rate, params) -> torch.Tensor:
    """Signal modification routine for band-pass filtering."""
    from scipy.signal import butter, sosfilt

    f_low_hz = params.get("f_low_hz", 500)
    f_high_hz = params.get("f_high_hz", 2000)
    audio_np = waveform[0].numpy()

    nyq = 0.5 * sample_rate
    low = f_low_hz / nyq
    high = f_high_hz / nyq
    sos = butter(5, [low, high], analog=False, btype='band', output='sos')
    filtered_audio = sosfilt(sos, audio_np)

    return torch.from_numpy(filtered_audio).unsqueeze(0)
```

A **Butterworth filter** has a maximally flat passband — no ripple, at the cost
of a gentler roll-off than alternatives like Chebyshev. Order 5 is a reasonable
compromise between steepness and stability.

Two details that are easy to get wrong:

**Frequencies are normalised to Nyquist.** `scipy.signal.butter` expects
critical frequencies in [0, 1] where 1 is Nyquist (half the sample rate), not
in hertz. Passing hertz directly gives either an error or a filter with
completely wrong cutoffs.

**`output='sos'`, not the default `'ba'`.** Second-order sections are
numerically stable for higher-order filters; transfer-function coefficients
(`b, a`) accumulate floating-point error and can make a high-order filter
unstable — it rings or blows up. `sos` plus `sosfilt` is the modern
recommendation and the difference is not cosmetic.

The import is function-local, so scipy is only loaded if this filter is
actually used.

### Pitch shift

```python
def apply_pitch_shift(waveform, sample_rate, pitch_shift_semitones) -> torch.Tensor:
    """Apply pitch shifting to the waveform using Librosa."""
    pitch_shift_semitones = max(-6, min(6, pitch_shift_semitones))
    if abs(pitch_shift_semitones) < 0.1:
        return waveform

    max_length = sample_rate * 30
    if waveform.shape[-1] > max_length:
        waveform = waveform[..., :max_length]

    try:
        audio_np = waveform[0].numpy() if waveform.dim() > 1 else waveform.numpy()
        shifted_audio = librosa.effects.pitch_shift(y=audio_np, sr=sample_rate,
                                                    n_steps=pitch_shift_semitones)
        return torch.from_numpy(shifted_audio).unsqueeze(0)
    except Exception as e:
        logger.error(f"Pitch shift failed: {e}. Returning original waveform.")
        return waveform
```

Pitch shifting changes pitch **without** changing duration, which is
non-trivial — it requires a phase vocoder: STFT, shift the frequency content,
resynthesise with phase correction. librosa implements it.

Four guards, each for a distinct reason:

- **Clamp to ±6 semitones.** Beyond half an octave the phase vocoder produces
  obvious artefacts, and you would be testing the model's response to
  artefacts rather than to pitch.
- **Skip below 0.1 semitones.** A no-op shift still costs a full
  STFT/resynthesis round trip. Returning early avoids paying for nothing —
  and avoids introducing vocoder artefacts for a change nobody asked for.
- **Cap at 30 seconds.** Phase vocoding is expensive and memory-hungry.
- **Return the original on failure.** A perturbation that cannot be applied
  should degrade to "unchanged", not fail the whole pipeline. The caller learns
  from the status field (§11.4).

### Time stretch

```python
def apply_time_stretch(waveform, stretch_factor) -> torch.Tensor:
    """Apply time stretching to the waveform using Librosa."""
    if abs(stretch_factor - 1.0) < 0.01:
        return waveform
    try:
        audio_np = waveform[0].numpy() if waveform.dim() > 1 else waveform.numpy()
        stretched_audio = librosa.effects.time_stretch(y=audio_np, rate=stretch_factor)
        return torch.from_numpy(stretched_audio).unsqueeze(0)
    except Exception as e:
        logger.error(f"Time stretch failed: {e}. Returning original waveform.")
        return waveform
```

The dual operation: change duration without changing pitch. Same phase-vocoder
machinery. `rate > 1` speeds up, `rate < 1` slows down.

**This one changes the array length**, which matters for anything that aligns
an attribution against the audio. A stretched clip's timeline no longer matches
the original's.

---

## 11.4 Composing perturbations

```python
def apply_perturbations(waveform, sample_rate, perturbations) -> Tuple[torch.Tensor, List[Dict]]:
    """Apply multiple perturbations to a waveform sequentially."""
    perturbed_waveform = waveform.clone()
    applied_perturbations = []

    for perturbation in perturbations:
        perturbation_type = perturbation.get("type")
        params = perturbation.get("params", {})

        try:
            if perturbation_type == "noise":
                noise_level = params.get("noise_level", 0.005)
                perturbed_waveform = add_gaussian_noise(perturbed_waveform, noise_level)
                applied_perturbations.append({"type": "noise",
                                              "params": {"noise_level": noise_level},
                                              "status": "applied"})
            elif perturbation_type == "time_masking":
                ...
            else:
                applied_perturbations.append({"type": perturbation_type,
                                              "params": params, "status": "unsupported"})
        except Exception as e:
            applied_perturbations.append({"type": perturbation_type, "params": params,
                                          "status": "failed", "error": str(e)})

    return perturbed_waveform, applied_perturbations
```

### The three-state report

Every perturbation reports `applied`, `unsupported`, or `failed` with an error.
This is the same honesty principle as provenance (§2.7): **the caller must be
able to tell what actually happened.**

Without it, a request for `[mute 1-2s, add noise]` where the mute silently
failed returns audio with noise only — and the caller believes both were
applied. Any conclusion drawn from that comparison is wrong.

The echoed `params` include resolved defaults, so the report says
`noise_level: 0.005` rather than leaving the caller to guess what default was
used. The report is a complete record of what was done.

### Order matters, and is the caller's responsibility

Perturbations apply sequentially to the accumulating result. `[pitch_shift,
time_mask]` and `[time_mask, pitch_shift]` produce different audio: masking
then shifting puts a pitch-shifted silence where the mask was; shifting then
masking silences a region of already-shifted audio.

The function does not reorder or optimise. The order given is the order applied,
and the report preserves it.

### One `try` per perturbation

The `try` is **inside** the loop, so one failure does not abandon the rest.
Combined with the status report, a partially successful run is a usable result
with an accurate description.

---

## 11.5 Non-destructive by construction

The requirement is that the original audio is always preserved and a mutation
produces a *derived* clip. Three mechanisms enforce it:

**Every transformation clones.** `waveform.clone()` or `masked_waveform =
waveform.clone()`. No function mutates its input, so the original tensor is
intact for the next perturbation or for comparison.

**Output goes to a new file with a new name:**

```python
    input_path = Path(file_path)
    output_filename = f"{input_path.stem}_perturbed_{uuid.uuid4().hex[:8]}.wav"
    output_path = Path(output_dir) / output_filename
```

The stem keeps it recognisable; the UUID fragment prevents collisions between
two perturbations of the same source.

**Output is always `.wav`**, regardless of input format. Uncompressed, so no
lossy re-encoding adds artefacts on top of the perturbation you are trying to
study.

---

## 11.6 `perturb_and_save`

```python
def perturb_and_save(file_path, perturbations, output_dir="uploads",
                     dataset=None, session_id=None) -> Dict[str, Any]:
    """Apply perturbations to an audio file and save the derived clip non-destructively."""
    try:
        if dataset and not Path(file_path).is_absolute():
            resolved_path = resolve_file(dataset, file_path, session_id)
        else:
            resolved_path = Path(file_path)
            if not resolved_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
    except FileNotFoundError as e:
        return {
            "original_file": file_path, "perturbed_file": "", "filename": "", "duration_ms": 0,
            "sample_rate": 0, "applied_perturbations": [], "success": False, "error": str(e)
        }
```

**Errors are returned, not raised.** Every failure path returns the same dict
shape with `success: False` and an error message.

Why: this function is called both from a synchronous route and from an RQ task
(§7.7). A consistent shape means neither caller needs exception handling, and
the RQ aggregator can fan in a failed mutation alongside successful siblings
without special cases.

The cost is that callers must check `success` — and a caller that forgets will
treat a failure as a success with empty fields. That is the trade: the shape is
uniform, the check is the caller's obligation.

```python
    preview_bytes = export_to_wav_bytes(perturbed_waveform, sample_rate)
    duration_ms = int(perturbed_waveform.shape[-1] / sample_rate * 1000)
    perturbed_file_path = str(output_path).replace("\\", "/")

    return {
        "original_file": file_path,
        "perturbed_file": perturbed_file_path,
        "filename": output_filename,
        "duration_ms": duration_ms,
        "sample_rate": sample_rate,
        "applied_perturbations": applied_perturbations,
        "preview_bytes": preview_bytes,
        "success": True
    }
```

**`preview_bytes`** lets the browser play the result immediately via the Web
Audio API, with no second request. The requirement asks for client-side preview
before dispatch; returning the bytes inline is what makes it instant.

**`.replace("\\", "/")`** normalises Windows path separators. Backslashes in
JSON require escaping and confuse URL construction in the frontend. A small
thing that causes a genuinely confusing class of cross-platform bug.

**`duration_ms` is computed from the perturbed waveform**, not the original —
correct, because `time_stretch` changes length.

---

## 11.7 Masking for the faithfulness audit

The same engine, driven by an attribution instead of a user.

```python
def mask_top_k_features(waveform, attributions, k_percent=10.0) -> torch.Tensor:
    """Mask the top-K% most salient timesteps/features based on attribution weights (FR16)."""
    masked = waveform.clone()
    if masked.dim() == 1:
        masked = masked.unsqueeze(0)

    channels, length = masked.shape
    attr_np = np.asarray(attributions, dtype=np.float32)

    if len(attr_np) == 0 or length == 0:
        return masked

    if len(attr_np) != length:
        attr_np = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(attr_np)), attr_np)

    k_percent = max(0.0, min(100.0, k_percent))
    k_count = int(np.ceil((k_percent / 100.0) * length))

    if k_count > 0:
        top_k_indices = np.argpartition(np.abs(attr_np), -k_count)[-k_count:]
        masked[:, top_k_indices] = 0.0

    return masked
```

Four details that each matter:

**Resolution mismatch.** The attribution is per *frame* (perhaps 3000 values);
the waveform is per *sample* (perhaps 192,000). `np.interp` over normalised
[0, 1] positions maps one to the other. Get this wrong and you mask the wrong
audio, and the faithfulness score becomes noise.

**`np.abs`.** A strongly negative attribution is strongly influential (§9.6).

**`np.argpartition`** finds the top *k* in O(n) without fully sorting. For
192,000 samples that is a real saving over `argsort`.

**`np.ceil`.** Round up, so a tiny `k_percent` masks at least one sample rather
than zero. Masking nothing would report a deletion score of 0 and look like a
maximally unfaithful attribution.

### Contiguous intervals instead of scattered samples

Masking individual samples produces a comb of one-sample gaps — audible as a
buzz, and not a realistic perturbation. `HighSaliencyMaskingEngine` groups the
selected indices into **contiguous intervals**:

```python
        top_k_indices = np.argpartition(np.abs(attr_np), -k_count)[-k_count:]
        top_k_indices.sort()

        intervals: List[Tuple[float, float]] = []
        step_sec = (total_samples / sample_rate) / len(attr_np)
        start_idx = top_k_indices[0]
        prev_idx = top_k_indices[0]

        for idx in top_k_indices[1:]:
            if idx == prev_idx + 1:
                prev_idx = idx
            else:
                t_start = round(start_idx * step_sec * 1000, 2)
                t_end = round((prev_idx + 1) * step_sec * 1000, 2)
                intervals.append((t_start, t_end))
                start_idx = idx
                prev_idx = idx

        t_start = round(start_idx * step_sec * 1000, 2)
        t_end = round((prev_idx + 1) * step_sec * 1000, 2)
        intervals.append((t_start, t_end))
        return intervals
```

A standard run-length grouping: sort, then walk, extending the current run
while indices are consecutive and emitting an interval when they are not. The
**final emission after the loop** is the part people forget — without it the
last run is silently dropped.

`prev_idx + 1` on the end bound makes the interval half-open in samples but
inclusive of the last selected frame's duration, which is what you want for a
time range.

Returning millisecond intervals rather than sample indices means the result is
human-readable and directly renderable on a timeline — the UI can show exactly
which regions were masked.

### Three masking modes

```python
            if mode.lower() == "noise":
                noise = torch.randn((channels, end_sample - start_sample)) * noise_level
                masked[:, start_sample:end_sample] = noise
            elif mode.lower() in ("mean", "blur"):
                mean_val = float(masked.mean())
                masked[:, start_sample:end_sample] = mean_val
            else:
                masked[:, start_sample:end_sample] = 0.0
```

Three baselines, and the choice is not neutral — it is the same baseline
question as Integrated Gradients (§2.3):

- **zero** — silence. The most common choice, and a genuine out-of-distribution
  input: models rarely see perfect digital silence in training.
- **noise** — replaces content while keeping energy. Less
  out-of-distribution.
- **mean** — a constant DC offset. Removes all variation.

A confidence drop under zero-masking might partly reflect the model's reaction
to unnatural silence rather than to the removed content. Offering all three
lets a researcher check whether the conclusion is robust to the baseline — and
if it is not, that is itself the finding.

---

## 11.8 Summary

- A counterfactual demonstrates what an attribution merely claims. The
  perturbation engine and the faithfulness auditor are the same code.
- Normalise at the boundary: `_load_waveform` handles transposition, downmix and
  resampling once, so ten downstream functions can assume `(1, N)` at 16 kHz.
- `numpy.T` is a view; `torch.from_numpy` needs contiguous memory. `.copy()` is
  required, not defensive.
- `keepdim=True` keeps mono as `(1, N)` and removes shape special-casing
  everywhere else.
- Frequency masking is FFT → mask bins → inverse FFT → `.real`, and is global in
  time. The 2-D version uses an STFT and an outer product of two boolean
  vectors, preserving phase.
- `istft` does not return the input length — reconcile explicitly or break every
  downstream alignment.
- `scipy.signal.butter` takes frequencies normalised to Nyquist, and
  `output='sos'` is numerically stable where `'ba'` is not at order 5.
- Pitch shift and time stretch are phase-vocoder operations: clamp the range,
  skip near-no-ops, cap the duration, and degrade to the original on failure.
- Every perturbation reports `applied` / `unsupported` / `failed` with resolved
  parameters. Without that report, a partially applied chain produces
  conclusions about an input you do not know.
- One `try` per perturbation inside the loop, so one failure does not abandon
  the rest.
- Non-destructive by three mechanisms: clone on every transform, a new UUID
  filename, and uncompressed WAV output.
- Errors are returned as a uniform shape rather than raised, because two
  different callers consume it — at the cost of the caller having to check.
- `preview_bytes` inline makes browser preview instant.
- Normalise path separators before they reach JSON.
- Masking must interpolate between frame and sample resolution, use absolute
  values, `argpartition` for O(n), and `ceil` so a small K masks at least one
  sample.
- Group masked samples into contiguous intervals — scattered single samples are
  an unrealistic perturbation — and remember to emit the final run.
- Three masking baselines exist because the baseline choice changes the answer,
  and robustness across them is the real test.

Next: [Chapter 12 — Latent projection](12-latent-projection.md).
