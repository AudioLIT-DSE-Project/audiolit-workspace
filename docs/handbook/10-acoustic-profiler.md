# Chapter 10 — The acoustic profiler

144 lines, no model, no GPU, no queue. The simplest module in the project and
one of the most useful, because it answers the question attribution cannot:
**what was physically in the audio?**

---

## 10.1 Why a model-free module exists in an interpretability tool

An attribution map says "the model relied on this region". It does not say what
is *in* that region. A researcher looking at a bright band at 2.3 seconds needs
to know: was that a pitch spike? A loudness peak? A particular formant?

The profiler provides the physical ground truth to compare against. That
comparison — model attention against physical signal — is the workbench's core
interaction, and it only works if both are on the same time axis.

Hence the module's opening:

```python
"""Acoustic Profiler — DSP feature extraction (LIT-125, FR10).

Time-aligned pitch (LIT-145) and energy (LIT-146) contours computed directly
from raw audio via librosa — no model involved. Both share one
frame_length/hop_length so the two contours line up on the same time axis,
which is what LIT-125's combined pipeline needs in order to overlay them.
"""
```

The shared framing is stated as the module's purpose, not an implementation
detail.

---

## 10.2 Parameters, and why these values

```python
DEFAULT_FRAME_LENGTH = 2048
DEFAULT_HOP_LENGTH = 512
# Typical human voice fundamental-frequency range, matches librosa's own
# pYIN example range for speech (C2-C7).
DEFAULT_FMIN = librosa.note_to_hz("C2")
DEFAULT_FMAX = librosa.note_to_hz("C7")
```

At 16 kHz:

- **2048 samples = 128 ms** per analysis frame. Long enough to resolve a
  fundamental down to ~65 Hz (you need several cycles of the lowest frequency
  you want to measure), short enough that a syllable is not entirely blurred.
- **512 samples = 32 ms** hop, so ~31 frames per second and 4× overlap between
  consecutive frames. Overlap is what makes the contour smooth rather than
  blocky.
- **`note_to_hz("C2")` ≈ 65 Hz** to **`note_to_hz("C7")` ≈ 2093 Hz**.

Using note names rather than raw numbers is worth copying. `65.4` and `2093.0`
are opaque; `C2` and `C7` say "this is a musical/vocal range" and are
self-documenting. It also cites librosa's own documented example range, so the
choice is traceable rather than invented.

**Why bound the search at all?** pYIN searches for periodicity within a range.
A wider range is slower and more prone to octave errors (§1.3). Bounding it to
the human vocal range is both faster and more accurate for speech.

---

## 10.3 Pitch: `track_pitch_contour`

```python
def track_pitch_contour(audio, sr, fmin=DEFAULT_FMIN, fmax=DEFAULT_FMAX,
                        frame_length=DEFAULT_FRAME_LENGTH, hop_length=DEFAULT_HOP_LENGTH,
                        voiced_prob_threshold=0.5) -> np.ndarray:
    """Frame-wise fundamental-frequency (F0) trajectory via pYIN (LIT-145, FR10).

    Returns a 1-D array, one value per frame at ``hop_length`` spacing.
    Unvoiced frames — silence, noise, or frames pYIN itself isn't confident
    about (``voiced_prob`` below ``voiced_prob_threshold``) — are ``np.nan``
    rather than 0.0, so they read as "no pitch" instead of a spurious 0 Hz
    value, matching how desktop tools like Praat render pitch gaps.
    """
    audio = np.asarray(audio, dtype=np.float32)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=hop_length,
    )
    f0 = np.asarray(f0, dtype=np.float64)
    unvoiced = ~np.asarray(voiced_flag, dtype=bool) | (np.asarray(voiced_prob) < voiced_prob_threshold)
    f0[unvoiced] = np.nan
    return f0
```

### The NaN decision

This is the most consequential line in the module, and §1.3 introduced it.
Restated as a principle:

> **A missing measurement and a measurement of zero are different claims.**

`0.0` Hz asserts "the pitch here is zero", which is not a thing that can be
true. `NaN` asserts "there is no pitch here", which is exactly right for
silence and for unvoiced consonants.

Downstream consequences:

- The frontend lifts the pen on NaN, so the contour shows gaps where pitch was
  genuinely unmeasurable. Zeros would draw a line crashing to the bottom of the
  chart during every `s` sound.
- `np.nanmean` computes statistics over voiced frames only. With zeros, any
  average would be dragged toward zero by every silent frame.
- The API converts NaN to JSON `null`, because NaN is not valid JSON.

The reference to Praat matters too: this is how established phonetics software
behaves, so a researcher's expectations are met rather than surprised.

### Two conditions for "unvoiced"

```python
    unvoiced = ~np.asarray(voiced_flag, dtype=bool) | (np.asarray(voiced_prob) < voiced_prob_threshold)
```

pYIN returns both a boolean flag and a probability. The code uses **both**,
combined with `|` (or): a frame is unvoiced if pYIN says so *or* if its
confidence is below 0.5.

The second condition is a deliberate tightening. pYIN's flag is generous; a
low-confidence "voiced" frame often carries a garbage F0 value. Requiring 50%
confidence trades some coverage for reliability, and the threshold is a
parameter so a caller can loosen it.

---

## 10.4 Energy: `estimate_rms_contour`

```python
def estimate_rms_contour(audio, frame_length=DEFAULT_FRAME_LENGTH,
                         hop_length=DEFAULT_HOP_LENGTH) -> np.ndarray:
    """Frame-wise RMS energy / localized amplitude contour (LIT-146, FR10).

    Returns a 1-D array, one value per frame, using the same
    ``frame_length``/``hop_length`` defaults (and the same STFT-style
    ``center=True`` framing) as `track_pitch_contour`, so the two contours
    come out the same length and line up frame-for-frame — the "aligned
    contours" LIT-125's combined pipeline needs to overlay pitch and
    intensity on one timeline.
    """
    audio = np.asarray(audio, dtype=np.float32)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    return rms.astype(np.float64)
```

Four lines of work and a docstring explaining the thing that matters: the
parameters must match.

**`center=True` framing** (librosa's default for both functions) pads the signal
by half a frame at each end so that frame *i* is *centred* on sample
`i * hop_length` rather than starting there. That is why both functions produce
`1 + len(audio) // hop_length` frames rather than `len(audio) // hop_length`.
Both using it is what makes the frame counts match exactly.

If one function centred and the other did not, the contours would differ in
length by one and be offset by half a frame — 16 ms. Small enough to look fine,
large enough to misattribute a pitch spike to the wrong syllable.

**`[0]`** because `librosa.feature.rms` returns shape `[1, n_frames]`
(channels-first convention).

---

## 10.5 The combined profile

```python
def extract_acoustic_profile(audio, sr, ...) -> dict:
    """Combined DSP acoustic profile (LIT-125, FR10): the STFT + pYIN F0 + RMS
    engine, packaged as one aligned, JSON-serializable timeline for the API
    serialization layer.

    Runs `track_pitch_contour` (LIT-145) and `estimate_rms_contour` (LIT-146)
    over the same frames and zips them into a millisecond-aligned timeline —
    the "aligned contours" both sub-tasks were built to share. Unvoiced F0
    frames become ``None`` (not NaN, which isn't valid JSON) so the result can
    go straight into an API response.
    """
```

### The timeline

```python
    step_s = hop_length / sr
    timeline = [
        {
            "t_ms": round(i * step_s * 1000.0, 3),
            "f0_hz": None if np.isnan(f0[i]) else float(f0[i]),
            "rms": float(rms[i]),
        }
        for i in range(len(rms))
    ]
```

`step_s = hop_length / sr` = 512/16000 = 0.032 s per frame. Frame *i* is at
`i * 32` ms.

`range(len(rms))` uses the RMS length as authoritative, which is safe *because*
the two contours are the same length by construction. If they could differ this
would silently truncate or index out of bounds — another reason the shared
parameters are load-bearing rather than tidy.

`round(..., 3)` gives microsecond precision in the JSON, which is far more than
needed but costs nothing and keeps the numbers short.

### The spectrogram

```python
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128,
                                       n_fft=frame_length, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    s_min, s_max = float(S_db.min()), float(S_db.max())
    S_norm = (S_db - s_min) / (s_max - s_min + 1e-8) if s_max > s_min else np.zeros_like(S_db)
```

Mel spectrogram → dB → normalise to [0, 1] so the frontend can paint it as
pixel brightness directly.

Two guards on the same degenerate case: `+ 1e-8` in the denominator and
`if s_max > s_min`. A silent clip has `min == max`; either guard alone would
prevent the division by zero, and having both means a change to one does not
reintroduce the bug.

`n_fft=frame_length` and the same `hop_length` again — the spectrogram shares
the timeline's time axis, so the overlay aligns.

### The waveform envelope, and a bug worth studying

```python
    # Normalised amplitude envelope for the canvas waveform layer. Nothing in
    # the backend produced one, so that layer drew nothing on every clip - the
    # canvas bails on `waveformData.length === 0`. Downsampled to one peak per
    # frame so it aligns with the spectrogram's time axis and stays small.
    n_frames = len(rms)
    needed = n_frames * hop_length
    padded = np.abs(audio)
    if len(padded) < needed:
        # librosa centres its frames, so the signal is a little shorter than
        # n_frames * hop_length. Pad rather than fall back to the raw signal:
        # the fallback returned 241,920 points for a 473-frame clip, which is
        # both unaligned with the spectrogram and far too large to ship as JSON.
        padded = np.pad(padded, (0, needed - len(padded)))
    peaks = padded[:needed].reshape(n_frames, hop_length).max(axis=1)
    peak_max = float(peaks.max()) if peaks.size else 0.0
    waveform = (peaks / peak_max) if peak_max > 0 else np.zeros_like(peaks)
```

Two bugs recorded here, and they are different in kind.

**Bug one: a layer that never drew.** The frontend canvas has a waveform layer
that bails on `waveformData.length === 0`. The backend never produced the
field. So the layer silently rendered nothing, forever. No error, no missing
data warning — the guard that was supposed to handle "no waveform yet"
handled "no waveform ever" identically.

**Bug two: the wrong fix for it.** An earlier attempt fell back to the raw
signal when lengths did not match. For a 473-frame clip that returned 241,920
points — one per sample. Unaligned with the spectrogram's 473 frames, and
roughly 2 MB of JSON for a field meant to be a small envelope.

The correct fix is to **pad**. librosa's `center=True` framing means the signal
is slightly shorter than `n_frames * hop_length`, so a few zeros at the end
make the reshape work and keep the alignment exact.

The downsampling itself is the neat part:

```python
peaks = padded[:needed].reshape(n_frames, hop_length).max(axis=1)
```

`reshape(n_frames, hop_length)` turns a flat array into one row per frame, then
`max(axis=1)` takes the peak within each frame. One line, no loop, and the
result is exactly `n_frames` long — aligned with everything else by
construction.

**Peak, not mean.** A waveform display should show the envelope. Averaging
absolute amplitude over 32 ms would flatten transients — a plosive `t` would
vanish. The peak preserves them, which is what makes a waveform look like a
waveform.

---

## 10.6 The route

```python
"""Acoustic Wave Profiler exposure (LIT-231, FR10).

Pure DSP - no model load, safe to run synchronously on the request path
(unlike model inference, which is why FR3's async gateway rule doesn't apply
here; SAD §5.1's "gateway never loads AI models" is about AI models
specifically). Wraps `acoustic_profiler_service.extract_acoustic_profile`,
which already returns a JSON-safe dict.
"""
```

The exception to the never-block rule, reasoned rather than assumed (§3.2).

```python
@router.post("/acoustic/profile")
def acoustic_profile(http_request: Request, request: AcousticProfileRequest) -> dict:
```

Note: `def`, not `async def`. FastAPI runs a synchronous handler in its
threadpool automatically, so a slow one cannot block the event loop. This is
the correct way to write a handler that does blocking work — simpler and safer
than `async def` plus `to_thread`.

```python
    keys = ck.acoustic_keys(ck.both_hashes(resolved_path))
    for ns, key in keys:
        cached = get_result_sync(ns, key)
        if cached:
            return cached

    try:
        audio, sr = sf.read(str(resolved_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load audio file: {e}")

    try:
        prof = extract_acoustic_profile(audio, sr)
        for ns, key in keys:
            cache_result_sync(ns, key, prof, ttl=86400)
        return prof
    except Exception as e:
        logger.error("Acoustic profiling failed for %s: %s", resolved_path, e)
        raise HTTPException(status_code=500, detail=f"Acoustic profiling failed: {e}")
```

Points worth noting:

- **`always_2d=False`** then an explicit `if audio.ndim > 1: mean(axis=1)`.
  Mono comes back 1-D, stereo 2-D and is downmixed. The alternative
  (`always_2d=True` then always average) is also fine; what matters is that one
  of them is chosen explicitly rather than assuming mono.
- **No resampling.** The profiler takes `sr` as a parameter and derives its time
  axis from it, so it works at any rate. This is the one place in the system
  that does not force 16 kHz, because no model is involved.
- **Two separate `try` blocks** with distinct messages — "failed to load" and
  "profiling failed" are different problems and the error should say which.
- **Keys from `cache_keys`**, with the comment about drift (§8.7).

---

## 10.7 What the frontend does with it

The response feeds three canvas layers (Chapter 15):

| Field | Layer | Rendering |
|---|---|---|
| `spectrogram` | base | `[128][n_frames]` painted as greyscale via `createImageData` |
| `waveform` | overlay | white polyline, one point per frame |
| `timeline[].f0_hz` | overlay | cyan contour, pen lifted on `null` |
| `timeline[].rms` | chart | plotted in the profiler panel |

The pitch layer maps hertz to a y-coordinate using the mel formula, so the
contour sits at the right height over a mel spectrogram:

```tsx
const mapHzToY = (hz: number, maxFreq = maxFreqHz) => {
  const mel = 2595 * Math.log10(1 + hz / 500);
  const maxMel = 2595 * Math.log10(1 + maxFreq / 500);
  return height - (mel / maxMel) * height;
};
```

Linear placement over a mel axis would put every pitch value at the wrong
height — and it would look *almost* right, which is worse.

```tsx
// Upper bound of the spectrogram's mel axis. librosa defaults fmax to sr/2,
// so a fixed 8000 puts the F0 line at the wrong height on any other rate.
maxFreqHz = 8000,
```

A parameter with a documented default rather than a constant, because librosa's
`fmax` follows the sample rate.

---

## 10.8 Summary

- No model, no GPU, no queue. The physical ground truth to compare attributions
  against.
- Shared `frame_length`/`hop_length` across pitch, energy and spectrogram is the
  module's *purpose*, not a tidiness choice: identical framing is what makes the
  contours the same length and the overlay aligned.
- `center=True` on all of them, or you get a half-frame offset that looks fine
  and misattributes spikes by 16 ms.
- Unvoiced F0 is `NaN`, never `0.0` — a missing measurement and a measurement of
  zero are different claims. `None` in JSON, pen lifted in the UI.
- Unvoiced is decided by pYIN's flag *or* a confidence threshold, because the
  flag alone admits garbage values.
- Note names (`C2`, `C7`) instead of raw hertz, citing librosa's own speech
  example.
- Degenerate (silent) clips are guarded twice on the normalisation.
- The waveform envelope is `reshape(n_frames, hop).max(axis=1)` — one line, no
  loop, aligned by construction, and peak rather than mean so transients
  survive.
- A frontend layer that bails on empty data renders nothing forever if the
  backend never sends the field; the guard cannot tell "not yet" from "never".
- Synchronous handlers (`def`, not `async def`) are the right shape for blocking
  work in FastAPI.
- This is the only place that does not force 16 kHz, because no model is
  involved.

Next: [Chapter 11 — The mutation engine](11-perturbation.md).
