# Chapter 17 — Building it from scratch

Everything so far explained what exists and why. This chapter is the
implementation plan: what to build in what order, what each stage should prove
before you move on, and which trap you will hit at each step.

You will not build the whole thing. You will build a version, discover which
parts matter for your purposes, and go deeper there. This chapter is ordered so
that each stage is independently useful.

---

## 17.1 Prerequisites

### What you need to know

- **Python** — functions, classes, dicts, list comprehensions, `try/except`,
  virtual environments, `pip`.
- **JavaScript/TypeScript** — enough to read a React component. You can learn
  React as you go; you cannot learn Python as you go here.
- **A terminal** — `cd`, running commands, environment variables.
- **git** — commit, branch, merge.

You do **not** need machine learning, signal processing, or web development
experience. Chapters 1–2 cover the theory; everything else is explained where
it is used.

### What you need installed

```bash
python --version   # 3.10 or 3.11
node --version     # 20
docker --version   # any recent version
```

Python 3.10 is what CI uses, so it is the version your code is judged on. 3.11
works locally.

### Documentation to keep open

You will read these constantly, and they are the only sources you need:

| Library | Why |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | the backend framework |
| [PyTorch](https://pytorch.org/docs/stable/index.html) | tensors, hooks, devices |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers) | models and pipelines |
| [librosa](https://librosa.org/doc/latest/index.html) | audio loading, STFT, pYIN, effects |
| [soundfile](https://python-soundfile.readthedocs.io/) | audio I/O |
| [Captum](https://captum.ai/api/) | Integrated Gradients, LIME, GradientShap |
| [RQ](https://python-rq.org/) | job queues |
| [Redis](https://redis.io/docs/latest/commands/) | the commands you will use |
| [NumPy](https://numpy.org/doc/stable/) | arrays, indexing, interpolation |
| [scikit-learn](https://scikit-learn.org/stable/) | PCA, t-SNE |
| [React](https://react.dev/) | components, hooks, effects |
| [MDN Canvas](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API) | the overlay drawing |
| [Playwright](https://playwright.dev/) | end-to-end tests |

---

## 17.2 The ten stages

Each stage ends with a **proof** — a specific thing you can demonstrate. Do not
move on until the proof passes. That discipline is what stops you from building
three stages on top of a broken assumption.

---

### Stage 1 — Audio in, audio out

**Build:** a FastAPI app with one `POST /upload` endpoint that validates,
stores and reports an audio file. Plus `GET /upload/file/{id}` to serve it back.

**Why first:** everything downstream needs a clip identified by an id. And
this endpoint contains the most validation per line of any part of the system,
so it teaches the codebase's style.

```bash
mkdir -p backend/app/api/routes
cd backend
python -m venv .venv
.venv/bin/pip install "fastapi>=0.141" "uvicorn[standard]" python-multipart soundfile librosa numpy
```

Follow Chapter 4. In order: UUID filenames, chunked streaming with a byte
counter, decode validation with soundfile then librosa, the duration cap,
`except HTTPException: raise` before the generic handler.

**The traps, in the order you will hit them:**

1. Forgetting `python-multipart`. FastAPI raises a confusing error about form
   data rather than saying "install this".
2. Using the uploaded filename as a path. Do not. Generate a UUID.
3. Checking size after writing. Count bytes as you stream.
4. Putting the generic `except Exception` before `except HTTPException`, which
   converts every deliberate 4xx into a 500.
5. Using a name in a log line that you never defined. This exact bug lived in
   this codebase for months (§4.3) — because the error path never ran.

**Proof:** upload a WAV, get a `file_id` and the correct duration. Upload a
`.txt` renamed to `.wav` and get **422** with a message about decoding. Upload
a 200 MB file and get **413**. Test all three, including the failures.

---

### Stage 2 — One prediction

**Build:** `POST /inferences/run` that transcribes an uploaded clip with
Whisper.

```bash
.venv/bin/pip install "torch>=2.6.0" transformers
```

```python
from transformers import pipeline
import librosa, numpy as np

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=-1)
audio, _ = librosa.load(path, sr=16000)
result = pipe(audio.astype(np.float32), chunk_length_s=30,
              generate_kwargs={"language": "english", "task": "transcribe"})
print(result["text"])
```

**The traps:**

1. **Not passing `sr=16000`.** The model interprets the array as 16 kHz
   whatever the file's rate, so a 44.1 kHz clip is heard at a third speed. No
   error — just nonsense. §1.1.
2. **Not forcing the language.** Whisper guesses, and on accented English it
   guesses wrong and loops. §1.7.
3. **Building the pipeline inside the request handler.** 5 seconds per call.
   Build it once at module level or in a cache. §6.2.
4. **Blocking the event loop.** Use `def` rather than `async def` for a
   handler that does blocking work, and FastAPI will run it in its threadpool.
   §10.6.
5. **`chunk_length_s` shorter than 30.** Changes the transcript. §6.3.

**Proof:** transcribe a clip and get sensible text. Transcribe the same clip
twice and confirm the second is fast (the model was not reloaded). Time both
and look at the numbers.

---

### Stage 3 — Caching

**Build:** Redis, and a cache around Stage 2.

```bash
docker run -d -p 6379:6379 --name my-redis redis:7-alpine \
  --maxmemory 512mb --maxmemory-policy volatile-lru
.venv/bin/pip install redis
```

Follow Chapter 8. Key on `sha256(audio_bytes) + model + task`, build keys in
functions, set a TTL, swallow exceptions on the cache path only.

**The traps:**

1. **Keying on the path instead of the content.** Two copies cache twice; an
   in-place edit serves stale. §8.2.
2. **Omitting the model from the key.** The first model to run owns the entry
   and every other model reads its answer. This happened here. §8.3.
3. **`allkeys-lru` when the same Redis is your queue broker.** It will evict
   your queues. §3.6.
4. **Non-canonical params in the key.** `json.dumps(params, sort_keys=True,
   separators=(',',':'))` or logically identical requests miss. §8.6.
5. **Letting a cache failure fail the request.** A cache is an optimisation.

**Proof:** first call slow, second call fast, same answer. Two different models
on one clip give two different cached answers. Kill Redis and confirm the API
still works, slowly.

---

### Stage 4 — Async work

**Build:** RQ, a worker process, and a WebSocket that streams progress.

```bash
.venv/bin/pip install rq
```

Follow Chapter 7. `SimpleWorker`, one queue per family, progress over Redis
pub/sub, a WebSocket relay, an HTTP polling fallback.

**The traps:**

1. **Using RQ's default forking `Worker`.** The model loads in a child that
   exits after each job, so you reload every time. Use `SimpleWorker`. §7.5.
2. **Calling blocking Redis inside `async def`.** `loop.run_in_executor` or
   you serialise the entire server. §7.8.
3. **Not sending initial state on WebSocket connect.** If the job finished
   before the client connected, pub/sub has no history and the client waits
   forever. §7.8.
4. **One Redis connection for both request path and workers.** RQ's blocking
   dequeue outlives a short `socket_timeout`, so idle workers die. §7.3.
5. **Trusting RQ's `perform_job` to raise on failure.** It returns `False`.
   §7.5.
6. **Two spellings of the progress channel prefix.** One constant, one module.
   §7.1.

**Proof:** enqueue a transcription, get a job id in under 50 ms, watch state
transitions arrive over the WebSocket, receive the result. Kill the worker
mid-job and confirm you see a failure rather than a hang. Block WebSockets in
your browser's dev tools and confirm polling takes over.

---

### Stage 5 — The frontend shell

**Build:** a React app that uploads a file, shows a waveform, and displays a
transcript.

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend && npm install wavesurfer.js
```

Follow Chapter 15 for `API_BASE`, `useTaskStatus` and `PlaybackContext`.

**The traps:**

1. **Deriving the WebSocket URL from `window.location`.** Points at the dev
   server and forces `ws://` on `https://`. §15.3.
2. **Forgetting CORS on the backend.** Add `CORSMiddleware` with a restricted
   origin list, never `*` with credentials. §3.9.
3. **Mixing `localhost` and `127.0.0.1`** between frontend and API. That is
   cross-site for `SameSite=Lax` cookies, and sessions break silently. §3.9.
4. **Not cleaning up timers and sockets** in the `useEffect` return. §15.6.
5. **Not clearing the previous task's result** when the id changes — the old
   answer flashes on screen. §15.6.

**Proof:** upload from the browser, see the waveform, see the transcript
appear, watch the status change live. Switch to a second clip and confirm no
flash of the first one's result.

---

### Stage 6 — The acoustic profiler

**Build:** `POST /acoustic/profile` returning pitch, energy, spectrogram and a
waveform envelope. Then draw it on a canvas.

Follow Chapter 10. The one thing that matters most: **shared
`frame_length`/`hop_length`** across all three computations.

**The traps:**

1. **Returning `0.0` for unvoiced pitch.** Use `NaN` internally and `null` in
   JSON, or your contour crashes to the bottom of the chart on every `s`. §10.3.
2. **Serialising `NaN` to JSON.** It is invalid and most parsers reject the
   whole document.
3. **Different hop lengths** between pitch and energy — the contours differ in
   length and are offset by half a frame. §10.4.
4. **Not sending the waveform envelope** when your canvas layer needs it. The
   layer silently draws nothing, and its empty-data guard cannot distinguish
   "not yet" from "never". §10.5.
5. **Placing the pitch line linearly over a mel-scaled spectrogram.** It looks
   almost right, which is worse than looking wrong. §15.5.

**Proof:** pitch and energy arrays are the same length. The pitch contour has
visible gaps during silence. The pitch line sits on the visible harmonic in the
spectrogram.

---

### Stage 7 — Your first explanation

**Build:** `POST /saliency/generate` with Integrated Gradients only. Draw the
heatmap over the spectrogram.

```bash
.venv/bin/pip install captum
```

**Start with IG, not Grad-CAM.** IG needs no conv layer, no
class-discriminative target, and no interpolation from a different resolution.
It is the one method that works the first time.

Follow Chapters 2 and 9.

**The traps:**

1. **Forgetting `requires_grad_(True)` on the input.** No gradient, no
   attribution. §9.4.
2. **Wrapping attribution in `torch.no_grad()`.** It needs the graph. §1.4.
3. **Averaging signed attributions.** Use `mean(|·|)` or +0.5 and −0.5 cancel
   to "unimportant". §9.6.
4. **Not normalising before drawing.** Attribution magnitudes are arbitrary;
   the canvas needs [0, 1].
5. **Using jet as your colour ramp.** Use Viridis. Jet invents banding that is
   not in the data. §15.5.
6. **No colourbar.** A heatmap without a scale is not readable.
7. **Drawing a matrix with `fillRect` per cell.** Use `createImageData` and
   direct byte writes. §15.5.
8. **Running attribution on the event loop.** `asyncio.to_thread`, and then a
   per-model lock, because attribution mutates shared model state. §9.12.

**Proof:** upload a clip with one loud word in silence. The heatmap should be
brightest near that word. If it is uniform, or brightest in the silence,
something is wrong — investigate before continuing.

That proof is the important one. **A plausible-looking heatmap is not
evidence.** Construct an input where you know what the answer should be.

---

### Stage 8 — Provenance, before you add more methods

**Build:** the `Provenance` enum, `provenance_fields`, and surface it in the UI.

This stage looks like paperwork. It is the most valuable stage in the plan.

```python
class Provenance(str, Enum):
    MEASURED = "measured"
    FALLBACK = "fallback"
    UNAVAILABLE = "unavailable"


def provenance_fields(source: Provenance, reason: str | None = None) -> dict:
    if source == Provenance.FALLBACK:
        if not reason or not reason.strip():
            raise ValueError("Provenance.FALLBACK requires a non-empty reason string")
        return {"provenance": source.value, "provenance_reason": reason.strip()}
    return {"provenance": source.value,
            "provenance_reason": reason.strip() if reason else None}
```

**Do it now, before adding Grad-CAM, LIME and SHAP.** Every one of those has a
failure mode where it returns a plausible but empty or fallback result. This
project shipped *every* Whisper Grad-CAM as a secretly-fallback energy map, for
a long time, because there was no provenance field to make it visible (§2.3).

The `raise` on a reasonless fallback is what makes the contract real rather
than aspirational.

**Proof:** deliberately break your IG path (return zeros). The response must
say `provenance: "fallback"` with a reason, and the UI must show it.

---

### Stage 9 — The remaining methods

**Build:** Grad-CAM, LIME, SHAP. Follow Chapters 2 and 9.

**Grad-CAM traps:**

1. **No class-discriminative target.** For a classifier use the predicted
   class logit. For ASR you must build something like
   `_WhisperTranscriptScore` — encoder energy gives a uniformly negative map
   that ReLU annihilates. §2.3.
2. **Hooking the wrong model instance's layer.** If your target wrapper loads a
   different class, hook *its* encoder. §2.3.
3. **Not removing hooks.** `try/finally`. A leaked hook fires on every later
   forward pass and holds tensors alive. §2.3.
4. **Not zeroing gradients.** They accumulate. §2.3.

**LIME traps:**

1. **No feature mask.** Every mel cell becomes a feature: 240,000 unknowns from
   50 samples, and the surrogate returns zeros. Group into time bands. §2.4.
2. **Lasso as the surrogate.** Against an unbounded target it zeroes every
   coefficient. Use Ridge. §2.4.
3. **Too few samples.** At least 4× your feature count.

**SHAP traps:**

1. **Not capping duration.** It is the most expensive method; give it a
   stricter cap.
2. **Not handling OOM.** Retry with fewer samples, then degrade with a recorded
   reason.

**Proof:** all four methods on the same clip produce **different** maps, each
labelled `measured`. If two are identical, your dispatch is falling through —
which is exactly what the differential test in §16.2 catches.

---

### Stage 10 — Perturbation and faithfulness

**Build:** the mutation engine (Chapter 11), then the deletion-score audit
(Chapter 13).

This closes the loop: attribution claims, perturbation tests, faithfulness
quantifies.

**Perturbation traps:**

1. **Mutating the input.** Clone. Always.
2. **`numpy.T` without `.copy()`** before `torch.from_numpy`. Non-contiguous
   memory. §11.2.
3. **Not reconciling `istft` output length.** It does not return the input
   length. §11.3.
4. **Passing hertz to `scipy.signal.butter`.** It wants frequencies normalised
   to Nyquist, and `output='sos'` not the default. §11.3.
5. **Not reporting what was applied.** A silently failed perturbation in a
   chain makes every conclusion wrong. §11.4.

**Faithfulness traps:**

1. **Computing degraded confidence arithmetically instead of re-running the
   model.** This project did it, and the metric would have certified random
   noise as faithful. There is no acceptable shortcut. §13.6.
2. **Comparing max confidence rather than the original class's probability**
   for a multi-class model. A flipped prediction then looks like a confidence
   *increase*. §13.7.
3. **Not interpolating** between frame-resolution attributions and
   sample-resolution audio. You mask the wrong audio. §11.7.
4. **Returning 0.0 when nothing could be measured.** 0.0 means "completely
   unfaithful"; use `None`. §13.6.
5. **Auditing a fallback attribution.** Refuse, and count the refusal. §13.6.

**Proof:** two audits on one clip — one with your real attribution, one with a
*random* attribution. The real one must score meaningfully higher. **If random
noise scores as faithful, your metric is broken, not your model.**

That is the single most important proof in the plan. It is the test the
fabricated metric would have failed.

---

## 17.3 What to build after that

Ordered by value, with the chapter that covers each:

| Feature | Chapter | Why |
|---|---|---|
| SER and deepfake detection | 6 | two more tasks, same pattern |
| Model registry with safetensors + pinning | 5 | required before accepting arbitrary user models |
| Fan-out/fan-in multitask | 7 | all three predictions at once |
| Latent projection | 12 | model-level rather than prediction-level interpretability |
| Dataset loaders | 14 | batch evaluation needs corpora |
| Accent-bias profiling | 13 | the bias finding is the research output |
| MongoDB metadata tier | 3 | durable records, reproducibility |
| Docker Compose | 3 | one-command deployment |
| Cross-browser E2E + accessibility | 16 | catches what unit tests cannot |

---

## 17.4 The ten decisions that matter most

If you take nothing else from this handbook:

**1. 16 kHz mono, everywhere, no exceptions.** Every model requires it.
Violating it produces confident nonsense with no error.

**2. Never substitute a model silently.** If you cannot load what was
requested, fail. A fallback to a default returns another model's answer under
the requested model's name, and the cache makes it permanent. This bug appeared
three times in this codebase, in three different places.

**3. Every explanation carries provenance.** Measured, fallback with a
mandatory reason, or unavailable. Build this before you build your second
attribution method, not after.

**4. Never fabricate a measurement.** If you cannot compute it, return `None`
and say why. A plausible number is worse than a missing one, and this is
doubly true for a metric whose purpose is detecting fabrication.

**5. `None`, not zero, for "could not measure".** Zero is a *value*, and for a
deletion score it means "completely unfaithful" — the opposite of "unknown".
Same for unvoiced pitch (`NaN`/`null`) and empty cohort statistics.

**6. Never let the gateway load a model.** Enqueue and return a job id.
Anything else stalls the server.

**7. Cache on content, and put every discriminator in the key.** SHA-256 over
audio bytes plus model plus task plus canonical params. One payload shape per
key family.

**8. A right key with a wrong value shape is worse than a cache miss.** The
consumer will not recompute — it will read your value and fail on it.

**9. Attribution mutates shared model state.** Serialise per model identity,
including against plain inference on the same model. Hooks fire on any forward
pass through their module.

**10. Test what a wrong answer would look like.** Type-and-shape assertions
catch almost none of this system's bugs. Construct inputs where you know the
answer; compare two calls; verify your test by breaking the code.

---

## 17.5 The mindset

The recurring theme across every incident in this handbook:

> **A contract asserted on one side and never checked on the other.**

- The pipeline cache existed; nothing called it.
- The duration was computed; nothing compared it.
- Provenance was documented; the fallback did not set it.
- The retention TTL was asserted in a comment; no code implemented it.
- The waveform layer existed; no backend field fed it.
- The key family documented a payload shape; a writer used a different one.
- The `Path:` field named a directory; the directory had been deleted.

Every one is a place where one side of a contract was written and the other
side was assumed. And in each case the system kept working, plausibly, while
being wrong.

The discipline that catches these is small and specific:

- **When you write a helper, check something calls it.** `grep` for the name.
- **When you compute a value, check something uses it.**
- **When you document a contract, check the other side honours it.**
- **When a guard fires, find out how often.** A fallback that fires every time
  is a bug, not a fallback.
- **When two tools disagree, chase the contradiction.** Do not average, and do
  not pick the number you expected.
- **When something matters, spend the one command it takes to check the tree.**
  `ls` the directory. Read the file. Run the merge.

And the disposition underneath all of it:

> **Refuse, and say why.**

An unsupported architecture raises with a code. An unknown reduction method
raises rather than falling back to PCA. A fallback attribution is refused by the
auditor and counted. A corpus with no loader raises naming its issue. A
fabricated metric was deleted rather than kept.

For a tool whose entire output is meant to be trusted as measurement, that is
not caution. It is the product.

---

## 17.6 Where to go for more

- **This handbook's chapters** — [1](01-foundations-audio-ml.md) through
  [16](16-testing.md).
- **`docs/README.md`** — conventions and errata. Read **E4** before citing any
  functional-requirement number; the submitted specification and the codebase
  number them differently.
- **`docs/SAD.md`** and **`docs/SRS.md`** — the architecture and requirements
  of record.
- **`docs/evaluation/DEFECT_LOG.md`** — every defect, how it was found, and the
  three findings that were wrong on first measurement.
- **`docs/testing/AudioLIT_Test_Plan_Report_Merged.md`** — the test and
  evaluation report, including the requirement-to-test matrix.
- **`docs/DEMO_RUNBOOK.md`** — how to actually start the stack.

### The papers

| Paper | What for |
|---|---|
| Sundararajan, Taly & Yan (2017), *Axiomatic Attribution for Deep Networks* | Integrated Gradients, and its axioms |
| Selvaraju et al. (2017), *Grad-CAM* | Grad-CAM |
| Ribeiro, Singh & Guestrin (2016), *"Why Should I Trust You?"* | LIME |
| Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions* | SHAP |
| Jain & Wallace (2019), *Attention is not Explanation* | why attention is not attribution |
| Wiegreffe & Pinter (2019), *Attention is not not Explanation* | the nuanced rebuttal |
| Radford et al. (2022), *Robust Speech Recognition via Large-Scale Weak Supervision* | Whisper |
| Baevski et al. (2020), *wav2vec 2.0* | Wav2Vec2 |
| Mauch & Dixon (2014), *pYIN* | the pitch tracker |

---

Next: [Chapter 18 — Testing and evaluation in practice](18-testing-and-evaluation-in-practice.md),
the full record of how this system was verified.
