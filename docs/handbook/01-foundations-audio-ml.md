# Chapter 1 — Digital audio and the machine learning behind it

Nothing in this chapter is specific to AudioLIT. It is the vocabulary and the
mechanics every later chapter assumes. If you already know signal processing
and transformers, skim the summary boxes and move on — but do read §1.9, which
fixes the three tasks and the exact model checkpoints this project uses.

---

## 1.1 Sound, and how a computer can hold it

Sound is a pressure wave. A microphone converts pressure into a voltage that
rises and falls with it. That voltage is *continuous*: at every instant it has
some value. A computer cannot store infinitely many values, so it does two
things to make the signal finite.

**Sampling** measures the voltage at regular intervals. The **sample rate** is
how many measurements per second, in hertz (Hz). 16,000 Hz — written 16 kHz —
means 16,000 measurements per second.

**Quantisation** rounds each measurement to one of a finite set of levels. 16
bits per sample gives 65,536 levels. Audio processed by machine-learning code
is usually converted to 32-bit *floating point* in the range −1.0 to +1.0
instead, because that is what neural networks consume.

So a one-second mono clip at 16 kHz is an array of 16,000 floating-point
numbers. Ten seconds is 160,000 numbers. That array is the whole signal — the
"waveform".

```
amplitude
   +1.0 |      ..**..
        |    ..      ..        ..**..
    0.0 |---*------------**---*------**-----> time (sample index)
        |                  ..      ..
   -1.0 |
         0    1    2    3    4    5   ... 15999
```

**Channels.** A mono recording has one such array. Stereo has two (left and
right). AudioLIT always reduces to mono by averaging the channels, because
every model it uses was trained on mono.

### Why 16 kHz specifically

The **Nyquist–Shannon sampling theorem** says that to represent a signal
containing frequencies up to *f* Hz without ambiguity, you must sample at more
than 2*f* Hz. Sample too slowly and high frequencies *alias* — they get
misrecorded as lower frequencies that were never there, an error you cannot
undo afterwards.

Turn it around: sampling at 16 kHz faithfully represents everything below
8 kHz (the **Nyquist frequency**, half the sample rate). Human speech carries
nearly all of its intelligibility below 8 kHz. Music needs more, which is why
CDs use 44.1 kHz. Speech models therefore standardise on 16 kHz: it is the
cheapest rate that loses nothing important about speech.

This is why **16 kHz mono appears everywhere in this codebase**. Whisper,
Wav2Vec2 and every deepfake checkpoint used here were trained at 16 kHz mono.
Feed a model 44.1 kHz audio and it does not error — it interprets the array as
if it were 16 kHz, hears everything at roughly a third speed and a much lower
pitch, and returns confident nonsense. There is no exception anywhere in the
system.

> **Resampling** converts between rates. Going from 44.1 kHz to 16 kHz cannot
> be done by throwing away samples: that aliases. It requires low-pass
> filtering first (removing everything above the new Nyquist frequency) and
> then interpolating. `librosa.resample` does both. AudioLIT never implements
> this itself; it calls librosa.

---

## 1.2 Frequency: from a waveform to a spectrum

A waveform tells you amplitude over time. It does *not* directly tell you
which pitches are present, and pitch is what speech is made of. Getting from
one to the other is the job of the **Fourier transform**.

The core idea: any signal can be written as a sum of pure sine waves of
different frequencies, amplitudes and phases. The Fourier transform finds that
recipe. Feed it a chunk of waveform, get back "how much of each frequency is
in here".

The **Discrete Fourier Transform (DFT)** is the version for sampled data. The
**Fast Fourier Transform (FFT)** is an algorithm that computes the DFT in
O(n log n) instead of O(n²) — which is the only reason any of this is
practical. You will never implement an FFT; NumPy and librosa have one.

### The short-time Fourier transform (STFT)

One Fourier transform over a whole 10-second clip tells you which frequencies
occur *somewhere* in those 10 seconds, with no idea where. Useless for speech,
which changes every few tens of milliseconds.

The fix is to cut the signal into short overlapping **frames** and transform
each one separately. That is the **STFT**. Three parameters define it:

- **`n_fft`** — frame length in samples. The number of samples in each chunk.
  This codebase uses 2048 for profiling and display, 512 for some
  attribution work.
- **`hop_length`** — how far the window advances between frames. 512 samples
  at 16 kHz is 32 ms, so you get about 31 frames per second.
- **window function** — each frame is multiplied by a smooth taper (Hann is
  the default) before transforming. Without it, cutting the signal creates
  artificial discontinuities at the frame edges that show up as spurious
  frequencies — *spectral leakage*.

There is an unavoidable trade-off here, and it is a real physical limit, not
an implementation shortcoming. A long frame resolves frequency finely but
smears time. A short frame localises time but cannot distinguish nearby
frequencies. You cannot have both. (This is the same mathematics as the
Heisenberg uncertainty principle, which is not a metaphor — it is literally
the same inequality about conjugate variables.)

The output of an STFT is a 2-D complex array: `[frequency_bins, time_frames]`.
Take the magnitude (`np.abs`) and you have a **spectrogram** — an image where
the x-axis is time, the y-axis is frequency, and brightness is energy.

```python
# saliency_service.audio_to_spectrogram
def audio_to_spectrogram(audio, n_fft: int = 512, hop_length: int = 256):
    """Magnitude STFT spectrogram [freq, time] for a 1-D audio array."""
    return np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
```

**This is the single most important idea in the whole project.** Once audio is
a spectrogram, it is an *image*, and every visual-interpretability technique
built for images — heatmaps, Grad-CAM, patch occlusion — becomes available.
AudioLIT is fundamentally an exercise in treating audio as images without
forgetting that it is audio.

### Mel scale: spectrograms shaped like hearing

Human pitch perception is roughly logarithmic. The difference between 100 Hz
and 200 Hz sounds large; between 5000 Hz and 5100 Hz, inaudible. A linear
frequency axis therefore spends most of its resolution where your ear has
almost none.

The **mel scale** warps frequency to match perception. One common formula,
which is exactly the one the frontend canvas uses:

```
mel = 2595 * log10(1 + hz / 500)
```

A **mel spectrogram** groups the linear FFT bins into a smaller number of
mel-spaced bands (128 in this codebase, or 80 for Whisper's own input). Then
energy is converted to decibels, because loudness perception is also
logarithmic:

```
dB = 10 * log10(power / reference)
```

`librosa.power_to_db(S, ref=np.max)` does this, using the loudest point as the
reference so the result is ≤ 0 dB.

The complete pipeline, straight out of `acoustic_profiler_service.py`:

```python
S = librosa.feature.melspectrogram(
    y=audio, sr=sr, n_mels=128, n_fft=frame_length, hop_length=hop_length
)
S_db = librosa.power_to_db(S, ref=np.max)
s_min, s_max = float(S_db.min()), float(S_db.max())
S_norm = (S_db - s_min) / (s_max - s_min + 1e-8) if s_max > s_min else np.zeros_like(S_db)
```

The last step rescales to [0, 1] so the frontend can paint it directly as
pixel brightness. Note the `+ 1e-8`: without it, a perfectly flat (silent)
clip divides by zero. The `if s_max > s_min` guard catches the same case
explicitly. Both are there; belt and braces.

> **MFCCs** (mel-frequency cepstral coefficients) apply a further discrete
> cosine transform to the log-mel spectrogram, compressing it to ~13 numbers
> per frame. They dominated speech recognition before deep learning because
> they are compact and decorrelated. Modern neural models prefer raw log-mel
> (Whisper) or the raw waveform (Wav2Vec2) because they would rather learn the
> compression themselves. AudioLIT computes MFCCs only as descriptive features,
> never as model input.

---

## 1.3 Two features worth computing directly

Not everything useful needs a neural network. Two signal-level measurements
appear throughout AudioLIT because they describe the *physical* signal a model
was reacting to, which is exactly what you need to interpret the model.

### Fundamental frequency (F0) — pitch

Voiced speech (vowels, `m`, `z`) is produced by the vocal folds opening and
closing periodically. The rate of that vibration is the **fundamental
frequency**, F0, and it is what you hear as pitch. Unvoiced sounds (`s`, `f`,
`t`) are turbulent noise with no periodicity and therefore no F0 at all.

Typical ranges: adult male speech ~85–180 Hz, adult female ~165–255 Hz,
children higher. AudioLIT searches C2 to C7 (about 65 Hz to 2093 Hz), which is
librosa's own recommended range for speech.

Estimating F0 is harder than it looks, because of **octave errors**: a signal
at 200 Hz also has strong energy at 400 Hz and 600 Hz (its harmonics), and a
naive algorithm happily reports the wrong one. **pYIN** (probabilistic YIN)
handles this by computing a probability distribution over candidate pitches per
frame and then using a hidden Markov model to pick the path through time that
is both locally likely and smooth. It also returns a per-frame *voicing
probability*.

AudioLIT's use of it contains one decision that matters far more than it
looks:

```python
f0, voiced_flag, voiced_prob = librosa.pyin(
    audio, fmin=fmin, fmax=fmax, sr=sr,
    frame_length=frame_length, hop_length=hop_length,
)
f0 = np.asarray(f0, dtype=np.float64)
unvoiced = ~np.asarray(voiced_flag, dtype=bool) | (np.asarray(voiced_prob) < voiced_prob_threshold)
f0[unvoiced] = np.nan
return f0
```

Unvoiced frames become **NaN, not 0.0**. This is not cosmetic. A 0 Hz pitch is
a *claim* — it says "the pitch here is zero", which is meaningless. NaN says
"there is no pitch here", which is true. Downstream, the frontend lifts the
pen on NaN so the contour shows gaps where pitch was genuinely unmeasurable,
the way Praat and other analysis tools do. Plotting zeros instead would draw a
line crashing to the bottom of the chart during every `s` sound — a visual
artefact the user would reasonably interpret as data.

The API layer converts NaN to JSON `null`, because NaN is not valid JSON:

```python
"f0_hz": None if np.isnan(f0[i]) else float(f0[i]),
```

### RMS energy — loudness

**Root mean square** amplitude per frame: square every sample, average, take
the square root. It measures energy rather than peak amplitude, so it tracks
perceived loudness much better than a maximum would.

```python
rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
```

RMS also serves as a silence detector. `dataset_ingestion.is_silent` rejects
clips below an RMS of 1e-4, which catches empty files and dead recordings
before they reach an evaluation batch and get scored as a model failure.

### Why they share parameters

Both functions default to `frame_length=2048, hop_length=512`. That is
deliberate and it is the entire point of the module: identical framing means
both contours come out the same length, frame for frame, and can be zipped
into a single timeline.

```python
step_s = hop_length / sr
timeline = [
    {"t_ms": round(i * step_s * 1000.0, 3),
     "f0_hz": None if np.isnan(f0[i]) else float(f0[i]),
     "rms": float(rms[i])}
    for i in range(len(rms))
]
```

If the two used different hop lengths, aligning pitch against loudness would
require resampling one to the other, and every such resample is a chance to
introduce a half-frame offset that nobody notices but that quietly
misattributes a pitch spike to the wrong syllable.

---

## 1.4 Neural networks, in the amount of depth you need

A **neural network** is a function with adjustable numbers in it. You choose a
shape, initialise the numbers randomly, then repeatedly nudge them so the
function's outputs get closer to the outputs you wanted.

A single **layer** is typically:

```
output = activation(W · input + b)
```

`W` (weights) and `b` (bias) are the adjustable **parameters**. The
`activation` is a simple non-linear function — without one, stacking layers
would be pointless, because a composition of linear maps is just another
linear map. Common choices: **ReLU** (`max(0, x)`), **GELU**, **tanh**.

Stack layers and you have a **deep** network. The intermediate outputs are
**hidden states** or **activations**, and they are exactly what
interpretability work reads.

### Training, and the one piece of it you must understand

- A **loss function** scores how wrong the output is. Cross-entropy for
  classification, and more specialised losses for sequence tasks.
- **Backpropagation** computes the gradient of the loss with respect to every
  parameter — how much the loss would change per unit change in that
  parameter. It is the chain rule from calculus, applied mechanically
  backwards through the network.
- **Gradient descent** steps each parameter a little way against its gradient.
  Repeat over many batches of data.

**AudioLIT never trains anything.** It downloads models other people trained.
But it uses gradients constantly, and this is the crucial connection:

> The same gradient machinery that trains a network can be pointed at the
> *input* instead of the parameters. `∂output/∂input` says how much each input
> value influences the output. That is an explanation, and it is the
> foundation of every gradient-based attribution method in Chapter 2.

In PyTorch, `input.requires_grad_(True)` asks for that gradient to be tracked;
`loss.backward()` computes it; `input.grad` holds it.

Two modes matter:

- `model.eval()` — inference mode. Disables dropout, freezes batch-norm
  statistics. **Always set this before inference**, or outputs vary run to
  run. Every loader in this codebase calls it.
- `torch.no_grad()` — stop tracking gradients, roughly halving memory.
  Correct for plain prediction, but **wrong for attribution**, which needs the
  graph. You will see both used, deliberately, a few lines apart.

---

## 1.5 Convolutional layers, and why Grad-CAM needs them

A **convolution** slides a small window (a **kernel**) across the input,
computing a weighted sum at each position. The same kernel is used everywhere,
which gives two properties:

- **Parameter sharing** — a 3-wide kernel has 3 weights whether the input is
  100 or 100,000 long.
- **Translation equivariance** — a pattern is detected the same way wherever
  it occurs.

A layer has many kernels, each producing one **channel** of output. Early
channels learn simple things (edges, onsets); deeper ones learn combinations.

Crucially, a convolutional layer's output **keeps its spatial layout**:
position *t* of the output corresponds to a region around position *t* of the
input. That is what makes Grad-CAM possible — you can point at a location in
the activation and know where in the audio it came from. A fully-connected
layer destroys that correspondence entirely.

This is why AudioLIT's Grad-CAM hunts for the last convolutional layer:

```python
def find_last_conv_layer(model):
    last_conv = None
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
            last_conv = module
    if last_conv is None:
        raise ValueError("model has no Conv1d/Conv2d layer to attach Grad-CAM to")
    return last_conv
```

The *last* one, because it holds the most abstract features that are still
spatially located. And it raises rather than guessing when there is no
convolution at all — a model with no conv layer cannot be Grad-CAM'd, and
pretending otherwise would produce a map of nothing.

Both Whisper and Wav2Vec2 have convolutional front ends, so both qualify.

---

## 1.6 Transformers and self-attention

Every model AudioLIT touches is a **transformer**. Understanding self-attention
is required, because "attention as explanation" is a central theme of this
project — and a central cautionary tale.

### The problem transformers solve

Recurrent networks process a sequence one step at a time, carrying a hidden
state forward. Two consequences: they cannot be parallelised across time, and
information from step 1 must survive hundreds of sequential updates to reach
step 500. Transformers replace recurrence with an operation that lets every
position look at every other position directly, in one parallel step.

### Self-attention, mechanically

Each position's vector is projected three ways:

- **Query (Q)** — what this position is looking for
- **Key (K)** — what this position offers
- **Value (V)** — what this position contributes if attended to

Then:

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

Step by step:

1. `Q·Kᵀ` — dot every query against every key. Result is `[seq_len, seq_len]`:
   a compatibility score for each pair of positions.
2. `/ √d_k` — divide by the square root of the key dimension. Without it, dot
   products of high-dimensional vectors grow large, softmax saturates, and
   gradients vanish.
3. `softmax` — turn each row into a probability distribution summing to 1.
   **This matrix is "the attention weights".**
4. `· V` — each position's output is the attention-weighted average of all
   values.

**Multi-head attention** runs several of these in parallel with different
projections and concatenates the results, letting different heads specialise.
Whisper-base has 6 encoder layers with 8 heads each; larger checkpoints have
more of both.

### The shape that causes real engineering problems

Attention weights for one clip are `[layers, heads, seq_len, seq_len]`. The
last two dimensions are both sequence length, so **memory grows quadratically
with clip duration**.

Wav2Vec2 emits about 50 frames per second. A 14-second clip is ~700 frames.
Across 24 layers and 16 heads: 24 × 16 × 700 × 700 × 4 bytes ≈ **750 MB** —
for one clip, in float32. Converting that to Python lists for JSON multiplies
it several times over.

This is not theoretical. It happened:

```python
#: SER attention leaves the model pooled to at most this many frames per axis.
#: wav2vec2 emits ~50 frames/s, so raw attention is [heads, T, T] per layer and
#: grows quadratically with clip length: a 14 s clip is ~740 MB of tensors
#: across 24 layers, and converting that to Python lists (which the fallback
#: branches below did, unbounded) took the warmup worker past 5 GB until the
#: kernel OOM-killed it mid-run. Pooling (not truncating) keeps the whole clip.
SER_ATTENTION_MAX_FRAMES = 100
```

The fix pools rather than truncates — average-pooling down to 100×100 keeps
the whole clip at lower resolution, whereas truncating would silently discard
the end of it. After pooling, rows are renormalised so each still sums to 1
and remains a valid attention distribution:

```python
pooled = torch.nn.functional.adaptive_avg_pool2d(attn, size)
return pooled / pooled.sum(dim=-1, keepdim=True).clamp_min(1e-12)
```

### Encoder, decoder, and the three kinds of attention

- An **encoder** reads the whole input at once, every position attending to
  every other.
- A **decoder** generates output one token at a time, attending to what it has
  produced so far (*causally* — it must not see the future) and to the
  encoder's output.

That gives three attention varieties, and confusing them produces meaningless
explanations:

| Kind | Who attends to what | What it tells you |
|---|---|---|
| Encoder self-attention | audio frames → audio frames | which parts of the audio relate to each other |
| Decoder self-attention | output tokens → earlier output tokens | how the text depends on itself |
| **Cross-attention** | output tokens → audio frames | **which audio each word came from** |

Cross-attention is the one that answers "where in the clip did this word come
from". You will see `transcribe_whisper` probe `decoder_attentions`,
`attentions`, `cross_attentions` and `encoder_attentions` in turn, because
different transformers versions populate different fields.

> **`attn_implementation="eager"`.** Modern transformers default to fused
> attention kernels (FlashAttention, SDPA) that are much faster but never
> materialise the `[seq, seq]` matrix — so `output_attentions=True` returns
> nothing. Every model load in this codebase that needs attention passes
> `attn_implementation="eager"` to force the slow, explicit implementation.
> Forget it and you get silent `None`s, which is exactly the trap that led to
> the fabricated-attention fallback described in Chapter 2.

---

## 1.7 Whisper: the ASR model

**Automatic Speech Recognition** turns audio into text. Whisper is OpenAI's
encoder–decoder transformer, trained on 680,000 hours of multilingual audio
scraped from the web.

**Input.** Whisper does not take a waveform. It takes an 80-bin log-mel
spectrogram, computed at 16 kHz with a 25 ms window and 10 ms hop, **padded or
truncated to exactly 30 seconds** — 3000 frames, always. `WhisperProcessor`
does all of this:

```python
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
# shape: [1, 80, 3000]
```

That fixed 3000 is why you see `80 x 3000` in the LIME discussion later: it is
240,000 input cells, and treating each as an independent feature is what made
LIME collapse.

**Architecture.** Two conv1d layers downsample the mel frames, then the
transformer encoder processes them, then an autoregressive decoder emits text
tokens conditioned on the encoder output.

**Sizes.** tiny (39M parameters), base (74M), small (244M), medium (769M),
large-v3 (1550M). AudioLIT's aliases:

```python
_WHISPER_ALIASES = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
}
```

**Special tokens.** Whisper's decoder is steered by tokens rather than flags:
`<|startoftranscript|>`, a language token like `<|en|>`, a task token
(`<|transcribe|>` or `<|translate|>`), and optionally `<|notimestamps|>`.

### The language-detection trap

If you do not supply a language token, Whisper *infers* one from a short probe
of the audio. On clean audio this works. On heavily accented English it does
not, and the failure mode is spectacular.

From `accent_bias_profiler.make_whisper_transcriber`:

```python
# Force English decoding. Without this Whisper runs language
# identification per utterance, and on heavily accented English it
# selects the speaker's L1 and transcribes into that language, then
# loops. On L2-ARCTIC that produced Vietnamese and Arabic output with
# WER 22.30 and 17.80 (insertions far outnumbering the reference
# words), which dragged two cohort means from ~0.17 to >1.3 and made
# the accent-bias ranking a measure of language misdetection rather
# than of accent.
result = asr_pipeline(
    audio.astype(np.float32),
    chunk_length_s=30,
    generate_kwargs={"language": "en", "task": "transcribe"},
)
```

Read the numbers. Word Error Rate of 22.30 means the model emitted more than
twenty times as many words as the reference contained — it fell into a
repetition loop. And the consequence was not merely bad transcripts: the
*accent-bias measurement itself* became invalid. It was ranking cohorts by how
often Whisper misidentified their language, then reporting that as accent bias.
A measurement instrument that silently measures the wrong quantity is worse
than one that breaks.

The corpus is read English throughout. The language was known. It should never
have been guessed. The same fix is applied on the main transcription path,
with a fallback for English-only checkpoints that reject the kwargs:

```python
try:
    result = _run_pipe(generate_kwargs={"language": "english", "task": "transcribe"})
except ValueError:
    result = _run_pipe()
```

### Word timestamps, and two decodes that disagree

Whisper can emit per-word timings, but doing so requires a
timestamp-constrained decode — a *different* decode from the plain one. The two
can produce different words.

```
ground truth : "mine's in the door"
plain decode : "Mines in the door."
timed decode : "Minds in the door."
```

Both decodes ran correctly. But the transcript panel shows one and the XAI
segments are labelled from the other, so the user sees a saliency bar labelled
with a word the transcript never contained. That reads as a broken system.

The resolution keeps the more accurate plain decode as canonical and relabels
the timed segments from it when the two agree on word count:

```python
canonical_words = canonical.split()
chunks = result.get("chunks") or []
diverged = str(result.get("text", "")).strip() != canonical.strip()

if diverged and len(canonical_words) == len(chunks):
    for chunk, word in zip(chunks, canonical_words):
        chunk["text"] = f" {word}"
    diverged = False

result["text"] = canonical
result["word_labels_diverged"] = diverged
```

When the word counts differ, it does **not** force a mapping — it keeps the
timed decode's own labels and sets `word_labels_diverged: True`. Forcing an
alignment there would put the right word on the wrong time interval, which is
a silent lie. Reporting the divergence is an honest partial answer.

---

## 1.8 Wav2Vec2: the SER and deepfake backbone

**Wav2Vec2** (Meta AI) takes the **raw waveform**, not a spectrogram, and is
used here for both emotion recognition and deepfake detection.

**Architecture.**
1. A **convolutional feature encoder** — seven conv1d layers — maps the
   waveform to ~50 frames per second.
2. A **transformer encoder** contextualises those frames.
3. For pretraining, a quantisation module and a contrastive objective.

**Self-supervised pretraining** is what makes it useful: it learns from
unlabelled audio by masking spans of its own convolutional output and learning
to identify the correct quantised representation for each masked span from
among distractors. Millions of hours of unlabelled speech can be used this way.
Then a small **head** is fine-tuned on a small labelled dataset for the actual
task.

`Wav2Vec2ForSequenceClassification` is that arrangement: the pretrained
encoder, mean-pooling over time, then a `projector` linear layer and a
`classifier` linear layer producing one logit per class.

### The head-mismatch failure, and why it is terrifying

This is the single most instructive bug in the project. Read the whole comment:

```python
# ECHO's inherited default was `r-f/wav2vec-english-speech-emotion-recognition`.
# It is unusable here for two independent reasons, both verified against the hub:
#
#   1. It publishes only `pytorch_model.bin` - no safetensors. The registry
#      fetches safetensors exclusively (SAD constraint C3), so it refuses the
#      checkpoint outright: every SER call raised ModelRegistryError.
#   2. Even loaded directly, its head does not match the class we load it with.
#      Its config declares `Wav2Vec2ForCTC` and `finetuning_task: wav2vec2_clf`,
#      and its weights carry a custom `classifier.dense` / `classifier.out_proj`
#      head. `Wav2Vec2ForSequenceClassification` expects `projector` +
#      `classifier`, so all four head tensors are randomly initialised and all
#      four trained ones discarded - predictions become chance-level noise that
#      changes with the torch seed.
```

Point 2 is the important one. Hugging Face's `from_pretrained` loads the
parameters whose names match the class you asked for and **randomly initialises
the rest, with a warning most people never read**. If the names do not match,
you get a model with a perfectly trained encoder and a completely random
classifier. It runs. It returns a label and a confidence. The confidence looks
high. The answer is noise, and it changes if you change the random seed.

Nothing crashes. No test that only checks "did we get a label back" catches it.
This is the archetype of the bug class this project is built to expose.

The defence is a declared list of parameters that *must* come from the
checkpoint:

```python
_EMO_REQUIRED_HEAD_KEYS = ("projector.weight", "projector.bias",
                           "classifier.weight", "classifier.bias")
```

If any is missing at load time, the classifier is random and every prediction
is noise — so the loader checks rather than hopes.

The replacement checkpoint is pinned by commit SHA, not by branch:

```python
_EMO_MODEL_ID = "firdhokk/speech-emotion-recognition-with-facebook-wav2vec2-large-xlsr-53"
_EMO_MODEL_REVISION = "611e6db8ee667aa07fe66596f9fc761e036ff5b9"
```

The comment is explicit about why: *"Do not float this to `main` — the head
layout is the thing that broke last time, and a silent upstream re-upload would
break it again."* A branch name is a moving target. A SHA is not.

---

## 1.9 The three tasks

### ASR — Automatic Speech Recognition

Audio → text. Model: Whisper. Metric: **Word Error Rate**, built from scratch
in Chapter 13.

### SER — Speech Emotion Recognition

Audio → one of several emotion categories, plus a probability for each. The
committed requirement is at least six categories (angry, disgust, fear, happy,
neutral, sad); the pinned checkpoint ships seven:

```python
_EMO_PINNED_LABELS = ("angry", "disgust", "fearful", "happy",
                      "neutral", "sad", "surprised")
```

Note "fearful" and "surprised", not "fear" and "surprise" — the previous
default spelled them differently, so anything hardcoding the old spellings
breaks. Runtime code therefore reads `config.id2label` from the checkpoint
rather than any hardcoded tuple. The tuple exists only so that category
coverage can be asserted in an offline test.

The inference is short and worth reading in full:

```python
feature_extractor, emo_model, emo_device = ensure_emo_model_loaded(model_id)
audio, rate = librosa.load(audio_path, sr=16000)
inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
input_values = inputs.input_values.to(emo_device)
attention_mask = inputs.attention_mask.to(emo_device) if "attention_mask" in inputs else None

with torch.no_grad():
    logits = emo_model(input_values=input_values, attention_mask=attention_mask).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]

id2label = emo_model.config.id2label if isinstance(emo_model.config.id2label, dict) else {}
probabilities = {
    id2label.get(i, id2label.get(str(i), f"emotion_{i}")): float(p)
    for i, p in enumerate(probs)
}
predicted_emotion = max(probabilities, key=probabilities.get) if probabilities else None
return {
    "predicted_emotion": predicted_emotion,
    "probabilities": probabilities,
    "confidence": float(probs.max()),
}
```

**Logits** are the raw pre-softmax scores, any real number. **Softmax**
exponentiates and normalises them into a probability distribution:
`softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)`. The `id2label.get(i, id2label.get(str(i), ...))`
double lookup exists because some checkpoints key that dict by integer and
others by string — a real incompatibility between checkpoints, not paranoia.

SER is genuinely hard and its ceiling is low. Human annotators agree with each
other only ~70–80% of the time on acted emotional speech, and acted emotion
differs systematically from spontaneous emotion. Treat any SER accuracy figure
with that in mind.

### ADD — Audio Deepfake Detection

Audio → bona-fide or spoof, with a probability. Synthetic speech leaves
artefacts: unnaturally smooth pitch, missing breath noise, phase
inconsistencies, characteristic spectral signatures of the vocoder that
generated it.

Label naming across checkpoints is a mess — real/fake, genuine/spoof,
bonafide/synthetic — so AudioLIT normalises:

```python
def _normalize_deepfake_label(raw) -> str:
    key = str(raw).strip().lower()
    if any(tok in key for tok in ("spoof", "fake", "synthetic", "generated", "deepfake")):
        return DEEPFAKE_SPOOF
    return DEEPFAKE_BONA_FIDE
```

Anything meaning "generated" maps to spoof; everything else to bona-fide. The
default is the safer direction to be wrong in for a forensic tool.

#### The checkpoint that was worse than a coin flip

The original default was benchmarked against an alternative on 200 labelled
clips — 100 genuine from common-voice/CREMA-D/RAVDESS, 100 spoof from ASVspoof
2021 DF:

```
melody-machine  38.5% accuracy, 48/100 false alarms, 75/100 misses,
                mean P(spoof) 0.48 on genuine vs 0.25 on spoof
                -- separation -0.23, i.e. anti-correlated with the truth
wav2vec2-add    88.5% accuracy, 0/100 false alarms, 23/100 misses,
                mean P(spoof) 0.08 on genuine vs 0.70 on spoof
                -- separation +0.62
```

38.5% on a balanced binary task is below the 50% you get by guessing. The
separation is *negative*: the model assigned **higher** spoof probability to
genuine audio than to spoof audio. It was systematically anti-correlated with
the truth — and reporting 0.9999 confidence while doing so.

Two things to take from this. First, a confident model can be worse than
random, and confidence carries no information about correctness. Second, the
comment records the diagnosis that mattered:

> *This is a checkpoint quality difference, not a preprocessing bug:
> `predict_deepfake` was verified to match a plain reference implementation on
> 12/12 clips.*

Before blaming the checkpoint, they proved the code was right. That ordering is
the discipline — otherwise you swap checkpoints to hide your own bug.

Both remain selectable, because a tool for studying model behaviour has
legitimate use for a bad model.

#### The confidence timeline

Clip-level detection says *whether*. The forensic question is *where*.
`predict_deepfake_timeline` slides a window across the clip:

```python
win = int(window_s * rate)
hop = max(1, int(win * (1.0 - overlap)))
if len(audio) <= win:
    starts = [0]
    win = len(audio)
else:
    starts = list(range(0, len(audio) - win + 1, hop))
    # keep the tail rather than silently dropping up to `hop` samples
    if starts[-1] + win < len(audio):
        starts.append(len(audio) - win)
```

Default: 1-second windows, 50% overlap. Two details worth copying. The tail
handling adds a final window flush to the end of the clip, because
`range(0, len-win+1, hop)` otherwise drops up to `hop` samples — and a
deepfake artefact in the last half second would be invisible. And the model is
loaded **once** before the loop: *"reloading per window turned a 5 s clip into
nine full model loads."*

---

## 1.10 Where models come from

**Hugging Face Hub** is a repository of pretrained models. Each repo has an id
like `openai/whisper-base`, a `config.json` describing the architecture, and
weight files.

Two weight formats, and the difference is a security boundary:

- **`.bin`** — Python `pickle`. Unpickling **executes arbitrary code by
  design**. Loading an untrusted `.bin` is equivalent to running an untrusted
  script.
- **`.safetensors`** — a flat header of tensor names, shapes and dtypes, plus
  raw bytes. No code path, no execution. Also memory-mappable, so loading is
  faster.

AudioLIT accepts safetensors only, and enforces it *before* downloading
weights. Chapter 5 covers the enforcement in detail.

The **transformers** library provides `AutoModel`, `AutoProcessor` and
`pipeline`. A `pipeline` bundles preprocessing, inference and postprocessing —
convenient, but it hides the model, which is why attribution code in this
project works with the model directly and only plain transcription uses the
pipeline.

---

## 1.11 Summary

| Concept | Why AudioLIT needs it |
|---|---|
| Sampling, 16 kHz mono | Every model's fixed input contract; violating it produces confident nonsense |
| STFT, spectrogram | Turns audio into an image so visual XAI applies |
| Mel scale, dB | Makes that image match human hearing |
| F0 (pYIN), RMS | Describes the physical signal a model reacted to; NaN for unvoiced is a correctness requirement |
| Gradients w.r.t. input | The mechanism behind every gradient-based explanation |
| Convolutional layers | Preserve spatial layout, which is what Grad-CAM needs |
| Self-attention | The models' core operation; its `[seq, seq]` shape is a real memory problem |
| Cross-attention | The only attention that maps words to audio positions |
| `eager` attention | Required or `output_attentions` silently returns nothing |
| Whisper | ASR; needs an explicit language token or it guesses and loops |
| Wav2Vec2 | SER and ADD backbone; head-name mismatches silently randomise the classifier |
| Softmax, logits | Turning raw scores into probabilities |
| safetensors vs pickle | A code-execution boundary, not a file-format preference |

Next: [Chapter 2 — Explainable AI for audio](02-xai-theory.md).
