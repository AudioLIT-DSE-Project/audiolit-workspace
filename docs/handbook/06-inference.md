# Chapter 6 — Inference: ASR, SER and deepfake detection

`Backend/app/domain/model_loader_service.py` is 1804 lines and the largest
module in the project. It is also the messiest, because most of it is inherited
from the ECHO baseline and has been repaired rather than rewritten. Reading it
teaches you as much about working in an existing codebase as about inference.

This chapter covers what each entry point does, the fixes that matter, and the
patterns worth copying.

---

## 6.1 Model selection: never substitute silently

```python
# Whisper model resolution (FR1).
#
# Every entry point below takes whatever the user selected - a short alias
# from the built-in list, or an arbitrary Hugging Face repo id added at
# runtime - and resolves it here. Nothing downstream may substitute a
# different checkpoint: doing so silently returns another model's output
# under the requested model's cache key, which breaks both FR1 (the selected
# model is the one that runs) and FR4 (a key identifies exactly one result).
DEFAULT_WHISPER_MODEL_ID = "openai/whisper-base"

_WHISPER_ALIASES = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
}


def resolve_whisper_model_id(model: str | None) -> str:
    """Map a selection to a concrete Hugging Face id.

    A bare size alias expands to the matching OpenAI checkpoint; anything
    else - including any custom `org/name` repo - passes through untouched.
    """
    if not model:
        return DEFAULT_WHISPER_MODEL_ID
    name = str(model).strip()
    if "/" in name:
        return name
    key = name.lower()
    if key.startswith("whisper-"):
        key = key[len("whisper-"):]
    return _WHISPER_ALIASES.get(key, name)
```

Read the top comment again, because it describes a bug class rather than a
feature. Suppose a code path cannot load the requested model and falls back to
`whisper-base`. The user sees a transcript. It is labelled with their model's
name. It is cached under their model's key. Every later request for that model
serves whisper-base's answer.

Nothing errors. Nothing looks wrong. The system is now systematically lying
about which model produced which output — and the cache makes the lie
persistent.

The resolver is deliberately permissive in one direction and strict in the
other: a bare alias expands, a `/` means "this is a real repo id, pass it
through". A name it does not recognise is returned unchanged rather than
replaced with a default, so an unknown model fails loudly downstream instead of
becoming whisper-base.

You will see `resolve_whisper_model_id` called at the top of almost every
Whisper entry point. That repetition is intentional: each entry point resolves
for itself rather than trusting a caller to have done it.

---

## 6.2 The dead cache that cost 73% of every call

```python
_pipeline_cache = {}

def _get_whisper_pipeline(model_id: str, device: int, torch_dtype: torch.dtype):
    if model_id not in _pipeline_cache:
        try:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=torch_dtype,
                device=device,
            )
        except NotImplementedError as e:
            if "meta tensor" in str(e):
                pipe = pipeline("automatic-speech-recognition", model=model_id,
                                torch_dtype=torch_dtype, device=-1)
                if torch.cuda.is_available():
                    try:
                        pipe.model = pipe.model.to("cuda:0")
                    except Exception:
                        pass
            else:
                raise
        _pipeline_cache[model_id] = pipe
    return _pipeline_cache[model_id]
```

This helper existed, complete with its cache and its meta-tensor fallback, and
**had zero callers**. `transcribe_whisper` constructed the pipeline inline
instead, duplicating the same logic. The cache stayed empty for the life of
every process.

```python
    # For regular transcription without attention, use the cached pipeline.
    #
    # This block used to construct the pipeline inline, rebuilding the model on
    # every call. `_get_whisper_pipeline` already existed with exactly this
    # construction logic, meta-tensor fallback included, and a `_pipeline_cache`
    # behind it, but nothing ever called it: the helper was dead code and the
    # cache stayed empty for the life of the process. Measured on whisper-base,
    # construction cost 5.22 s against 1.89 s of actual inference, so 73% of
    # every transcription was rebuilding a model it already had. Those are the
    # repeated "Loading weights" entries in the API log.
    pipe = _get_whisper_pipeline(model_id, device, torch_dtype)
```

Three things worth extracting from this:

**The symptom was visible the whole time.** Repeated "Loading weights" lines in
the log. Nobody read them as a bug, because logs full of framework chatter are
normal.

**Nothing failed.** The output was correct. Only slow. Bugs that produce correct
output are the longest-lived ones, because no test that checks correctness will
find them.

**The fix was one line.** Finding it was the work. Which is why it is now
guarded by tests that assert reuse rather than correctness:

```python
def test_repeated_calls_build_the_model_once(self, monkeypatch):
    _install_counting_pipeline(monkeypatch)
    for _ in range(4):
        ml._get_whisper_pipeline("openai/whisper-base", -1, torch.float32)
    assert _FakePipe.builds == 1, (
        f"built the pipeline {_FakePipe.builds} times for one model; the "
        "cache is not being consulted"
    )
```

And the one that catches the specific regression that happened — the caller
being disconnected from the helper again:

```python
def test_transcribe_whisper_populates_the_pipeline_cache(self, monkeypatch, tmp_path):
    ...
    ml.transcribe_whisper("openai/whisper-base", str(clip))
    assert ml._pipeline_cache, (
        "transcribe_whisper left the pipeline cache empty, so it built its "
        "own pipeline instead of using the cached one"
    )
```

**This is the general technique for performance bugs: make the reuse itself
observable.** A test that only checks the transcript cannot tell you how many
times the model was built. A counting stub can.

---

## 6.3 `transcribe_whisper`: two very different paths

```python
def transcribe_whisper(model_id, audio_file, chunk_length_s=30, batch_size=8,
                       return_timestamps=False, return_attention=False):
    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    audio, sample_rate = librosa.load(audio_file, sr=16000)
    audio = audio.astype(np.float32)
```

Note the conventions:

- **`device` is an int**, not a string: `0` for the first GPU, `-1` for CPU.
  That is the `transformers.pipeline` convention, not PyTorch's (`"cuda:0"` /
  `"cpu"`). Both appear in this codebase, for different APIs, and mixing them
  up is a silent bug — `pipeline(device="cpu")` is not an error but does not
  mean what you think.
- **`float16` on GPU, `float32` on CPU.** Half precision halves memory and is
  much faster on tensor cores. On CPU it is usually *slower*, because CPUs lack
  native fp16 arithmetic and emulate it.
- **`sr=16000` on the load.** librosa resamples if needed. Non-negotiable
  (§1.1).

Then the function splits. `return_attention=True` takes a long, defensive
branch using the raw model. Otherwise it uses the cached pipeline.

### The plain path

```python
    def _run_pipe(**generate_kwargs):
        if return_timestamps:
            return pipe(audio, return_timestamps="word",
                        chunk_length_s=chunk_length_s, batch_size=batch_size, **generate_kwargs)
        else:
            return pipe(audio, return_timestamps=return_timestamps,
                        chunk_length_s=chunk_length_s, batch_size=batch_size, **generate_kwargs)
```

`chunk_length_s=30` matches Whisper's native 30-second context window (§1.7).
The comment records what happens if you shorten it:

> *Decoding in 5 s chunks gave the decoder a fraction of Whisper's 30 s
> context, so the two paths disagreed on the actual words: on common-voice
> sample-000037, `/inferences/run` returned "Mines in the door." while the
> word-timestamp path returned "Minds in the door.". The transcript panel and
> the XAI word segments are meant to describe one prediction, and a saliency
> map labelled with words the displayed transcript never contained is the
> "garbage output" that shows up in the UI. Word timestamps do not require
> short chunks.*

Someone presumably shortened the chunk length hoping for better timestamps. It
changed the transcript instead — and the visible symptom appeared somewhere
else entirely, in the XAI panel's labels.

Then the forced-language logic from §1.7:

```python
    try:
        result = _run_pipe(generate_kwargs={"language": "english", "task": "transcribe"})
    except ValueError:
        result = _run_pipe()
```

`except ValueError` rather than a bare `except`: English-only checkpoints
(`*.en`) reject the language kwarg with a `ValueError` specifically. Catching
everything would also swallow real failures and retry them unforced, hiding
the actual error.

### The attention path: a ladder of four attempts

The attention branch tries four increasingly desperate strategies. Summarised:

1. **Generated ids as decoder input.** Generate the transcript, then re-run the
   model with those tokens as `decoder_input_ids` and
   `output_attentions=True`. Probes `decoder_attentions`, `attentions`,
   `cross_attentions`, `encoder_attentions` in turn.
2. **Minimal decoder input.** Just the `<|startoftranscript|>` token.
3. **Direct encoder call** on a freshly loaded `WhisperModel` with
   `attn_implementation="eager"`.
4. **`AutoProcessor`/`AutoModel`** with eager attention, falling back to the
   Whisper-specific classes.

Why four? Because `transformers` has changed which fields it populates across
versions, and the project must work across a range of them. This is genuinely
defensive code, not paranoia — though it is also a strong argument for pinning
your library versions.

If all four fail, the fabricated-pattern fallback from §2.6 fires, flagged:

```python
            prov_source = (
                Provenance.FALLBACK
                if (attention_data and attention_is_fallback)
                else (Provenance.MEASURED if attention_data else Provenance.UNAVAILABLE)
            )
            prov_reason = (
                "Fabricated structured attention pattern - NOT real attention"
                if (attention_data and attention_is_fallback)
                else ("All attention extraction methods failed" if not attention_data else None)
            )
            prov_info = provenance_fields(prov_source, prov_reason)
```

Three-way: fabricated data → FALLBACK with a reason; real data → MEASURED; no
data → UNAVAILABLE. Exactly the contract from §2.7.

---

## 6.4 SER: lazy loading with a per-checkpoint cache

```python
def ensure_emo_model_loaded(model_id: str | None = None, revision: str | None = None):
    """Lazily load the emotion model + feature extractor via the ModelRegistry (LIT-207).

    Replaces the old eager `feature_extractor = ...` / `emo_model = ...`
    import-time singletons. Call this at the top of every entry point that
    touches `emo_model`/`feature_extractor` so nothing loads until actually used.

    Other modules must go through this function (or the module, e.g.
    `model_loader_service.emo_model`) rather than `from ... import emo_model`
    -- a bare name import binds a snapshot at the importer's own import time
    and would permanently see the pre-load None value once these globals are
    reassigned here.
    """
    global feature_extractor, emo_model

    target = model_id or _EMO_MODEL_ID
    if target == _EMO_MODEL_ID:
        if emo_model is None:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                _EMO_MODEL_ID, revision=revision or _EMO_MODEL_REVISION
            )
            loaded = _model_registry.get(
                _EMO_MODEL_ID,
                revision=revision or _EMO_MODEL_REVISION,
                model_class=Wav2Vec2ForSequenceClassification,
            )
            emo_model = loaded.model
        return feature_extractor, emo_model, emo_device
```

### The import trap, which is a genuine Python subtlety

The docstring warns against `from ... import emo_model`. Here is exactly why.

`emo_model` starts as `None` at module level. `ensure_emo_model_loaded`
**reassigns the global**. Now:

```python
# WRONG - binds the value at import time
from app.domain.model_loader_service import emo_model
# emo_model is permanently None in this module, forever

# RIGHT - looks up the attribute at use time
import app.domain.model_loader_service as ml
ml.emo_model   # sees whatever the global currently is
```

`from X import y` copies the *current value* of `X.y` into your namespace. A
later reassignment of `X.y` is invisible to you. Importing the module and using
`ml.emo_model` performs the attribute lookup each time, so it sees the update.

This is also what makes monkeypatching work in tests: patching
`ml.emo_model` is visible to any code that reads it through the module, and
invisible to any code that bound the name directly. It is why the project
convention is *"import the module, not the name."*

A function-local `from ... import x` is fine, because it binds at call time.

### Per-checkpoint caching

```python
    # A user-selected SER checkpoint. Cached per model id rather than in the
    # module globals: those hold exactly one model, so the first one loaded
    # would answer for every later selection - the SER twin of the whisper-base
    # substitution fixed for ASR.
    if target not in _emo_model_cache:
        _emo_model_cache[target] = (
            Wav2Vec2FeatureExtractor.from_pretrained(target, revision=revision or "main"),
            _model_registry.get(target, revision=revision or "main",
                                model_class=Wav2Vec2ForSequenceClassification).model,
        )
    custom_extractor, custom_model = _emo_model_cache[target]
    return custom_extractor, custom_model, emo_device
```

The module globals hold exactly one model. If a user selected a custom SER
checkpoint and the globals were reused, the first-loaded model would answer for
every subsequent selection — the same substitution bug as §6.1, in a different
place. A dict keyed by model id fixes it.

Note the asymmetry: the default checkpoint lives in the globals, custom ones in
the dict. That is a compatibility artefact — existing test code monkeypatches
`ml.emo_model` — rather than a design choice, and the comment elsewhere says so.

### Binding locals to redirect a function

```python
def _predict_emotion_wave2vec(audio_path, return_attention=False, model_id=None):
    # Bind the loader's return value to local names. The body below refers to
    # `feature_extractor`/`emo_model`/`emo_device` as free names, so binding
    # them here redirects the whole function at the selected checkpoint instead
    # of the module-global default.
    feature_extractor, emo_model, emo_device = ensure_emo_model_loaded(model_id)
```

A neat trick with a real risk. The function body (hundreds of inherited lines)
refers to those three names. Without the local assignment they resolve to the
module globals — the default model. Assigning them locally at the top makes
every reference in the body use the selected model, with no other edits.

It works because Python resolves names as local if they are assigned anywhere
in the function. It is the minimum-diff way to retrofit model selection onto
inherited code. It is also fragile: someone adding a line that reads the global
deliberately would be silently redirected. The comment is what keeps it safe.

### Attention pooling at the source

```python
@contextmanager
def _pooled_attention(model, max_frames: int = SER_ATTENTION_MAX_FRAMES):
    """Pool each layer's attention the moment its attention module returns it.

    With ``output_attentions`` every layer's weights are collected into one
    tuple, so peak memory is all layers at full size at once. transformers 5
    does that collecting with its own persistent forward hooks on each
    attention module (installed on first use), so pooling has to happen *at*
    that module and *before* its hook: ours are prepended, and PyTorch chains
    forward hooks, so the capture hook receives the pooled tensor.
    """
    handles = []

    def hook(_module, _inputs, output):
        if isinstance(output, tuple) and len(output) > 1 and torch.is_tensor(output[1]):
            return (output[0], _pool_attention(output[1], max_frames), *output[2:])
        return None  # leave the output untouched

    named_modules = model.named_modules() if isinstance(model, torch.nn.Module) else ()
    for name, module in named_modules:
        if name.endswith("encoder.layers"):
            for layer in module:
                targets = [layer]
                if isinstance(getattr(layer, "attention", None), torch.nn.Module):
                    targets.append(layer.attention)
                for target in targets:
                    handles.append(target.register_forward_hook(hook, prepend=True))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
```

This solves the 750 MB problem from §1.6, and it does so with two PyTorch
features most people never touch.

**A forward hook can modify the output.** If a hook returns a non-`None` value,
PyTorch substitutes it for the module's output. So this hook returns the tuple
with element 1 (the attention weights) replaced by a pooled version.

**`prepend=True` controls hook ordering.** Hooks are chained: each receives the
previous one's output. `transformers` installs its own collector hook on each
attention module. Prepending means the pooling hook runs *first*, so the
collector only ever sees the small tensor. Append instead and the collector
grabs the full-size tensor before pooling happens — the memory is already
allocated and the fix does nothing.

Both the encoder layer and its `attention` submodule are hooked, because
different `transformers` versions collect from different places.

`isinstance(model, torch.nn.Module)` guards against test stand-ins that are
not real modules — they simply get no pooling rather than an `AttributeError`.

---

## 6.5 `predict_ser`: the clean entry point

Most of the SER code is inherited and defensive. `predict_ser` is new, and it
is what new code should look like:

```python
def predict_ser(audio_path, model_id=None):
    """Speech Emotion Recognition inference (FR6).

    ``model_id`` selects the SER checkpoint; None is the project default. The
    faithfulness auditor calls this, so a hardcoded model would have scored a
    custom model's explanation against the default model's confidence.
    """
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

The docstring's second sentence is the interesting part. The faithfulness
auditor (Chapter 13) calls this to measure confidence before and after masking.
If `predict_ser` hardcoded the default model, the auditor would score a
*custom* model's explanation against the *default* model's confidence — a
faithfulness number computed across two different models. Meaningless, and
plausible-looking.

`model_id` threading through is not a nice-to-have. It is what makes the audit
valid.

Other details:

- **`torch.no_grad()`** — no gradients needed for prediction; roughly halves
  memory.
- **`attention_mask` is conditional.** Some feature extractors emit one, some
  do not. Passing `None` is correct for single unpadded inputs.
- **`id2label.get(i, id2label.get(str(i), ...))`** — some checkpoints key that
  dict by int, some by string. A real incompatibility.
- **`max(probabilities, key=probabilities.get)`** rather than `argmax` on the
  tensor. Same answer, but it reads the label directly rather than
  round-tripping through an index.

---

## 6.6 Deepfake detection

`predict_deepfake` mirrors `predict_ser` with one addition — label
normalisation:

```python
    id2label = model_.config.id2label if isinstance(model_.config.id2label, dict) else {}
    class_probs = {DEEPFAKE_BONA_FIDE: 0.0, DEEPFAKE_SPOOF: 0.0}
    for i, prob in enumerate(probs):
        raw = id2label.get(i, id2label.get(str(i), f"label_{i}"))
        class_probs[_normalize_deepfake_label(raw)] += float(prob.item())

    predicted_label = (
        DEEPFAKE_SPOOF
        if class_probs[DEEPFAKE_SPOOF] >= class_probs[DEEPFAKE_BONA_FIDE]
        else DEEPFAKE_BONA_FIDE
    )
```

Note the `+=`. A checkpoint could have more than two classes — several spoof
types, say — and their probabilities *accumulate* into the right bucket rather
than the last one overwriting the others. Assignment instead of accumulation
would silently discard probability mass.

`>=` on the tie: ties go to spoof. For a forensic tool, flagging a coin-flip as
suspicious is the safer direction.

### The timeline, and the loop-invariant load

```python
def predict_deepfake_timeline(audio_path, model_key=_DEFAULT_ADD_MODEL_KEY,
                              window_s=1.0, overlap=0.5):
    """Per-window deepfake confidence across a clip (FR7.2).

    The model is loaded once and reused across windows; reloading per window
    turned a 5 s clip into nine full model loads.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if window_s <= 0:
        raise ValueError("window_s must be positive")

    feature_extractor_, model_, device_ = ensure_add_model_loaded(model_key)
    audio, rate = librosa.load(audio_path, sr=16000)
```

Input validation first, and it is precise: `overlap < 1.0` strictly, because
`overlap = 1.0` gives `hop = 0` and an infinite loop.

The model load is hoisted out of the loop. Nine model loads for a five-second
clip is the kind of thing that happens when you write the loop first and the
loading inside it, and it is invisible until someone profiles.

Windowing, with the tail fix:

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

`range(0, len - win + 1, hop)` stops before the end unless the length divides
evenly, dropping up to `hop` samples — half a second at the defaults. An
artefact in the last half second would be invisible. The extra window is
flush-right against the end, so it overlaps the previous one more than `overlap`
specifies. That is the correct trade: a slightly uneven final window beats a
blind spot.

`max(1, ...)` on the hop prevents a zero hop from any rounding.

Output is rounded before returning:

```python
        timeline.append({
            "start_s": round(start / rate, 3),
            "end_s": round((start + win) / rate, 3),
            "synthetic_probability": round(class_probs[DEEPFAKE_SPOOF], 4),
            "confidence": round(max(class_probs.values()), 4),
            "predicted_label": label,
        })
```

Rounding at the serialisation boundary keeps JSON small and avoids
`0.30000000000000004` in the UI. Never round intermediate computations — only
output.

---

## 6.7 Embeddings

Three extractors, one per family. All follow the same shape:

```python
def extract_wav2vec2_embeddings(audio_file_path: str, model_id=None) -> np.ndarray:
    feature_extractor, emo_model, emo_device = ensure_emo_model_loaded(model_id)
    audio, rate = librosa.load(audio_file_path, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(emo_device)
    attention_mask = inputs.attention_mask.to(emo_device) if "attention_mask" in inputs else None

    with torch.no_grad():
        outputs = emo_model.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        # outputs.last_hidden_state shape: [batch, time_frames, hidden_size]
        hidden_states = outputs.last_hidden_state
        pooled_embeddings = torch.mean(hidden_states, dim=1)  # [batch, hidden_size]
        embeddings = pooled_embeddings.cpu().numpy().squeeze()  # [hidden_size]
    return embeddings
```

Two decisions:

**`emo_model.wav2vec2`, not `emo_model`.** Calling the base encoder skips the
classification head. An embedding should be a general-purpose representation,
not one squeezed through a 7-way classifier — the head throws away everything
not needed to distinguish those seven classes.

**Mean pooling over time.** A clip of *T* frames gives `[T, hidden]`; a scatter
plot needs one point per clip. Averaging over time is the standard reduction.
It loses all temporal structure — two clips with the same sounds in a different
order pool identically — but it gives a fixed-size vector regardless of
duration, which is what a projection needs.

The Whisper extractor carries a pointed comment:

```python
        # Deliberately no fall back to whisper-base on failure: the caller
        # stores this vector under the *requested* model's cache key, so
        # substituting another checkpoint would poison the projection with
        # embeddings from a model the user never asked for (FR1, FR4).
```

The same substitution rule as §6.1, now protecting a scatter plot. A silent
fallback would put whisper-base's embeddings into a projection labelled as the
custom model's latent space — and since the vectors have the same
dimensionality, nothing would complain.

---

## 6.8 Dispatch: which function runs

`orchestration/inference_service.py` routes a model name to a function:

```python
MODEL_FUNCTIONS = {
    "whisper-base": transcribe_whisper_base,
    "wav2vec2": wave2vec,
    "melody-machine": predict_melody_machine,
    "wav2vec2-add": predict_wav2vec2_add,
}
```

Plus resolution for anything not in that table:

```python
    # Whisper is dispatched by family rather than by name so that any custom
    # checkpoint runs as itself. Binding the selected id here is what stops
    # `transcribe_whisper_base` from quietly transcribing with whisper-base
    # and caching the result under the requested model's key (FR1, FR4).
    func = MODEL_FUNCTIONS.get(model)
    if func is transcribe_whisper_base or (func is None and "whisper" in model.lower()):
        func = functools.partial(transcribe_whisper_base, model=model)
    elif not func:
        from app.domain.model_registry_service import registry
        try:
            loaded_model = registry.get(model)
            if loaded_model.family == "whisper":
                ...
```

`functools.partial(transcribe_whisper_base, model=model)` pre-binds the model
argument. Without it, `transcribe_whisper_base` is called with its default and
transcribes with whisper-base — the substitution bug again, for the third time,
in a third place.

It appears three times because it is the natural failure mode of a system where
one function serves many models with a default argument. If you are building
this, the structural fix is to make the model argument *required* everywhere and
never give it a default.

### `force_refresh`

```python
    """``force_refresh`` skips the cache lookup and recomputes, then overwrites
    the cache entry with the fresh result — backs the per-row "Regenerate"
    button in AudioDataTable.tsx, which would otherwise just get the same
    cached prediction handed back unchanged (a deterministic model on an
    unchanged file always predicts the same thing; the point of that button
    is forcing a real, visible recompute, not silently no-op-ing).
    """
```

A subtle UX point. "Regenerate" on a deterministic model over an unchanged file
produces an identical answer, so a cache-respecting implementation appears to do
nothing. The user clicks again. Nothing happens again. The button looks broken.
`force_refresh` makes the recompute real, which is what the user asked for even
though the answer is the same.

### The embedding family router

Already quoted in §2.10, but it belongs here too because it is a dispatch bug:

```python
def _embedding_family(model: str) -> str:
    """Which extractor owns ``model``: ``add`` / ``wav2vec2`` / ``whisper``.

    Routing used to be three substring tests in an if/elif chain, which got two
    cases wrong. ``wav2vec2-add`` contains "wav2vec", so the deepfake models were
    handed to the emotion extractor - same 1024 dims, no error, wrong latent
    space. And a custom SER checkpoint whose name says neither "whisper" nor
    "wav2vec" (``myorg/custom-ser``) fell through to the *Whisper* extractor,
    which is how a custom emotion model ended up plotted in Whisper's space.

    The registry knows each model's family, so ask it. Substrings stay as the
    fast path for the built-in keys and as the fallback when the model is not
    registered, since guessing beats failing the request outright.
    """
    if model in ADD_MODEL_KEYS:
        return "add"
    lowered = model.lower()
    if "whisper" in lowered:
        return "whisper"
    if "wav2vec" in lowered:
        return "wav2vec2"
    try:
        from app.domain.model_registry_service import registry
        family = getattr(registry.get(model), "family", None)
        if family in ("whisper", "wav2vec2"):
            return family
    except Exception:
        logger.warning("Could not resolve family for %s; defaulting to whisper", model)
    return "whisper"
```

Exact-match first, then substrings as a fast path, then the authoritative
registry lookup, then a logged default. The layering means the common cases
never pay for a registry call and the uncommon ones get a correct answer.

---

## 6.9 Summary

- **Never substitute a model silently.** It appears three times in this module
  and is the most consequential bug class in the system, because the cache makes
  the substitution permanent and nothing errors.
- A helper with a cache and zero callers cost 73% of every transcription.
  Correct-but-slow bugs survive longest; guard reuse with counting stubs, not
  correctness assertions.
- `device` is an int for `transformers.pipeline` and a string for PyTorch.
  Mixing them silently does the wrong thing.
- `chunk_length_s=30` matches Whisper's context; shortening it changes the
  transcript and the symptom surfaces in the XAI panel.
- Force the language on multilingual checkpoints; catch `ValueError` narrowly
  for `*.en` models.
- `from X import y` binds a value; module reassignment is invisible to it.
  Import the module, not the name.
- Assigning globals to locals at the top of a function redirects the whole body
  — minimum-diff retrofit, but fragile without a comment.
- Forward hooks can rewrite a module's output, and `prepend=True` puts yours
  before a library's collector. That is what makes attention pooling work.
- Accumulate (`+=`) when folding many model classes into fewer buckets;
  assignment silently discards probability mass.
- Hoist model loads out of loops.
- Window the tail explicitly or lose up to one hop of audio.
- Round only at the serialisation boundary.
- Embeddings come from the base encoder, not through the classification head,
  and are mean-pooled over time.
- Thread `model_id` everywhere — the faithfulness audit is only valid if the
  model being explained is the model being measured.
- Dispatch by authoritative family where one exists; substrings are a fast path,
  not a type system.

Next: [Chapter 7 — Orchestration](07-orchestration.md).
