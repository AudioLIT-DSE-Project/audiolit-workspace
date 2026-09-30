# Chapter 2 — Explainable AI for audio: the theory

Chapter 1 gave you models that turn audio into predictions. This chapter is
about the question AudioLIT exists to answer: **why did the model say that?**

It is a theory chapter. Chapter 9 shows the implementation. Read this one
first, because the implementation is full of guards that only make sense once
you know what can go wrong conceptually.

---

## 2.1 What "explanation" means here, precisely

A model with 74 million parameters gives no human-readable account of itself.
"Explainable AI" (XAI) is the family of techniques that extract one anyway.

AudioLIT is concerned almost entirely with **local, post-hoc, attribution-based
explanation**. Unpacking those three words:

- **Local** — explains *this one prediction on this one input*, not the model's
  global behaviour. "Why did it say 'door' here", not "what does it think
  'door' sounds like in general".
- **Post-hoc** — the model is already trained and is not modified. The
  alternative is an *intrinsically* interpretable model (a decision tree, a
  linear model), which you can read directly but which cannot transcribe
  speech.
- **Attribution-based** — the output is a number per input region, saying how
  much that region contributed. For audio, that means a score per time frame,
  or per time-frequency cell.

So the deliverable is: given a clip and a prediction, a **heatmap over the
spectrogram**. Bright where the model relied on it.

### The distinction that the whole project turns on

There are two questions that look identical and are not:

1. **Where is the energy in this audio?**
2. **Which parts of this audio did the model use?**

Question 1 is signal processing. You can answer it without a model at all — it
is just `|STFT|`. Question 2 requires the model.

They frequently *look* the same, because models do tend to attend to loud,
spectrally rich regions. A heatmap answering question 1 is therefore a
plausible-looking answer to question 2, and a user cannot tell them apart by
eye.

This is the central hazard of the entire system, and it has already caused
real defects in this codebase. Chapter 9 documents a case where **every Whisper
Grad-CAM heatmap ever rendered was secretly an energy map** — the attribution
silently collapsed to zero, a fallback fired, and the label still said
"Grad-CAM". Hold onto this while reading §2.9 on provenance.

---

## 2.2 Why interpretability matters more for audio than for images

If an image classifier mislabels a photo, you can look at the photo. You have
the same perceptual access to the input that the model does.

Audio is different in three ways:

- **It is temporal.** You cannot see it at a glance. Reviewing 10 seconds
  takes 10 seconds.
- **You are a poor spectral analyser.** You cannot hear that energy at 4.2 kHz
  in frames 300–340 drove a decision. You have no introspective access to the
  representation the model uses.
- **Its failures are demographically structured.** ASR accuracy varies
  systematically by accent, dialect and gender. A system deployed on
  "everyone" quietly works much worse for some people than others, and the
  aggregate accuracy number hides it completely.

That third point is why accent-bias profiling (Chapter 13) is a first-class
feature and not an afterthought. And the deployment contexts named in the
project's own problem statement — healthcare, customer service, security
biometrics — are exactly the ones where an unexamined failure mode does harm.

---

## 2.3 Family 1: Gradient-based attribution

### Plain (vanilla) gradients

The simplest possible explanation. Take the gradient of the output score with
respect to the input:

```
attribution(xᵢ) = ∂score / ∂xᵢ
```

Interpretation: if I nudged this input value slightly, how much would the score
move? Large magnitude means influential.

One forward and one backward pass. Fast. And flawed in two ways:

- **Saturation.** If the model is already maximally confident, the gradient is
  near zero everywhere — including at the features that caused the confidence.
  The gradient measures *local sensitivity*, not *contribution*. A feature can
  be the entire reason for a decision and have zero gradient.
- **Noise.** Neural networks are locally jagged. Gradients fluctuate sharply
  between adjacent inputs, producing speckled maps.

Both are fixed by the same idea: stop looking at one point.

### Integrated Gradients

**Integrated Gradients** (Sundararajan, Taly & Yan, ICML 2017) integrates the
gradient along a straight path from a **baseline** to the input:

```
IG(xᵢ) = (xᵢ − x'ᵢ) × ∫₀¹ ∂F(x' + α(x − x')) / ∂xᵢ  dα
```

where `x` is your input and `x'` is the baseline — an input representing
"absence", conventionally all zeros (silence, for audio).

In practice the integral is a Riemann sum over `n_steps` interpolated inputs:

```
IG(xᵢ) ≈ (xᵢ − x'ᵢ) × (1/n) × Σₖ ∂F(x' + (k/n)(x − x')) / ∂xᵢ
```

So: construct *n* versions of your input fading from silence up to the real
clip, take the gradient at each, average them, scale by the difference from
baseline. Saturation is defeated because somewhere along that path the model
was *not* saturated, and that region's gradients carry the signal.

IG is the method with the strongest theoretical backing, because it satisfies
two axioms:

- **Sensitivity** — if changing one feature changes the prediction, that
  feature gets non-zero attribution.
- **Implementation invariance** — two networks that compute the same function
  get the same attributions, regardless of internal structure.

And it satisfies **completeness**: the attributions sum exactly to
`F(x) − F(x')`. The explanation accounts for the entire difference between
"silence" and "this clip". That is a genuinely strong property — most methods
cannot state what their numbers add up to.

Cost: `n_steps` forward *and* backward passes. AudioLIT adapts:

```python
n_steps = 4 if not torch.cuda.is_available() else 16
internal_batch_size = 1
```

Four steps on CPU is low — the paper suggests 20 to 300. It is an explicit
latency-versus-fidelity trade (the requirement budgets attribution generation
tightly), and a 4-step IG is still categorically better than a plain gradient.
`internal_batch_size=1` processes interpolations one at a time to bound memory.

> **The baseline choice is not neutral.** IG explains your input *relative to
> the baseline*. Zeros mean silence, so an IG map answers "what distinguishes
> this clip from silence". If you used average noise as the baseline you would
> get a different, equally valid, differently-meaning explanation. Anyone
> reading an IG map must know which baseline produced it. AudioLIT uses zeros
> throughout.

### Grad-CAM

**Gradient-weighted Class Activation Mapping** (Selvaraju et al., ICCV 2017)
attributes to a *convolutional layer's feature maps* rather than to the raw
input.

The algorithm:

1. Forward pass, capturing activations `A` of the chosen conv layer. Shape
   `[channels, spatial...]`.
2. Backward pass from the target score, capturing gradients `∂score/∂A`.
3. **Channel importance** = spatially averaged gradient:
   `αᶜ = mean over spatial dims of ∂score/∂Aᶜ`
4. **Weighted sum, then ReLU**:
   `CAM = ReLU( Σᶜ αᶜ · Aᶜ )`

Why ReLU? Negative values mean "this region argued *against* the target
class". Grad-CAM deliberately keeps only positive evidence — regions
supporting the prediction.

The implementation is compact and worth reading as a whole, because it is the
clearest example of PyTorch hooks in the codebase:

```python
def compute_grad_cam(model, inputs, target_layer=None, target_index=None):
    if target_layer is None:
        target_layer = find_last_conv_layer(model)

    captured = {}

    def _forward_hook(_module, _inp, output):
        captured["activations"] = output.detach()

    def _backward_hook(_module, _grad_in, grad_out):
        captured["gradients"] = grad_out[0].detach()

    fwd_handle = target_layer.register_forward_hook(_forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(_backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        if target_index is None:
            target_index = int(torch.argmax(logits, dim=-1)[0])
        logits[0, target_index].backward()

        acts = captured["activations"][0]   # [C, *spatial]
        grads = captured["gradients"][0]    # [C, *spatial]
        spatial_dims = tuple(range(1, grads.dim()))
        weights = grads.mean(dim=spatial_dims)                    # [C]
        weights = weights.view([-1] + [1] * (acts.dim() - 1))     # broadcast
        cam = torch.relu((weights * acts).sum(dim=0))             # [*spatial]

        cam = cam - cam.min()
        peak = cam.max()
        if peak > 0:
            cam = cam / peak
        return cam.cpu().numpy()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()
```

Three things to notice:

- **A `register_forward_hook` fires on *any* forward pass through that
  module**, not only the one that registered it. That is a global side effect
  on a shared object, and it is why concurrency matters (§2.10).
- **`finally: handle.remove()`** — hooks must always be removed. A leaked hook
  keeps firing on every later inference, silently capturing tensors from
  unrelated requests and holding them in memory.
- **`model.zero_grad(set_to_none=True)`** before the backward pass. Gradients
  accumulate in PyTorch; without clearing them you get this pass plus every
  previous one summed together.

#### The problem Grad-CAM has with ASR

Grad-CAM needs a **class-discriminative scalar** to differentiate. A classifier
supplies one: the logit for the predicted class. Whisper has no classifier — it
generates a token sequence.

The obvious substitute is the encoder's energy: `hidden.pow(2).mean()`. It is a
scalar and it differentiates. It is also completely wrong, and the code says
exactly how wrong:

```python
# Attribute the model's score for *its own transcript*, not encoder
# energy. Grad-CAM's ReLU keeps only channels that push the target
# score up, which is meaningful for a class score and meaningless
# for `hidden.pow(2).mean()`: measured on whisper-base, the
# pre-ReLU map came out uniformly negative (max -9.7e-6, 0/1500
# positions positive), so ReLU zeroed the whole map, the
# "empty or constant" guard below fired on every clip, and every
# Whisper heatmap ever rendered was the energy fallback wearing a
# Grad-CAM label.
```

Measured: **0 of 1500 positions positive**. ReLU annihilated the entire map.
Every single time. And because the fallback produced a plausible-looking
heatmap, nobody noticed.

The fix constructs a genuine class-discriminative target — the log-probability
Whisper assigns to the transcript it actually produced:

```python
class _WhisperTranscriptScore(torch.nn.Module):
    """Scores ``input_features`` by the log-probability Whisper assigns its own transcript.

    Grad-CAM needs a *class-discriminative* scalar. Whisper has no classifier
    head, so the score used here is the summed log-probability of the token
    sequence the model itself generated: raising it means "more evidence for
    exactly what the model said", which is the question a saliency map over an
    ASR prediction is supposed to answer.
    """

    def forward(self, x):
        n = x.shape[0]
        dec = self.decoder_input_ids.expand(n, -1)
        tgt = self.target_ids.expand(n, -1)
        logits = self.model(input_features=x, decoder_input_ids=dec).logits  # [B, L, V]
        logp = torch.log_softmax(logits, dim=-1)
        picked = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)              # [B, L]
        return picked.sum(dim=1, keepdim=True)                               # [B, 1]
```

Walk through it: run the decoder over the generated tokens, take
log-softmax over the vocabulary, `gather` the log-probability of each token
that was actually emitted, sum them. The result is one number meaning "total
evidence for this exact transcript". Raising it means the model is more certain
of precisely what it said. That is a proper Grad-CAM target.

The energy version survives only as an explicitly-labelled fallback:

```python
class _WhisperEncoderEnergy(torch.nn.Module):
    """Encoder-energy scalar - the pre-existing target, kept only as a fallback."""
    def forward(self, x):
        return self.encoder(x).last_hidden_state.pow(2).mean(dim=(1, 2), keepdim=True)
```

And a further subtlety: scoring the transcript needs the language-model head,
which the bare `WhisperModel` does not have. So the target wrapper loads
`WhisperForConditionalGeneration` — and the Grad-CAM hooks must attach to
*that* instance's encoder, not the other one:

```python
wrapper = _WhisperTranscriptScore(cond_model, seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0))
return wrapper, find_last_conv_layer(cond_model.model.encoder)
```

Hooking the wrong instance's encoder gives you hooks that never fire, and
`captured["activations"]` raises `KeyError`. Which encoder actually runs the
forward pass is the one that matters.

---

## 2.4 Family 2: Perturbation-based attribution

Gradient methods need access to the model's internals. Perturbation methods
need only the ability to call it. They ask the counterfactual question
directly: **change the input, see if the output changes.**

### Occlusion

The simplest version. Divide the spectrogram into patches, replace each with a
baseline, record how far the score drops.

```python
def occlusion_attribution(score_fn, spectrogram, n_freq_patches=8,
                          n_time_patches=8, baseline="mean"):
    base_score = float(score_fn(spectrogram))
    fill = float(spectrogram.mean()) if baseline == "mean" else float(baseline)

    freq_bounds = spectrogram_patch_bounds(spectrogram.shape[0], n_freq_patches)
    time_bounds = spectrogram_patch_bounds(spectrogram.shape[1], n_time_patches)
    importance = np.zeros((len(freq_bounds), len(time_bounds)), dtype=np.float32)

    for i, (f0, f1) in enumerate(freq_bounds):
        for j, (t0, t1) in enumerate(time_bounds):
            occluded = spectrogram.copy()
            occluded[f0:f1, t0:t1] = fill
            importance[i, j] = base_score - float(score_fn(occluded))
    return importance
```

Positive where occluding the patch dropped the score (it supported the
prediction); negative where occluding *raised* it (it argued against).

`score_fn` is a parameter, so this works for any model that produces a scalar.
That genericity is the whole appeal.

The patch-bounds helper is worth a look, because off-by-one tiling bugs in
attribution grids are both easy and invisible:

```python
def spectrogram_patch_bounds(length, n_patches):
    """Patch sizes differ by at most 1 so they tile the axis exactly - no gaps
    or overlaps - even when ``length`` isn't divisible by ``n_patches``."""
    if n_patches < 1:
        raise ValueError("n_patches must be >= 1")
    n_patches = min(n_patches, length)
    edges = np.linspace(0, length, n_patches + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_patches)]
```

`np.linspace` over edges guarantees exact tiling. `min(n_patches, length)`
prevents asking for more patches than there are samples, which would produce
empty patches that occlude nothing and score as unimportant.

Cost: `n_freq × n_time + 1` forward passes. 8×8 is 65 — heavy, but no
gradients needed.

### LIME

**Local Interpretable Model-agnostic Explanations** (Ribeiro, Singh &
Guestrin, KDD 2016). The idea: a complex model may be wildly non-linear
globally but is approximately linear in a small neighbourhood. So sample that
neighbourhood, fit a linear model, and read off its coefficients.

1. Define **interpretable features** — groups of input a human can reason
   about.
2. Generate samples by randomly switching groups on and off.
3. Run the real model on each sample.
4. Fit a weighted linear model predicting the model's output from the on/off
   pattern.
5. The linear coefficients are the attributions.

**Step 1 is where LIME succeeds or fails**, and AudioLIT got it wrong first:

```python
# LIME fits a linear surrogate over *interpretable* features. Left
# unmasked, Captum treats every one of the 80x3000 mel cells as its own
# feature and draws Captum's default 50 samples - 240,000 unknowns from
# 50 equations. The surrogate came back all-zero, so the "empty or
# constant" guard fired and Whisper LIME was silently the energy
# fallback on every clip, exactly like Grad-CAM was.
```

240,000 unknowns from 50 equations. Wildly underdetermined; the regression
returns zeros. Same silent-fallback outcome as the Grad-CAM bug, different
cause.

The fix defines features that are *meaningful for audio*: contiguous bands of
time.

```python
def _time_band_feature_mask(input_features, n_bands=32):
    """Group a ``[B, n_mels, T]`` input into ``n_bands`` contiguous time bands.

    Every mel bin at the same time band shares a group id, so LIME perturbs a
    slice of *time* - the unit a listener can actually interpret - instead of
    one mel cell at a time.
    """
    t = input_features.shape[-1]
    n_bands = max(1, min(n_bands, t))
    band_of_frame = (torch.arange(t, device=input_features.device) * n_bands) // t
    shape = [1] * (input_features.dim() - 1) + [t]
    return band_of_frame.view(shape).expand_as(input_features).long()
```

`(arange(t) * n_bands) // t` assigns each frame to a band by integer division —
a neat trick that distributes any remainder evenly instead of leaving a short
final band. `expand_as` broadcasts the per-frame id across all mel bins, so a
whole vertical slice shares one group. 32 bands, sampled 4× over:

```python
n_samples=max(LIME_TIME_BANDS * 4, 64)
```

128 samples for 32 unknowns: comfortably over-determined.

There is a second, subtler failure in the same area — the surrogate model
itself:

```python
def _lime_surrogate():
    """The linear model LIME fits over the perturbation samples.

    Captum defaults to Lasso(alpha=0.01). Against a target whose scale is ~2
    (encoder energy), that penalty shrinks *every* coefficient to exactly zero,
    so `Lime.attribute` returned an all-zero map and the energy fallback fired
    on every request. Ridge with a token penalty keeps the fit well-posed
    without erasing it; 4x as many samples as bands leaves it over-determined.
    """
    from captum._utils.models.linear_model import SkLearnRidge
    return SkLearnRidge(alpha=0.01)
```

**Lasso** (L1 regularisation) drives coefficients to exactly zero — that is
its purpose, feature selection. If your target variable has magnitude ~2 and
your penalty is 0.01, the penalty dominates and *every* coefficient is zeroed.
**Ridge** (L2) shrinks coefficients toward zero without ever reaching it,
keeping the fit well-posed while preserving relative magnitudes.

The lesson generalises beyond LIME: **a regularisation strength is only
meaningful relative to the scale of your target.** Copying a default from a
tutorial where the target was a probability in [0,1] into a setting where it is
an unbounded energy is how you get a silently blank explanation.

### SHAP

**SHapley Additive exPlanations** (Lundberg & Lee, NeurIPS 2017) borrows from
cooperative game theory. Shapley values answer: across all possible orderings
in which features could be added to the model, what is a feature's average
marginal contribution?

They are the *unique* attribution satisfying four fairness axioms —
efficiency, symmetry, dummy and additivity. That uniqueness is why SHAP is
theoretically attractive.

Exact computation requires evaluating all 2ⁿ subsets. For 240,000 features
that is not a number, it is a joke. Every practical SHAP is an approximation.
AudioLIT uses Captum's **GradientShap**, which samples baselines, adds noise,
and averages gradients — effectively expected gradients:

```python
gs = GradientShap(model_forward)
baseline = torch.zeros_like(input_features)
attributions = gs.attribute(
    input_features, baselines=baseline,
    n_samples=max(2, min(16, SALIENCY_SHAP_SAMPLES)), stdevs=0.09,
)
```

SHAP is the most expensive method here, which is why it has a stricter
duration cap than the others:

```python
MAX_SALIENCY_SECONDS = int(os.getenv("MAX_SALIENCY_SECONDS", "12"))
MAX_SALIENCY_SECONDS_SHAP = int(os.getenv("MAX_SALIENCY_SECONDS_SHAP", "6"))
```

---

## 2.5 The four methods compared

| | Grad-CAM | Integrated Gradients | LIME | SHAP |
|---|---|---|---|---|
| Needs internals | yes (a conv layer) | yes (gradients) | no | gradients, in this variant |
| Passes needed | 1 fwd + 1 bwd | n_steps × (fwd+bwd) | n_samples fwd | n_samples × (fwd+bwd) |
| Resolution | conv layer's grid | full input | your feature groups | full input |
| Theory | heuristic | 3 axioms, completeness | local linear fit | 4 axioms, unique |
| Fails when | no conv layer; non-discriminative target | baseline is wrong for the question | features badly grouped; surrogate mis-tuned | too few samples |
| Cost here | lowest | low (4–16 steps) | medium | highest |

**Why implement all four?** Because they disagree, and disagreement is
information. If four methods with different assumptions all highlight the same
region, that is strong evidence. If they disagree wildly, the explanation is
not robust and you should not trust any of them. A single-method tool cannot
tell you which situation you are in.

This is also why the code refuses to fall back silently between methods. If
you ask for t-SNE you must not get PCA; if you ask for LIME you must not get
Grad-CAM. The test suite asserts this directly:

```python
def test_unsupported_method_raises_rather_than_falling_back(self):
    # A silent fallback to PCA would show the user a t-SNE-labelled plot
    # that is not t-SNE, which is exactly the class of defect this project
    # treats as most serious.
    with pytest.raises(ValueError, match="Unsupported reduction method"):
        reduce_dimensions(_embeddings(), method="definitely-not-a-method", n_components=2)
```

---

## 2.6 Attention is not an explanation

Attention weights are seductive. The model literally computes "how much
position *i* attends to position *j*", it is already there, it needs no extra
computation, and it visualises beautifully.

The research consensus is that **attention weights are not faithful
explanations**. The key papers:

- *Attention is not Explanation* (Jain & Wallace, NAACL 2019) — you can often
  find *very different* attention distributions that produce *identical*
  predictions. If many attention patterns give the same output, no single one
  explains it.
- *Attention is not not Explanation* (Wiegreffe & Pinter, EMNLP 2019) — a
  partial rebuttal: attention is not arbitrary and does carry information, but
  it is not a faithful attribution.

The mechanistic reasons:

- Attention weights the *values*, and a value vector's magnitude matters too.
  High attention on a near-zero value contributes nothing.
- Information mixes across layers. Layer 8's attention over layer 7's
  representations tells you little about the original input, because each
  layer-7 position already blends many input positions.
- With many heads, you have many distributions and no principled way to
  combine them. Averaging them is a choice, not a derivation.

AudioLIT's position: attention is exposed as an **observation** and never as
an attribution. It appears in its own visualisation panel, separate from the
saliency overlay. Attribution methods are the ones used for the faithfulness
audit. The distinction is architectural, not just documentary.

### The fabricated-attention incident

This is the defect the project treats as the most serious it has had, and it
belongs in a theory chapter because it is a theory failure as much as a coding
one.

Whisper attention extraction has a ladder of fallbacks — different transformers
versions expose attention differently, so the code tries `decoder_attentions`,
then minimal decoder input, then direct encoder call, then `AutoModel`. If all
of them fail, the original code did this:

```python
# Final fallback - create structured attention patterns
seq_length = min(1500, len(audio) // 320)
cfg = getattr(model, "config", None)
num_layers = getattr(cfg, "decoder_layers", None) or 6
num_heads = getattr(cfg, "decoder_attention_heads", None) or 8

for layer_idx in range(num_layers):
    layer_data = []
    for head_idx in range(num_heads):
        attention_matrix = torch.zeros(seq_length, seq_length)
        attention_matrix.fill_diagonal_(0.6)                   # self-attention
        for i in range(seq_length):
            for j in range(max(0, i-3), min(seq_length, i+4)):  # local context
                if i != j:
                    distance = abs(i - j)
                    attention_matrix[i, j] = 0.4 / (1 + distance)
        attention_matrix = torch.softmax(attention_matrix, dim=-1)
        layer_data.append(attention_matrix.tolist())
    attention_data.append(layer_data)
```

This fabricates a plausible attention pattern — strong diagonal, decaying
local context, softmax-normalised so each row sums to 1 like a real
distribution. It has the right shape, the right layer and head counts, the
right statistical properties. **It is indistinguishable from real attention by
inspection, and it contains zero information about the audio.**

The user sees an attention heatmap. It looks like attention. It is a picture of
an assumption about how attention generally behaves.

For an interpretability tool this is the worst possible failure. The output's
entire purpose is to be trusted as measurement, and it is fiction.

The fix does not delete the fallback — a shaped placeholder has uses, and
returning nothing at all breaks downstream consumers. It makes it
**impossible to mistake**:

```python
attention_is_fallback = True
logger.warning(
    "Fabricated %d layers of structured attention - NOT real "
    "attention; flagged via attention_is_fallback (FR17)",
    len(attention_data),
)
```

And the flag travels with the data, all the way to the UI:

```python
result_dict = {
    "text": transcript,
    "attention": attention_data if attention_data else None,
    # FR17.1: a caller must be able to tell a genuine extraction
    # from the synthesised stand-in below, which has the same
    # shape and would otherwise be indistinguishable.
    "attention_is_fallback": bool(attention_data) and attention_is_fallback,
    "provenance": prov_info["provenance"],
    "provenance_reason": prov_info["provenance_reason"],
}
```

Note the earlier bug fixed in the same block: the placeholder used to be shaped
`12 layers / 16 heads` from a hardcoded guess, *"a shape no Whisper-tiny ever
produces"*. It now reads the loaded checkpoint's own config. A fabrication
with the wrong shape is at least detectable; one with the right shape is not.
Fixing the shape made the honesty flag more necessary, not less.

---

## 2.7 The provenance contract

The fabricated-attention incident produced a general mechanism. Every XAI
response in AudioLIT carries a provenance field:

```python
class Provenance(str, Enum):
    MEASURED = "measured"        # produced by the model on this input
    FALLBACK = "fallback"        # synthesised stand-in, NOT model output
    UNAVAILABLE = "unavailable"  # could not be produced at all
```

Three states, and the distinction between the second and third matters:
`FALLBACK` means "here is something, but it is not a measurement";
`UNAVAILABLE` means "there is nothing". Collapsing them would let a fallback
masquerade as a genuine empty result.

The helper enforces the one rule that makes the contract non-optional:

```python
def provenance_fields(source: Provenance, reason: str | None = None) -> dict:
    if source == Provenance.FALLBACK:
        if not reason or not reason.strip():
            raise ValueError("Provenance.FALLBACK requires a non-empty reason string explaining why fallback was used.")
        return {"provenance": source.value, "provenance_reason": reason.strip()}
    return {
        "provenance": source.value,
        "provenance_reason": reason.strip() if reason else None,
    }
```

**A fallback without a stated reason raises.** You cannot declare a result
non-measured and stay silent about why. The reason string is not documentation,
it is a required field, checked at runtime.

This is the load-bearing design decision of the whole interpretability layer.
It converts "we hope nobody ships a fabricated explanation" into "you cannot
ship one without saying so in a field the UI renders".

### Provenance has teeth downstream

The faithfulness auditor **refuses to score a fallback**:

```python
prov = str(item.get("provenance", "measured")).lower()
if prov != "measured":
    reason = item.get("provenance_reason", "attribution is not measured")
    refused += 1
    item_results.append({
        "file_path": file_path,
        "error": f"cannot audit a fallback attribution: {reason}",
        "provenance": item.get("provenance"),
        "provenance_reason": reason,
    })
    continue
```

Auditing a fabricated attribution would produce a faithfulness score for
fiction. The correct answer is to decline, and to count the declines
(`audio_refused`) so the refusal appears in the summary rather than vanishing.

---

## 2.8 Measuring whether an explanation is honest

An explanation is a claim. Claims can be tested. This is **faithfulness**: does
the attribution actually reflect what the model used?

### The deletion score

The core idea, and it is beautifully simple:

1. Get the model's confidence on the original input.
2. Remove the regions the attribution says are most important (the top K%).
3. Re-run the model.
4. Confidence should drop a lot. If it does not, the attribution was wrong
   about what mattered.

```
deletion_score = (original_confidence − masked_confidence) / original_confidence
```

Normalised by the original confidence so it is comparable across inputs — a
drop from 0.9 to 0.45 and from 0.5 to 0.25 both score 0.5.

Masking in AudioLIT zeroes the top-K% most salient timesteps:

```python
def mask_top_k_features(waveform, attributions, k_percent=10.0):
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

Details that matter: the attribution is at frame resolution and the waveform at
sample resolution, so `np.interp` resamples between them — get this wrong by a
frame and you mask the wrong audio. `np.argpartition` finds the top *k*
without fully sorting, O(n) rather than O(n log n). `np.abs` is used because a
strongly negative attribution is also strongly influential.

### The deletion AUC

A single K is a single point and can be lucky. Sweep K and integrate:

```python
def compute_deletion_auc(degradation_curve: Dict[str, float]) -> float:
    if not degradation_curve:
        return 0.0
    x_vals, y_vals = [], []
    if "top_0pct" not in degradation_curve:
        x_vals.append(0.0)
        y_vals.append(0.0)
    for pct_key, score in sorted(degradation_curve.items(),
                                 key=lambda t: int(t[0].split("_")[1].replace("pct", ""))):
        pct_num = int(pct_key.split("_")[1].replace("pct", ""))
        x_vals.append(pct_num / 100.0)
        y_vals.append(score)
    if len(x_vals) < 2:
        return float(y_vals[0]) if y_vals else 0.0
    return round(float(np.trapz(y_vals, x_vals)), 4)
```

`np.trapz` is trapezoidal integration. The origin `(0, 0)` is inserted
explicitly if absent — deleting nothing must drop nothing, and omitting that
anchor biases the area. The default sweep:

```python
top_k_percentages = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
```

A faithful attribution gives a curve that rises steeply and early. An unfaithful
one rises slowly, or not at all.

### The fabricated metric

The deletion score has a failure mode of its own, and it happened here. Read
the comment in full:

```python
"""Deletion-score faithfulness by masking and RE-RUNNING INFERENCE (FR16.1).

FR16.1 requires the top-K salient regions be zero-masked, inference re-run,
and the resulting confidence drop reported. An earlier version computed

    degraded_conf = orig_conf * (1.0 - k_pct * (0.5 + saliency_weight))

which never called a model. That expression is monotone in ``k_pct`` by
construction, so it produced a plausible degradation curve for *any*
attribution - including a random one - and the auditor could not tell a
faithful explanation from noise. It measured the saliency values, not the
model. Deleted, not kept as a fallback: a fabricated metric is worse than a
missing one, and this is the requirement whose whole purpose is to catch
exactly that.
"""
```

Look at the expression. It computes degraded confidence from the original
confidence and the deletion percentage *arithmetically*. No model call. It is
monotone in `k_pct` by construction, so it *always* produces a rising
degradation curve — including for random attributions. The tool built to detect
unfaithful explanations would have certified pure noise as faithful.

This is the second-order version of the same disease as the fabricated
attention: not a fake explanation, but a **fake measurement of explanation
quality**. And note the disposition: *deleted, not kept as a fallback*. The
honest options were "measure it properly" or "report nothing". A plausible
number was not among them.

### Other faithfulness metrics (not implemented, and why)

- **Insertion score** — start from a blank input, add the most salient regions
  first, watch confidence rise. Complements deletion. Explicitly out of scope.
- **Infidelity** — expected squared difference between attribution-predicted
  and actual output changes under perturbation.
- **Sensitivity-n** — how well attributions predict the effect of removing
  random subsets of size n.
- **IoU against ground-truth masks** — where you know which region matters
  (e.g. you inserted the artefact), measure overlap.

All are in the non-committed scope list. The project implements deletion score
and deletion AUC, properly, rather than several metrics badly.

---

## 2.9 Bias as an interpretability question

Attribution explains one prediction. **Bias profiling** explains a pattern
across many — which is a different and equally important kind of
interpretability.

The method for ASR:

1. Take a corpus with accent labels. L2-ARCTIC: 24 non-native English speakers
   across six first languages (Hindi, Korean, Mandarin, Arabic, Spanish,
   Vietnamese), all reading the same English prompts.
2. Transcribe every utterance.
3. Compute Word Error Rate against ground truth.
4. Group by first language, average within group.
5. Report the spread.

```python
wers_list = list(cohort_breakdown.values())
bias_discrepancy = round(max(wers_list) - min(wers_list), 4) if wers_list else 0.0
```

The **bias discrepancy index** is simply worst cohort minus best cohort. Crude
but honest and readable: "this model is *this much* worse for the group it
serves worst".

Because all speakers read identical prompts, the content is controlled. A WER
difference between cohorts is attributable to accent rather than to text
difficulty. That is what makes the comparison meaningful — and it is why the
language-misdetection bug in §1.7 was so damaging: it broke exactly that
controlled comparison.

---

## 2.10 Two practical theory traps

### Concurrency corrupts attribution

Attribution mutates a shared model object: it registers hooks, zeroes
gradients, runs backward passes. Two simultaneous requests against the same
cached model interleave all of that.

```python
# Every model_type here reuses one process-wide cached nn.Module ... and
# Grad-CAM/Captum mutate that shared module in place -
# register_forward_hook/register_full_backward_hook on the same target
# layer, then model.zero_grad()/.backward(). Two threads doing that at once on
# the same module race each other's hooks and gradient buffers, which is
# exactly what produced the intermittent "size of tensor a (2) must match ...
# dimension 1" 500s: one thread's hook captured activations/gradients that
# belonged to a different thread's forward/backward pass.
```

And a second, worse case — a *plain inference* racing an attribution:

```python
# a live L2-ARCTIC + whisper-base repro produced a Grad-CAM "size of
# tensor a (2) must match the size of tensor b (0)" crash on one thread and
# garbage (non-audio-matching) transcripts on the other, from ordinary
# concurrent page load (the Saliency tab's XAI fetch racing the transcript
# fetch).
```

A hook registered by the attribution thread fires on the *transcription*
thread's forward pass too, because hooks are per-module, not per-caller. One
request crashed; the other returned a transcript of nothing in particular.

The fix is a per-model-identity lock, shared between the saliency path and the
inference path so both derive the *same* key:

```python
def model_lock_key(model: str) -> Optional[str]:
    model_type = detect_model_type(model)
    if model_type == "whisper":
        return f"whisper:{resolve_whisper_model_id(model)}"
    elif model_type == "wav2vec2":
        return "wav2vec2:ser-emotion"
    elif model_type == "add":
        return f"add:{model}"
    return None
```

Per-identity, not global: unrelated models still run in parallel. And the
function lives in one module and is imported by both callers, because two
copies of this logic that drift apart give you a lock that protects nothing.

### Model routing by substring

```python
_ADD_MODEL_KEYS = ("melody-machine", "wav2vec2-add")

def detect_model_type(model: str) -> str:
    if model in _ADD_MODEL_KEYS or "deepfake" in model.lower():
        return "add"
    elif "whisper" in model.lower():
        return "whisper"
    elif "wav2vec" in model.lower():
        return "wav2vec2"
    return "unknown"
```

The ADD check must come first. `"wav2vec2-add"` contains `"wav2vec"`, so the
generic branch would claim it and run saliency against the emotion model's
weights while labelling the result with the deepfake model's name.

The same trap, in the embedding router, with a worse outcome:

```python
# Routing used to be three substring tests in an if/elif chain, which got two
# cases wrong. ``wav2vec2-add`` contains "wav2vec", so the deepfake models were
# handed to the emotion extractor - same 1024 dims, no error, wrong latent
# space. And a custom SER checkpoint whose name says neither "whisper" nor
# "wav2vec" (``myorg/custom-ser``) fell through to the *Whisper* extractor,
# which is how a custom emotion model ended up plotted in Whisper's space.
```

**Same 1024 dims, no error, wrong latent space.** The shapes matched, so
nothing complained; the scatter plot rendered; the points were in a space that
had nothing to do with the requested model.

The fix asks the registry (which knows each model's declared family) and keeps
substrings only as a fast path for built-in keys and a last-resort fallback.

The general lesson: **string matching on model names is a heuristic pretending
to be a type system.** Where a real answer is available — a config field, a
registry — ask for it.

---

## 2.11 Summary

- Explanation here means local, post-hoc, attribution-based: a heatmap over the
  spectrogram for one prediction.
- "Where is the energy" and "what did the model use" are different questions
  whose answers look identical. Confusing them is this project's central
  hazard, and it has happened.
- Gradient methods: plain gradients (saturate), Integrated Gradients (axiomatic,
  baseline-dependent), Grad-CAM (needs a conv layer *and* a class-discriminative
  target).
- Perturbation methods: occlusion (simple, expensive), LIME (needs meaningful
  feature groups and a correctly-scaled surrogate), SHAP (axiomatic, always
  approximated).
- All four are implemented because their disagreement is diagnostic.
- Attention is exposed as observation, never as attribution — the literature
  does not support the latter.
- Every XAI result carries provenance: measured, fallback (with a mandatory
  reason), or unavailable. A fallback without a reason raises at runtime.
- Faithfulness is measured by deletion score and deletion AUC, with real
  masking and real re-inference. The arithmetic shortcut that preceded it
  would have certified random noise as faithful, and was deleted rather than
  kept.
- Bias profiling is interpretability across many predictions: WER per accent
  cohort on controlled text, reported as a max-minus-min discrepancy.
- Attribution mutates shared model state, so it must be serialised per model
  identity — including against plain inference on the same model.

Next: [Chapter 3 — System architecture](03-architecture.md).
