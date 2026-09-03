# AudioLIT FR Remediation Plan

> **CLOSED 2026-08-20.** Every order below is done and verified against
> `develop`. Re-audited mechanically at that date: 13/13 met, backend 581
> accounted for and matching collected, frontend lint/test/build clean, removal
> ledger empty. Two orders were closed by other work rather than by this plan
> (PR-9 by LIT-237, PR-2/3/4/5 partly by LIT-238/239/147/240), and two gaps were
> found *after* it was written — FR16.1's fabricated deletion score and the
> canvas waveform layer — both now closed and recorded in §5.
>
> Kept as the record of what was wrong and why, not as a to-do list.

**Audited at:** `ceee3c8` on `feature/lit-232-dataset-warmup-eta-cancellation-and-memory-cleanup`
**Baseline suite:** 486 passed, 6 skipped, 0 failed
**Scope:** the 14 gaps found auditing all 14 committed FRs against the working tree.

This document is the specification. Antigravity executes one work order per branch,
one PR per work order, in the wave order below. Each order states the gap, why it
matters, the exact edit sites, and the acceptance test. Do not batch orders; the
collision map in §3 exists because this repo has already shipped two individually
green PRs that broke in combination.

---

## 1. How to run a work order

For every order, in this order:

1. `git fetch origin && git checkout develop && git pull`
2. `git checkout -b <branch>` — use the Linear issue's own `gitBranchName` field.
   Branch names below are **provisional**; replace them with the real ones.
3. Make only the edits the order lists. Do not opportunistically fix neighbouring
   code — that is what the next order is for.
4. Verify locally, both suites, before pushing:
   ```
   cd Backend  && pytest -q
   cd Frontend && npm ci && npm run lint && npm run build
   ```
5. Stage the work. **Do not commit** — the developer commits.
6. After the developer commits: push, open the PR against `develop` with the LIT id
   in title and body, then wait for `gh pr checks <n>` to report a terminal result.
7. Do not merge. Every PR needs one approving review from another team member.

**Definition of done for every order:** the stated acceptance test passes, the full
backend suite is still green, the frontend lints and builds, and no test was
weakened, skipped, or deleted to get there.

---

## 2. Wave order and dependencies

```
WAVE 0   PR-1  provenance contract ─────┐
                                        │
WAVE 1   PR-2  FR8.2  real Grad-CAM ────┤   (all three touch saliency_service.py)
         PR-3  FR9.1  IG relabel     ───┤   strictly sequential
         PR-4  FR8/16 energy fallback ──┤
         PR-5  FR17.1 attention badge ──┘

WAVE 2   PR-6  FR11   remove mock data ─┐   (both touch EmbeddingPlot.tsx)
         PR-7  FR11.3 real-label colour ┘   strictly sequential

WAVE 3   PR-8  FR8.4  viridis + opacity      (independent, parallelisable)
         PR-9  FR2.3  licence notice  ** CANCELLED - closed by LIT-237 (#113) **
         PR-10 FR10.2 playback sync
         PR-11 FR12.2 Web Audio preview

WAVE 4   PR-12 FR7.2  ADD timeline + panel   (independent, larger)
         PR-13 FR1.4  VRAM fallback
         PR-14 FR3.2  async upload  (LIT-157)
         PR-15 FR4.1  content-addressed cache

ANYTIME  PR-16 docs: CLAUDE.md app/core correction
```

**Hard rule:** PR-2 → PR-3 → PR-4 land in that order. PR-6 → PR-7 in that order.
Waves 3 and 4 may run in parallel with each other and with anything else.

---

## 3. File collision map

Order these against each other; never run two in the same row concurrently.

| File | Touched by |
|---|---|
| `Backend/app/domain/saliency_service.py` | PR-1, PR-2, PR-3, PR-4 |
| `Frontend/src/components/visualization/EmbeddingPlot.tsx` | PR-6, PR-7 |
| `Frontend/src/components/visualization/SaliencyVisualization.tsx` | PR-3, PR-5, PR-8 |
| `Frontend/src/components/visualization/XAIOverlayCanvas.tsx` | PR-5, PR-8 |
| `Backend/app/api/routes/inferences.py` | PR-12, PR-15 |
| `Backend/app/infrastructure/cache_keys.py` | PR-3 (version bump), PR-15 |

---

# WAVE 0

## PR-1 — Shared provenance contract for XAI outputs

**Branch (provisional):** `chore/xai-provenance-contract`
**FR:** none directly; unblocks PR-2, PR-4, PR-5.
**Type:** additive, no behaviour change.

### The gap

Three separate places in this codebase return fabricated data in the exact shape of
genuine model output: the attention fallback (flagged as of `ceee3c8`, with an
ad-hoc boolean), the saliency energy-map fallback (unflagged), and the embedding
mock data (unflagged). Each was patched or will be patched with its own ad-hoc
field, which means the UI needs three different checks and the next fallback added
will invent a fourth convention.

### Why it matters

FR17 is a whole requirement dedicated to the principle that fabricated output must
never be presented as measured. That principle is not attention-specific. One
contract makes it enforceable; three booleans make it a matter of remembering.

### What to code

Create `Backend/app/domain/provenance.py`:

```python
class Provenance(str, Enum):
    MEASURED = "measured"        # produced by the model on this input
    FALLBACK = "fallback"        # synthesised stand-in, NOT model output
    UNAVAILABLE = "unavailable"  # could not be produced at all

def provenance_fields(source: Provenance, reason: str | None = None) -> dict:
    """Returns {"provenance": ..., "provenance_reason": ...} for merging
    into any XAI response payload."""
```

Migrate the one existing consumer: in
`Backend/app/domain/model_loader_service.py`, the ASR result dict currently sets
`attention_is_fallback: bool`. Keep that key (it is already in cached payloads with
a 24 h TTL) **and** add the new `provenance` / `provenance_reason` fields alongside
it. Do not remove the boolean in this PR.

### Where

- **New:** `Backend/app/domain/provenance.py`
- **Edit:** `Backend/app/domain/model_loader_service.py` — the `result_dict` built in
  `transcribe_whisper` under `return_attention`, and the fabricated-pattern branch
  that sets `attention_is_fallback = True`.

### How to test

New `Backend/tests/test_provenance.py`:
- `provenance_fields(MEASURED)` has no reason; `FALLBACK` requires a non-empty reason
  (raise `ValueError` if missing — a fallback with no explanation is the bug).
- A real attention extraction reports `MEASURED`; the fabricated-pattern path
  reports `FALLBACK` with a reason mentioning that it is synthesised.
- The legacy `attention_is_fallback` boolean still agrees with the new field.

### Acceptance

Every attention response carries a `provenance` value, and no code path can emit
`FALLBACK` without a reason string.

---

# WAVE 1 — Interpretability correctness

## PR-2 — FR8.2: wire the real Grad-CAM into the saliency endpoint

**Branch (provisional):** `feature/lit-xxx-fr8-2-wire-real-gradcam`
**FR:** FR8.2 · **Severity:** High · **Depends on:** PR-1

### The gap

`compute_grad_cam()` at `Backend/app/domain/saliency_service.py:596` is a genuine,
correct Grad-CAM: it hooks the target layer's activations and gradients, pools the
gradients into per-channel weights, and returns a ReLU'd weighted combination. It
has seven call sites — all in `tests/test_grad_cam.py`. Production callers: zero.

```
$ grep -rn "compute_grad_cam" Backend/app Backend/tests
app/domain/saliency_service.py:596   def compute_grad_cam(     ← definition
tests/test_grad_cam.py               7 call sites              ← tests only
```

`generate_saliency()` at line 558 dispatches to IG, LIME, or SHAP. Grad-CAM is not
a branch. Separately, the ADD family raises outright:

```python
elif model_type == "add":
    raise ValueError(f"Saliency/Grad-CAM is not yet supported for deepfake-detection models ({model}).")
```

FR8.2 names the deepfake classifier specifically: *"gradient-weighted feature maps
from the deepfake classifier's final layer, projected onto the spectrogram grid."*
So the one model family the requirement is written about is the one family the
endpoint refuses.

### Why it matters

FR8.2 is described in the SRS as *"genuinely new"* — the headline addition to the
inherited attribution stack. The algorithm is written and tested. Only the wiring is
missing, which makes this the cheapest high-severity fix on the list.

### What to code

1. Add a real `"gradcam"` branch to `generate_saliency()` that calls
   `compute_grad_cam()`. Do **not** reuse the existing `method == "gradcam"` block —
   that block is Integrated Gradients and PR-3 renames it.
2. Target layer selection: use the existing `find_last_conv_layer()` helper.
   - **Wav2Vec2 / ADD:** the conv feature extractor is the correct target and is a
     genuine `Conv1d` stack. This is the FR8.2 path — implement it first and treat
     it as the acceptance case.
   - **Whisper:** the encoder's `conv1`/`conv2` front-end. Acceptable target.
   - If no conv layer resolves, return `Provenance.UNAVAILABLE` with a typed error.
     **Do not fall back to IG and call it Grad-CAM** — that is the FR9 defect again.
3. Enable the ADD branch of `generate_saliency()` for `method == "gradcam"` only.
   Leave the `ValueError` in place for LIME/SHAP/IG on ADD models.
4. Project the CAM onto the spectrogram grid (time × mel) and return it in the same
   response shape as the other methods, tagged `Provenance.MEASURED`.

### Where

- `Backend/app/domain/saliency_service.py` — `generate_saliency()` (line ~558), and
  the ADD branch; `compute_grad_cam()` and `find_last_conv_layer()` already exist.

### How to test

Extend `Backend/tests/test_grad_cam.py` and add to `test_saliency_service.py`:
- `generate_saliency(..., method="gradcam")` on an ADD model returns segments and
  does **not** raise.
- The returned CAM is not numerically equal to the IG attribution for the same input
  — this is the test that proves the two methods are now actually different.
- CAM values are in `[0, 1]` and the returned matrix's time axis length matches the
  spectrogram's.
- A model with no conv layer yields `UNAVAILABLE`, not a silent IG result.

### Acceptance

Requesting `gradcam` runs `compute_grad_cam`, works on the deepfake classifier, and
produces a different attribution from `integrated_gradients` on the same input.

---

## PR-3 — FR9.1: stop calling Integrated Gradients "GradCAM"

**Branch (provisional):** `feature/lit-xxx-fr9-1-ig-label-correction`
**FR:** FR9.1 · **Severity:** High · **Depends on:** PR-2 (must land after)

### The gap

FR9 exists for exactly one purpose: to correct ECHO 1.0's mislabelling of Integrated
Gradients as GradCAM. The mislabel is unchanged on both sides of the wire.

```python
# Backend/app/domain/saliency_service.py:82,99   (and :342-343 for wav2vec2)
if method == "gradcam":
    ...
    ig = IntegratedGradients(model_forward)
```
```tsx
// Frontend/src/components/visualization/SaliencyVisualization.tsx:196
<SelectItem value="gradcam">GradCAM</SelectItem>
// Frontend/src/components/panels/FaithfulnessAuditPanel.tsx:146  — same
```

The picker offers GradCAM, LIME, SHAP. There is no Integrated Gradients option, so
the one method that *is* IG is the one labelled as something else.

### Why it matters

Every saliency figure produced for the report is captioned with the wrong method
name. FR16's faithfulness audit runs against `"gradcam"`, so the deletion scores
currently measure IG while reporting Grad-CAM. After PR-2, leaving this unfixed is
worse: `"gradcam"` would be ambiguous between two real methods.

### What to code

1. Rename the existing IG branches from `method == "gradcam"` to
   `method == "integrated_gradients"` in **both** `generate_whisper_saliency` (line
   ~82) and `generate_wav2vec2_saliency` (line ~342).
2. Add `"integrated_gradients"` to the UI method pickers, labelled
   **"Integrated Gradients"**. Keep `"gradcam"` as an option labelled **"Grad-CAM"**
   — after PR-2 it is a real Grad-CAM.
3. **Bump the saliency cache schema version.** `Backend/app/api/routes/saliency.py:16`
   currently has `SALIENCY_SCHEMA_VERSION = "v2"`. Set it to `"v3"`.
   This is mandatory, not cosmetic: cached entries keyed `saliency_v2_..._gradcam_...`
   hold IG results. Without the bump, a `gradcam` request after PR-2 gets a stale IG
   result back from cache under a key that now means Grad-CAM — FR4.1 requires that a
   key-format change cannot cause collisions.
4. Reject unknown method strings with a 400 rather than silently falling through to
   the `else: attributions = torch.zeros_like(...)` branch at line ~163, which
   currently returns an all-zero map for a typo.

### Where

- `Backend/app/domain/saliency_service.py` — lines ~82, ~342, and the `else` at ~163
- `Backend/app/api/routes/saliency.py:16` — schema version
- `Frontend/src/components/visualization/SaliencyVisualization.tsx:40,196`
- `Frontend/src/components/panels/FaithfulnessAuditPanel.tsx:79,146`
- `Frontend/src/components/layout/MainLayout.tsx:578` — hardcoded `method: "gradcam"`;
  decide deliberately which method the default warmup should precompute and say so in
  the PR body.

### How to test

- Backend: `method="integrated_gradients"` returns the attribution the old
  `"gradcam"` string used to return (assert equality against a pinned fixture).
- Backend: an unknown method returns 400, not a zero map.
- A `v2` cache entry is not served to a `v3` request.
- Frontend: both options render; `npm run lint && npm run build` clean.

### Acceptance

No code path and no UI string associates the name "GradCAM" with Integrated
Gradients, and both methods are independently selectable.

---

## PR-4 — FR8/FR16: a failed attribution must not silently become an energy map

**Branch (provisional):** `feature/lit-xxx-fr8-energy-fallback-provenance`
**FR:** FR8, FR16 · **Severity:** High · **Depends on:** PR-1, lands after PR-3

### The gap

When the requested attribution returns empty or near-constant, the saliency service
substitutes the encoder's activation energy and returns it in the ordinary response
shape with no flag:

```python
# Backend/app/domain/saliency_service.py:182-195
use_energy_fallback = (saliency_scores.size == 0 or ... < 1e-6)
if use_energy_fallback:
    logger.info("Using Whisper energy-map fallback for saliency")
    enc = model.encoder(input_features).last_hidden_state
    energy = enc.abs().mean(dim=2)...        # not an attribution
```

The SHAP branch at line ~160 also sets `attributions = None` on failure, which routes
straight into this fallback.

### Why it matters

Energy is a property of the signal, not of what the model attended to. It correlates
with loudness, so it looks plausible on a waveform — which is what makes it dangerous
rather than merely wrong. And FR16 computes deletion scores by masking the top-K
regions of this map: masking the loudest frames and observing a confidence drop
measures nothing about explanation faithfulness. A silent fallback turns the
faithfulness auditor into a loudness detector.

### What to code

1. Tag the energy-map path `Provenance.FALLBACK` with reason
   `"attribution was empty or constant; showing encoder energy, not attribution"`.
2. Tag the SHAP/OOM failure path `Provenance.FALLBACK` with its own reason.
3. **FR16 guard:** `evaluate_batch_faithfulness_scores` and the deletion-score
   endpoint must refuse to score an attribution whose provenance is not `MEASURED`.
   Return a typed, explicit "cannot audit a fallback attribution" result rather than a
   number. A deletion score computed on a fallback map is a fabricated metric.
4. Surface the flag in the saliency response so the UI can render it (the badge
   component lands in PR-5 and should be reused here).

### Where

- `Backend/app/domain/saliency_service.py:160-195`
- `Backend/app/domain/evaluation_service.py` — `evaluate_batch_faithfulness_scores`
- `Backend/app/api/routes/saliency.py`, `Backend/app/api/routes/evaluation.py`

### How to test

- Force an all-zero attribution; assert the response is `FALLBACK` with a reason.
- Assert the faithfulness auditor **refuses** a `FALLBACK` attribution and returns no
  deletion score.
- Assert a normal attribution is `MEASURED` and still scores as before — the existing
  faithfulness tests must not change their expected numbers.

### Acceptance

No response can present an energy map as an attribution, and no deletion score can be
computed from one.

---

## PR-5 — FR17.1: render the fabricated-attention distinction in the UI

**Branch (provisional):** `feature/lit-xxx-fr17-1-attention-provenance-badge`
**FR:** FR17.1 · **Severity:** High · **Depends on:** PR-1

### The gap

Commit `ceee3c8` added `attention_is_fallback` to the ASR response, satisfying the
backend half of FR17.1. The requirement continues: *"and the UI shall render that
distinction visibly."*

```
$ grep -rn "attention_is_fallback" Frontend/src
(no matches)
```

### Why it matters

Half-satisfied is indistinguishable from unsatisfied at the screen. Until the
visualisation shows it, a synthesised attention pattern still looks exactly like a
real one — which is the entire defect FR17 was written to remove.

### What to code

1. A small shared `<ProvenanceBadge provenance reason />` component. Three states:
   measured (quiet, or nothing at all), fallback (clearly marked, warning tone,
   reason in a tooltip), unavailable (muted, explains that nothing was produced).
   PR-4 reuses this for saliency, so put it somewhere shared, not inside the
   attention component.
2. Render it in `AttentionVisualization.tsx` wherever an attention map is drawn,
   including the Timeline tab.
3. When provenance is `FALLBACK`, the heatmap itself must be visually distinguished —
   not merely captioned. Desaturate it or apply a hatch overlay, so a reader
   screenshotting the panel cannot accidentally present synthetic attention as real.

### Where

- **New:** `Frontend/src/components/ui/ProvenanceBadge.tsx`
- `Frontend/src/components/visualization/AttentionVisualization.tsx`

### How to test

Extend `Frontend/src/tests/ui-components.test.tsx`: given a response with
`provenance: "fallback"`, the badge renders and the reason text is present; given
`measured`, no warning is shown.

### Acceptance

A fabricated attention pattern is visibly distinguishable from a genuine one on
screen, without reading the network tab.

---

# WAVE 2 — Latent projection integrity

## PR-6 — FR11: delete the random-data fallback

**Branch (provisional):** `feature/lit-xxx-fr11-remove-mock-projection`
**FR:** FR11 · **Severity:** High

### The gap

When embedding data is unavailable, the projection panel renders synthetic scatter —
random coordinates with randomly assigned emotion labels — with nothing marking it as
fabricated. It is a live fallback, not dead code:

```tsx
// Frontend/src/components/visualization/EmbeddingPlot.tsx:50
const generateMockData = () => {
  for (let i = 0; i < n; i++) {
    x.push(Math.random() * 20 - 10);
    colors.push(['neutral','happy','sad','angry'][Math.floor(Math.random()*4)]);

// …:162   ← reached whenever real data is missing
const mockData = generateMockData();
```

### Why it matters

This is the same class of defect FR17 exists to eliminate, applied to the projection
viewer: fabricated output in the shape of genuine model output. A reader cannot tell
a real cluster from a random one — and in a projection, apparent cluster structure is
the entire finding.

### What to code

1. Delete `generateMockData()` and its call site entirely. Do not flag it — there is
   no legitimate reason to show random points in a latent projection.
2. Replace with an explicit empty state: what is missing, and the action that fixes it
   ("No embeddings for this selection. Run inference or warm the dataset.").
3. Also remove the 3-D branch that generates random `z` coordinates for real 2-D data
   at line ~164 — the same fabrication in another form.

### Where

- `Frontend/src/components/visualization/EmbeddingPlot.tsx:49-66, 162-167`

### How to test

Render `EmbeddingPlot` with empty/undefined embedding data: assert the empty state is
shown and **no plot points exist**. Assert `Math.random` is not called during render.

### Acceptance

The projection panel either shows real embeddings or shows nothing, and never shows
invented points.

---

## PR-7 — FR11.3: colour points by real labels, not filename spelling

**Branch (provisional):** `feature/lit-xxx-fr11-3-label-colouring`
**FR:** FR11.3 · **Severity:** High · **Depends on:** PR-6 (same file)

### The gap

FR11.3 requires colour-coding by emotion label (SER), bona-fide/synthetic (ADD), and
accent or speaker group. What exists is substring matching on filenames, with a
geometric fallback:

```tsx
// Frontend/src/components/visualization/EmbeddingPlot.tsx:123-155
if (filename.includes('01-03') || filename.includes('happy')) return 'happy';
...
// For Common Voice or other datasets, use spatial clustering
if (px > q3X && py > q3Y) return 'region1';     // colour = quartile position
```

No ADD colouring and no accent grouping exist at all.

### Why it matters

Colouring by position makes every projection look clustered, because the colours *are*
the geometry. It is a chart that cannot disagree with itself — it will show apparent
structure on random data. The RAVDESS filename matching is also silently wrong for
any corpus that does not use RAVDESS's naming convention, which is six of the seven.

### What to code

**Backend** — the embeddings endpoint must return a label set per point. It already
resolves each file; extend the response with:
```
labels: { emotion: str|null, deepfake: "bona_fide"|"synthetic"|null,
          accent: str|null, speaker: str|null }
```
Source each from what already exists, and return `null` where genuinely unknown —
never guess:
- `emotion` — from the cached SER prediction (`wav2vec2_detailed_*`), else null.
- `deepfake` — from the cached ADD prediction, else null.
- `accent` / `speaker` — from the corpus metadata already parsed by the loaders in
  `dataset_ingestion.py` (`SampleMetadata` carries these for L2-ARCTIC and CREMA-D).

**Frontend** — a "Colour by" selector: Emotion / Bona-fide vs Synthetic / Accent /
Speaker / None. Colour from `labels`, never from the filename. Points whose label is
`null` render in an explicit "unlabelled" neutral, and the legend says so.

Delete the filename matching and the spatial-quartile fallback outright.

### Where

- `Backend/app/orchestration/inference_service.py` — embedding extraction response
- `Backend/app/api/routes/inferences.py` — `/inferences/embeddings`
- `Frontend/src/components/panels/EmbeddingPanel.tsx` — selector
- `Frontend/src/components/visualization/EmbeddingPlot.tsx:118-160` — colour mapping

### How to test

- Backend: a file with a cached SER prediction returns its emotion; one without
  returns `null`, not a guess.
- Frontend: with all labels `null`, every point is "unlabelled" — assert no point is
  assigned a colour derived from its coordinates.
- Assert `filename.includes` no longer appears in the colour path.

### Acceptance

Point colour is a function of model output or corpus metadata only, and unknown is
rendered as unknown.

---

# WAVE 3 — Presentation and compliance

## PR-8 — FR8.4: perceptually uniform colour scale and adjustable opacity

**Branch (provisional):** `feature/lit-xxx-fr8-4-viridis-overlay`
**FR:** FR8.4 · **Severity:** High

### The gap

```tsx
// Frontend/src/components/visualization/XAIOverlayCanvas.tsx:27-33
const getHeatmapColor = (value) => {
  if (v < 0.25) return [0, 0, 255 * (v*4)];         // blue → cyan
  if (v < 0.5)  return [0, 255*((v-.25)*4), 255];   // cyan → green
  ...
// …:185   style={{ opacity: 0.85 }}    ← fixed, no control
```

FR8.4 asks for *"a perceptually uniform, accessible colour scale"* with *"adjustable
transparency."* This is the jet family, at a hardcoded alpha.

### Why it matters

Jet's luminance is non-monotonic, so it invents banding that is not in the data,
loses ordering in greyscale print, and is not colourblind-safe. It is the standard
counterexample to "perceptually uniform" — and the SRS names that property
explicitly, so this is a stated requirement rather than a taste preference.

### What to code

1. Replace `getHeatmapColor` with a **viridis** (or magma) lookup table — a 256-entry
   const array of RGB triples, sampled at `Math.round(v * 255)`. No runtime dependency
   needed; inline the LUT.
2. Add an opacity slider (0–100%, default ~70%) wired to the overlay alpha. The
   existing `Slider` UI primitive is already in the component library.
3. Add a colourbar legend showing the scale and its min/max values — a heatmap without
   a scale is not readable.

### Where

- `Frontend/src/components/visualization/XAIOverlayCanvas.tsx:27-33, 115-150, 185`
- `Frontend/src/components/visualization/SaliencyVisualization.tsx` — slider control

### How to test

Assert the LUT is monotonically increasing in relative luminance across its 256
entries — this is the property jet fails and the one FR8.4 actually requires. Assert
the slider changes rendered alpha.

### Acceptance

The overlay uses a perceptually uniform ramp with a visible legend and a working
opacity control.

---

## PR-9 — FR2.3: show the licence notice on dataset load

> **CANCELLED 2026-08-19 — already closed by LIT-237, merged as #113.**
> `Backend/app/api/routes/datasets.py` exposes `license` / `non_commercial` per
> corpus, and `Frontend/src/components/dataset/DatasetLicenseNotice.tsx` renders
> the notice, both citing FR2.3. Building this would duplicate merged work.
>
> The audit was taken at `ceee3c8`; #113 landed after it. **Re-run triage against
> `origin/develop` before starting any remaining order** — PR-12 (FR7.2) is the
> next most likely to have been overtaken, since LIT-152 was queued to complete
> the ADD surface.

**Branch (provisional):** `feature/lit-xxx-fr2-3-licence-notice`
**FR:** FR2.3 · **Severity:** Medium

### The gap

The first half of FR2.3 is done well — every corpus carries its licence:

```python
# Backend/app/infrastructure/dataset_ingestion.py:933-939
"ravdess": CorpusSpec("ravdess", TaskFamily.SER, RAVDESS_LICENSE, ...)
```

The second half has no implementation:
```
$ grep -rn "licen" Backend/app/api/ Frontend/src
(no matches in either)
```

### Why it matters

RAVDESS, L2-ARCTIC, ESD and ASVspoof 2021 DF are non-commercial or research-use-only.
The notice is the compliance artefact for an academic deliverable, and the data it
needs is already sitting in `CorpusSpec` — this is one route and one banner.

### What to code

1. Expose `licence` (and a `non_commercial: bool`) on the dataset listing/metadata
   route, read from `CorpusSpec`.
2. On dataset selection, show a dismissible notice for any corpus flagged
   non-commercial, naming the licence. Non-blocking; it informs, it does not gate.

### Where

- `Backend/app/api/routes/datasets.py` (and/or `dataset_management.py`)
- `Frontend/src/components/panels/AudioDatasetPanel.tsx` or `Toolbar.tsx`

### How to test

Backend: the dataset listing returns the correct licence string per corpus and
`non_commercial: true` for exactly RAVDESS, L2-ARCTIC, ESD, ASVspoof 2021 DF.
Frontend: selecting RAVDESS renders the notice; selecting Common Voice (CC0) does not.

### Acceptance

Loading a non-commercial corpus displays its licence.

---

## PR-10 — FR10.2: synchronise the acoustic profiler to playback

**Branch (provisional):** `feature/lit-xxx-fr10-2-playhead-sync`
**FR:** FR10.2 · **Severity:** Medium

### The gap

FR10.1 is fully met — pYIN F0, RMS envelope and the STFT log-mel are computed and
validated against Librosa. FR10.2 additionally requires the pane be *"time-synchronised
with audio playback and with XAI overlays."*

```
$ grep -rn "currentTime\|playhead" Frontend/src/components/panels/AcousticProfilePanel.tsx
(no matches — the chart renders a static timeline)
```

### Why it matters

Relating what the model attends to against the physical signal is the pane's stated
purpose. Without a shared playhead the comparison is done by eye across two
independent axes, which is exactly the manual alignment the pane was meant to remove.

### What to code

1. Lift `currentTime` into shared state (a context, or props from the page that owns
   the `AudioPlayer`). `AudioPlayer.tsx` already tracks `currentTime`.
2. Draw a playhead line on the acoustic profile chart and on `XAIOverlayCanvas`,
   driven by that value.
3. Clicking a position on either chart seeks the audio — synchronisation should work
   in both directions.

### Where

- `Frontend/src/components/audio/AudioPlayer.tsx` — emit time updates
- `Frontend/src/components/panels/AcousticProfilePanel.tsx` — playhead + seek
- `Frontend/src/components/visualization/XAIOverlayCanvas.tsx` — playhead

### How to test

Assert the playhead's x position is a correct function of `currentTime` and clip
duration, and that a click at a known x seeks to the expected time.

### Acceptance

One playhead, shared by the player, the profiler, and the XAI overlay.

---

## PR-11 — FR12.2: build the Web Audio preview that only exists as a mock

**Branch (provisional):** `feature/lit-xxx-fr12-2-web-audio-preview`
**FR:** FR12.2 · **Severity:** Medium

### The gap

FR12.2 requires a client-side preview letting the user audition or mute a selected
region *before any network call*. `AudioContext` appears nowhere in application code:

```
$ grep -rn "AudioContext" Frontend/src --include=*.tsx --include=*.ts
Frontend/src/tests/ui-components.test.tsx:13,59,64    ← mocks only
```

### Why it matters

The test suite asserts against a mock of a feature that was never built, which is
precisely how this stayed invisible through previous reviews. Treat the existing mock
as evidence of intent, not of implementation.

### What to code

In `PerturbationTools.tsx`, which already converts canvas pixels to ms/Hz
(`startTimeMs`, `startFreqHz` at lines ~81-84):
1. Decode the clip once into an `AudioBuffer` via `AudioContext.decodeAudioData`.
2. "Preview region" — play only `[startTimeMs, endTimeMs]` through a
   `BufferSourceNode`.
3. "Preview muted" — play the clip with the selected region silenced, so the user
   hears the counterfactual before dispatching it.
4. No network call on either action. Clean up the context and nodes on unmount.

### Where

- `Frontend/src/components/analysis/PerturbationTools.tsx`

### How to test

Replace the mock-only assertions in `ui-components.test.tsx` with tests that the
preview controls call `decodeAudioData` and start a source node with the expected
offset/duration, and that **no `fetch` occurs**. Add an unmount test asserting the
context is closed.

### Acceptance

A region can be auditioned and muted locally, with no request to the backend.

---

# WAVE 4 — Larger structural work

## PR-12 — FR7.2: deepfake confidence timeline and forensic panel

**Branch (provisional):** `feature/lit-xxx-fr7-2-add-forensic-timeline`
**FR:** FR7.2 · **Severity:** Medium

### The gap

`predict_deepfake()` returns a single clip-level verdict. The `timeline` field is
declared and never populated, and no UI component renders an ADD panel:

```python
# Backend/app/api/routes/results.py:24
timeline: Optional[List[Dict[str, Any]]] = Field(None, ...)   # never set
```
```
$ grep -rln "deepfake" Frontend/src/components/panels Frontend/src/components/visualization
(no matches)
```

### Why it matters

FR7 is the headline new model task and a core novelty claim. Clip-level detection
works (FR7.1 is met); the forensic view that makes it *interpretable* — where in the
clip the synthesis is — does not exist.

### What to code

**Backend:** add windowed ADD inference — slide a window (suggest 1 s, 50% overlap,
make both parameters arguments) across the clip, run the detector per window, and
return `[{start_s, end_s, synthetic_probability, confidence}]`. Reuse the loaded
model across windows; do not reload per window. Populate `timeline`. Cache it under
its own key family in `cache_keys.py` and warm it from the dataset warmup task
alongside the other families.

**Frontend:** a `DeepfakeForensicPanel` showing the clip-level verdict plus the
timeline as an area/step chart on the same time axis as the acoustic profiler, using
the shared playhead from PR-10 if that has landed.

### Where

- `Backend/app/domain/model_loader_service.py` — windowed variant of `predict_deepfake`
- `Backend/app/api/routes/inferences.py`, `Backend/app/infrastructure/cache_keys.py`
- `Backend/app/orchestration/task_orchestrator.py` — warm the new family
- **New:** `Frontend/src/components/panels/DeepfakeForensicPanel.tsx`

### How to test

Backend: a 5 s clip at 1 s/50% yields 9 windows with monotonically increasing
`start_s`; probabilities are in `[0,1]`; window count is a correct function of clip
length. Assert the model is loaded once, not per window.

### Acceptance

The ADD panel shows where in the clip the detector suspects synthesis.

---

## PR-13 — FR1.4: fall back to CPU on VRAM overflow

**Branch (provisional):** `feature/lit-xxx-fr1-4-vram-cpu-fallback`
**FR:** FR1.4 · **Severity:** Medium (low urgency on the team's current CPU-only machines)

### The gap

FR1.1–FR1.3 are implemented carefully — safetensors enforced before download, commit
SHA pinned, `UNSUPPORTED_ARCHITECTURE` a real typed error. FR1.4's lazy CPU fallback
is absent:

```python
# Backend/app/domain/model_registry_service.py:200-202
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)      # no OutOfMemoryError handling
```

### Why it matters

Low practical urgency — the team runs CPU-only (`torch 2.13.0+cpu`), so this will not
surface until a GPU box is used, and then it will surface as a hard crash on model
load. Cheap to fix now, and FR1.4 requires a *user-visible* warning, not just a
silent retry.

### What to code

Wrap `.to(device)` in a `try/except torch.cuda.OutOfMemoryError` (and the
`RuntimeError` "out of memory" variant). On overflow: `torch.cuda.empty_cache()`,
retry on CPU, and record `device_fallback: True` with a reason on the `LoadedModel`
so the API and UI can surface the warning FR1.4 asks for.

### Where

- `Backend/app/domain/model_registry_service.py:200-202`, `LoadedModel` dataclass
- Whichever model-status route feeds `ModelDownloadBanner.tsx`

### How to test

Monkeypatch `.to()` to raise `torch.cuda.OutOfMemoryError` on first call: assert the
model still loads on CPU and `device_fallback` is `True` with a reason. Guard the test
so it runs without a GPU.

### Acceptance

VRAM overflow degrades to CPU with a visible warning instead of failing the load.

---

## PR-14 — FR3.2: stop blocking the request on inference (LIT-157)

**Branch (provisional):** use LIT-157's own `gitBranchName`
**FR:** FR3.2 · **Severity:** Medium · **Contract change — coordinate before starting**

### The gap

FR3.2 states the web-server thread shall never block on model inference, and §3.6.2
sets the threshold at roughly 200 ms. `/upload` awaits a full forward pass:

```python
# Backend/app/api/routes/upload.py:61
prediction = await run_inference(model, str(file_path))
# …:69
embedding_result = await extract_single_embedding(...)
```

### Why it matters

The RQ fabric, the WebSocket channel and the long-polling fallback are all built and
working — this is the last route bypassing them. It is also the route a demo starts
with, so it is the most visible violation of the architecture the SAD describes.

### What to code

1. `/upload` stores the file, enqueues ASR/SER/ADD via `task_orchestrator`, and
   returns `{job_id, websocket_url, schema_version}` immediately.
2. Frontend consumes it through the existing `useTaskStatus` hook, which already
   handles the WebSocket and the long-poll fallback.
3. Keep the response schema versioned (FR3.3) and update any caller that assumes the
   prediction is present in the upload response.

**This changes an HTTP contract.** Announce it before starting; check `gh pr list
--state open` for anything touching `upload.py`, `MainLayout.tsx`, or
`PredictionPanel.tsx` first.

### Where

- `Backend/app/api/routes/upload.py`
- `Frontend/src/components/layout/MainLayout.tsx`,
  `Frontend/src/components/audio/AudioUploader.tsx`,
  `Frontend/src/hooks/useTaskStatus.ts`

### How to test

Assert `/upload` returns within a small bound with a `job_id` and that no model
inference ran in the request. Assert the job completes and results arrive over the
progress channel. Existing upload tests will need updating — update them, do not
delete them.

### Acceptance

No route runs model inference on the request path.

---

## PR-15 — FR4.1: put the hot paths on the content-addressed cache

**Branch (provisional):** `feature/lit-xxx-fr4-1-content-addressed-keys`
**FR:** FR4.1 · **Severity:** Medium · **Largest order — schedule last**

### The gap

FR4's stated novelty is a SHA-256 key over (audio bytes, model, task, parameters).
That scheme exists and is well built — cache schema version, msgpack/lz4 codecs,
dedup lock — but only one route uses it:

```
$ grep -rn "cache_manager\|cached_inference" Backend/app --exclude=core/redis.py
app/api/routes/results.py:39,86      ← the only production caller
```

Every inference, attribution, acoustic and embedding path uses the inherited ECHO
scheme instead: MD5 of the resolved **file path**.

### Why it matters

MD5-of-path is keyed on a filename. Identical audio at two paths caches twice; edited
audio at one path serves a stale result. FR4.4's reproducibility guarantee — *"identical
requests shall produce byte-identical cached responses"* — rests on content addressing
that the hot paths do not use. Note the `path+size+mtime` hash used as a secondary key
is a partial mitigation, not content addressing: it detects edits but still keys on
location.

### What to code

Staged, so it does not destabilise the warmup work just landed in `ceee3c8`:

1. Add `content_sha256(path)` to `Backend/app/infrastructure/cache_keys.py` — SHA-256
   over the audio bytes, chunked (`generate_audio_hash_from_bytes` in
   `app/core/redis.py` already does exactly this; reuse it rather than reimplementing).
2. Add the SHA-256 key as a **third** hash in the existing `both_hashes` tuple, so
   writers populate it and readers try it first, falling back to the path hashes.
   Nothing breaks during the transition.
3. Bump the cache schema version, per FR4.1.
4. Once the read paths prefer the content hash, stop writing the path-hash keys and
   note the removal in the PR body.

Do **not** attempt to migrate every route to `RedisCacheManager` in one PR.

### Where

- `Backend/app/infrastructure/cache_keys.py`, `Backend/app/api/routes/inferences.py`,
  `Backend/app/orchestration/inference_service.py`,
  `Backend/app/orchestration/task_orchestrator.py`

### How to test

- The same audio bytes at two different paths produce the same key and one cache entry.
- Editing a file in place invalidates its entry.
- Extend `tests/test_warmup_cache_contract.py`: warmed entries are still found after
  the key change, and payload shapes are unchanged.
- `tests/test_redis_cache.py` already covers determinism and corruption — keep it green.

### Acceptance

Cache identity is a function of audio content, model, task, and parameters — not of
where the file happens to sit.

---

# ANYTIME

## PR-16 — Correct the CLAUDE.md repo-structure claim

**Branch (provisional):** `docs/claude-md-app-core-correction`

### The gap

`CLAUDE.md` states that LIT-230 removed `app/core/` and that its return is *"the exact
bug LIT-230 fixed."* `Backend/app/core/redis.py` exists on this branch and is imported
by `Backend/app/api/routes/results.py:7`.

### Why it matters

The file's own warning applies to itself: *"a stale CLAUDE.md is worse than no
CLAUDE.md, because it reads as authoritative."* A session following it today would
either try to delete a module that one route depends on, or treat a real structural
question as already settled.

### What to code

Correct the repo-structure section to describe what is actually on disk, and state
plainly what `app/core/redis.py` is (the FR4 content-addressed cache manager) and what
the intended end state is — either it moves to `app/infrastructure/` or the claim that
`app/core/` is gone is retired. Flag the decision in a Linear comment rather than
deciding it unilaterally.

---

# 4. Summary

| PR | FR | Severity | Size | Depends on |
|---|---|---|---|---|
| 1 | — | foundation | S | — |
| 2 | FR8.2 | High | M | 1 |
| 3 | FR9.1 | High | S | 2 |
| 4 | FR8/16 | High | M | 1, 3 |
| 5 | FR17.1 | High | S | 1 |
| 6 | FR11 | High | S | — |
| 7 | FR11.3 | High | M | 6 |
| 8 | FR8.4 | High | S | — |
| ~~9~~ | ~~FR2.3~~ | — | — | **CANCELLED — closed by LIT-237 (#113)** |
| 10 | FR10.2 | Medium | M | — |
| 11 | FR12.2 | Medium | M | — |
| 12 | FR7.2 | Medium | L | — |
| 13 | FR1.4 | Medium | S | — |
| 14 | FR3.2 | Medium | L | — |
| 15 | FR4.1 | Medium | L | — |
| 16 | — | docs | S | — |

**Suggested first session:** PR-1 → PR-2 → PR-3. That sequence closes the two findings
that would not survive a demo question, and PR-2 is mostly wiring work that is already
written and tested.

**A note on scope discipline.** Several orders touch files that a neighbouring order
also touches. The instruction to make only the listed edits is not bureaucratic: this
repo has already shipped two individually green PRs whose *combination* broke pytest
collection for everyone, and a stale `Path:` field that caused a 383-line module to be
written twice. Sequence matters more than speed here.


---

# 5. Closure record — 2026-08-20

Re-audited against `develop` at `b709c39` + this branch. Probes are claims about
the tree, re-run mechanically rather than read off a status column.

| FR | What closed it | Evidence |
|---|---|---|
| FR1.4 | OOM caught, retried on CPU, `device_fallback` on `LoadedModel` | 2 tests incl. "a non-OOM error is not swallowed" |
| FR3.2 | `/upload` no longer awaits `run_inference`/`extract_single_embedding` | test asserts zero model calls on the request path |
| FR4.1 | `content_sha256` added; `both_hashes` returns content-first | same audio at two paths shares a key; in-place edit changes it |
| FR7.2 | `predict_deepfake_timeline` + `DeepfakeForensicPanel` | 9 windows for a 5 s clip, model loaded **once**, not per window |
| FR8.4 | viridis LUT, opacity slider, colourbar legend | luminance asserted monotonic across 64 steps |
| FR10.2 | `PlaybackContext`; playhead on canvas, F0 and RMS charts; click-to-seek | one publisher (`WaveformViewer`), three subscribers |
| FR11 | `generateMockData` deleted, explicit empty state | `Math.random` count in EmbeddingPlot: 0 |
| FR11.3 | `_point_labels` in the embeddings route; colour-by selector | labels are model output or corpus metadata, `null` otherwise |
| FR12.2 | real `AudioContext` preview, region + muted, no network call | `AudioContext` in components: 7 (was 0, mocks only) |
| FR16.1 | auditor routed through `perturbation_service.compute_deletion_score` | test fails if no inference ran; null, never 0.0, when unmeasurable |
| FR17.1 | `ProvenanceBadge` + desaturation of fabricated overlays | badge renders for `fallback`, nothing for `measured` |
| — | waveform layer given a producer in the acoustic profile | 473 frames, aligned with spectrogram and F0 |
| docs | CLAUDE.md corrected on `app/core/` | it exists; the claim that it is gone was stale |

## Defects found while closing, not in the original audit

1. **The deletion score never ran a model.** `evaluate_batch_faithfulness_scores`
   derived it from `orig_conf * (1.0 - k_pct * (0.5 + saliency_weight))` — monotone
   in `k_pct` by construction, so a random attribution scored like a real one. FR16
   is the requirement whose purpose is catching exactly that. The real masking and
   re-inference already existed, unwired, in `perturbation_service`.

2. **ADD overwrote the ASR transcript.** `deepfake_keys` shares the transcript
   family's key shape, and warmup passed the *selected ASR model*, so `tasks=[asr,add]`
   on a Whisper model wrote the deepfake dict over the transcript it had just cached.
   Introduced by the very module built to stop payload-shape collisions.

3. **`resolve_audio_reference` called with swapped positional arguments** in the
   cached-results route, so every analytics lookup missed and the panel fell back
   to ground-truth metadata that looked like a model prediction.

4. **Acoustic cache had no schema version**, so the FR10.1 spectrogram was
   computed correctly and never served to any file cached before it.

5. **Three of four analytics lookups read invented key names** (`ser_{h}`,
   `add_{h}`) that no writer produces.

Each is the same shape: a key or a contract asserted in one place and not checked
against the other side. `ag verify`'s removal ledger and the payload-shape tests
in `test_warmup_cache_contract.py` exist to make that class mechanically visible.
