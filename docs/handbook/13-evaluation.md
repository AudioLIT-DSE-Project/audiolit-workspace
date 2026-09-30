# Chapter 13 — Evaluation: accent bias and faithfulness auditing

Two measurements, both of which required building a metric from scratch, and
both of which produced a wrong answer on the first attempt in ways that are
worth studying closely.

---

## 13.1 Word Error Rate

WER is the standard ASR metric:

```
WER = (Substitutions + Deletions + Insertions) / Reference_Word_Count
```

- **Substitution** — a word was recognised as a different word.
- **Deletion** — a reference word was missed.
- **Insertion** — a word was invented.

0.0 is perfect. **WER can exceed 1.0**, because insertions are not bounded by
the reference length. That fact matters later (§13.4).

### Computing it: edit distance

The minimum number of edits is the **Levenshtein distance** at word level,
computed by dynamic programming.

```python
def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Computes Word Error Rate (WER) using word-level Levenshtein distance.
    WER = (Substitutions + Deletions + Insertions) / Total_Reference_Words

    Punctuation and casing are normalised away first; see _normalise_for_wer.
    """
    ref_words = _normalise_for_wer(reference)
    hyp_words = _normalise_for_wer(hypothesis)

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    r_len = len(ref_words)
    h_len = len(hyp_words)

    dp = np.zeros((r_len + 1, h_len + 1), dtype=int)
    for i in range(r_len + 1):
        dp[i, 0] = i
    for j in range(h_len + 1):
        dp[0, j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                sub_cost = dp[i - 1, j - 1] + 1
                del_cost = dp[i - 1, j] + 1
                ins_cost = dp[i, j - 1] + 1
                dp[i, j] = min(sub_cost, del_cost, ins_cost)

    edit_distance = dp[r_len, h_len]
    return float(edit_distance / r_len)
```

`dp[i][j]` is the minimum edits to turn the first *i* reference words into the
first *j* hypothesis words.

**Base cases:** `dp[i][0] = i` (delete all *i* reference words), `dp[0][j] = j`
(insert all *j* hypothesis words).

**Recurrence:** if the words match, no cost — inherit the diagonal. Otherwise
take the cheapest of substitute (diagonal + 1), delete (up + 1), insert
(left + 1).

**The empty-reference case** is a deliberate judgement, not an oversight. Empty
reference with empty hypothesis is perfect (0.0). Empty reference with *any*
output is total failure (1.0) — the denominator would be zero, and 1.0 is the
sensible convention.

O(r × h) time and space. For sentence-length inputs, trivial.

### Normalisation, and a bug that understated the bias by 56%

```python
def _normalise_for_wer(text: str) -> List[str]:
    """Lowercase, strip punctuation, split into words.

    ASR output carries casing and punctuation that a raw comparison scores as
    errors on an otherwise perfect transcription: Whisper returns "a child."
    where the reference says "a child", and the trailing full stop turns a
    correct word into a substitution. Both sides are normalised identically
    before scoring, matching the transform `accent_bias_profiler` already
    applies (lowercase, remove punctuation, collapse whitespace).

    Leaving this out inflated every L2-ARCTIC cohort by roughly a constant
    amount (overall mean 0.2587 against the profiler's 0.1474) and, because
    the inflation was near-constant across cohorts, it compressed the
    bias-discrepancy index from 0.1091 to 0.0481 - understating the very
    disparity the index exists to measure.
    """
    return re.sub(r"[^\w\s]", " ", text.lower()).split()
```

Read the numbers, because this is the most instructive measurement error in the
project.

Without normalisation, mean WER was **0.2587**. With it, **0.1474**. So the
metric was reporting roughly 75% more errors than existed — every trailing full
stop counted as a substitution.

That alone would be a simple bug. But look at the second-order effect. The
inflation was **near-constant across cohorts** — every cohort's transcripts had
about the same punctuation density. And the bias discrepancy index is
`max_wer − min_wer`. Adding a constant *c* to every cohort leaves the
difference unchanged...

...except it does not, because the bias index went from **0.1091 to 0.0481**.
It was **compressed by 56%**.

Why? Because the inflation was not exactly constant, and more importantly
because WER is a *ratio*. Punctuation adds a roughly fixed number of errors, so
it adds a larger proportional amount to a low-WER cohort than to a high-WER
one. The floor rises more than the ceiling, and the gap closes.

**The metric was systematically understating the exact quantity it existed to
measure.** And the direction of the error is the dangerous one: it made the
model look *fairer* than it is. A bug that flatters your system is much less
likely to be investigated than one that embarrasses it.

The general lesson: **a bias metric built on a ratio is sensitive to anything
that shifts the baseline**, and a constant additive error in a ratio is not a
constant error in the ratio.

The fix also unified two implementations that had drifted. `accent_bias_profiler`
already normalised, using jiwer:

```python
# ASR output includes casing/punctuation that a raw string comparison would
# penalize as "errors" even on an otherwise perfect transcription -- both
# sides are normalized the same way before scoring. jiwer's own
# `wer_standardize` transform stops short of removing punctuation attached to
# words (e.g. "cat," survives it unchanged), so it isn't enough on its own.
_WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])
```

Note the comment about `wer_standardize`: the library's own standard transform
is *not enough*, because it does not remove punctuation attached to words. The
obvious "use the library default" would have left the bug in place.

**Two WER implementations existed in the codebase and disagreed with each
other.** That disagreement is what exposed the bug — a hand-rolled one without
normalisation and a jiwer-based one with it. Finding two numbers for the same
quantity is a gift; chase the contradiction rather than picking the one you
prefer.

---

## 13.2 Cohort aggregation

```python
def calculate_group_wer(cohort_results: List[Dict[str, str]]) -> Dict[str, Any]:
    """Calculates group-wise Word Error Rate (WER) indices for ASR accent-bias
    profiling (FR15)."""
    cohort_data: Dict[str, List[float]] = {}

    for item in cohort_results:
        cohort = item.get("cohort", "unknown").lower()
        ref = item.get("reference", "")
        hyp = item.get("hypothesis", "")

        wer_val = calculate_wer(ref, hyp)
        cohort_data.setdefault(cohort, []).append(wer_val)

    cohort_breakdown: Dict[str, float] = {}
    all_wers: List[float] = []

    for cohort, wers in cohort_data.items():
        avg_wer = float(np.mean(wers)) if wers else 0.0
        cohort_breakdown[cohort] = round(avg_wer, 4)
        all_wers.extend(wers)

    overall_mean_wer = float(np.mean(all_wers)) if all_wers else 0.0
    overall_std_wer = float(np.std(all_wers)) if all_wers else 0.0

    wers_list = list(cohort_breakdown.values())
    bias_discrepancy = round(max(wers_list) - min(wers_list), 4) if wers_list else 0.0

    return {
        "overall_mean_wer": round(overall_mean_wer, 4),
        "overall_std_wer": round(overall_std_wer, 4),
        "bias_discrepancy_index": bias_discrepancy,
        "cohort_breakdown": cohort_breakdown,
        "total_samples_evaluated": len(cohort_results)
    }
```

**The bias discrepancy index is `max − min` across cohort means.** Crude, and
deliberately so: "this model is *this much* worse for the group it serves
worst" is a sentence anyone can read. A statistical test would be more rigorous
and far less legible.

`overall_mean_wer` averages over **all samples**, not over cohort means. With
unequal cohort sizes those differ, and per-sample is the right choice for an
overall figure — averaging cohort means would weight a 10-sample cohort equally
with a 500-sample one.

`.lower()` on the cohort name prevents `"Hindi"` and `"hindi"` becoming two
cohorts.

---

## 13.3 Cohort sampling, and a sampling bug worth avoiding

```python
def load_accent_cohorts(corpus="l2-arctic", samples_per_cohort=None, seed=0, **loader_kwargs):
    """Load ``corpus`` samples grouped by accent/L1.

    Without ``samples_per_cohort``, returns every sample per accent group.
    With it, reservoir-samples each cohort independently in a single pass
    over the corpus (the same algorithm `DatasetLoader.subsample` already
    uses, applied per group) -- a single global cap would let large cohorts
    exhaust it before smaller-represented accents are ever seen, silently
    starving them out of the profile.
    """
```

The trap: with a global cap of 100 samples, streaming the corpus in order fills
those 100 from whichever cohorts appear first. A cohort near the end of the
catalog contributes **zero samples** — and a bias report that silently omits a
cohort is worse than one that reports it badly.

**Per-cohort reservoir sampling** fixes it. Reservoir sampling picks *k* items
uniformly at random from a stream of unknown length in one pass:

```python
        rng = random.Random(seed)
        seen: Dict[str, int] = {}
        for meta in samples:
            key = meta.accent or "unknown"
            reservoir = cohorts.setdefault(key, [])
            i = seen.get(key, 0)
            seen[key] = i + 1
            if i < samples_per_cohort:
                reservoir.append(meta)
            else:
                j = rng.randint(0, i)
                if j < samples_per_cohort:
                    reservoir[j] = meta
```

The algorithm: fill the reservoir with the first *k*. For the *i*-th item
thereafter, keep it with probability *k/i*, replacing a random existing entry.
Every item ends up with equal probability *k/n*, and you never know *n*.

Here it runs **per cohort** — each has its own counter and reservoir — so every
accent gets up to *k* samples regardless of catalog order.

`seed` makes it reproducible, which matters for a research measurement.

### Validating before scoring

```python
    """Streams through ``validated_stream(deep=True)`` (FR2.1) rather than
    ``iter_metadata`` directly, so a missing, undecodable, or all-silence
    clip is excluded before it can be transcribed and scored as a WER
    outlier attributed to the model rather than to a bad source file.
    Rejections are logged with a per-reason count rather than one line per
    clip, since a batch run over hundreds of utterances shouldn't flood the
    log.
    """
```

A silent clip transcribes to nothing, scoring WER 1.0 — and that 1.0 lands on
the model rather than on the broken file. Over hundreds of utterances a handful
of bad files shifts a cohort mean noticeably.

The rejection logging aggregates by reason:

```python
    rejected: Dict[str, int] = {}

    def _on_reject(report) -> None:
        rejected[report.reason] = rejected.get(report.reason, 0) + 1
    ...
    if rejected:
        logger.warning(
            "load_accent_cohorts(%s): excluded %d sample(s) failing integrity check: %s",
            corpus, sum(rejected.values()), rejected,
        )
```

One summary line, not one per clip. A log nobody reads because it is 400 lines
long is not a log.

---

## 13.4 Per-sample scoring

```python
def score_sample(meta: SampleMetadata, transcribe: TranscribeFn) -> Optional[SampleWERResult]:
    """Transcribe one sample and score it against its ground truth.

    Returns ``None`` (rather than raising) for samples this diagnostic can't
    score -- no ground-truth transcript, or no accent label -- since a batch
    run over hundreds of utterances shouldn't die on one bad row.
    """
    if not meta.label or not meta.label.strip():
        logger.warning("Skipping %s: no ground-truth transcript", meta.sample_id)
        return None
    if not meta.accent:
        logger.warning("Skipping %s: no accent/L1 label", meta.sample_id)
        return None

    hypothesis = transcribe(str(meta.audio_path))
    wer = jiwer.wer(meta.label, hypothesis,
                    reference_transform=_WER_TRANSFORM,
                    hypothesis_transform=_WER_TRANSFORM)

    return SampleWERResult(
        sample_id=meta.sample_id, speaker_id=meta.speaker_id, accent=meta.accent,
        reference=meta.label, hypothesis=hypothesis, wer=wer,
    )
```

### Dependency injection for the transcriber

```python
TranscribeFn = Callable[[str], str]
"""Transcribes one audio file path to text. Injected rather than hardcoded so
the scoring/batching primitives below don't depend on a specific ASR
entrypoint or a loaded model in tests -- callers pass a real transcriber
(see `make_whisper_transcriber`) or a fake."""
```

The scoring logic takes a *function*, not a model. So the whole
batching-and-scoring path is testable with a two-line fake transcriber — no
weights, no network, no GPU. It also means the same code can profile any ASR
system, not just Whisper.

This is the single highest-leverage testability decision in the evaluation
layer.

### The transcriber, and the bug that invalidated the measurement

```python
def make_whisper_transcriber(model_id: str) -> TranscribeFn:
    """Build a transcribe function backed by one cached Whisper pipeline.

    A profiling run transcribes many utterances in one pass;
    `model_loader_service.transcribe_whisper` rebuilds its HF pipeline on
    every call, which is fine for one-off request-path inference but far too
    slow across a batch, so this loads the pipeline once and reuses it.
    """
    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr_pipeline = pipeline("automatic-speech-recognition", model=model_id,
                            torch_dtype=torch_dtype, device=device)

    def _transcribe(audio_path: str) -> str:
        audio, _ = librosa.load(audio_path, sr=16_000)
        # Force English decoding. ...
        result = asr_pipeline(audio.astype(np.float32), chunk_length_s=30,
                              generate_kwargs={"language": "en", "task": "transcribe"})
        return result["text"]

    return _transcribe
```

A **closure** over one pipeline: constructed once, reused for every call.

And the language bug from §1.7, which belongs here because of what it did to the
*measurement*:

> *On L2-ARCTIC that produced Vietnamese and Arabic output with WER 22.30 and
> 17.80 (insertions far outnumbering the reference words), which dragged two
> cohort means from ~0.17 to >1.3 and made the accent-bias ranking a measure of
> language misdetection rather than of accent.*

WER 22.30 means 22 times more errors than reference words — a repetition loop.
Two cohorts went from 0.17 to over 1.3.

Now consider what the bias report said. The Vietnamese and Arabic cohorts
appeared catastrophically worse than the others. A researcher would conclude
"this model is severely biased against Vietnamese- and Arabic-accented
English". The real finding was "this model's *language identifier* fails on
these accents, and then the decoder loops".

Both are real problems. They are different problems, with different fixes. The
measurement named the wrong one, confidently, with numbers.

**This is the deepest lesson in the chapter:** a measurement instrument can be
working perfectly and measuring the wrong quantity. WER was computed correctly.
The transcription was the problem. And nothing in the WER number could reveal
that — only looking at the actual transcripts did.

---

## 13.5 The report

```python
@dataclass(frozen=True)
class CohortWERSummary:
    """Aggregated WER statistics for one accent cohort.

    Stats are ``None`` (not NaN) when a cohort has zero scoreable samples,
    so the report stays valid JSON for the dashboard chart the acceptance
    criteria call for -- a literal `NaN` token isn't standard JSON and most
    frontend JSON parsers reject it.
    """
    accent: str
    sample_count: int
    scored_count: int
    mean_wer: Optional[float]
    median_wer: Optional[float]
    stdev_wer: Optional[float]
    min_wer: Optional[float]
    max_wer: Optional[float]
```

**`None`, not NaN** — the same decision as unvoiced pitch (§10.3), for the same
reason plus a technical one: `NaN` is not valid JSON and most parsers reject the
whole document.

**`sample_count` and `scored_count` are separate fields.** A cohort with 50
samples of which 12 were scoreable is a different situation from one with 12
samples, and the report must not hide the difference.

```python
    return CohortWERSummary(
        accent=accent, sample_count=sample_count, scored_count=len(wers),
        mean_wer=statistics.mean(wers),
        median_wer=statistics.median(wers),
        stdev_wer=statistics.stdev(wers) if len(wers) > 1 else 0.0,
        min_wer=min(wers), max_wer=max(wers),
    )
```

**Median as well as mean**, because WER distributions are skewed — a few
disastrous utterances drag the mean while the median stays representative.
Reporting both lets a reader see the skew.

**`stdev` needs at least two samples**; `statistics.stdev` raises on one, so
`0.0` is substituted.

```python
@dataclass(frozen=True)
class AccentBiasReport:
    """Full diagnostic pass output -- ``cohorts`` ranked worst-WER first."""
    corpus: str
    model_id: str
    cohorts: List[CohortWERSummary]
    sample_results: List[SampleWERResult]
```

**Ranked worst-first**, so the group the model serves worst is the first thing
you read.

**`sample_results` includes every individual reference and hypothesis.** That is
what makes the report auditable — and it is exactly what allowed the language
bug to be found. A report of only aggregates would have shown two high numbers
and no way to see they were Vietnamese text.

**Include the raw data.** Aggregates hide the reason.

`corpus` and `model_id` are recorded, so a report is self-describing.

---

## 13.6 Faithfulness auditing

Theory in §2.8; the implementation is `evaluate_batch_faithfulness_scores`.

### The fabricated metric

The docstring is the most important text in the module:

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

Look at the formula. It computes the degraded confidence from the original
confidence and the masking percentage, arithmetically. No model call anywhere.

Because it is monotone in `k_pct` by construction, it **always** produces a
rising degradation curve. Feed it a random attribution and it reports the
attribution as faithful. The tool built to detect unfaithful explanations would
have certified pure noise.

*"It measured the saliency values, not the model."* That sentence is the
diagnosis. And *"deleted, not kept as a fallback"* is the disposition: the
honest options were to measure properly or report nothing. A plausible number
was not among them.

### Refusing to audit a fallback

```python
    for item in eval_items:
        file_path = item.get("file_path", "")
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

        saliency_scores = item.get("saliency_scores", [])
        if not saliency_scores:
            refused += 1
            item_results.append({"file_path": file_path,
                                 "error": "cannot audit an empty attribution"})
            continue
```

The provenance contract with teeth (§2.7). Auditing a fabricated attribution
would produce a faithfulness score for fiction, so it declines — and **counts
the declines**, so refusals appear in the summary rather than vanishing.

### The sweep

```python
        degradation_curve: Dict[str, float] = {"top_0pct": 0.0}
        top_k_scores: List[float] = []
        orig_conf = None
        failure = None

        for k_pct in top_k_percentages:
            measured = measure_deletion(
                audio_path=file_path,
                attributions=saliency_scores,
                model_type=item.get("model_type", model_type),
                model_id=item.get("model_id", "default"),
                k_percent=k_pct * 100.0,
            )
            if not measured.get("success"):
                failure = measured.get("error", "deletion measurement failed")
                break
            orig_conf = measured["initial_confidence"]
            del_score = measured["deletion_score"]
            degradation_curve[f"top_{int(k_pct * 100)}pct"] = round(del_score, 4)
            top_k_scores.append(del_score)
```

`{"top_0pct": 0.0}` seeded up front — masking nothing drops nothing, and the
AUC needs that anchor (§2.8).

`model_id` threaded through, because the audit must measure the model whose
explanation is being audited (§6.5).

The loop `break`s on the first failure rather than continuing with a partial
curve, and the item is then reported as an error. A curve with a hole in it
would integrate to a meaningless AUC.

### `None` rather than zero

```python
    scored = len(overall_deletion_scores)
    # null, never 0.0. A zero deletion score reads as "the attribution is
    # completely unfaithful"; the opposite of "nothing could be measured".
    return {
        "mean_deletion_score": round(float(np.mean(overall_deletion_scores)), 4) if scored else None,
        "mean_deletion_auc": round(float(np.mean(overall_deletion_aucs)), 4) if scored else None,
        "audio_scored": scored,
        "audio_refused": refused,
        "total_audio_evaluated": scored,
        "item_results": item_results,
    }
```

The third appearance of this principle (after NaN pitch and `None` cohort
stats), and here the stakes are highest: **0.0 means "the attribution is
completely unfaithful". `None` means "nothing could be measured".** They are
opposite claims. Defaulting to 0.0 would report every unmeasurable audit as a
maximally damning result.

The summary function repeats the guard rather than trusting the caller:

```python
            # Defaults are None, not 0.0: an unmeasurable audit must not be
            # summarised as a perfectly unfaithful one.
            "faithfulness_deletion_audit": {
                "mean_deletion_score": faithfulness_result.get("mean_deletion_score"),
                "mean_deletion_auc": faithfulness_result.get("mean_deletion_auc"),
                ...
            }
```

`.get(key)` without a default returns `None` — deliberately, because
`.get(key, 0.0)` is the mistake the comment is warning against.

---

## 13.7 The per-item measurement

```python
def compute_deletion_score(audio_path, attributions, model_type="ser",
                           model_id="default", k_percent=10.0, output_dir="uploads"):
    """Calculate single-method deletion-score faithfulness (FR16).

    Scrubs top-K% salient features, executes inference on the masked sample,
    and returns initial vs masked confidence drop and deletion score.
    """
    resolved_path = Path(audio_path)
    if not resolved_path.exists():
        return {"success": False, "error": f"Audio file not found: {audio_path}"}

    waveform, sample_rate = _load_waveform(str(resolved_path))
    masked_waveform = mask_top_k_features(waveform, attributions, k_percent=k_percent)

    masked_filename = f"faithfulness_masked_{uuid.uuid4().hex[:8]}.wav"
    masked_path = Path(output_dir) / masked_filename
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _save_waveform(str(masked_path), masked_waveform, sample_rate)
```

The masked audio is **written to disk** rather than passed in memory, because
the prediction functions take a path. Slightly wasteful, and it means the
retention sweep (§4.6) will eventually clean these up. It also means the masked
audio is *inspectable* — you can listen to exactly what the auditor removed,
which is genuinely useful when a faithfulness score surprises you.

```python
        if model_type.lower() in ("add", "deepfake"):
            from .model_loader_service import predict_deepfake
            orig_res = predict_deepfake(str(resolved_path))
            masked_res = predict_deepfake(str(masked_path))
            orig_conf = float(orig_res.get("confidence", 0.0))
            masked_conf = float(masked_res.get("confidence", 0.0))
            target_class = orig_res.get("predicted_label", "bona-fide")
        else:
            from .model_loader_service import predict_ser
            orig_res = predict_ser(str(resolved_path))
            masked_res = predict_ser(str(masked_path))

            target_class = orig_res.get("predicted_emotion", "neutral")
            orig_probs = orig_res.get("probabilities", {})
            masked_probs = masked_res.get("probabilities", {})
            orig_conf = float(orig_probs.get(target_class, orig_res.get("confidence", 0.0)))
            masked_conf = float(masked_probs.get(target_class, 0.0))
```

**The SER branch is subtler than the ADD branch, and correctly so.**

For ADD, `confidence` is the max class probability and the two classes are
symmetric, so comparing confidences is fine.

For SER, you must track the **same class** across both runs. Suppose the
original predicts "angry" at 0.8. After masking, the model predicts "neutral"
at 0.9. Comparing *confidences* gives 0.8 → 0.9 — an *increase*, implying the
attribution was worse than useless.

But the right question is what happened to P(angry), which might have gone 0.8
→ 0.1. That is a large drop and a faithful attribution.

So the SER branch pins `target_class` from the original prediction and looks up
*that class's* probability in both. `masked_probs.get(target_class, 0.0)`
defaults to 0.0 — correct, because if the class is absent from the masked
distribution its probability really is (effectively) zero.

Getting this wrong would make faithfulness scores wrong in a *direction that
depends on whether the prediction flipped* — which is worse than uniformly
wrong, because it correlates with exactly the cases you care about most.

```python
        confidence_drop = max(0.0, orig_conf - masked_conf)
        deletion_score = round(confidence_drop / orig_conf, 4) if orig_conf > 0 else 0.0

        return {
            "success": True, "model_type": model_type, "model_id": model_id,
            "target_class": target_class, "k_percent": k_percent,
            "initial_confidence": round(orig_conf, 4),
            "masked_confidence": round(masked_conf, 4),
            "confidence_drop": round(confidence_drop, 4),
            "deletion_score": deletion_score,
            "faithfulness_verdict": "faithful" if confidence_drop > 0.05 else "unfaithful",
            "masked_audio_file": str(masked_path).replace("\\", "/"),
        }
```

`max(0.0, ...)` clamps a *negative* drop to zero. Masking salient regions
should not raise confidence; when it does, the attribution is unfaithful, and
0.0 is the correct score. A negative deletion score would be uninterpretable.

The 0.05 verdict threshold is arbitrary and should be read as such. It is a
convenience label; the number is the real output.

Every intermediate is returned — initial confidence, masked confidence, raw
drop, normalised score, the target class, and the path to the masked audio.
**The full working is shown**, so a surprising score can be traced rather than
trusted.

---

## 13.8 Summary

- WER is word-level Levenshtein over the reference length, and can exceed 1.0
  because insertions are unbounded.
- Normalisation is not optional. Without it, mean WER was inflated 75% —
  and because WER is a ratio, the near-constant inflation **compressed the bias
  index by 56%**, understating the disparity the index exists to measure, in the
  flattering direction.
- jiwer's own `wer_standardize` is insufficient; punctuation attached to words
  survives it.
- Two disagreeing implementations of a metric is how the bug was found. Chase
  the contradiction.
- Per-cohort reservoir sampling, because a global cap starves whichever cohorts
  appear late in the catalog.
- Validate clips before scoring, or file corruption is attributed to the model.
  Aggregate rejection logs by reason.
- Inject the transcriber as a function: the whole scoring path becomes testable
  with a fake, and the profiler works for any ASR system.
- A measurement instrument can work perfectly and measure the wrong quantity.
  Language misdetection turned the accent-bias ranking into a
  language-identification ranking, with confident numbers.
- Report `None`, never 0.0 or NaN, for unmeasurable statistics. NaN also breaks
  JSON parsers.
- Report `sample_count` and `scored_count` separately; report median alongside
  mean; include every raw reference and hypothesis, because aggregates hide the
  reason.
- The faithfulness metric that preceded the real one never called a model and
  was monotone by construction — it would have certified random noise. Deleted,
  not kept as a fallback.
- The auditor refuses fallback attributions and counts refusals.
- For multi-class models, track the *original predicted class's* probability
  across both runs, not the max confidence — otherwise a flipped prediction
  looks like a confidence increase.
- Return every intermediate value, including the masked audio path, so a
  surprising score can be traced.

Next: [Chapter 14 — Datasets](14-datasets.md).
