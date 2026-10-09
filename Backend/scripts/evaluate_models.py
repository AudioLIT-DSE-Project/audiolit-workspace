"""Reproducible evaluation runner for the figures quoted in T&E section 6 (LIT-252).

`docs/evaluation/TESTING_AND_EVALUATION.md` section 6 and
`docs/evaluation/DS_ERROR_ANALYSIS.md` quote four numbers: FR15 mean WER and the
bias-discrepancy index, and FR16 mean confidence drop and mean deletion AUC.
Both halves were computable before this script -- `accent_bias_runner` has its
own CLI, and `evaluation_service` has the aggregation -- but nothing regenerated
*both* in one pass, so the documented figures could not be reproduced by a
single command. That is what this adds; it orchestrates the existing services
and deliberately reimplements none of their maths.

Usage
-----
Full run (needs the corpus, the models, and time)::

    python scripts/evaluate_models.py \
        --asr-model-id openai/whisper-base \
        --manifest eval_manifest.json \
        --output docs/evaluation/results

Either half alone::

    python scripts/evaluate_models.py --asr-model-id openai/whisper-base --skip-faithfulness
    python scripts/evaluate_models.py --manifest eval_manifest.json --skip-accent-bias

Verify the plumbing with no corpus, no model download and no GPU::

    python scripts/evaluate_models.py --self-test

Manifest format (FR16 input) -- a JSON list; ``saliency_scores`` is optional and
is computed through the saliency service when absent::

    [{"file_path": "samples/clip1.wav", "model": "whisper-base", "method": "gradcam"}]

Design notes
------------
No checkpoint is hardcoded: the ASR model is a required argument for the FR15
pass, and each manifest item names its own model, so a custom checkpoint is
evaluated as itself.

A stage that cannot run reports ``status: "unavailable"`` with the reason, and
the process exits non-zero. It never emits 0.0 for an unmeasured quantity --
that is the failure mode ``evaluate_batch_faithfulness_scores`` was rewritten to
remove, and a fabricated evaluation number is worse than a missing one in a tool
whose output is read as measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Import as `app.*` so this runs from Backend/ without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --------------------------------------------------------------------------
# FR15 -- ASR accent-bias profiling
# --------------------------------------------------------------------------

def run_accent_bias_stage(
    asr_model_id: str,
    corpus: str = "l2-arctic",
    samples_per_cohort: Optional[int] = None,
    seed: int = 0,
    transcribe: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """Group-wise WER across accent cohorts, plus the bias-discrepancy index.

    ``transcribe`` is injected for the self-test; production passes None and a
    real Whisper pipeline is built. The cohort ranking comes from
    ``run_accent_bias_diagnostic``; the discrepancy index comes from
    ``calculate_group_wer``. Both are existing, tested code -- the WER maths is
    not repeated here.
    """
    from app.domain.accent_bias_runner import run_accent_bias_diagnostic
    from app.domain.evaluation_service import calculate_group_wer

    if transcribe is None:
        from app.domain.accent_bias_profiler import make_whisper_transcriber
        transcribe = make_whisper_transcriber(asr_model_id)

    report = run_accent_bias_diagnostic(
        transcribe,
        corpus=corpus,
        model_id=asr_model_id,
        samples_per_cohort=samples_per_cohort,
        seed=seed,
    )

    # calculate_group_wer re-derives WER from reference/hypothesis pairs, giving
    # the bias_discrepancy_index in the exact form section 6 quotes it.
    cohort_results = [
        {"cohort": r.accent, "reference": r.reference, "hypothesis": r.hypothesis}
        for r in report.sample_results
    ]
    group_wer = calculate_group_wer(cohort_results)

    if group_wer.get("total_samples_evaluated", 0) == 0:
        return {
            "status": "unavailable",
            "reason": (
                f"no scoreable samples for corpus {corpus!r} -- is the L2-ARCTIC "
                f"data present and are transcripts prepared "
                f"(scripts/prepare_l2arctic_transcripts.py)?"
            ),
        }

    # Per-sample rows are kept so an outlier can be inspected without paying for
    # another full transcription pass. The mean WER is not robust to Whisper
    # hallucinating on a single clip (insertions can push one sample's WER well
    # above 1.0), and without these rows that skew is invisible in the summary.
    worst_first = sorted(report.sample_results, key=lambda r: r.wer, reverse=True)

    return {
        "status": "measured",
        "model_id": asr_model_id,
        "corpus": corpus,
        "group_wer": group_wer,
        "sample_results": [
            {
                "sample_id": r.sample_id,
                "accent": r.accent,
                "wer": r.wer,
                "reference": r.reference,
                "hypothesis": r.hypothesis,
            }
            for r in worst_first
        ],
        "ranked_cohorts": [
            {
                "accent": c.accent,
                "sample_count": c.sample_count,
                "scored_count": c.scored_count,
                "mean_wer": c.mean_wer,
                "median_wer": c.median_wer,
                "stdev_wer": c.stdev_wer,
            }
            for c in report.cohorts
        ],
    }


# --------------------------------------------------------------------------
# FR16 -- attribution faithfulness
# --------------------------------------------------------------------------

def _saliency_for(item: Dict[str, Any]) -> Dict[str, Any]:
    """Compute one item's attribution through the normal saliency dispatcher.

    Uses `generate_saliency`, the same entry point the request path uses, so a
    custom checkpoint is explained by itself rather than by a default.
    """
    from app.domain.saliency_service import generate_saliency

    return generate_saliency(
        item["file_path"],
        model=item["model"],
        method=item.get("method", "gradcam"),
    )


def _extract_scores(saliency: Dict[str, Any]) -> Optional[List[float]]:
    """Pull a flat attribution vector out of a saliency payload.

    `generate_saliency` returns `series` (a flat attribution aligned to the
    waveform) and `saliency_matrix` (mel bins x frames). `compute_deletion_score`
    masks top-K features over the waveform, so `series` is the one that matches
    its contract; the matrix is flattened only as a fallback for a payload that
    omits the series.

    The other key names are kept because different families have used them, and
    probing is cheaper than assuming. Returns None when nothing usable is
    present, so the caller refuses the item rather than scoring a guess.
    """
    for key in ("series", "saliency_scores", "scores", "attribution", "attributions"):
        val = saliency.get(key)
        if isinstance(val, list) and val and isinstance(val[0], (int, float)):
            return [float(v) for v in val]

    matrix = saliency.get("saliency_matrix")
    if isinstance(matrix, list) and matrix and isinstance(matrix[0], list):
        return [float(v) for row in matrix for v in row]

    return None


def run_faithfulness_stage(
    manifest: List[Dict[str, Any]],
    top_k_percentages: Optional[List[float]] = None,
    model_type: str = "ser",
    saliency_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = _saliency_for,
) -> Dict[str, Any]:
    """Deletion-score faithfulness over the manifest.

    Attribution is computed per item (unless the manifest supplies it), then
    handed to ``evaluate_batch_faithfulness_scores``, which masks the top-K
    salient regions and RE-RUNS inference. Items whose attribution came from a
    fallback are refused by that function, not scored -- auditing a fallback
    would measure the fallback, not the model.
    """
    # Checked before the import: an empty manifest is answerable without pulling
    # in the app package, which transitively imports FastAPI and redis.
    if not manifest:
        return {"status": "unavailable", "reason": "manifest is empty", "skipped": []}

    from app.domain.evaluation_service import evaluate_batch_faithfulness_scores

    eval_items: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    for item in manifest:
        if "saliency_scores" in item:
            eval_items.append(dict(item))
            continue
        try:
            saliency = saliency_fn(item)
        except Exception as exc:  # a broken checkpoint must not abort the batch
            skipped.append({"file_path": item.get("file_path", "?"), "reason": str(exc)})
            continue

        scores = _extract_scores(saliency)
        if scores is None:
            skipped.append({
                "file_path": item.get("file_path", "?"),
                "reason": "saliency payload carried no usable attribution vector",
            })
            continue

        eval_items.append({
            **item,
            "saliency_scores": scores,
            "provenance": saliency.get("provenance"),
        })

    if not eval_items:
        return {
            "status": "unavailable",
            "reason": "no manifest item produced a usable attribution",
            "skipped": skipped,
        }

    result = evaluate_batch_faithfulness_scores(
        eval_items,
        top_k_percentages=top_k_percentages,
        model_type=model_type,
    )
    result["status"] = "measured"
    result["skipped"] = skipped
    return result


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _fmt(val: Any, digits: int = 4) -> str:
    return f"{val:.{digits}f}" if isinstance(val, (int, float)) else "not measured"


def render_markdown(report: Dict[str, Any]) -> str:
    """Render the summary table that section 6's figures can be checked against."""
    lines = [
        "# Model Evaluation Results",
        "",
        f"Generated {report['generated_at']} by `Backend/scripts/evaluate_models.py` (LIT-252).",
        "Regenerate with the command in `command`, below. Figures here are the",
        "reproducible source for `TESTING_AND_EVALUATION.md` section 6.",
        "",
        f"```\n{report['command']}\n```",
        "",
        "## FR15 -- ASR accent bias",
        "",
    ]

    ab = report["fr15_accent_bias"]
    if ab.get("status") != "measured":
        lines += [f"**Not measured.** {ab.get('reason', 'stage skipped')}", ""]
    else:
        gw = ab["group_wer"]
        lines += [
            f"Model `{ab['model_id']}` on corpus `{ab['corpus']}`, "
            f"{gw['total_samples_evaluated']} samples scored.",
            "",
            f"- Overall mean WER: **{_fmt(gw['overall_mean_wer'])}**",
            f"- Bias discrepancy index (max cohort - min cohort): **{_fmt(gw['bias_discrepancy_index'])}**",
            "",
            "| Cohort | Samples | Scored | Mean WER | Median WER |",
            "|--------|---------|--------|----------|------------|",
        ]
        for c in ab["ranked_cohorts"]:
            lines.append(
                f"| {c['accent']} | {c['sample_count']} | {c['scored_count']} | "
                f"{_fmt(c['mean_wer'])} | {_fmt(c['median_wer'])} |"
            )
        lines += ["", "Ranked worst-WER first; that ordering is the bias signal.", ""]

        # A mean far above the median means a few clips dominate the cohort --
        # usually ASR hallucination (insertions push a single sample's WER past
        # 1.0), not a uniformly worse accent. Reporting the mean alone would
        # attribute an outlier to the whole cohort.
        skewed = [
            c for c in ab["ranked_cohorts"]
            if isinstance(c.get("mean_wer"), (int, float))
            and isinstance(c.get("median_wer"), (int, float))
            and c["median_wer"] > 0 and c["mean_wer"] > 3 * c["median_wer"]
        ]
        if skewed:
            lines += [
                "> **Outlier warning.** These cohorts have a mean WER more than",
                "> 3x their median, so a small number of clips dominates the mean:",
                "",
            ]
            lines += [
                f"> - {c['accent']}: mean {_fmt(c['mean_wer'])} vs median {_fmt(c['median_wer'])}"
                for c in skewed
            ]
            lines += [
                "",
                "> Inspect `sample_results` in the JSON (sorted worst-WER first) before",
                "> reading these cohort means as an accent-bias result.",
                ">",
                "> A known cause is ASR language misdetection: the profiler does not force",
                "> the decode language, so heavily accented English can be transcribed into",
                "> the speaker's L1 and then loop, producing WER far above 1.0 on a single",
                "> clip. See defect D13 in `docs/evaluation/DEFECT_LOG.md`.",
                "",
            ]

    lines += ["## FR16 -- attribution faithfulness", ""]
    fa = report["fr16_faithfulness"]
    if fa.get("status") != "measured":
        lines += [f"**Not measured.** {fa.get('reason', 'stage skipped')}", ""]
    else:
        lines += [
            f"- Mean deletion score (confidence drop): **{_fmt(fa.get('mean_deletion_score'))}**",
            f"- Mean deletion AUC: **{_fmt(fa.get('mean_deletion_auc'))}**",
            f"- Audio scored: {fa.get('audio_scored', 0)} / "
            f"{fa.get('total_audio_evaluated', 0)} "
            f"(refused as fallback attribution: {fa.get('audio_refused', 0)})",
            "",
        ]
        if fa.get("skipped"):
            lines.append(f"{len(fa['skipped'])} item(s) skipped before scoring:")
            lines += [f"- `{s['file_path']}` -- {s['reason']}" for s in fa["skipped"]]
            lines.append("")

    lines += [
        "---",
        "",
        "A stage reading *not measured* was not run or produced nothing scoreable.",
        "It is never reported as a zero: an unmeasured quantity and a measured",
        "zero mean opposite things in a faithfulness audit.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _self_test() -> int:
    """Exercise both stages with injected fakes -- no corpus, model or GPU.

    Verifies the wiring this script owns: that each stage is reachable, that a
    stage with nothing to measure reports `unavailable` rather than zero, and
    that the report renders. It does not re-test the underlying services, which
    have their own suites.
    """
    print("self-test: running both stages against fakes\n")
    failures: List[str] = []

    # -- FR15 with a transcriber that returns the reference verbatim (WER 0).
    try:
        ab = run_accent_bias_stage(
            asr_model_id="self-test/fake-asr",
            transcribe=lambda path: "the quick brown fox",
        )
        status = ab.get("status")
        if status == "measured":
            print(f"  FR15: measured, "
                  f"{ab['group_wer']['total_samples_evaluated']} samples")
        elif status == "unavailable":
            print(f"  FR15: unavailable (expected without the corpus)")
            print(f"        reason: {ab['reason'][:70]}...")
        else:
            failures.append(f"FR15 returned unexpected status {status!r}")
    except Exception as exc:
        failures.append(f"FR15 stage raised: {type(exc).__name__}: {exc}")

    # -- FR16 with a fake attribution, so no model is loaded to produce it.
    try:
        fa = run_faithfulness_stage(
            manifest=[{"file_path": "self-test.wav", "model": "self-test/fake"}],
            saliency_fn=lambda item: {"saliency_scores": [0.1, 0.9, 0.2, 0.8]},
        )
        if fa.get("status") in ("measured", "unavailable"):
            print(f"  FR16: {fa['status']}")
        else:
            failures.append(f"FR16 returned unexpected status {fa.get('status')!r}")
    except Exception as exc:
        failures.append(f"FR16 stage raised: {type(exc).__name__}: {exc}")

    # -- An empty manifest must refuse, not report a confident zero.
    try:
        empty = run_faithfulness_stage(manifest=[], saliency_fn=lambda i: {})
        if empty.get("status") != "unavailable":
            failures.append("an empty manifest was not reported as unavailable")
        elif empty.get("mean_deletion_score") == 0.0:
            failures.append("an empty manifest produced a 0.0 score instead of nothing")
        else:
            print("  empty manifest correctly refused rather than scored 0.0")
    except Exception as exc:
        failures.append(f"empty-manifest check raised: {type(exc).__name__}: {exc}")

    # -- The renderer must survive both stages being unavailable.
    try:
        md = render_markdown({
            "generated_at": "self-test",
            "command": "self-test",
            "fr15_accent_bias": {"status": "unavailable", "reason": "self-test"},
            "fr16_faithfulness": {"status": "unavailable", "reason": "self-test"},
        })
        if "not measured" not in md.lower():
            failures.append("renderer did not mark unavailable stages as not measured")
        else:
            print("  renderer marks unavailable stages as 'not measured'")
    except Exception as exc:
        failures.append(f"renderer raised: {type(exc).__name__}: {exc}")

    print()
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        return 1
    print("self-test passed")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the FR15/FR16 evaluation figures quoted in T&E section 6.",
    )
    parser.add_argument("--asr-model-id", default=None,
                        help="HF model id for the FR15 pass, e.g. openai/whisper-base")
    parser.add_argument("--corpus", default="l2-arctic")
    parser.add_argument("--samples-per-cohort", type=int, default=None,
                        help="Cap samples per cohort for a faster pass")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", default=None,
                        help="JSON list of FR16 items (see module docstring)")
    parser.add_argument("--model-type", default="ser", choices=["ser", "add", "deepfake"],
                        help="Model family for the FR16 masking pass")
    parser.add_argument("--top-k", type=float, nargs="*", default=None,
                        help="Deletion curve points, e.g. --top-k 0.1 0.3 0.5")
    parser.add_argument("--output", default=None,
                        help="Directory for evaluation_results.json / .md (default: stdout)")
    parser.add_argument("--skip-accent-bias", action="store_true")
    parser.add_argument("--skip-faithfulness", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="Verify the wiring with fakes; no corpus, model or GPU needed")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    if args.skip_accent_bias and args.skip_faithfulness:
        parser.error("both stages skipped -- nothing to do")
    if not args.skip_accent_bias and not args.asr_model_id:
        parser.error("--asr-model-id is required unless --skip-accent-bias")
    if not args.skip_faithfulness and not args.manifest:
        parser.error("--manifest is required unless --skip-faithfulness")

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python " + " ".join(["scripts/evaluate_models.py"] + (argv or sys.argv[1:])),
        "fr15_accent_bias": {"status": "skipped", "reason": "--skip-accent-bias"},
        "fr16_faithfulness": {"status": "skipped", "reason": "--skip-faithfulness"},
    }

    if not args.skip_accent_bias:
        print(f"FR15: profiling {args.asr_model_id} on {args.corpus} ...", file=sys.stderr)
        report["fr15_accent_bias"] = run_accent_bias_stage(
            asr_model_id=args.asr_model_id,
            corpus=args.corpus,
            samples_per_cohort=args.samples_per_cohort,
            seed=args.seed,
        )

    if not args.skip_faithfulness:
        manifest = json.loads(Path(args.manifest).read_text())
        print(f"FR16: auditing {len(manifest)} item(s) ...", file=sys.stderr)
        report["fr16_faithfulness"] = run_faithfulness_stage(
            manifest=manifest,
            top_k_percentages=args.top_k,
            model_type=args.model_type,
        )

    # Combined view in the schema the evaluation store already expects.
    if (report["fr15_accent_bias"].get("status") == "measured"
            or report["fr16_faithfulness"].get("status") == "measured"):
        from app.domain.evaluation_service import compute_multi_task_performance_summary
        report["combined_summary"] = compute_multi_task_performance_summary(
            report["fr15_accent_bias"].get("group_wer", {}),
            report["fr16_faithfulness"],
        )

    markdown = render_markdown(report)

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "evaluation_results.json").write_text(json.dumps(report, indent=2))
        (out / "evaluation_results.md").write_text(markdown)
        print(f"\nWrote {out / 'evaluation_results.json'}", file=sys.stderr)
        print(f"Wrote {out / 'evaluation_results.md'}", file=sys.stderr)
    else:
        print(markdown)

    # Non-zero when a stage that was asked for could not be measured, so CI or a
    # rerun notices instead of silently accepting a half-empty report.
    requested = [
        report["fr15_accent_bias"] if not args.skip_accent_bias else None,
        report["fr16_faithfulness"] if not args.skip_faithfulness else None,
    ]
    if any(s is not None and s.get("status") != "measured" for s in requested):
        print("\nAt least one requested stage was not measured.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
