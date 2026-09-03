import logging
import numpy as np
import torch
from typing import List, Dict, Any, Optional

# Setup logger
logger = logging.getLogger("audiolit.evaluation")
logger.setLevel(logging.INFO)

def mask_top_k_features(features: torch.Tensor, saliency_map: torch.Tensor, top_k_percent: float = 0.2) -> torch.Tensor:
    """
    Masks the top-k highest saliency features by replacing them with zeros (FR16).
    """
    if features.shape != saliency_map.shape:
        raise ValueError("Features and saliency_map must have the same shape.")
        
    flat_saliency = saliency_map.flatten()
    k = max(1, int(len(flat_saliency) * top_k_percent))
    
    # Find threshold for top-k highest saliency values
    threshold = torch.topk(flat_saliency, k).values[-1]
    mask = saliency_map < threshold
    
    return features * mask.float()

def compute_deletion_score(original_confidence: float, degraded_confidence: float) -> float:
    """
    Calculates the deletion score (confidence drop ratio) when salient features are masked (FR16).
    Deletion score = (Original Confidence - Degraded Confidence) / Original Confidence
    """
    if original_confidence <= 0.0:
        return 0.0
    drop = original_confidence - degraded_confidence
    return float(max(0.0, min(1.0, drop / original_confidence)))

def compute_deletion_auc(degradation_curve: Dict[str, float]) -> float:
    """
    Computes the Area Under Curve (AUC) metric for progressive feature deletion trajectory (FR16).
    Uses trapezoidal numerical integration over top-k removal percentages [0.0, 1.0].
    """
    if not degradation_curve:
        return 0.0
        
    x_vals: List[float] = []
    y_vals: List[float] = []
    
    # Always include origin (0% removal = 0 drop)
    if "top_0pct" not in degradation_curve:
        x_vals.append(0.0)
        y_vals.append(0.0)
        
    for pct_key, score in sorted(degradation_curve.items(), key=lambda t: int(t[0].split("_")[1].replace("pct", ""))):
        try:
            pct_num = int(pct_key.split("_")[1].replace("pct", ""))
            x_vals.append(pct_num / 100.0)
            y_vals.append(score)
        except (ValueError, IndexError):
            continue
            
    if len(x_vals) < 2:
        return float(y_vals[0]) if y_vals else 0.0
        
    auc_val = float(np.trapz(y_vals, x_vals))
    return round(auc_val, 4)

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Computes Word Error Rate (WER) using word-level Levenshtein distance.
    WER = (Substitutions + Deletions + Insertions) / Total_Reference_Words
    """
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()
    
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
        
    r_len = len(ref_words)
    h_len = len(hyp_words)
    
    # Initialize DP matrix for edit distance
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

def calculate_group_wer(cohort_results: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Calculates group-wise Word Error Rate (WER) indices for ASR accent-bias profiling (FR15).
    Expects cohort_results list with dicts containing: 'cohort', 'reference', 'hypothesis'.
    Returns group WER metrics, cohort breakdowns, and bias discrepancy index.
    """
    cohort_data: Dict[str, List[float]] = {}
    
    for item in cohort_results:
        cohort = item.get("cohort", "unknown").lower()
        ref = item.get("reference", "")
        hyp = item.get("hypothesis", "")
        
        wer_val = calculate_wer(ref, hyp)
        if cohort not in cohort_data:
            cohort_data[cohort] = []
        cohort_data[cohort].append(wer_val)
        
    cohort_breakdown: Dict[str, float] = {}
    all_wers: List[float] = []
    
    for cohort, wers in cohort_data.items():
        avg_wer = float(np.mean(wers)) if wers else 0.0
        cohort_breakdown[cohort] = round(avg_wer, 4)
        all_wers.extend(wers)
        
    overall_mean_wer = float(np.mean(all_wers)) if all_wers else 0.0
    overall_std_wer = float(np.std(all_wers)) if all_wers else 0.0
    
    # Calculate accent bias discrepancy index (max_wer - min_wer)
    wers_list = list(cohort_breakdown.values())
    bias_discrepancy = round(max(wers_list) - min(wers_list), 4) if wers_list else 0.0
    
    return {
        "overall_mean_wer": round(overall_mean_wer, 4),
        "overall_std_wer": round(overall_std_wer, 4),
        "bias_discrepancy_index": bias_discrepancy,
        "cohort_breakdown": cohort_breakdown,
        "total_samples_evaluated": len(cohort_results)
    }

def evaluate_batch_faithfulness_scores(
    eval_items: List[Dict[str, Any]],
    top_k_percentages: Optional[List[float]] = None,
    model_type: str = "ser",
) -> Dict[str, Any]:
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
    exactly that (A2).

    Real masking and inference live in ``perturbation_service.compute_deletion_score``.

    Each item needs ``file_path``, ``saliency_scores`` and, where the attribution
    came from a fallback, its ``provenance`` - fallbacks are refused rather than
    scored.
    """
    from .perturbation_service import compute_deletion_score as measure_deletion

    if top_k_percentages is None:
        top_k_percentages = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    item_results: List[Dict[str, Any]] = []
    overall_deletion_scores: List[float] = []
    overall_deletion_aucs: List[float] = []
    refused = 0

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
            item_results.append({
                "file_path": file_path,
                "error": "cannot audit an empty attribution",
            })
            continue

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

        if failure is not None or not top_k_scores:
            refused += 1
            item_results.append({
                "file_path": file_path,
                "error": failure or "no deletion measurements succeeded",
            })
            continue

        mean_item_del = float(np.mean(top_k_scores))
        item_auc = compute_deletion_auc(degradation_curve)
        overall_deletion_scores.append(mean_item_del)
        overall_deletion_aucs.append(item_auc)

        item_results.append({
            "file_path": file_path,
            "original_confidence": round(float(orig_conf), 4),
            "mean_deletion_score": round(mean_item_del, 4),
            "deletion_auc": item_auc,
            "degradation_curve": degradation_curve,
            "provenance": "measured",
        })

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


def compute_multi_task_performance_summary(
    group_wer_result: Dict[str, Any],
    faithfulness_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compiles committed data-science performance metrics into structured evaluation store schemas.
    Note (SRS ?????????4.4 & Scope Note): Insertion score and full insertion AUC curve are stretch extensions
    and are excluded here.
    """
    return {
        "evaluation_summary": {
            "status": "completed",
            "asr_accent_bias_wer": {
                "overall_mean_wer": group_wer_result.get("overall_mean_wer", 0.0),
                "bias_discrepancy_index": group_wer_result.get("bias_discrepancy_index", 0.0),
                "cohort_breakdown": group_wer_result.get("cohort_breakdown", {})
            },
            # Defaults are None, not 0.0: an unmeasurable audit must not be
            # summarised as a perfectly unfaithful one.
            "faithfulness_deletion_audit": {
                "mean_deletion_score": faithfulness_result.get("mean_deletion_score"),
                "mean_deletion_auc": faithfulness_result.get("mean_deletion_auc"),
                "audio_scored": faithfulness_result.get("audio_scored", 0),
                "audio_refused": faithfulness_result.get("audio_refused", 0),
                "total_audio_evaluated": faithfulness_result.get("total_audio_evaluated", 0)
            }
        }
    }
