import pytest
import numpy as np
from app.domain.evaluation_service import (
    compute_deletion_auc,
    evaluate_batch_faithfulness_scores,
    compute_multi_task_performance_summary,
)

def test_compute_deletion_auc_flat():
    # 0% drop across curve -> AUC = 0.0
    curve = {
        "top_0pct": 0.0,
        "top_20pct": 0.0,
        "top_50pct": 0.0,
        "top_100pct": 0.0
    }
    assert compute_deletion_auc(curve) == 0.0

def test_compute_deletion_auc_linear():
    # Linear drop: y = x, AUC from 0 to 1 = 0.5
    curve = {
        "top_0pct": 0.0,
        "top_10pct": 0.10,
        "top_50pct": 0.50,
        "top_100pct": 1.00
    }
    auc = compute_deletion_auc(curve)
    assert 0.49 <= auc <= 0.51

def test_evaluate_batch_faithfulness_scores_with_auc():
    eval_items = [
        {
            "file_path": "sample1.wav",
            "saliency_scores": [0.9, 0.7, 0.5, 0.3, 0.1],
            "original_confidence": 0.95
        },
        {
            "file_path": "sample2.wav",
            "saliency_scores": [0.8, 0.6, 0.4, 0.2, 0.1],
            "original_confidence": 0.85
        }
    ]
    
    result = evaluate_batch_faithfulness_scores(eval_items)
    
    assert "mean_deletion_score" in result
    assert "mean_deletion_auc" in result
    assert result["mean_deletion_auc"] >= 0.0
    assert result["total_audio_evaluated"] == 2
    assert "deletion_auc" in result["item_results"][0]

def test_compute_multi_task_performance_summary_with_auc():
    group_wer = {
        "overall_mean_wer": 0.10,
        "bias_discrepancy_index": 0.04,
        "cohort_breakdown": {"arabic": 0.12, "spanish": 0.08}
    }
    faithfulness = {
        "mean_deletion_score": 0.42,
        "mean_deletion_auc": 0.38,
        "total_audio_evaluated": 2
    }
    
    summary = compute_multi_task_performance_summary(group_wer, faithfulness)
    
    assert "evaluation_summary" in summary
    audit = summary["evaluation_summary"]["faithfulness_deletion_audit"]
    assert audit["mean_deletion_score"] == 0.42
    assert audit["mean_deletion_auc"] == 0.38
