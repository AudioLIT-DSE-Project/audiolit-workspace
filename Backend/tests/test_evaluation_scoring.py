import pytest
import numpy as np
from app.domain.evaluation_service import (
    calculate_wer,
    calculate_group_wer,
    evaluate_batch_faithfulness_scores,
    compute_multi_task_performance_summary,
)

def test_calculate_wer_exact_match():
    ref = "the quick brown fox"
    hyp = "the quick brown fox"
    assert calculate_wer(ref, hyp) == 0.0

def test_calculate_wer_single_substitution():
    ref = "the quick brown fox"
    hyp = "the fast brown fox"
    # 1 substitution out of 4 words -> 0.25
    assert calculate_wer(ref, hyp) == 0.25

def test_calculate_wer_empty_ref():
    assert calculate_wer("", "") == 0.0
    assert calculate_wer("", "word") == 1.0

def test_calculate_group_wer_cohorts():
    cohort_samples = [
        {"cohort": "arabic", "reference": "speech recognition test", "hypothesis": "speech recognition test"},
        {"cohort": "arabic", "reference": "speech recognition test", "hypothesis": "speech reorganisation test"},
        {"cohort": "spanish", "reference": "audio deepfake detection", "hypothesis": "audio deepfake detection"},
        {"cohort": "spanish", "reference": "audio deepfake detection", "hypothesis": "audio fake detection"},
    ]
    
    result = calculate_group_wer(cohort_samples)
    
    assert "overall_mean_wer" in result
    assert "bias_discrepancy_index" in result
    assert "cohort_breakdown" in result
    assert "arabic" in result["cohort_breakdown"]
    assert "spanish" in result["cohort_breakdown"]
    assert result["total_samples_evaluated"] == 4

def test_evaluate_batch_faithfulness_scores():
    eval_samples = [
        {
            "file_path": "audio1.wav",
            "saliency_scores": [0.9, 0.8, 0.4, 0.2, 0.1],
            "original_confidence": 0.90
        },
        {
            "file_path": "audio2.wav",
            "saliency_scores": [0.7, 0.6, 0.5, 0.3, 0.2],
            "original_confidence": 0.80
        }
    ]
    
    result = evaluate_batch_faithfulness_scores(eval_samples)
    
    assert "mean_deletion_score" in result
    assert result["total_audio_evaluated"] == 2
    assert len(result["item_results"]) == 2
    assert "degradation_curve" in result["item_results"][0]

def test_compute_multi_task_performance_summary():
    group_wer = {
        "overall_mean_wer": 0.12,
        "bias_discrepancy_index": 0.05,
        "cohort_breakdown": {"arabic": 0.14, "spanish": 0.09}
    }
    faithfulness = {
        "mean_deletion_score": 0.35,
        "total_audio_evaluated": 2
    }
    
    summary = compute_multi_task_performance_summary(group_wer, faithfulness)
    
    assert "evaluation_summary" in summary
    assert summary["evaluation_summary"]["status"] == "completed"
    assert summary["evaluation_summary"]["asr_accent_bias_wer"]["overall_mean_wer"] == 0.12
    assert summary["evaluation_summary"]["faithfulness_deletion_audit"]["mean_deletion_score"] == 0.35
