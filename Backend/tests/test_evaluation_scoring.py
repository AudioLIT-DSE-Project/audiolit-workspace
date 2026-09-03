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

def test_evaluate_batch_faithfulness_scores(sample_audio_file, monkeypatch):
    """FR16.1: the score comes from masking + re-running inference.

    This test used to pass file paths that did not exist and still assert a
    deletion score for both of them. That only worked because the old
    implementation never touched a model - it derived the "score" from the
    saliency values with a closed-form formula. Now a real measurement is
    required, so the inference is stubbed rather than the maths faked.
    """
    import app.domain.perturbation_service as ps

    calls = []

    def fake_ser(path):
        calls.append(path)
        # The masked clip is written to a new file, so a lower confidence on
        # any path other than the original models a genuine confidence drop.
        is_masked = "faithfulness_masked_" in str(path)
        return {"predicted_emotion": "happy",
                "probabilities": {"happy": 0.4 if is_masked else 0.9},
                "confidence": 0.4 if is_masked else 0.9}

    monkeypatch.setattr("app.domain.model_loader_service.predict_ser", fake_ser)

    result = evaluate_batch_faithfulness_scores(
        [{"file_path": str(sample_audio_file),
          "saliency_scores": [0.9, 0.8, 0.4, 0.2, 0.1],
          "provenance": "measured"}],
        top_k_percentages=[0.2],
    )

    assert calls, "no inference ran - the score would be fabricated"
    assert result["audio_scored"] == 1
    assert result["audio_refused"] == 0
    assert result["mean_deletion_score"] is not None
    assert result["item_results"][0]["degradation_curve"]


def test_faithfulness_refuses_a_fallback_attribution(sample_audio_file):
    """A deletion score over a fallback map measures loudness, not faithfulness."""
    result = evaluate_batch_faithfulness_scores([{
        "file_path": str(sample_audio_file),
        "saliency_scores": [0.9, 0.5],
        "provenance": "fallback",
        "provenance_reason": "attribution empty; showing encoder energy",
    }])
    assert result["audio_scored"] == 0
    assert result["audio_refused"] == 1
    assert "cannot audit a fallback attribution" in result["item_results"][0]["error"]


def test_unmeasurable_batch_reports_null_not_zero():
    """0.0 reads as 'completely unfaithful'. Nothing measured is not that."""
    result = evaluate_batch_faithfulness_scores(
        [{"file_path": "does-not-exist.wav", "saliency_scores": [0.5]}]
    )
    assert result["mean_deletion_score"] is None
    assert result["mean_deletion_auc"] is None
    assert result["audio_scored"] == 0


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
