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

def test_evaluate_batch_faithfulness_scores_with_auc(sample_audio_file, monkeypatch):
    """AUC over a curve of REAL measurements (FR16.1).

    This previously passed two filenames that did not exist and still asserted
    an AUC for both, because the old implementation derived the curve from the
    saliency values without running a model. Real audio and a stubbed model now
    stand in for that.
    """
    def fake_ser(path):
        masked = "faithfulness_masked_" in str(path)
        return {"predicted_emotion": "happy",
                "probabilities": {"happy": 0.3 if masked else 0.95},
                "confidence": 0.3 if masked else 0.95}

    monkeypatch.setattr("app.domain.model_loader_service.predict_ser", fake_ser)

    result = evaluate_batch_faithfulness_scores(
        [{"file_path": str(sample_audio_file),
          "saliency_scores": [0.9, 0.7, 0.5, 0.3, 0.1],
          "provenance": "measured"}],
        top_k_percentages=[0.2, 0.5],
    )

    assert result["mean_deletion_auc"] is not None
    assert result["mean_deletion_auc"] >= 0.0
    assert result["audio_scored"] == 1
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
