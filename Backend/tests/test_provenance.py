import pytest
from unittest.mock import MagicMock
import numpy as np
import torch

from app.domain.provenance import Provenance, provenance_fields
from app.domain.model_loader_service import transcribe_whisper


def test_provenance_fields_measured():
    res = provenance_fields(Provenance.MEASURED)
    assert res["provenance"] == "measured"
    assert res["provenance_reason"] is None


def test_provenance_fields_fallback_valid():
    reason_str = "Synthesised stand-in fallback used"
    res = provenance_fields(Provenance.FALLBACK, reason_str)
    assert res["provenance"] == "fallback"
    assert res["provenance_reason"] == reason_str


def test_provenance_fields_fallback_missing_reason_raises():
    with pytest.raises(ValueError, match="requires a non-empty reason string"):
        provenance_fields(Provenance.FALLBACK)

    with pytest.raises(ValueError, match="requires a non-empty reason string"):
        provenance_fields(Provenance.FALLBACK, "   ")


def test_provenance_fields_unavailable():
    res = provenance_fields(Provenance.UNAVAILABLE, "Model failed to compute")
    assert res["provenance"] == "unavailable"
    assert res["provenance_reason"] == "Model failed to compute"


def test_transcribe_whisper_provenance_measured(monkeypatch):
    """Test transcribe_whisper returning Provenance.MEASURED when real attention is extracted."""
    dummy_audio = np.zeros(16000, dtype=np.float32)
    monkeypatch.setattr("librosa.load", lambda file, sr=16000: (dummy_audio, 16000))

    mock_processor = MagicMock()
    mock_processor.return_value.input_features.to.return_value = torch.zeros((1, 80, 3000))
    mock_processor.get_decoder_prompt_ids.return_value = None
    mock_processor.batch_decode.return_value = ["dummy transcript"]

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = torch.zeros((1, 5), dtype=torch.long)

    dummy_att_layer = torch.ones((1, 8, 10, 10))
    mock_outputs = MagicMock()
    mock_outputs.keys.return_value = ["decoder_attentions"]
    mock_outputs.decoder_attentions = [dummy_att_layer]
    mock_model.return_value = mock_outputs

    monkeypatch.setattr(
        "app.domain.model_loader_service._get_whisper_cond_gen",
        lambda model_id: (mock_processor, mock_model),
    )

    res = transcribe_whisper("base", "dummy.wav", return_attention=True)

    assert res["text"] == "dummy transcript"
    assert res["provenance"] == "measured"
    assert res["provenance_reason"] is None
    assert res["attention_is_fallback"] is False
    assert res["attention"] is not None
    assert len(res["attention"]) > 0


def test_transcribe_whisper_provenance_fallback(monkeypatch):
    """Test transcribe_whisper returning Provenance.FALLBACK when attention extraction fails and structured pattern is used."""
    dummy_audio = np.zeros(16000, dtype=np.float32)
    monkeypatch.setattr("librosa.load", lambda file, sr=16000: (dummy_audio, 16000))

    mock_processor = MagicMock()
    mock_processor.return_value.input_features.to.return_value = torch.zeros((1, 80, 3000))
    mock_processor.get_decoder_prompt_ids.return_value = None
    mock_processor.batch_decode.return_value = ["dummy transcript"]

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = torch.zeros((1, 5), dtype=torch.long)
    mock_model.side_effect = Exception("Attention extraction failed")

    mock_cfg = MagicMock()
    mock_cfg.decoder_layers = 2
    mock_cfg.decoder_attention_heads = 2
    mock_model.config = mock_cfg

    monkeypatch.setattr(
        "app.domain.model_loader_service._get_whisper_cond_gen",
        lambda model_id: (mock_processor, mock_model),
    )

    res = transcribe_whisper("base", "dummy.wav", return_attention=True)

    assert res["provenance"] == "fallback"
    assert res["provenance_reason"] is not None
    assert "Fabricated structured attention" in res["provenance_reason"]
    assert res["attention_is_fallback"] is True
    assert res["attention"] is not None


def test_transcribe_whisper_provenance_unavailable(monkeypatch):
    """Test transcribe_whisper returning Provenance.UNAVAILABLE when all attention extraction and pattern generation fail."""
    dummy_audio = np.zeros(16000, dtype=np.float32)
    monkeypatch.setattr("librosa.load", lambda file, sr=16000: (dummy_audio, 16000))

    mock_processor = MagicMock()
    mock_processor.return_value.input_features.to.return_value = torch.zeros((1, 80, 3000))
    mock_processor.get_decoder_prompt_ids.return_value = None
    mock_processor.batch_decode.return_value = ["dummy transcript"]

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = torch.zeros((1, 5), dtype=torch.long)
    mock_model.side_effect = Exception("Attention extraction failed")

    type(mock_model).config = property(fget=MagicMock(side_effect=Exception("No config")))

    monkeypatch.setattr(
        "app.domain.model_loader_service._get_whisper_cond_gen",
        lambda model_id: (mock_processor, mock_model),
    )

    res = transcribe_whisper("base", "dummy.wav", return_attention=True)

    assert res["provenance"] == "unavailable"
    assert res["provenance_reason"] == "All attention extraction methods failed"
    assert res["attention_is_fallback"] is False
    assert res["attention"] is None


def test_saliency_energy_fallback_provenance(monkeypatch, tmp_path):
    """Test saliency service returning Provenance.FALLBACK when attributions are flat or empty."""
    from app.domain.saliency_service import generate_whisper_saliency

    dummy_audio = np.zeros(16000, dtype=np.float32)
    audio_file = tmp_path / "test.wav"
    import soundfile as sf
    sf.write(str(audio_file), dummy_audio, 16000)

    monkeypatch.setattr("librosa.load", lambda file, sr=16000: (dummy_audio, 16000))
    monkeypatch.setattr("app.domain.model_loader_service.resolve_whisper_model_id", lambda name: "openai/whisper-tiny")

    # Mock whisper model whose encoder returns dummy hidden states
    mock_model = MagicMock()
    mock_model.encoder.return_value.last_hidden_state = torch.ones((1, 50, 64))
    monkeypatch.setattr("app.domain.model_loader_service.get_whisper_base_models", lambda name: (mock_model, MagicMock()))

    # Force IG to return zero attributions
    monkeypatch.setattr("captum.attr.IntegratedGradients.attribute", lambda self, inputs, **kwargs: torch.zeros_like(inputs))

    res = generate_whisper_saliency(str(audio_file), model_size="tiny", method="integrated_gradients")

    assert res["provenance"] == "fallback"
    assert res["provenance_reason"] is not None
    assert "showing encoder energy" in res["provenance_reason"]


def test_evaluation_faithfulness_refuses_fallback_attribution():
    """Test evaluate_batch_faithfulness_scores refusing attributions with non-MEASURED provenance."""
    from app.domain.evaluation_service import evaluate_batch_faithfulness_scores

    eval_items = [
        {
            "file_path": "fallback_sample.wav",
            "saliency_scores": [0.1, 0.2, 0.3],
            "original_confidence": 0.9,
            "provenance": "fallback",
            "provenance_reason": "attribution was empty or constant",
        },
        {
            "file_path": "measured_sample.wav",
            "saliency_scores": [0.8, 0.9, 0.7],
            "original_confidence": 0.9,
            "provenance": "measured",
        }
    ]

    res = evaluate_batch_faithfulness_scores(eval_items)

    assert len(res["item_results"]) == 2
    assert "error" in res["item_results"][0]
    assert "cannot audit a fallback attribution" in res["item_results"][0]["error"]
    # The measured item is refused too, but for a different and honest reason:
    # its file does not exist, so nothing could be masked or re-run. FR16.1
    # means a score only exists where a measurement happened.
    assert "error" in res["item_results"][1]
    assert "cannot audit a fallback" not in res["item_results"][1]["error"]
    assert res["audio_scored"] == 0

