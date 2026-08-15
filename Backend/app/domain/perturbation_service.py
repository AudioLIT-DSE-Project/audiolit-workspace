import io
import torch
import soundfile as sf
import os
import uuid
import librosa
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Sequence
from ..infrastructure.dataset_service import resolve_file

# Setup logger
logger = logging.getLogger("audiolit.perturbation")
logger.setLevel(logging.INFO)

def _load_waveform(path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio via soundfile, returning the (channels, time) float32 tensor.
    Enforces 16kHz mono constraint for downstream inference endpoints (SRS FR12.3).
    """
    try:
        # always_2d=True ensures (samples, channels) shape even for mono
        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        
        # Transpose to (channels, samples) to match PyTorch's expected orientation
        waveform = torch.from_numpy(data.T.copy())
        
        # Enforce 16kHz mono constraint
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True) # Downmix to mono
            
        if sample_rate != 16000:
            audio_np = waveform[0].numpy()
            audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)
            waveform = torch.from_numpy(audio_np).unsqueeze(0)
            sample_rate = 16000
            
        return waveform, sample_rate
    except Exception as e:
        logger.error(f"Failed to load audio file {path}: {e}")
        raise

def _save_waveform(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save a (channels, time) float32 tensor via soundfile."""
    try:
        # Transpose back to (samples, channels) for soundfile
        data = waveform.detach().cpu().numpy().T
        sf.write(path, data, sample_rate)
    except Exception as e:
        logger.error(f"Failed to save audio file {path}: {e}")
        raise

def export_to_wav_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
    """Exports tensor to in-memory WAV bytes for Web Audio preview (SRS FR12.2)."""
    buf = io.BytesIO()
    data = waveform.detach().cpu().numpy().T
    sf.write(buf, data, sample_rate, format='WAV', subtype='PCM_16')
    buf.seek(0)
    return buf.read()

def add_gaussian_noise(waveform: torch.Tensor, noise_level: float = 0.005) -> torch.Tensor:
    """Add Gaussian noise to the waveform."""
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

def apply_time_masking(waveform: torch.Tensor, mask_start_percent: float, mask_end_percent: float) -> torch.Tensor:
    """Apply time masking to a portion of the waveform."""
    channels, length = waveform.shape
    start_idx = int(length * mask_start_percent / 100)
    end_idx = int(length * mask_end_percent / 100)
    masked_waveform = waveform.clone()
    masked_waveform[:, start_idx:end_idx] = 0
    return masked_waveform

def apply_frequency_masking(waveform: torch.Tensor, sample_rate: int, mask_freq_start: float, mask_freq_end: float) -> torch.Tensor:
    """Apply frequency masking to the waveform."""
    fft = torch.fft.fft(waveform, dim=-1)
    freqs = torch.fft.fftfreq(waveform.shape[-1], 1/sample_rate)
    freq_mask = (freqs >= mask_freq_start) & (freqs <= mask_freq_end)
    fft[:, freq_mask] = 0
    return torch.fft.ifft(fft, dim=-1).real

def apply_2d_time_freq_mask(waveform: torch.Tensor, sample_rate: int, params: Dict[str, Any]) -> torch.Tensor:
    """NumPy-Driven 2D Spectrogram Slice Masking (mute regions)."""
    t_start_ms = params.get("t_start_ms", 0)
    t_end_ms = params.get("t_end_ms", (waveform.shape[-1] / sample_rate) * 1000)
    f_low_hz = params.get("f_low_hz", 0)
    f_high_hz = params.get("f_high_hz", sample_rate / 2)

    if waveform.dim() > 1:
        audio_np = waveform[0].numpy()
    else:
        audio_np = waveform.numpy()
    
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(audio_np, n_fft=n_fft, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sample_rate, hop_length=hop_length)

    freq_mask = (freqs >= f_low_hz) & (freqs <= f_high_hz)
    time_mask = (times * 1000 >= t_start_ms) & (times * 1000 <= t_end_ms)

    # Outer product creates the 2D mute region
    mask_2d = np.outer(freq_mask, time_mask)
    stft[mask_2d] = 0.0

    y_masked = librosa.istft(stft, hop_length=hop_length)

    if len(y_masked) < len(audio_np):
        y_masked = np.pad(y_masked, (0, len(audio_np) - len(y_masked)))
    else:
        y_masked = y_masked[:len(audio_np)]

    return torch.from_numpy(y_masked).unsqueeze(0)

def apply_band_pass_filter(waveform: torch.Tensor, sample_rate: int, params: Dict[str, Any]) -> torch.Tensor:
    """Signal modification routine for band-pass filtering."""
    from scipy.signal import butter, sosfilt
    
    f_low_hz = params.get("f_low_hz", 500)
    f_high_hz = params.get("f_high_hz", 2000)
    audio_np = waveform[0].numpy()
    
    nyq = 0.5 * sample_rate
    low = f_low_hz / nyq
    high = f_high_hz / nyq
    sos = butter(5, [low, high], analog=False, btype='band', output='sos')
    filtered_audio = sosfilt(sos, audio_np)
    
    return torch.from_numpy(filtered_audio).unsqueeze(0)

def apply_pitch_shift(waveform: torch.Tensor, sample_rate: int, pitch_shift_semitones: float) -> torch.Tensor:
    """Apply pitch shifting to the waveform using Librosa."""
    pitch_shift_semitones = max(-6, min(6, pitch_shift_semitones))
    if abs(pitch_shift_semitones) < 0.1:
        return waveform
    
    max_length = sample_rate * 30
    if waveform.shape[-1] > max_length:
        waveform = waveform[..., :max_length]
    
    try:
        audio_np = waveform[0].numpy() if waveform.dim() > 1 else waveform.numpy()
        shifted_audio = librosa.effects.pitch_shift(y=audio_np, sr=sample_rate, n_steps=pitch_shift_semitones)
        return torch.from_numpy(shifted_audio).unsqueeze(0)
    except Exception as e:
        logger.error(f"Pitch shift failed: {e}. Returning original waveform.")
        return waveform

def apply_time_stretch(waveform: torch.Tensor, stretch_factor: float) -> torch.Tensor:
    """Apply time stretching to the waveform using Librosa."""
    if abs(stretch_factor - 1.0) < 0.01:
        return waveform
    
    try:
        audio_np = waveform[0].numpy() if waveform.dim() > 1 else waveform.numpy()
        stretched_audio = librosa.effects.time_stretch(y=audio_np, rate=stretch_factor)
        return torch.from_numpy(stretched_audio).unsqueeze(0)
    except Exception as e:
        logger.error(f"Time stretch failed: {e}. Returning original waveform.")
        return waveform

def apply_perturbations(waveform: torch.Tensor, sample_rate: int, perturbations: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Apply multiple perturbations to a waveform sequentially."""
    perturbed_waveform = waveform.clone()
    applied_perturbations = []
    
    for perturbation in perturbations:
        perturbation_type = perturbation.get("type")
        params = perturbation.get("params", {})
        
        try:
            if perturbation_type == "noise":
                noise_level = params.get("noise_level", 0.005)
                perturbed_waveform = add_gaussian_noise(perturbed_waveform, noise_level)
                applied_perturbations.append({"type": "noise", "params": {"noise_level": noise_level}, "status": "applied"})
                
            elif perturbation_type == "time_masking":
                mask_start = params.get("mask_start_percent", 20)
                mask_end = params.get("mask_end_percent", 40)
                perturbed_waveform = apply_time_masking(perturbed_waveform, mask_start, mask_end)
                applied_perturbations.append({"type": "time_masking", "params": {"mask_start_percent": mask_start, "mask_end_percent": mask_end}, "status": "applied"})
                
            elif perturbation_type == "frequency_masking":
                mask_freq_start = params.get("mask_freq_start", 1000)
                mask_freq_end = params.get("mask_freq_end", 2000)
                perturbed_waveform = apply_frequency_masking(perturbed_waveform, sample_rate, mask_freq_start, mask_freq_end)
                applied_perturbations.append({"type": "frequency_masking", "params": {"mask_freq_start": mask_freq_start, "mask_freq_end": mask_freq_end}, "status": "applied"})
                
            elif perturbation_type == "time_freq_mask":
                perturbed_waveform = apply_2d_time_freq_mask(perturbed_waveform, sample_rate, params)
                applied_perturbations.append({"type": "time_freq_mask", "params": params, "status": "applied"})
                
            elif perturbation_type == "band_pass_filter":
                perturbed_waveform = apply_band_pass_filter(perturbed_waveform, sample_rate, params)
                applied_perturbations.append({"type": "band_pass_filter", "params": params, "status": "applied"})
                
            elif perturbation_type == "pitch_shift":
                pitch_shift_semitones = params.get("pitch_shift_semitones", 2)
                perturbed_waveform = apply_pitch_shift(perturbed_waveform, sample_rate, pitch_shift_semitones)
                applied_perturbations.append({"type": "pitch_shift", "params": {"pitch_shift_semitones": pitch_shift_semitones}, "status": "applied"})
                
            elif perturbation_type == "time_stretch":
                stretch_factor = params.get("stretch_factor", 1.1)
                perturbed_waveform = apply_time_stretch(perturbed_waveform, stretch_factor)
                applied_perturbations.append({"type": "time_stretch", "params": {"stretch_factor": stretch_factor}, "status": "applied"})
                
            else:
                applied_perturbations.append({"type": perturbation_type, "params": params, "status": "unsupported"})
                
        except Exception as e:
            applied_perturbations.append({"type": perturbation_type, "params": params, "status": "failed", "error": str(e)})
    
    return perturbed_waveform, applied_perturbations

def perturb_and_save(file_path: str, perturbations: List[Dict[str, Any]], output_dir: str = "uploads", dataset: str = None, session_id: str = None) -> Dict[str, Any]:
    """Apply perturbations to an audio file and save the derived clip non-destructively."""
    try:
        if dataset and not Path(file_path).is_absolute():
            resolved_path = resolve_file(dataset, file_path, session_id)
        else:
            resolved_path = Path(file_path)
            if not resolved_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
    except FileNotFoundError as e:
        return {
            "original_file": file_path, "perturbed_file": "", "filename": "", "duration_ms": 0,
            "sample_rate": 0, "applied_perturbations": [], "success": False, "error": str(e)
        }
    
    try:
        waveform, sample_rate = _load_waveform(str(resolved_path))
    except Exception as e:
        return {
            "original_file": file_path, "perturbed_file": "", "filename": "", "duration_ms": 0,
            "sample_rate": 0, "applied_perturbations": [], "success": False, "error": f"Failed to load audio file: {str(e)}"
        }
    
    perturbed_waveform, applied_perturbations = apply_perturbations(waveform, sample_rate, perturbations)
    
    input_path = Path(file_path)
    output_filename = f"{input_path.stem}_perturbed_{uuid.uuid4().hex[:8]}.wav"
    output_path = Path(output_dir) / output_filename
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        _save_waveform(str(output_path), perturbed_waveform, sample_rate)
    except Exception as e:
        return {
            "original_file": file_path, "perturbed_file": "", "filename": "", "duration_ms": 0,
            "sample_rate": 0, "applied_perturbations": applied_perturbations, "success": False,
            "error": f"Failed to save perturbed audio: {str(e)}"
        }
    
    preview_bytes = export_to_wav_bytes(perturbed_waveform, sample_rate)
    duration_ms = int(perturbed_waveform.shape[-1] / sample_rate * 1000)
    perturbed_file_path = str(output_path).replace("\\", "/")
    
    return {
        "original_file": file_path,
        "perturbed_file": perturbed_file_path,
        "filename": output_filename,
        "duration_ms": duration_ms,
        "sample_rate": sample_rate,
        "applied_perturbations": applied_perturbations,
        "preview_bytes": preview_bytes, 
        "success": True
    }


def mask_top_k_features(
    waveform: torch.Tensor,
    attributions: Sequence[float] | np.ndarray,
    k_percent: float = 10.0
) -> torch.Tensor:
    """Mask the top-K% most salient timesteps/features based on attribution weights (FR16).

    Args:
        waveform: Input audio tensor of shape (channels, samples) or (samples,).
        attributions: Saliency/attribution score array matching or resampled to the time dimension.
        k_percent: Percentage of top salient timesteps to zero out (0.0 to 100.0).

    Returns:
        Masked waveform tensor with top-K% salient regions zeroed.
    """
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


def compute_deletion_score(
    audio_path: str,
    attributions: Sequence[float] | np.ndarray,
    model_type: str = "ser",
    model_id: str = "default",
    k_percent: float = 10.0,
    output_dir: str = "uploads",
) -> Dict[str, Any]:
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

    try:
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

        confidence_drop = max(0.0, orig_conf - masked_conf)
        deletion_score = round(confidence_drop / orig_conf, 4) if orig_conf > 0 else 0.0

        return {
            "success": True,
            "model_type": model_type,
            "model_id": model_id,
            "target_class": target_class,
            "k_percent": k_percent,
            "initial_confidence": round(orig_conf, 4),
            "masked_confidence": round(masked_conf, 4),
            "confidence_drop": round(confidence_drop, 4),
            "deletion_score": deletion_score,
            "faithfulness_verdict": "faithful" if confidence_drop > 0.05 else "unfaithful",
            "masked_audio_file": str(masked_path).replace("\\", "/"),
        }
    except Exception as exc:
        return {"success": False, "error": f"Faithfulness evaluation failed: {exc}"}

