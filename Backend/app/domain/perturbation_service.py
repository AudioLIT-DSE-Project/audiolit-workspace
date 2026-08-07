import torch
import soundfile as sf
import os
import uuid
import librosa
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from ..infrastructure.dataset_service import resolve_file

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

def apply_frequency_masking(waveform, sample_rate, mask_freq_start, mask_freq_end):
    """
    Apply frequency masking to the waveform
    waveform: Tensor [channels, time]
    sample_rate: Sample rate of the audio
    mask_freq_start: Start frequency in Hz
    mask_freq_end: End frequency in Hz
    """
    # Convert to frequency domain
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
        print("DEBUG: Skipping pitch shift - too small")
        return waveform
    
    max_length = sample_rate * 30
    if waveform.shape[-1] > max_length:
        print(f"DEBUG: Truncating audio to {max_length} samples to prevent long processing")
        waveform = waveform[..., :max_length]
    
    try:
        # Use librosa directly as it's more reliable and faster
        # Convert to numpy for librosa
        if waveform.dim() > 1:
            # Take first channel if stereo
            audio_np = waveform[0].numpy()
        else:
            audio_np = waveform.numpy()
        
        print(f"DEBUG: Using librosa.effects.pitch_shift with n_steps={pitch_shift_semitones}")
        print(f"DEBUG: Audio length: {len(audio_np)} samples")
        
        # Apply pitch shift using librosa with timeout protection
        import signal
        import time
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Pitch shift operation timed out")
        
        # Set a timeout of 30 seconds
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            shifted_audio = librosa.effects.pitch_shift(
                y=audio_np, 
                sr=sample_rate, 
                n_steps=pitch_shift_semitones
            )
            signal.alarm(0)  # Cancel the alarm
            
            # Convert back to torch tensor
            result = torch.from_numpy(shifted_audio).unsqueeze(0)  # Add channel dimension
            print(f"DEBUG: Librosa pitch shift completed successfully, output shape: {result.shape}")
            return result
            
        except TimeoutError:
            signal.alarm(0)  # Cancel the alarm
            print("DEBUG: Pitch shift timed out, returning original waveform")
            return waveform
            
    except Exception as e:
        print(f"DEBUG: Pitch shift failed with error: {e}")
        print("DEBUG: Returning original waveform")
        return waveform

def apply_time_stretch(waveform, stretch_factor):
    """
    Apply time stretching to the waveform
    waveform: Tensor [channels, time]
    stretch_factor: Factor to stretch time (1.0 = no change, >1.0 = slower, <1.0 = faster)
    """
    print(f"DEBUG: apply_time_stretch called with stretch_factor={stretch_factor}")
    print(f"DEBUG: Input waveform shape: {waveform.shape}")
    
    # Skip if no stretch needed
    if abs(stretch_factor - 1.0) < 0.01:
        print("DEBUG: Skipping time stretch - factor too close to 1.0")
        return waveform
    
    try:
        # Use librosa for time stretching as it's more reliable
        # Convert to numpy for librosa
        if waveform.dim() > 1:
            # Take first channel if stereo
            audio_np = waveform[0].numpy()
        else:
            audio_np = waveform.numpy()
        
        print(f"DEBUG: Using librosa.effects.time_stretch with rate={stretch_factor}")
        # Apply time stretch using librosa
        stretched_audio = librosa.effects.time_stretch(
            y=audio_np, 
            rate=stretch_factor
        )
        
        # Convert back to torch tensor
        result = torch.from_numpy(stretched_audio).unsqueeze(0)  # Add channel dimension
        print(f"DEBUG: Time stretch completed successfully, output shape: {result.shape}")
        return result
        
    except Exception as e:
        print(f"DEBUG: Time stretch failed with error: {e}")
        print("DEBUG: Returning original waveform")
        return waveform

def apply_perturbations(waveform, sample_rate, perturbations: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Apply multiple perturbations to a waveform
    waveform: Tensor [channels, time]
    sample_rate: Sample rate of the audio
    perturbations: List of perturbation dictionaries
    Returns: (perturbed_waveform, applied_perturbations)
    """
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
                print(f"DEBUG: Processing pitch_shift perturbation with {pitch_shift_semitones} semitones")
                perturbed_waveform = apply_pitch_shift(perturbed_waveform, sample_rate, pitch_shift_semitones)
                applied_perturbations.append({
                    "type": "pitch_shift",
                    "params": {"pitch_shift_semitones": pitch_shift_semitones},
                    "status": "applied"
                })
                print(f"DEBUG: Pitch shift perturbation completed")
                
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
    """
    Apply perturbations to an audio file and save the result
    file_path: Path to the input audio file (can be dataset path or absolute path)
    perturbations: List of perturbation dictionaries
    output_dir: Directory to save the perturbed audio
    dataset: Dataset name if file_path is a dataset file
    session_id: Session ID for custom dataset resolution
    Returns: Dictionary with file info and metadata
    """
    # Resolve the file path - handle both dataset files and uploaded files
    try:
        if dataset and not Path(file_path).is_absolute():
            # This is a dataset file, resolve it using the dataset service
            resolved_path = resolve_file(dataset, file_path, session_id)
        else:
            # This is an uploaded file or absolute path
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
    
    # Generate output filename
    input_path = Path(file_path)
    output_filename = f"{input_path.stem}_perturbed_{uuid.uuid4().hex[:8]}{input_path.suffix}"
    output_path = Path(output_dir) / output_filename
    
    # Ensure output directory exists
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
        "success": True
    }