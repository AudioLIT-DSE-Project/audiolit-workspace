import logging
import os
import torch
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from captum.attr import IntegratedGradients, GradientShap, Lime
from app.domain.model_loader_service import (
    transcribe_whisper_base,
    transcribe_whisper_large,
    transcribe_whisper_with_timestamps,
    predict_emotion_wave2vec,
    get_whisper_base_models,
    get_whisper_large_models,
)
import app.domain.model_loader_service as model_loader_service

logger = logging.getLogger(__name__)
MAX_SALIENCY_SECONDS = int(os.getenv("MAX_SALIENCY_SECONDS", "12"))  # cap analysis window
MAX_SALIENCY_SECONDS_SHAP = int(os.getenv("MAX_SALIENCY_SECONDS_SHAP", "6"))  # stricter for SHAP
SALIENCY_SHAP_SAMPLES = int(os.getenv("SALIENCY_SHAP_SAMPLES", "8"))

def detect_model_type(model: str) -> str:
    if "whisper" in model.lower():
        return "whisper"
    elif "wav2vec" in model.lower():
        return "wav2vec2"
    return "unknown"


#################################################################################################################
def generate_whisper_saliency(audio_file_path: str, model_size: str = "base", method: str = "gradcam", existing_prediction: Dict = None) -> Dict:
    logger.info(f"Generating Whisper saliency for {audio_file_path} using {method} method")
    
    if existing_prediction and "chunks" in existing_prediction:
        data = existing_prediction
        audio = data["audio"]
        chunks = data["chunks"]
        logger.info(f"Using existing prediction with {len(chunks)} chunks")
    else:
        logger.info("Transcribing audio with timestamps for saliency analysis")
        data = transcribe_whisper_with_timestamps(audio_file_path, model_size)
        audio = data["audio"]
        chunks = data["chunks"]
        logger.info(f"Transcription completed with {len(chunks) if chunks else 0} chunks")
    
    # Crop to a safe max duration to avoid OOM
    if isinstance(audio, (list, tuple)):
        audio = np.asarray(audio)
    if hasattr(audio, "shape") and audio is not None:
        max_seconds = MAX_SALIENCY_SECONDS_SHAP if method == "shap" else MAX_SALIENCY_SECONDS
        max_len = int(max_seconds * 16000)
        if len(audio) > max_len:
            audio = audio[:max_len]
            # Keep only chunks inside the window
            chunks = [c for c in chunks if c.get("timestamp", [0, 0])[0] < max_seconds]
    
    if model_size == "base":
        processor, model = get_whisper_base_models()
    else:
        processor, model = get_whisper_large_models()
    
    device = next(model.parameters()).device
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    input_features.requires_grad_(True)
    
    def model_forward(inputs):
        # Reduce to a scalar per batch: energy of encoder activations
        enc = model.encoder(inputs).last_hidden_state  # [B, T, H]
        return enc.pow(2).mean(dim=(1, 2))             # [B]
    
    if method == "gradcam":
        # Optimize memory usage for GPU
        torch.cuda.empty_cache()
        
        # Use gradient checkpointing to save memory
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        
        # Use smaller batch size and fewer steps to fit in GPU memory
        n_steps = 16  # Reduced from 32 to 16
        internal_batch_size = 1
        
        # Monitor GPU memory
        if torch.cuda.is_available():
            logger.info(f"GPU memory before saliency: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        try:
            ig = IntegratedGradients(model_forward)
            attributions = ig.attribute(
                input_features,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                # Clear cache and try again with even lower memory settings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning("First attempt failed, trying with even lower memory settings...")
                
                # Reduce memory usage further
                n_steps = 8
                
                # Try with gradient accumulation
                try:
                    ig = IntegratedGradients(model_forward)
                    attributions = ig.attribute(
                        input_features,
                        n_steps=n_steps,
                        internal_batch_size=internal_batch_size,
                    )
                except RuntimeError as e2:
                    logger.error(f"Saliency computation failed on GPU: {str(e2)}")
                    raise RuntimeError("Failed to compute saliency on GPU after optimization attempts") from e2
            else:
                raise
    elif method == "lime":
        lime = Lime(model_forward)
        attributions = lime.attribute(input_features)
    elif method == "shap":
        # Use Captum GradientShap on the model's current device with small n_samples
        gs = GradientShap(model_forward)
        baseline = torch.zeros_like(input_features)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            attributions = gs.attribute(
                input_features,
                baselines=baseline,
                n_samples=max(2, min(16, SALIENCY_SHAP_SAMPLES)),
                stdevs=0.09,
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning("Whisper SHAP OOM; retrying with fewer samples")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                attributions = gs.attribute(
                    input_features,
                    baselines=baseline,
                    n_samples=max(2, min(8, SALIENCY_SHAP_SAMPLES // 2 if SALIENCY_SHAP_SAMPLES > 2 else 2)),
                    stdevs=0.07,
                )
            else:
                logger.exception("Whisper SHAP failed; falling back to energy map")
                attributions = None
        except Exception:
            logger.exception("Whisper SHAP failed; falling back to energy map")
            attributions = None
    else:
        attributions = torch.zeros_like(input_features)
    
    # Reduce to 1D timeline and normalize to [0,1] for visible intensities
    if attributions is not None:
        saliency_np = attributions.detach().cpu().numpy().squeeze()
        if saliency_np.ndim == 2:
            if saliency_np.shape[0] in (64, 80, 128):
                agg = np.mean(np.abs(saliency_np), axis=0)
            else:
                agg = np.mean(np.abs(saliency_np), axis=1)
        elif saliency_np.ndim == 1:
            agg = np.abs(saliency_np)
        else:
            while saliency_np.ndim > 1:
                saliency_np = saliency_np.mean(axis=0)
            agg = np.abs(saliency_np)
        max_abs = float(np.max(agg)) if agg.size > 0 else 0.0
        saliency_scores = (agg / max_abs) if max_abs > 0 else np.zeros_like(agg)
    else:
        saliency_scores = np.array([])

    # Fallback: if scores are empty or nearly constant, use encoder energy map
    use_energy_fallback = (
        saliency_scores.size == 0 or
        (np.max(saliency_scores) - np.min(saliency_scores) if saliency_scores.size > 0 else 0.0) < 1e-6
    )
    if use_energy_fallback:
        logger.info("Using Whisper energy-map fallback for saliency")
        with torch.no_grad():
            enc = model.encoder(input_features).last_hidden_state  # [B, T, H]
            energy = enc.abs().mean(dim=2).squeeze(0).detach().cpu().numpy()
        if energy.size > 0:
            e_min, e_ptp = float(np.min(energy)), float(np.ptp(energy))
            saliency_scores = (energy - e_min) / (e_ptp + 1e-9)
        else:
            saliency_scores = np.zeros(1, dtype=np.float32)

    # Create dense series with smoothing and percentile clipping
    series = saliency_scores.astype(np.float32)
    if series.size > 0:
        win = max(3, int(series.size / 64))
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win, dtype=np.float32) / float(win)
        series = np.convolve(series, kernel, mode="same")
        p95 = float(np.percentile(series, 95))
        if p95 > 0:
            series = np.clip(series, 0, p95)
        smin, smax = float(np.min(series)), float(np.max(series))
        series = (series - smin) / (smax - smin + 1e-9)
    
    segments = []
    # Map timestamps to attribution timeline robustly
    total_duration = float(len(audio)) / 16000.0 if hasattr(audio, "__len__") and len(audio) > 0 else 0.0
    T = len(saliency_scores)
    fps = (T / total_duration) if total_duration > 0 else 1.0
    
    # Process word-level chunks if available with simplified logic
    if chunks and total_duration > 0:
        logger.info(f"Processing {len(chunks)} word-level chunks for saliency segmentation")
        
        # Debug: Log first few chunks to understand structure
        if len(chunks) > 0:
            logger.info(f"First chunk structure: {chunks[0]}")
            if len(chunks) > 5:
                logger.info(f"Sample of chunks: {chunks[:3]} ... {chunks[-2:]}")
        
        for chunk in chunks:
            start_time = chunk.get("timestamp", [0, 0])[0]
            end_time = chunk.get("timestamp", [0, 0])[1]
            word = chunk.get("text", "")
            
            # Skip invalid chunks
            if end_time <= start_time or start_time < 0 or end_time > total_duration:
                continue
            
            # Convert to attribution frames
            start_frame = max(0, min(T - 1, int(start_time * fps)))
            end_frame = max(start_frame + 1, min(T, int(end_time * fps)))
            
            # Calculate segment saliency
            if end_frame > start_frame:
                segment_saliency = float(np.mean(saliency_scores[start_frame:end_frame]))
                segments.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "word": word.strip(),
                    "saliency": segment_saliency,
                    "intensity": float(abs(segment_saliency))
                })
        
        # Sort by start time to ensure proper order
        segments.sort(key=lambda x: x["start_time"])
        
        logger.info(f"Created {len(segments)} segments from word-level timestamps")

    # Fallback: if no segments were created, create uniform time-based segments
    if len(segments) == 0 and T > 0 and total_duration > 0:
        logger.info("No word-level segments found, creating uniform time-based segments")
        # Create 10-20 segments based on audio duration (aim for ~0.3-1 second segments)
        num_segments = max(8, min(32, int(total_duration * 2)))
        
        for i in range(num_segments):
            start_time = (i / num_segments) * total_duration
            end_time = ((i + 1) / num_segments) * total_duration
            
            start_frame = max(0, min(T - 1, int(start_time * fps)))
            end_frame = max(start_frame + 1, min(T, int(end_time * fps)))
            
            segment_saliency = float(np.mean(saliency_scores[start_frame:end_frame]))
            segments.append({
                "start_time": start_time,
                "end_time": end_time,
                "word": f"segment_{i+1}",
                "saliency": segment_saliency,
                "intensity": float(abs(segment_saliency))
            })
        
        logger.info(f"Created {len(segments)} uniform time-based segments")

    # Final normalization across segments for visibility
    if len(segments) > 0:
        # Collect raw saliency values
        raw_saliencies = [s.get("saliency", 0.0) for s in segments]
        
        # Use absolute values for intensity (magnitude of importance)
        abs_vals = np.abs(raw_saliencies)
        
        # Robust normalization to prevent all-zero intensities
        max_abs = float(np.max(abs_vals)) if len(abs_vals) > 0 else 0.0
        if max_abs > 1e-9:
            # Scale to [0,1] based on maximum absolute value
            for i, segment in enumerate(segments):
                segment["intensity"] = float(abs_vals[i] / max_abs)
        else:
            # Fallback: use relative ranking if all values are very small
            sorted_indices = np.argsort(-abs_vals)  # Sort descending by magnitude
            for rank, idx in enumerate(sorted_indices):
                # Assign intensity based on ranking: highest gets 1.0, lowest gets 0.1
                segments[idx]["intensity"] = float(1.0 - (rank / len(segments)) * 0.9)
        
        # Ensure minimum visibility for all segments
        for segment in segments:
            segment["intensity"] = max(0.1, segment["intensity"])  # Minimum 10% intensity
    
    return {
        "model": f"whisper-{model_size}",
        "method": method,
        "segments": segments,
        "total_duration": total_duration,
        "series": series.tolist()
    }

################################################################################################################

def generate_wav2vec2_saliency(audio_file_path: str, method: str = "gradcam", existing_prediction: Dict = None) -> Dict:
    model_loader_service.ensure_emo_model_loaded()
    audio, rate = librosa.load(audio_file_path, sr=16000)
    # Crop to safe max duration to bound memory
    max_seconds = MAX_SALIENCY_SECONDS_SHAP if method == "shap" else MAX_SALIENCY_SECONDS
    max_len = int(max_seconds * rate)
    if len(audio) > max_len:
        audio = audio[:max_len]
    inputs = model_loader_service.feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
    
    input_values = inputs.input_values.to(model_loader_service.emo_device)
    attention_mask = inputs.attention_mask.to(model_loader_service.emo_device) if "attention_mask" in inputs else None
    
    input_values.requires_grad_(True)
    
    # Determine class to attribute (predicted emotion)
    with torch.no_grad():
        tmp_out = model_loader_service.emo_model(input_values=input_values, attention_mask=attention_mask)
        tmp_probs = torch.nn.functional.softmax(tmp_out.logits, dim=-1)
        target_idx = int(torch.argmax(tmp_probs, dim=-1).item())

    def model_forward(inputs, mask=None, cls_idx: int = 0):
        outputs = model_loader_service.emo_model(input_values=inputs, attention_mask=mask)
        return outputs.logits[:, cls_idx]
    
    if method == "gradcam":
        ig = IntegratedGradients(model_forward)
        try:
            attributions = ig.attribute(
                input_values,
                additional_forward_args=(attention_mask, target_idx),
                n_steps=32,
                internal_batch_size=1,
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning("CUDA OOM during Wav2Vec2 saliency. Falling back to CPU with fewer steps.")
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    cpu_device = torch.device("cpu")
                    # Move model and inputs to CPU
                    if hasattr(model_loader_service.emo_model, 'to'):
                        model_loader_service.emo_model.to(cpu_device)
                    input_values_cpu = input_values.detach().to(cpu_device)
                    input_values_cpu.requires_grad_(True)
                    attention_mask_cpu = attention_mask.detach().to(cpu_device) if attention_mask is not None else None
                    attributions = ig.attribute(
                        input_values_cpu,
                        additional_forward_args=(attention_mask_cpu, target_idx),
                        n_steps=16,
                        internal_batch_size=1,
                    )
                except Exception:
                    raise
            else:
                raise
    elif method == "lime":
        lime = Lime(model_forward)
        attributions = lime.attribute(input_values, additional_forward_args=(attention_mask, target_idx))
    elif method == "shap":
        # Use Captum GradientShap on model's current device with small n_samples
        gs = GradientShap(model_forward)
        baseline = torch.zeros_like(input_values)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            attributions = gs.attribute(
                input_values,
                baselines=baseline,
                additional_forward_args=(attention_mask, target_idx),
                n_samples=max(2, min(16, SALIENCY_SHAP_SAMPLES)),
                stdevs=0.09,
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning("Wav2Vec2 SHAP OOM; retrying with fewer samples")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                attributions = gs.attribute(
                    input_values,
                    baselines=baseline,
                    additional_forward_args=(attention_mask, target_idx),
                    n_samples=max(2, min(8, SALIENCY_SHAP_SAMPLES // 2 if SALIENCY_SHAP_SAMPLES > 2 else 2)),
                    stdevs=0.07,
                )
            else:
                logger.exception("Wav2Vec2 SHAP failed; falling back to energy map")
                attributions = None
        except Exception:
            logger.exception("Wav2Vec2 SHAP failed; falling back to energy map")
            attributions = None
    else:
        attributions = torch.zeros_like(input_values)
    
    # Normalize to [0,1] for visible intensities
    if attributions is not None:
        tmp = attributions.detach().cpu().numpy().squeeze()
        if tmp.ndim > 1:
            tmp = np.mean(np.abs(tmp), axis=0)
        else:
            tmp = np.abs(tmp)
        mx = float(np.max(tmp)) if tmp.size > 0 else 0.0
        saliency_scores = (tmp / mx) if mx > 0 else np.zeros_like(tmp)
    else:
        saliency_scores = np.array([])

    # Fallback: if SHAP produced empty/flat attributions, use encoder energy
    if saliency_scores.size == 0 or (np.max(saliency_scores) - np.min(saliency_scores) if saliency_scores.size > 0 else 0.0) < 1e-6:
        logger.info("Using Wav2Vec2 energy-map fallback for saliency")
        with torch.no_grad():
            hs = model_loader_service.emo_model.wav2vec2(input_values=input_values, attention_mask=attention_mask).last_hidden_state  # [B,T,H]
            energy = hs.abs().mean(dim=2).squeeze(0).detach().cpu().numpy()
        if energy.size > 0:
            e_min, e_ptp = float(np.min(energy)), float(np.ptp(energy))
            saliency_scores = (energy - e_min) / (e_ptp + 1e-9)
        else:
            saliency_scores = np.zeros(1, dtype=np.float32)

    # Create dense series with smoothing and percentile clipping
    series = saliency_scores.astype(np.float32)
    if series.size > 0:
        win = max(3, int(series.size / 64))
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win, dtype=np.float32) / float(win)
        series = np.convolve(series, kernel, mode="same")
        p95 = float(np.percentile(series, 95))
        if p95 > 0:
            series = np.clip(series, 0, p95)
        smin, smax = float(np.min(series)), float(np.max(series))
        series = (series - smin) / (smax - smin + 1e-9)
    
    with torch.no_grad():
        model_device = next(model_loader_service.emo_model.parameters()).device
        iv = input_values.to(model_device)
        am = attention_mask.to(model_device) if attention_mask is not None else None
        outputs = model_loader_service.emo_model(input_values=iv, attention_mask=am)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_emotion = torch.argmax(probs, dim=-1).item()
        id2label = model_loader_service.emo_model.config.id2label
        emotion = id2label.get(predicted_emotion, str(predicted_emotion))
    
    segment_duration = len(audio) / 16000
    num_segments = 32
    segment_length = segment_duration / num_segments if num_segments > 0 else segment_duration
    
    segments = []
    # Map segment times to attribution indices using derived fps from saliency timeline
    T = len(saliency_scores)
    fps = (T / segment_duration) if segment_duration > 0 else 1.0
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = (i + 1) * segment_length
        
        start_frame = max(0, min(T - 1, int(start_time * fps)))
        end_frame = max(start_frame + 1, min(T, int(end_time * fps)))
        segment_saliency = np.mean(saliency_scores[start_frame:end_frame])
        segments.append({
            "start_time": start_time,
            "end_time": end_time,
            "saliency": float(segment_saliency),
            "intensity": float(abs(segment_saliency))
        })

    # Final normalization across segments for visibility
    if len(segments) > 0:
        # Use robust intensity calculation
        raw_saliencies = [s.get("saliency", 0.0) for s in segments]
        abs_vals = np.abs(raw_saliencies)
        
        # Robust normalization to prevent all-zero intensities
        max_abs = float(np.max(abs_vals)) if len(abs_vals) > 0 else 0.0
        if max_abs > 1e-9:
            # Scale to [0,1] based on maximum absolute value
            for i, segment in enumerate(segments):
                segment["intensity"] = float(abs_vals[i] / max_abs)
        else:
            # Fallback: use relative ranking if all values are very small
            sorted_indices = np.argsort(-abs_vals)  # Sort descending by magnitude
            for rank, idx in enumerate(sorted_indices):
                # Assign intensity based on ranking: highest gets 1.0, lowest gets 0.1
                segments[idx]["intensity"] = float(1.0 - (rank / len(segments)) * 0.9)
        
        # Ensure minimum visibility for all segments
        for segment in segments:
            segment["intensity"] = max(0.1, segment["intensity"])  # Minimum 10% intensity
    
    return {
        "model": "wav2vec2",
        "method": method,
        "emotion": emotion,
        "segments": segments,
        "total_duration": segment_duration,
        "series": series.tolist()
    }

def generate_saliency(audio_file_path: str, model: str, method: str = "gradcam", existing_prediction: Dict = None) -> Dict:
    model_type = detect_model_type(model)
    
    if model_type == "whisper":
        model_size = "base" if "base" in model else "large"
        return generate_whisper_saliency(audio_file_path, model_size, method, existing_prediction)
    elif model_type == "wav2vec2":
        return generate_wav2vec2_saliency(audio_file_path, method, existing_prediction)
    else:
        raise ValueError(f"Unsupported model: {model}")


# --- Grad-CAM (LIT-148, FR8) --------------------------------------------------
# Class-discriminative attribution over a convolutional layer's spatial dims:
# hook the target conv layer's activations (forward) and gradients (backward),
# global-average-pool the gradients into per-channel weights, take a ReLU'd
# weighted combination of the feature maps, and normalize to [0, 1]. Works for
# Conv1d (attribution over time) and Conv2d (2D spectrogram attribution).


def find_last_conv_layer(model: "torch.nn.Module") -> "torch.nn.Module":
    """Return the last Conv1d/Conv2d module in ``model`` (Grad-CAM's usual target).

    Grad-CAM hooks the final convolutional layer because it holds the highest-
    level spatial features while still being spatially resolved.
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
            last_conv = module
    if last_conv is None:
        raise ValueError("model has no Conv1d/Conv2d layer to attach Grad-CAM to")
    return last_conv


def compute_grad_cam(
    model: "torch.nn.Module",
    inputs: "torch.Tensor",
    target_layer: "torch.nn.Module" = None,
    target_index: Optional[int] = None,
) -> np.ndarray:
    """Grad-CAM attribution map for one input over ``target_layer``'s feature maps.

    Registers a forward hook (activations) and a full backward hook (gradients)
    on ``target_layer`` (defaults to the model's last conv layer), runs a
    forward + backward pass for ``target_index`` (argmax class if None), and
    returns the ReLU'd, [0, 1]-normalized weighted feature-map combination as a
    numpy score matrix (shape = the conv layer's spatial dims). Hooks are always
    removed, so this can be called repeatedly without leaking them.
    """
    if target_layer is None:
        target_layer = find_last_conv_layer(model)

    captured = {}

    def _forward_hook(_module, _inp, output):
        captured["activations"] = output.detach()

    def _backward_hook(_module, _grad_in, grad_out):
        captured["gradients"] = grad_out[0].detach()

    fwd_handle = target_layer.register_forward_hook(_forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(_backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        if target_index is None:
            target_index = int(torch.argmax(logits, dim=-1)[0])
        logits[0, target_index].backward()

        acts = captured["activations"][0]   # [C, *spatial]
        grads = captured["gradients"][0]     # [C, *spatial]
        spatial_dims = tuple(range(1, grads.dim()))
        weights = grads.mean(dim=spatial_dims)                       # [C]
        weights = weights.view([-1] + [1] * (acts.dim() - 1))        # broadcast over spatial
        cam = torch.relu((weights * acts).sum(dim=0))                # [*spatial]

        cam = cam - cam.min()
        peak = cam.max()
        if peak > 0:
            cam = cam / peak
        return cam.cpu().numpy()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()


# --- Spectrogram-patch LIME/SHAP-style attribution (LIT-130, FR8) -------------
# Occlusion-based explanation over a 2D STFT spectrogram: divide it into a
# time-frequency patch grid, replace each patch with a baseline, and record how
# far the model's score drops. The result is a [n_freq_patches, n_time_patches]
# importance grid aligned to the spectrogram — higher = the patch the score
# depends on most (e.g. which frequency bins drove a deepfake alert). It's
# model-agnostic (takes a score function), so it works for the deepfake
# classifier or any other scalar-scoring model.


def audio_to_spectrogram(audio, n_fft: int = 512, hop_length: int = 256) -> np.ndarray:
    """Magnitude STFT spectrogram ``[freq, time]`` for a 1-D audio array."""
    audio = np.asarray(audio, dtype=np.float32)
    return np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))


def spectrogram_patch_bounds(length: int, n_patches: int) -> List[Tuple[int, int]]:
    """Split range ``[0, length)`` into ``n_patches`` contiguous (start, end) bounds.

    Patch sizes differ by at most 1 so they tile the axis exactly — no gaps or
    overlaps — even when ``length`` isn't divisible by ``n_patches``. Never emits
    more patches than there are samples along the axis.
    """
    if n_patches < 1:
        raise ValueError("n_patches must be >= 1")
    n_patches = min(n_patches, length)
    edges = np.linspace(0, length, n_patches + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_patches)]


def occlusion_attribution(
    score_fn,
    spectrogram,
    n_freq_patches: int = 8,
    n_time_patches: int = 8,
    baseline: Union[str, float] = "mean",
) -> np.ndarray:
    """LIME/SHAP-style patch attribution for a 2-D spectrogram.

    ``score_fn(spectrogram) -> float`` returns the model score to explain (e.g.
    the deepfake/spoof probability). Each time-frequency patch is occluded with a
    ``baseline`` value (``"mean"`` of the spectrogram, or a fixed float) and the
    drop from the un-occluded score is recorded, yielding a
    ``[n_freq_patches, n_time_patches]`` importance grid: positive where a patch
    supported the score, negative where it suppressed it.
    """
    spectrogram = np.asarray(spectrogram, dtype=np.float32)
    if spectrogram.ndim != 2:
        raise ValueError("spectrogram must be 2-D [freq, time]")

    base_score = float(score_fn(spectrogram))
    fill = float(spectrogram.mean()) if baseline == "mean" else float(baseline)

    freq_bounds = spectrogram_patch_bounds(spectrogram.shape[0], n_freq_patches)
    time_bounds = spectrogram_patch_bounds(spectrogram.shape[1], n_time_patches)
    importance = np.zeros((len(freq_bounds), len(time_bounds)), dtype=np.float32)

    for i, (f0, f1) in enumerate(freq_bounds):
        for j, (t0, t1) in enumerate(time_bounds):
            occluded = spectrogram.copy()
            occluded[f0:f1, t0:t1] = fill
            importance[i, j] = base_score - float(score_fn(occluded))
    return importance
