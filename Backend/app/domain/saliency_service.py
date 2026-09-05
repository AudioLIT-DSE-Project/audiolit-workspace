import logging
import os
import threading
import torch
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from captum.attr import IntegratedGradients, GradientShap, Lime
from app.domain.provenance import Provenance, provenance_fields
from app.domain.model_loader_service import (
    transcribe_whisper_base,
    transcribe_whisper_with_timestamps,
    predict_emotion_wave2vec,
    get_whisper_base_models,
    resolve_whisper_model_id,
)
import app.domain.model_loader_service as model_loader_service

logger = logging.getLogger(__name__)

# generate_saliency() runs on FastAPI's default thread pool (asyncio.to_thread),
# so concurrent /saliency/generate calls for the same model (e.g. the four XAI
# method tabs firing together, or a live selection racing a background dataset
# scan) execute on real, simultaneous OS threads. Every model_type here reuses
# one process-wide cached nn.Module (ModelRegistry for Whisper,
# model_loader_service.emo_model for wav2vec2, the ADD registry for
# melody-machine/wav2vec2-add) and Grad-CAM/Captum mutate that shared module in
# place - register_forward_hook/register_full_backward_hook on the same target
# layer, then model.zero_grad()/.backward(). Two threads doing that at once on
# the same module race each other's hooks and gradient buffers, which is
# exactly what produced the intermittent "size of tensor a (2) must match ...
# dimension 1" 500s: one thread's hook captured activations/gradients that
# belonged to a different thread's forward/backward pass. Serializing per
# model identity (not one global lock) keeps unrelated models running in
# parallel while making concurrent requests against the *same* shared model
# safe.
_model_locks: Dict[str, threading.Lock] = {}
_model_locks_guard = threading.Lock()


def lock_for_model(model_key: str) -> threading.Lock:
    with _model_locks_guard:
        lock = _model_locks.get(model_key)
        if lock is None:
            lock = threading.Lock()
            _model_locks[model_key] = lock
        return lock
MAX_SALIENCY_SECONDS = int(os.getenv("MAX_SALIENCY_SECONDS", "12"))  # cap analysis window
MAX_SALIENCY_SECONDS_SHAP = int(os.getenv("MAX_SALIENCY_SECONDS_SHAP", "6"))  # stricter for SHAP
SALIENCY_SHAP_SAMPLES = int(os.getenv("SALIENCY_SHAP_SAMPLES", "8"))

# ADD model keys (model_loader_service._ADD_MODEL_REGISTRY) checked before the
# generic "wav2vec" substring below -- "wav2vec2-add" would otherwise match the
# wav2vec2 (SER) branch and silently run saliency against the wrong model's
# weights instead of being rejected as unsupported.
_ADD_MODEL_KEYS = ("melody-machine", "wav2vec2-add")


def detect_model_type(model: str) -> str:
    if model in _ADD_MODEL_KEYS or "deepfake" in model.lower():
        return "add"
    elif "whisper" in model.lower():
        return "whisper"
    elif "wav2vec" in model.lower():
        return "wav2vec2"
    return "unknown"


def model_lock_key(model: str) -> Optional[str]:
    """Lock key for the shared nn.Module `model` actually resolves to.

    Single source of truth for both generate_saliency() below and
    inference_service.run_inference() (plain ASR/SER/ADD inference) - the two
    must derive identical keys for the same model, or a plain transcription
    forward pass can interleave with a saliency Grad-CAM's registered hooks on
    the very same cached module (register_forward_hook fires on ANY forward
    pass through that layer, not just the one that registered it) and corrupt
    both: a live L2-ARCTIC + whisper-base repro produced a Grad-CAM "size of
    tensor a (2) must match the size of tensor b (0)" crash on one thread and
    garbage (non-audio-matching) transcripts on the other, from ordinary
    concurrent page load (the Saliency tab's XAI fetch racing the transcript
    fetch). Returns None for a model type inference_service has no shared
    cached instance to protect.
    """
    model_type = detect_model_type(model)
    if model_type == "whisper":
        return f"whisper:{resolve_whisper_model_id(model)}"
    elif model_type == "wav2vec2":
        return "wav2vec2:ser-emotion"
    elif model_type == "add":
        return f"add:{model}"
    return None


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
    
    total_duration = float(len(audio)) / 16000.0 if hasattr(audio, "__len__") and len(audio) > 0 else 0.0

    # Attribute over the checkpoint the user selected, not whisper-base -
    # a heatmap from a different model than the prediction it explains is
    # worse than no heatmap (FR1, FR8).
    processor, model = get_whisper_base_models(resolve_whisper_model_id(model_size))

    device = next(model.parameters()).device
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    input_features.requires_grad_(True)
    
    def model_forward(inputs):
        # Reduce to a scalar per batch: energy of encoder activations
        enc = model.encoder(inputs).last_hidden_state  # [B, T, H]
        return enc.pow(2).mean(dim=(1, 2))             # [B]
    
    shap_fallback_reason = None
    if method == "gradcam":
        try:
            target_layer = find_last_conv_layer(model.encoder)
            
            class WhisperEncoderGradCamWrapper(torch.nn.Module):
                def __init__(self, encoder):
                    super().__init__()
                    self.encoder = encoder
                def forward(self, x):
                    hidden = self.encoder(x).last_hidden_state
                    return hidden.pow(2).mean(dim=(1, 2), keepdim=True)

            wrapper = WhisperEncoderGradCamWrapper(model.encoder)
            cam_np = compute_grad_cam(wrapper, input_features, target_layer=target_layer)
            cam_tensor = torch.from_numpy(cam_np).to(device=input_features.device, dtype=input_features.dtype)
            if cam_tensor.dim() == 1:
                cam_tensor = cam_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, T_conv]
                interp = torch.nn.functional.interpolate(
                    cam_tensor,
                    size=input_features.shape[-1],
                    mode="linear",
                    align_corners=False
                ).squeeze(0)  # [1, T_input]
                attributions = interp.repeat(input_features.shape[-2], 1)  # [80, T_input]
            else:
                attributions = cam_tensor
        except ValueError as e:
            return {
                "model": "whisper",
                "method": "gradcam",
                "segments": [],
                "total_duration": total_duration,
                "series": [],
                "base_spectrogram": [],
                "saliency_matrix": [],
                **provenance_fields(
                    Provenance.UNAVAILABLE,
                    reason=f"No convolutional layer found for Grad-CAM: {e}",
                ),
            }
    elif method in ("integrated_gradients", "ig"):
        # Optimize memory usage for GPU
        torch.cuda.empty_cache()
        
        # Use gradient checkpointing to save memory
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        
        # Use smaller batch size and fewer steps on CPU to guarantee fast response (< 1s)
        n_steps = 4 if not torch.cuda.is_available() else 16
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
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning("First attempt failed, trying with even lower memory settings...")
                n_steps = 8
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
                shap_fallback_reason = "SHAP computation failed or OOM; showing encoder energy map, not attribution"
        except Exception:
            logger.exception("Whisper SHAP failed; falling back to energy map")
            attributions = None
            shap_fallback_reason = "SHAP computation failed or OOM; showing encoder energy map, not attribution"
    else:
        raise ValueError(f"Unsupported saliency method '{method}'. Supported methods: 'gradcam', 'integrated_gradients', 'lime', 'shap'")
    
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
    fallback_reason = shap_fallback_reason
    if use_energy_fallback:
        if fallback_reason is None:
            fallback_reason = "attribution was empty or constant; showing encoder energy, not attribution"
        logger.info(f"Using Whisper energy-map fallback for saliency ({fallback_reason})")
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
    
    # Generate the base log-mel spectrogram for visual reference
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=2048, hop_length=512, n_mels=128)
    log_mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    log_mel_spect_norm = (log_mel_spect - log_mel_spect.min()) / (log_mel_spect.max() - log_mel_spect.min() + 1e-9)
    
    # Map 1D attributions to 2D time-frequency matrix
    n_frames = log_mel_spect_norm.shape[1]
    if series.size > 0:
        attr_resampled = np.interp(
            np.linspace(0, len(series) - 1, n_frames),
            np.arange(len(series)),
            np.abs(series)
        )
        mel_saliency = np.tile(attr_resampled, (128, 1))
    else:
        mel_saliency = np.zeros_like(log_mel_spect_norm)

    prov = Provenance.FALLBACK if (use_energy_fallback or shap_fallback_reason is not None) else Provenance.MEASURED
    res_reason = fallback_reason if prov == Provenance.FALLBACK else None
    return {
        "model": resolve_whisper_model_id(model_size),
        "method": method,
        "segments": segments,
        "total_duration": total_duration,
        "series": series.tolist(),
        "base_spectrogram": log_mel_spect_norm.tolist(),
        "saliency_matrix": mel_saliency.tolist(),
        **provenance_fields(prov, reason=res_reason)
    }

################################################################################################################

def generate_wav2vec2_saliency(audio_file_path: str, method: str = "gradcam", existing_prediction: Dict = None) -> Dict:
    model_loader_service.ensure_emo_model_loaded()
    audio, rate = librosa.load(audio_file_path, sr=16000)
    segment_duration = len(audio) / rate if rate > 0 else 0.0
    # Crop to safe max duration to bound memory
    max_seconds = MAX_SALIENCY_SECONDS_SHAP if method == "shap" else MAX_SALIENCY_SECONDS
    max_len = int(max_seconds * rate)
    if len(audio) > max_len:
        audio = audio[:max_len]
        segment_duration = len(audio) / rate
    inputs = model_loader_service.feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
    
    input_values = inputs.input_values.to(model_loader_service.emo_device)
    attention_mask = inputs.attention_mask.to(model_loader_service.emo_device) if "attention_mask" in inputs else None
    
    input_values.requires_grad_(True)
    
    # Determine class to attribute (predicted emotion)
    with torch.no_grad():
        tmp_out = model_loader_service.emo_model(input_values=input_values, attention_mask=attention_mask)
        tmp_probs = torch.nn.functional.softmax(tmp_out.logits, dim=-1)
        target_idx = int(torch.argmax(tmp_probs, dim=-1).item())
        id2label = getattr(model_loader_service.emo_model.config, "id2label", {})
        emotion = id2label.get(target_idx, f"emotion_{target_idx}")

    id2label = getattr(model_loader_service.emo_model.config, "id2label", {0: "neutral", 1: "happy", 2: "sad"})
    if isinstance(id2label, dict):
        emotion = id2label.get(target_idx, id2label.get(str(target_idx), f"emotion_{target_idx}"))
    else:
        emotion = f"emotion_{target_idx}"
    segment_duration = float(len(audio) / rate)

    def model_forward(inputs, mask=None, cls_idx: int = 0):
        outputs = model_loader_service.emo_model(input_values=inputs, attention_mask=mask)
        return outputs.logits[:, cls_idx]
    
    shap_fallback_reason = None
    if method == "gradcam":
        try:
            target_layer = find_last_conv_layer(model_loader_service.emo_model)
            cam_np = compute_grad_cam(
                model_loader_service.emo_model,
                input_values,
                target_layer=target_layer,
                target_index=target_idx,
            )
            attributions = torch.from_numpy(cam_np).to(device=input_values.device, dtype=input_values.dtype)
            if attributions.shape != input_values.shape:
                attributions = torch.nn.functional.interpolate(
                    attributions.view(1, 1, -1),
                    size=input_values.shape[-1],
                    mode="linear",
                    align_corners=False,
                ).squeeze(0)
        except ValueError as e:
            return {
                "model": "wav2vec2",
                "method": "gradcam",
                "emotion": emotion,
                "segments": [],
                "total_duration": segment_duration,
                "series": [],
                "base_spectrogram": [],
                "saliency_matrix": [],
                **provenance_fields(
                    Provenance.UNAVAILABLE,
                    reason=f"No convolutional layer found for Grad-CAM: {e}",
                ),
            }
    elif method in ("integrated_gradients", "ig"):
        ig = IntegratedGradients(model_forward)
        n_steps_val = 4 if not torch.cuda.is_available() else 16
        try:
            attributions = ig.attribute(
                input_values,
                additional_forward_args=(attention_mask, target_idx),
                n_steps=n_steps_val,
                internal_batch_size=1,
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning("CUDA OOM during Wav2Vec2 saliency. Falling back to CPU with fewer steps.")
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    cpu_device = torch.device("cpu")
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
                shap_fallback_reason = "SHAP computation failed or OOM; showing encoder energy map, not attribution"
        except Exception:
            logger.exception("Wav2Vec2 SHAP failed; falling back to energy map")
            attributions = None
            shap_fallback_reason = "SHAP computation failed or OOM; showing encoder energy map, not attribution"
    else:
        raise ValueError(f"Unsupported saliency method '{method}'. Supported methods: 'gradcam', 'integrated_gradients', 'lime', 'shap'")
    
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
    use_energy_fallback = (saliency_scores.size == 0 or (np.max(saliency_scores) - np.min(saliency_scores) if saliency_scores.size > 0 else 0.0) < 1e-6)
    fallback_reason = shap_fallback_reason
    if use_energy_fallback:
        if fallback_reason is None:
            fallback_reason = "attribution was empty or constant; showing encoder energy, not attribution"
        logger.info(f"Using Wav2Vec2 energy-map fallback for saliency ({fallback_reason})")
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
        raw_saliancies = [s.get("saliency", 0.0) for s in segments]
        abs_vals = np.abs(raw_saliancies)
        
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
    
    # --- NEW: 2D Spectrogram-aligned saliency mapping (SRS FR9 / FR8.2) ---
    # Generate the base log-mel spectrogram for visual reference
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=2048, hop_length=512, n_mels=128)
    log_mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    
    # Normalize log mel spectrogram to 0-1 for UI canvas rendering
    log_mel_spect_norm = (log_mel_spect - log_mel_spect.min()) / (log_mel_spect.max() - log_mel_spect.min() + 1e-9)
    
    # Map 1D time-domain attributions to 2D time-frequency spectrogram bins
    stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    n_frames = stft.shape[1]
    
    # Ensure attribution length matches STFT frames
    attr_resampled = np.interp(
        np.linspace(0, len(saliency_scores) - 1, n_frames),
        np.arange(len(saliency_scores)),
        np.abs(saliency_scores)
    )
    
    # Broadcast 1D attribution across frequency bins to create 2D mask
    attr_2d = np.tile(attr_resampled, (stft.shape[0], 1))
    
    # Weight the STFT bins by the attribution mask
    saliency_2d = stft * attr_2d
    max_val_2d = np.max(saliency_2d)
    if max_val_2d > 0:
        saliency_2d = saliency_2d / max_val_2d
        
    # Downsample to mel bins to match base spectrogram shape
    mel_saliency = np.zeros_like(log_mel_spect)
    fft_bins_per_mel = stft.shape[0] // log_mel_spect.shape[0] + 1 if log_mel_spect.shape[0] > 0 else 1
    for i in range(log_mel_spect.shape[0]):
        start = i * fft_bins_per_mel
        end = start + fft_bins_per_mel
        if end > stft.shape[0]:
            end = stft.shape[0]
        if start < end:
            mel_saliency[i, :] = np.mean(saliency_2d[start:end, :], axis=0)
        else:
            mel_saliency[i, :] = 0.0

    prov = Provenance.FALLBACK if (use_energy_fallback or shap_fallback_reason is not None) else Provenance.MEASURED
    res_reason = fallback_reason if prov == Provenance.FALLBACK else None
    return {
        "model": "wav2vec2",
        "method": method,
        "emotion": emotion,
        "segments": segments,
        "total_duration": segment_duration,
        "series": series.tolist(),
        "base_spectrogram": log_mel_spect_norm.tolist(),  # Added for XAI canvas
        "saliency_matrix": mel_saliency.tolist(),         # Added for XAI canvas
        **provenance_fields(prov, reason=res_reason)
    }

def generate_add_gradcam_saliency(audio_file_path: str, model_name: str = "melody-machine", model_key: Optional[str] = None) -> Dict:
    key = model_key or model_name
    audio, sr = librosa.load(audio_file_path, sr=16000)
    spect = audio_to_spectrogram(audio)
    spect_norm = (spect - spect.min()) / (spect.max() - spect.min() + 1e-8)

    feature_extractor, add_model, device = model_loader_service.ensure_add_model_loaded(key)

    try:
        target_layer = find_last_conv_layer(add_model)
        if isinstance(target_layer, torch.nn.Conv2d):
            inputs = torch.from_numpy(spect_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        else:
            # Both real ADD checkpoints (MelodyMachine, Wav2Vec2 XLSR) are
            # Wav2Vec2ForSequenceClassification, whose forward() always takes
            # the raw waveform through the feature extractor - which conv
            # layer Grad-CAM hooks for activations/gradients doesn't change
            # what the model itself accepts as input. The previous branch
            # keyed the input shape off target_layer.in_channels == 1, but
            # find_last_conv_layer() returns the *last* Conv1d, and Wav2Vec2's
            # feature encoder only has in_channels == 1 on its *first* layer
            # (every later layer is in_channels == config.conv_dim, e.g. 512)
            # - so that check was never true for a real checkpoint, and every
            # request fell through to feeding a 2-D spectrogram into a model
            # built for 1-D waveform input, raising a Conv1d shape RuntimeError.
            fe_inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = fe_inputs.input_values.to(device)
        cam = compute_grad_cam(add_model, inputs, target_layer=target_layer)
        
        timeline = cam.mean(axis=0) if cam.ndim == 2 else cam
        timeline_norm = (timeline - timeline.min()) / (timeline.max() - timeline.min() + 1e-8)
        
        total_duration = len(audio) / sr
        n_frames = len(timeline_norm)
        segments = []
        for idx, w in enumerate(timeline_norm):
            t_start = (idx / n_frames) * total_duration
            t_end = ((idx + 1) / n_frames) * total_duration
            segments.append({
                # Every other saliency path (whisper, wav2vec2, and this
                # module's own generate_add_saliency) keys segments as
                # start_time/end_time - the frontend's SaliencyVisualization
                # reads exactly those two fields (segment.start_time.toFixed(1)).
                # This function alone used "start"/"end", which rendered fine
                # as long as Grad-CAM never actually succeeded for a real ADD
                # checkpoint (see the fixed input-shape bug above) - once it
                # does, start_time comes back undefined and .toFixed() throws,
                # taking down the whole page (no error boundary previously).
                "start_time": round(t_start, 3),
                "end_time": round(t_end, 3),
                "intensity": float(w)
            })
            
        series = timeline_norm.astype(np.float32)
        mel_saliency = cam.astype(np.float32) if cam.ndim == 2 else np.tile(series, (128, 1))
        log_mel_spect_norm = spect_norm.astype(np.float32)
        
        return {
            "model": key,
            "method": "gradcam",
            "segments": segments,
            "total_duration": total_duration,
            "series": series.tolist(),
            "base_spectrogram": log_mel_spect_norm.tolist(),
            "saliency_matrix": mel_saliency.tolist(),
            **provenance_fields(Provenance.MEASURED)
        }
    except ValueError as e:
        return {
            "model": key,
            "method": "gradcam",
            "segments": [],
            "total_duration": len(audio) / sr if sr else 0.0,
            "series": [],
            "base_spectrogram": [],
            "saliency_matrix": [],
            **provenance_fields(
                Provenance.UNAVAILABLE,
                reason=f"no Conv1d/Conv2d layer found for Grad-CAM: {e}",
            ),
        }


def generate_add_saliency(audio_file_path: str, model_name: str = "melody-machine", method: str = "integrated_gradients") -> Dict:
    """Captum IG/LIME/SHAP for the deepfake-detection (ADD) classifiers (SRS FR9: IG
    "for ASR, SER, and ADD"; FR8.1 for the LIME/SHAP case).

    Mirrors generate_wav2vec2_saliency's already-working Captum wiring - both ADD
    checkpoints (melody-machine, wav2vec2-add) are Wav2Vec2ForSequenceClassification,
    the same architecture family as the SER model, so the same attribution-over-
    raw-waveform approach applies; only the model source (ensure_add_model_loaded)
    and the predicted class (bona-fide/spoof instead of an emotion) differ.
    """
    feature_extractor, add_model, device = model_loader_service.ensure_add_model_loaded(model_name)
    audio, rate = librosa.load(audio_file_path, sr=16000)

    max_seconds = MAX_SALIENCY_SECONDS_SHAP if method == "shap" else MAX_SALIENCY_SECONDS
    max_len = int(max_seconds * rate)
    if len(audio) > max_len:
        audio = audio[:max_len]
    segment_duration = len(audio) / rate if rate > 0 else 0.0

    inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None
    input_values.requires_grad_(True)

    with torch.no_grad():
        tmp_out = add_model(input_values=input_values, attention_mask=attention_mask)
        target_idx = int(torch.argmax(tmp_out.logits, dim=-1).item())
    id2label = getattr(add_model.config, "id2label", {})
    raw_label = id2label.get(target_idx, id2label.get(str(target_idx), f"class_{target_idx}"))
    predicted_label = model_loader_service._normalize_deepfake_label(raw_label)

    def model_forward(inputs, mask=None, cls_idx: int = 0):
        outputs = add_model(input_values=inputs, attention_mask=mask)
        return outputs.logits[:, cls_idx]

    shap_fallback_reason = None
    if method in ("integrated_gradients", "ig"):
        ig = IntegratedGradients(model_forward)
        n_steps_val = 4 if not torch.cuda.is_available() else 16
        attributions = ig.attribute(
            input_values,
            additional_forward_args=(attention_mask, target_idx),
            n_steps=n_steps_val,
            internal_batch_size=1,
        )
    elif method == "lime":
        lime = Lime(model_forward)
        attributions = lime.attribute(input_values, additional_forward_args=(attention_mask, target_idx))
    elif method == "shap":
        gs = GradientShap(model_forward)
        baseline = torch.zeros_like(input_values)
        try:
            attributions = gs.attribute(
                input_values,
                baselines=baseline,
                additional_forward_args=(attention_mask, target_idx),
                n_samples=max(2, min(16, SALIENCY_SHAP_SAMPLES)),
                stdevs=0.09,
            )
        except Exception:
            logger.exception("ADD SHAP failed; falling back to encoder energy map")
            attributions = None
            shap_fallback_reason = "SHAP computation failed or OOM; showing encoder energy map, not attribution"
    else:
        raise ValueError(
            f"Unsupported saliency method '{method}'. Supported methods: 'gradcam', 'integrated_gradients', 'lime', 'shap'"
        )

    if attributions is not None:
        tmp = attributions.detach().cpu().numpy().squeeze()
        tmp = np.abs(tmp) if tmp.ndim <= 1 else np.mean(np.abs(tmp), axis=0)
        mx = float(np.max(tmp)) if tmp.size > 0 else 0.0
        saliency_scores = (tmp / mx) if mx > 0 else np.zeros_like(tmp)
    else:
        saliency_scores = np.array([])

    use_energy_fallback = (
        saliency_scores.size == 0
        or (np.max(saliency_scores) - np.min(saliency_scores) if saliency_scores.size > 0 else 0.0) < 1e-6
    )
    fallback_reason = shap_fallback_reason
    if use_energy_fallback:
        if fallback_reason is None:
            fallback_reason = "attribution was empty or constant; showing encoder energy, not attribution"
        with torch.no_grad():
            hs = add_model.wav2vec2(input_values=input_values, attention_mask=attention_mask).last_hidden_state
            energy = hs.abs().mean(dim=2).squeeze(0).detach().cpu().numpy()
        if energy.size > 0:
            e_min, e_ptp = float(np.min(energy)), float(np.ptp(energy))
            saliency_scores = (energy - e_min) / (e_ptp + 1e-9)
        else:
            saliency_scores = np.zeros(1, dtype=np.float32)

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

    T = len(saliency_scores)
    num_segments = 32
    segment_length = segment_duration / num_segments if num_segments > 0 else segment_duration
    fps = (T / segment_duration) if segment_duration > 0 else 1.0
    segments = []
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = (i + 1) * segment_length
        start_frame = max(0, min(T - 1, int(start_time * fps)))
        end_frame = max(start_frame + 1, min(T, int(end_time * fps)))
        segment_saliency = float(np.mean(saliency_scores[start_frame:end_frame])) if T > 0 else 0.0
        segments.append({
            "start_time": start_time,
            "end_time": end_time,
            "saliency": segment_saliency,
            "intensity": float(abs(segment_saliency)),
        })

    if len(segments) > 0:
        abs_vals = np.abs([s["saliency"] for s in segments])
        max_abs = float(np.max(abs_vals)) if len(abs_vals) > 0 else 0.0
        if max_abs > 1e-9:
            for i, segment in enumerate(segments):
                segment["intensity"] = float(abs_vals[i] / max_abs)
        else:
            sorted_indices = np.argsort(-abs_vals)
            for rank, idx in enumerate(sorted_indices):
                segments[idx]["intensity"] = float(1.0 - (rank / len(segments)) * 0.9)
        for segment in segments:
            segment["intensity"] = max(0.1, segment["intensity"])

    mel_spect = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=2048, hop_length=512, n_mels=128)
    log_mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    log_mel_spect_norm = (log_mel_spect - log_mel_spect.min()) / (log_mel_spect.max() - log_mel_spect.min() + 1e-9)

    stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    n_frames = stft.shape[1]
    if saliency_scores.size > 0:
        attr_resampled = np.interp(
            np.linspace(0, len(saliency_scores) - 1, n_frames),
            np.arange(len(saliency_scores)),
            np.abs(saliency_scores),
        )
    else:
        attr_resampled = np.zeros(n_frames, dtype=np.float32)
    attr_2d = np.tile(attr_resampled, (stft.shape[0], 1))
    saliency_2d = stft * attr_2d
    max_val_2d = np.max(saliency_2d)
    if max_val_2d > 0:
        saliency_2d = saliency_2d / max_val_2d

    mel_saliency = np.zeros_like(log_mel_spect)
    fft_bins_per_mel = stft.shape[0] // log_mel_spect.shape[0] + 1 if log_mel_spect.shape[0] > 0 else 1
    for i in range(log_mel_spect.shape[0]):
        start = i * fft_bins_per_mel
        end = min(start + fft_bins_per_mel, stft.shape[0])
        if start < end:
            mel_saliency[i, :] = np.mean(saliency_2d[start:end, :], axis=0)

    prov = Provenance.FALLBACK if (use_energy_fallback or shap_fallback_reason is not None) else Provenance.MEASURED
    res_reason = fallback_reason if prov == Provenance.FALLBACK else None
    return {
        "model": model_name,
        "method": method,
        "predicted_label": predicted_label,
        "segments": segments,
        "total_duration": segment_duration,
        "series": series.tolist(),
        "base_spectrogram": log_mel_spect_norm.tolist(),
        "saliency_matrix": mel_saliency.tolist(),
        **provenance_fields(prov, reason=res_reason),
    }


def generate_saliency(audio_file_path: str, model: str, method: str = "gradcam", existing_prediction: Dict = None) -> Dict:
    model_type = detect_model_type(model)

    # Lock keyed by the actual shared nn.Module identity each branch below
    # will mutate (see lock_for_model's docstring comment above), not by the
    # raw `model` string - whisper aliases ("whisper-base" / "openai/whisper-
    # base") must serialize against each other since they resolve to the same
    # cached instance, and every wav2vec2/SER call shares the one emo_model
    # singleton regardless of `model`. Shared with inference_service.
    # run_inference via model_lock_key() - see its docstring for why plain
    # inference must serialize against this too, not just saliency-vs-saliency.
    lock_key = model_lock_key(model)
    if lock_key is None:
        raise ValueError(f"Unsupported model: {model}")

    with lock_for_model(lock_key):
        if method == "gradcam" and model_type == "add":
            return generate_add_gradcam_saliency(audio_file_path, model)

        if model_type == "whisper":
            return generate_whisper_saliency(audio_file_path, model, method, existing_prediction)
        elif model_type == "wav2vec2":
            return generate_wav2vec2_saliency(audio_file_path, method, existing_prediction)
        else:  # model_type == "add", method != "gradcam"
            return generate_add_saliency(audio_file_path, model, method)


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


# --- Integrated Gradients core (LIT-126, FR9) --------------------------------
# The primary gradient-based attribution: run Captum IntegratedGradients over an
# input tensor and map the model's output score back onto each input dimension,
# then collapse to a millisecond-aligned 1-D timeline. Model-agnostic (takes a
# forward function returning a per-batch score), so it explains the deepfake /
# emotion / any scalar-scoring model.


def integrated_gradients(
    forward_fn,
    inputs: "torch.Tensor",
    baselines: "torch.Tensor" = None,
    target=None,
    n_steps: int = 50,
) -> "torch.Tensor":
    """Integrated Gradients attribution via Captum (FR9).

    ``forward_fn(inputs)`` returns either a per-batch scalar score or a
    ``[batch, n_outputs]`` tensor (then pass ``target`` to pick the output).
    Returns an attribution tensor shaped like ``inputs``, crediting the score to
    each input dimension; the baseline defaults to zeros (the standard IG
    reference).
    """
    ig = IntegratedGradients(forward_fn)
    if baselines is None:
        baselines = torch.zeros_like(inputs)
    return ig.attribute(inputs, baselines=baselines, target=target, n_steps=n_steps)


def attribution_timeline(attributions, sample_rate: int = 16000, hop_length: int = None) -> list:
    """Collapse an attribution tensor to a normalized, time-aligned saliency stream.

    Non-time dimensions are reduced by ``mean(|·|)``, the result is scaled to
    ``[0, 1]``, and returned as ``[{"t_ms": float, "weight": float}, ...]`` — the
    millisecond-aligned JSON stream FR9 asks for. Each step spans one STFT frame
    (``hop_length / sample_rate``) when ``hop_length`` is given, else one input
    sample (``1 / sample_rate``).
    """
    arr = attributions
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.squeeze(np.asarray(arr, dtype=np.float64))
    while arr.ndim > 1:                      # collapse all but the time axis
        arr = np.mean(np.abs(arr), axis=0)
    weights = np.abs(np.atleast_1d(arr))

    peak = float(weights.max()) if weights.size else 0.0
    if peak > 0:
        weights = weights / peak

    step_s = (hop_length / sample_rate) if hop_length else (1.0 / sample_rate)
    return [{"t_ms": round(i * step_s * 1000.0, 3), "weight": float(w)} for i, w in enumerate(weights)]


def generate_perturbation_matrix(
    spectrogram: np.ndarray,
    n_patches_freq: int = 8,
    n_patches_time: int = 8,
    n_variants: int = 500,
    perturbation_type: str = "zero",
    noise_level: float = 0.1,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate a matrix of perturbed spectrograms for LIME/SHAP evaluation (SRS FR8).
    
    Divides the base spectrogram into a grid of time-frequency patches, then 
    generates ``n_variants`` copies where random subsets of patches are perturbed
    to evaluate performance degradation.
    
    Args:
        spectrogram: 2D numpy array [freq, time].
        n_patches_freq: Number of frequency patches.
        n_patches_time: Number of time patches.
        n_variants: Number of perturbed spectrograms to generate (DoD: 500).
        perturbation_type: "zero", "mean", or "noise".
        noise_level: Std dev for noise if perturbation_type="noise".
        random_state: Optional seed for reproducibility.
        
    Returns:
        A 3D numpy array of shape (n_variants, freq, time).
    """
    if spectrogram.ndim != 2:
        raise ValueError("spectrogram must be 2-D [freq, time]")
        
    rng = np.random.default_rng(random_state)
    freq_bounds = spectrogram_patch_bounds(spectrogram.shape[0], n_patches_freq)
    time_bounds = spectrogram_patch_bounds(spectrogram.shape[1], n_patches_time)
    
    total_patches = len(freq_bounds) * len(time_bounds)
    
    # Vectorized generation: create a 3D tensor of shape (n_variants, freq, time)
    variants = np.tile(spectrogram, (n_variants, 1, 1))
    
    # Determine perturbation fill value
    if perturbation_type == "mean":
        fill_val = float(spectrogram.mean())
    else:
        fill_val = 0.0
        
    for i in range(n_variants):
        # Randomly select ~50% of patches to perturb (standard LIME/SHAP practice)
        n_perturb = total_patches // 2
        patch_indices = rng.choice(total_patches, size=n_perturb, replace=False)
        
        for idx in patch_indices:
            f_idx = idx // n_patches_time
            t_idx = idx % n_patches_time
            f0, f1 = freq_bounds[f_idx]
            t0, t1 = time_bounds[t_idx]
            
            if perturbation_type == "noise":
                variants[i, f0:f1, t0:t1] = rng.normal(
                    loc=fill_val, scale=noise_level, size=(f1-f0, t1-t0)
                ).astype(np.float32)
            else:
                variants[i, f0:f1, t0:t1] = fill_val
                
    return variants
