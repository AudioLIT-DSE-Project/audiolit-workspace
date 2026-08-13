"use client"

import React, { useState, useEffect, useCallback, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Checkbox } from "@/components/ui/checkbox"
import { RangeSlider } from "@/components/ui/range-slider"
import { Volume2, VolumeX, Filter, Scissors, Plus, Play, Zap, XCircle } from "lucide-react"
import { WaveformViewer } from "../audio/WaveformViewer"
import { API_BASE } from '@/lib/api'
import { useTaskStatus } from '@/hooks/useTaskStatus'
import { GlobalTaskProgress } from '../layout/GlobalTaskProgress'

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
}

interface PerturbationResult {
  perturbed_file: string;
  filename: string;
  duration_ms: number;
  sample_rate: number;
  applied_perturbations: Array<{
    type: string;
    params: Record<string, any>;
    status: string;
    error?: string;
  }>;
  success: boolean;
  error?: string;
}

interface PerturbationToolsProps {
  selectedFile: UploadedFile | null;
  onPerturbationComplete?: (result: PerturbationResult) => void;
  onPredictionRefresh?: (file: UploadedFile, prediction: string) => void;
  model?: string;
  dataset?: string;
  originalDataset?: string;
}

const getAudioUrl = (selectedFile: UploadedFile, dataset?: string, originalDataset?: string): string => {
  const isUploadedFile = selectedFile.file_path && (
    selectedFile.file_path.includes('uploads/') || 
    selectedFile.file_path.includes('uploads\\') ||
    selectedFile.message === "Perturbed file" ||
    selectedFile.message === "File uploaded successfully" ||
    selectedFile.message === "File uploaded and processed successfully"
  ) && selectedFile.message !== "Selected from dataset";
  
  if (isUploadedFile) {
    return `${API_BASE}/upload/file/${selectedFile.file_id}`;
  } else {
    const datasetToUse = originalDataset && originalDataset !== "custom" ? originalDataset : dataset;
    if (datasetToUse && datasetToUse !== "custom") {
      const filename = encodeURIComponent(selectedFile.filename);
      return `${API_BASE}/${encodeURIComponent(datasetToUse)}/file/${filename}`;
    } else {
      return `${API_BASE}/upload/file/${selectedFile.file_id}`;
    }
  }
};

const getPerturbedAudioUrl = (perturbedFilePath: string): string => {
  const filename = perturbedFilePath.split('/').pop() || perturbedFilePath.split('\\').pop();
  return `${API_BASE}/upload/file/${filename}`;
};

// --- LIT-177: 2D Spectrogram Grid Selector & Coordinate Resolution Handler ---

export interface SpectrogramBoundaryFrame {
  id: string;
  startTimeMs: number;
  endTimeMs: number;
  startFreqHz: number;
  endFreqHz: number;
}

const GRID_DRAG_THRESHOLD_PX = 4;
const DEFAULT_MAX_FREQ_HZ = 8000; // Nyquist fallback when sample_rate is unknown

const gridClamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const getCanvasLogicalSize = (canvas: HTMLCanvasElement | null) => {
  if (!canvas) return { width: 0, height: 0 };
  const dpr = window.devicePixelRatio || 1;
  return { width: canvas.width / dpr, height: canvas.height / dpr };
};

// Pixel -> signal translation, using the audio track's duration and Nyquist
// frequency (sample_rate / 2). Mirrors the mel-scale mapping already used by
// XAIOverlayCanvas.tsx's mapTimeToX/mapHzToY, inverted.
const pixelXToTimeMs = (x: number, width: number, durationSec: number) =>
  width > 0 ? (x / width) * durationSec * 1000 : 0;

const pixelYToFreqHz = (y: number, height: number, maxFreqHz: number) => {
  if (height <= 0) return 0;
  const maxMel = 2595 * Math.log10(1 + maxFreqHz / 500);
  const mel = gridClamp((height - y) / height, 0, 1) * maxMel;
  return 500 * (Math.pow(10, mel / 2595) - 1);
};

// Signal -> pixel, used to redraw persisted frames and grid labels.
const timeMsToPixelX = (timeMs: number, width: number, durationSec: number) =>
  durationSec > 0 ? (timeMs / 1000 / durationSec) * width : 0;

const freqHzToPixelY = (hz: number, height: number, maxFreqHz: number) => {
  const maxMel = 2595 * Math.log10(1 + maxFreqHz / 500);
  const mel = 2595 * Math.log10(1 + hz / 500);
  return maxMel > 0 ? height - (mel / maxMel) * height : height;
};

interface SpectrogramGridSelectorProps {
  durationSec: number;
  maxFreqHz?: number;
  height?: number;
  onFrameCreated?: (frame: SpectrogramBoundaryFrame) => void;
}

export const SpectrogramGridSelector: React.FC<SpectrogramGridSelectorProps> = ({
  durationSec,
  maxFreqHz = DEFAULT_MAX_FREQ_HZ,
  height = 160,
  onFrameCreated,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const gridCanvasRef = useRef<HTMLCanvasElement>(null);
  const selectionCanvasRef = useRef<HTMLCanvasElement>(null);
  const dragStateRef = useRef<{ startX: number; startY: number; currentX: number; currentY: number; dragging: boolean } | null>(null);
  const rafIdRef = useRef<number | null>(null);
  const [frames, setFrames] = useState<SpectrogramBoundaryFrame[]>([]);
  const framesRef = useRef(frames);
  framesRef.current = frames;
  const onFrameCreatedRef = useRef(onFrameCreated);
  onFrameCreatedRef.current = onFrameCreated;

  const drawGrid = useCallback(() => {
    const canvas = gridCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const { width: w, height: h } = getCanvasLogicalSize(canvas);

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#0f172a'; // slate-900 backdrop
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)'; // slate-400 @ 25%
    ctx.fillStyle = 'rgba(226, 232, 240, 0.7)'; // slate-200 @ 70%
    ctx.font = '10px monospace';
    ctx.lineWidth = 1;

    // Frequency gridlines, evenly spaced across the Nyquist range.
    const freqBands = 4;
    for (let i = 0; i <= freqBands; i++) {
      const hz = (maxFreqHz / freqBands) * i;
      const y = freqHzToPixelY(hz, h, maxFreqHz);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
      ctx.fillText(`${Math.round(hz)} Hz`, 2, Math.max(9, y - 2));
    }

    // Time gridlines, roughly every 0.5-1s depending on clip length.
    if (durationSec > 0) {
      const step = durationSec > 4 ? 1 : 0.5;
      for (let t = 0; t <= durationSec; t += step) {
        const x = timeMsToPixelX(t * 1000, w, durationSec);
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
        ctx.fillText(`${t.toFixed(1)}s`, x + 2, h - 2);
      }
    }
    ctx.restore();
  }, [durationSec, maxFreqHz]);

  const drawSelections = useCallback(() => {
    const canvas = selectionCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const { width: w, height: h } = getCanvasLogicalSize(canvas);

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    // Saved frames persist so multiple selections stay visible simultaneously.
    ctx.fillStyle = 'rgba(16, 185, 129, 0.18)'; // emerald-500 @ 18%
    ctx.strokeStyle = 'rgba(5, 150, 105, 0.9)'; // emerald-600
    ctx.lineWidth = 1.5;
    framesRef.current.forEach((frame) => {
      const x1 = timeMsToPixelX(frame.startTimeMs, w, durationSec);
      const x2 = timeMsToPixelX(frame.endTimeMs, w, durationSec);
      const y1 = freqHzToPixelY(frame.endFreqHz, h, maxFreqHz);
      const y2 = freqHzToPixelY(frame.startFreqHz, h, maxFreqHz);
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    });

    // Active drag, drawn on top while the mouse is still down.
    const drag = dragStateRef.current;
    if (drag) {
      const x1 = Math.min(drag.startX, drag.currentX);
      const x2 = Math.max(drag.startX, drag.currentX);
      const y1 = Math.min(drag.startY, drag.currentY);
      const y2 = Math.max(drag.startY, drag.currentY);
      ctx.fillStyle = 'rgba(59, 130, 246, 0.25)'; // blue-500 @ 25%
      ctx.strokeStyle = 'rgba(29, 78, 216, 0.9)'; // blue-700
      ctx.lineWidth = 2;
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
    ctx.restore();
  }, [durationSec, maxFreqHz]);

  const renderSelectionFrame = useCallback(() => {
    drawSelections();
    if (dragStateRef.current?.dragging) {
      rafIdRef.current = requestAnimationFrame(renderSelectionFrame);
    }
  }, [drawSelections]);

  const handleWindowMouseMove = useCallback((event: MouseEvent) => {
    const container = containerRef.current;
    const drag = dragStateRef.current;
    if (!container || !drag) return;
    const rect = container.getBoundingClientRect();
    drag.currentX = gridClamp(event.clientX - rect.left, 0, rect.width);
    drag.currentY = gridClamp(event.clientY - rect.top, 0, rect.height);
  }, []);

  const handleWindowMouseUp = useCallback(() => {
    window.removeEventListener('mousemove', handleWindowMouseMove);
    window.removeEventListener('mouseup', handleWindowMouseUp);
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }

    const drag = dragStateRef.current;
    const container = containerRef.current;
    if (drag && container) {
      drag.dragging = false;
      const x1 = Math.min(drag.startX, drag.currentX);
      const x2 = Math.max(drag.startX, drag.currentX);
      const y1 = Math.min(drag.startY, drag.currentY);
      const y2 = Math.max(drag.startY, drag.currentY);

      if (x2 - x1 >= GRID_DRAG_THRESHOLD_PX && y2 - y1 >= GRID_DRAG_THRESHOLD_PX) {
        const { width: w, height: h } = container.getBoundingClientRect();
        const frame: SpectrogramBoundaryFrame = {
          id: `frame-${Date.now()}-${Math.round(Math.random() * 1e4)}`,
          startTimeMs: pixelXToTimeMs(x1, w, durationSec),
          endTimeMs: pixelXToTimeMs(x2, w, durationSec),
          startFreqHz: pixelYToFreqHz(y2, h, maxFreqHz),
          endFreqHz: pixelYToFreqHz(y1, h, maxFreqHz),
        };
        // DoD (LIT-177): verification logs of resolved timestamp/frequency bounds.
        console.log('[LIT-177] Spectrogram selection resolved:', frame);
        setFrames((prev) => [...prev, frame]);
        onFrameCreatedRef.current?.(frame);
      }
      dragStateRef.current = null;
    }

    drawSelections();
  }, [drawSelections, durationSec, maxFreqHz, handleWindowMouseMove]);

  const handleMouseDown = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    if (event.button !== 0 || durationSec <= 0) return;
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const x = gridClamp(event.clientX - rect.left, 0, rect.width);
    const y = gridClamp(event.clientY - rect.top, 0, rect.height);
    dragStateRef.current = { startX: x, startY: y, currentX: x, currentY: y, dragging: true };
    window.addEventListener('mousemove', handleWindowMouseMove);
    window.addEventListener('mouseup', handleWindowMouseUp);
    rafIdRef.current = requestAnimationFrame(renderSelectionFrame);
  }, [durationSec, handleWindowMouseMove, handleWindowMouseUp, renderSelectionFrame]);

  // Unmount safety net in case a drag is still in progress.
  useEffect(() => {
    return () => {
      window.removeEventListener('mousemove', handleWindowMouseMove);
      window.removeEventListener('mouseup', handleWindowMouseUp);
      if (rafIdRef.current !== null) cancelAnimationFrame(rafIdRef.current);
    };
  }, [handleWindowMouseMove, handleWindowMouseUp]);

  // Redraw once `frames` actually commits — the drawSelections() call inside
  // handleWindowMouseUp reads framesRef synchronously, one render behind the
  // setFrames() call that triggered it, so newly-saved frames need this
  // effect to actually appear on screen.
  useEffect(() => {
    drawSelections();
  }, [frames, drawSelections]);

  // Keep both canvases DPR-scaled and sized to the container; redraw on resize.
  useEffect(() => {
    const container = containerRef.current;
    const gridCanvas = gridCanvasRef.current;
    const selectionCanvas = selectionCanvasRef.current;
    if (!container || !gridCanvas || !selectionCanvas) return;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const { clientWidth } = container;
      [gridCanvas, selectionCanvas].forEach((canvas) => {
        canvas.width = Math.max(1, Math.round(clientWidth * dpr));
        canvas.height = Math.max(1, Math.round(height * dpr));
        canvas.style.width = `${clientWidth}px`;
        canvas.style.height = `${height}px`;
      });
      drawGrid();
      drawSelections();
    };

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    return () => observer.disconnect();
  }, [drawGrid, drawSelections, height]);

  return (
    <div
      ref={containerRef}
      className="relative w-full rounded border border-slate-700 overflow-hidden select-none cursor-crosshair"
      style={{ height }}
      onMouseDown={handleMouseDown}
    >
      <canvas ref={gridCanvasRef} className="absolute inset-0" />
      <canvas ref={selectionCanvasRef} className="absolute inset-0 pointer-events-none" />
    </div>
  );
};

// --- LIT-178: Frontend Mutation Event Trigger & Asynchronous State Dispatcher ---

// Maps onto the perturbation types perturbation_service.py already supports
// (apply_2d_time_freq_mask / apply_band_pass_filter / noise) — no backend
// changes needed.
type FrameMutationType = 'time_freq_mask' | 'band_pass_filter' | 'noise';

const buildFrameMutationPayload = (
  frame: SpectrogramBoundaryFrame,
  mutationType: FrameMutationType,
  noiseLevelPercent: number
) => {
  switch (mutationType) {
    case 'time_freq_mask':
      return {
        type: 'time_freq_mask',
        params: {
          t_start_ms: frame.startTimeMs,
          t_end_ms: frame.endTimeMs,
          f_low_hz: frame.startFreqHz,
          f_high_hz: frame.endFreqHz,
        },
      };
    case 'band_pass_filter':
      return {
        type: 'band_pass_filter',
        params: {
          f_low_hz: frame.startFreqHz,
          f_high_hz: frame.endFreqHz,
        },
      };
    case 'noise':
      return {
        type: 'noise',
        params: { noise_level: noiseLevelPercent / 100 },
      };
  }
};

export const PerturbationTools: React.FC<PerturbationToolsProps> = ({
  selectedFile,
  onPerturbationComplete,
  onPredictionRefresh,
  model,
  dataset,
  originalDataset,
}) => {
  const [noiseLevel, setNoiseLevel] = useState([10])
  const [maskRange, setMaskRange] = useState<[number, number]>([20, 40])
  const [pitchShift, setPitchShift] = useState([2])
  const [timeStretch, setTimeStretch] = useState([110])
  
  const [selectedPerturbations, setSelectedPerturbations] = useState({
    noise: false,
    timeMasking: false,
    pitchShift: false,
    timeStretch: false,
  })
  
  const [error, setError] = useState<string | null>(null)
  const [perturbationResult, setPerturbationResult] = useState<PerturbationResult | null>(null)

  // LIT-178: region-scoped mutation state, driven by LIT-177's spectrogram frames
  const [spectrogramFrames, setSpectrogramFrames] = useState<SpectrogramBoundaryFrame[]>([])
  const [activeFrameId, setActiveFrameId] = useState<string | null>(null)
  const [frameMutationType, setFrameMutationType] = useState<FrameMutationType>('time_freq_mask')
  const [frameNoiseLevel, setFrameNoiseLevel] = useState([10])
  const activeFrame = spectrogramFrames.find(f => f.id === activeFrameId) ?? null

  // RQ Task IDs
  const [mutationTaskId, setMutationTaskId] = useState<string | null>(null)
  const [inferenceTaskId, setInferenceTaskId] = useState<string | null>(null)

  // Track mutation job state
  const { state: mutationState, result: mutationResult } = useTaskStatus(mutationTaskId)
  // Track inference job state
  const { state: inferenceState, result: inferenceResult } = useTaskStatus(inferenceTaskId)

  useEffect(() => {
    setPerturbationResult(null);
    setError(null);
    setMutationTaskId(null);
    setInferenceTaskId(null);
    setSpectrogramFrames([]);
    setActiveFrameId(null);
  }, [selectedFile]);

  const handlePerturbationToggle = (perturbationType: keyof typeof selectedPerturbations) => {
    setSelectedPerturbations(prev => ({ ...prev, [perturbationType]: !prev[perturbationType] }));
  }

  // Effect: When mutation job succeeds, trigger inference
  useEffect(() => {
    if (mutationState === 'SUCCESS' && mutationResult) {
      // Adapt result to expected shape
      const perturbedData = mutationResult as any;
      setPerturbationResult(perturbedData);
      if (onPerturbationComplete) onPerturbationComplete(perturbedData);

      // Start inference on the perturbed file
      const runInference = async () => {
        try {
          const response = await fetch(`${API_BASE}/api/inference/multitask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              audio_ref: perturbedData.perturbed_file,
              tasks: model?.includes('whisper') ? ['asr'] : ['ser']
            })
          });
          if (response.ok) {
            const data = await response.json();
            setInferenceTaskId(data.job_id);
          } else {
            setError("Failed to enqueue inference job for perturbed audio.");
          }
        } catch (err) {
          setError("Error triggering inference job.");
        }
      };
      runInference();
    } else if (mutationState === 'FAILURE') {
      setError("Perturbation task failed in worker.");
    }
  }, [mutationState, mutationResult]);

  // Effect: When inference job succeeds, notify parent
  useEffect(() => {
    if (inferenceState === 'SUCCESS' && inferenceResult && perturbationResult) {
      const perturbedFile: UploadedFile = {
        file_id: perturbationResult.filename,
        filename: perturbationResult.filename,
        file_path: perturbationResult.perturbed_file,
        message: "Perturbed file",
        duration: perturbationResult.duration_ms / 1000,
        sample_rate: perturbationResult.sample_rate
      };
      if (onPredictionRefresh) {
        onPredictionRefresh(perturbedFile, JSON.stringify(inferenceResult));
      }
      setInferenceTaskId(null);
      setMutationTaskId(null);
    } else if (inferenceState === 'FAILURE') {
      setError("Inference task failed in worker.");
      setInferenceTaskId(null);
      setMutationTaskId(null);
    }
  }, [inferenceState, inferenceResult, perturbationResult]);

  const handleAddPerturbations = async () => {
    if (!selectedFile) { setError("No file selected"); return; }
    if (!Object.values(selectedPerturbations).some(Boolean)) { setError("Please select at least one perturbation type"); return; }
    
    setError(null);

    try {
      const perturbations = [];
      if (selectedPerturbations.noise) perturbations.push({ type: "noise", params: { noise_level: noiseLevel[0] / 100.0 } });
      if (selectedPerturbations.timeMasking) perturbations.push({ type: "time_masking", params: { mask_start_percent: maskRange[0], mask_end_percent: maskRange[1] } });
      if (selectedPerturbations.pitchShift) perturbations.push({ type: "pitch_shift", params: { pitch_shift_semitones: pitchShift[0] } });
      if (selectedPerturbations.timeStretch) perturbations.push({ type: "time_stretch", params: { stretch_factor: timeStretch[0] / 100.0 } });

      const isUploadedFile = selectedFile.file_path && (
        selectedFile.file_path.includes('uploads/') || 
        selectedFile.file_path.startsWith('uploads/') ||
        selectedFile.message === "Perturbed file" ||
        selectedFile.message === "File uploaded successfully" ||
        selectedFile.message === "File uploaded and processed successfully"
      ) && selectedFile.message !== "Selected from dataset";

      // Enqueue mutation job via RQ
      const response = await fetch(`${API_BASE}/api/inference/mutation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio_ref: isUploadedFile ? selectedFile.file_path : selectedFile.filename,
          mutation: { 
            perturbations, 
            is_uploaded: isUploadedFile,
            dataset: originalDataset || dataset 
          }
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const result = await response.json();
      setMutationTaskId(result.job_id); // Start tracking via WebSocket
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error occurred";
      setError(errorMessage);
    }
  };

  // LIT-178: dispatch a mutation scoped to the currently active spectrogram
  // region (LIT-177's boundary frames), via the same async RQ mutation
  // endpoint and job-tracking as handleAddPerturbations.
  const handleApplyFrameMutation = async () => {
    if (!selectedFile) { setError("No file selected"); return; }
    if (!activeFrame) { setError("No spectrogram region selected"); return; }

    setError(null);

    try {
      const perturbation = buildFrameMutationPayload(activeFrame, frameMutationType, frameNoiseLevel[0]);

      const isUploadedFile = selectedFile.file_path && (
        selectedFile.file_path.includes('uploads/') ||
        selectedFile.file_path.startsWith('uploads/') ||
        selectedFile.message === "Perturbed file" ||
        selectedFile.message === "File uploaded successfully" ||
        selectedFile.message === "File uploaded and processed successfully"
      ) && selectedFile.message !== "Selected from dataset";

      const response = await fetch(`${API_BASE}/api/inference/mutation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio_ref: isUploadedFile ? selectedFile.file_path : selectedFile.filename,
          mutation: {
            perturbations: [perturbation],
            is_uploaded: isUploadedFile,
            dataset: originalDataset || dataset
          }
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const result = await response.json();
      setMutationTaskId(result.job_id); // Start tracking via WebSocket

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error occurred";
      setError(errorMessage);
    }
  };

  const isProcessing = mutationTaskId !== null || inferenceTaskId !== null;

  return (
    <div className="space-y-4">
      {error && (
        <Card className="border-destructive/20 bg-destructive/5">
          <CardContent className="pt-4">
            <div className="flex items-center gap-2 text-destructive text-xs">
              <XCircle className="h-4 w-4" />
              {error}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Show dynamic progress bar when processing */}
      {isProcessing && (
        <GlobalTaskProgress 
          taskId={inferenceTaskId || mutationTaskId} 
          onComplete={() => {}} 
        />
      )}

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Perturbation Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Noise */}
          <div className="space-y-3 p-3 border rounded-lg">
            <div className="flex items-center space-x-2">
              <Checkbox id="noise-checkbox" checked={selectedPerturbations.noise} onCheckedChange={() => handlePerturbationToggle('noise')} className="border-blue-400 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600" />
              <Volume2 className="h-4 w-4 text-blue-600" />
              <label htmlFor="noise-checkbox" className="text-sm font-medium">Add Gaussian Noise</label>
            </div>
            {selectedPerturbations.noise && (
              <div className="space-y-2 pl-6">
                <div className="flex items-center justify-between">
                  <span className="text-xs">Noise Level</span>
                  <Badge variant="outline" className="text-xs border-blue-300 text-blue-700">{noiseLevel[0]}%</Badge>
                </div>
                <Slider value={noiseLevel} onValueChange={setNoiseLevel} max={50} step={1} className="w-full [&_[role=slider]]:border-blue-500 [&_[role=slider]]:bg-blue-600" />
              </div>
            )}
          </div>

          {/* Time Masking */}
          <div className="space-y-3 p-3 border rounded-lg">
            <div className="flex items-center space-x-2">
              <Checkbox id="masking-checkbox" checked={selectedPerturbations.timeMasking} onCheckedChange={() => handlePerturbationToggle('timeMasking')} className="border-blue-400 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600" />
              <Scissors className="h-4 w-4 text-blue-600" />
              <label htmlFor="masking-checkbox" className="text-sm font-medium">Apply Time Masking</label>
            </div>
            {selectedPerturbations.timeMasking && (
              <div className="space-y-3 pl-6">
                <div className="flex items-center justify-between">
                  <span className="text-xs">Mask Region</span>
                  <Badge variant="outline" className="text-xs border-blue-300 text-blue-700">{maskRange[0]}% - {maskRange[1]}%</Badge>
                </div>
                <RangeSlider value={maskRange} onValueChange={setMaskRange} min={0} max={100} step={1} className="w-full" formatLabel={(value) => `${value}%`} />
              </div>
            )}
          </div>

          {/* Pitch Shift */}
          <div className="space-y-3 p-3 border rounded-lg">
            <div className="flex items-center space-x-2">
              <Checkbox id="pitch-checkbox" checked={selectedPerturbations.pitchShift} onCheckedChange={() => handlePerturbationToggle('pitchShift')} className="border-blue-400 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600" />
              <Plus className="h-4 w-4 text-blue-600" />
              <label htmlFor="pitch-checkbox" className="text-sm font-medium">Apply Pitch Shift</label>
            </div>
            {selectedPerturbations.pitchShift && (
              <div className="space-y-2 pl-6">
                <div className="flex items-center justify-between">
                  <span className="text-xs">Pitch Shift</span>
                  <Badge variant="outline" className="text-xs border-blue-300 text-blue-700">{pitchShift[0] > 0 ? "+" : ""}{pitchShift[0]} semitones</Badge>
                </div>
                <Slider value={pitchShift} onValueChange={setPitchShift} min={-6} max={6} step={1} className="w-full [&_[role=slider]]:border-blue-500 [&_[role=slider]]:bg-blue-600" />
              </div>
            )}
          </div>

          {/* Time Stretch - Hidden for Whisper */}
          {!model?.includes('whisper') && (
            <div className="space-y-3 p-3 border rounded-lg">
              <div className="flex items-center space-x-2">
                <Checkbox id="time-checkbox" checked={selectedPerturbations.timeStretch} onCheckedChange={() => handlePerturbationToggle('timeStretch')} className="border-blue-400 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600" />
                <Play className="h-4 w-4 text-blue-600" />
                <label htmlFor="time-checkbox" className="text-sm font-medium">Apply Time Stretch</label>
              </div>
              {selectedPerturbations.timeStretch && (
                <div className="space-y-2 pl-6">
                  <div className="flex items-center justify-between">
                    <span className="text-xs">Time Stretch</span>
                    <Badge variant="outline" className="text-xs border-blue-300 text-blue-700">{timeStretch[0]}%</Badge>
                  </div>
                  <Slider value={timeStretch} onValueChange={setTimeStretch} min={50} max={200} step={5} className="w-full [&_[role=slider]]:border-blue-500 [&_[role=slider]]:bg-blue-600" />
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="pt-4">
          <Button onClick={handleAddPerturbations} disabled={isProcessing || !selectedFile || !Object.values(selectedPerturbations).some(Boolean)} className="w-full h-10 bg-blue-600 hover:bg-blue-700 text-white font-medium shadow-md" size="lg">
            <Zap className="h-4 w-4 mr-2" />
            {isProcessing ? "Processing..." : "Apply Perturbations"}
          </Button>
        </CardContent>
      </Card>

      {(selectedFile || perturbationResult) && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Audio Waveforms</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {selectedFile && (
              <div className="space-y-2">
                <div className="text-xs font-medium flex items-center gap-2">Original Audio <Badge variant="outline" className="text-[10px]">O</Badge></div>
                <WaveformViewer audioUrl={getAudioUrl(selectedFile, dataset, originalDataset)} />
              </div>
            )}
            {perturbationResult && perturbationResult.success && (
              <div className="space-y-2">
                <div className="text-xs font-medium flex items-center gap-2">Perturbed Audio <Badge variant="secondary" className="text-[10px]">P</Badge></div>
                <WaveformViewer audioUrl={getPerturbedAudioUrl(perturbationResult.perturbed_file)} />
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {selectedFile && typeof selectedFile.duration === 'number' && selectedFile.duration > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Spectrogram Region Selector</CardTitle>
          </CardHeader>
          <CardContent>
            <SpectrogramGridSelector
              key={selectedFile.file_id}
              durationSec={selectedFile.duration}
              maxFreqHz={selectedFile.sample_rate ? selectedFile.sample_rate / 2 : DEFAULT_MAX_FREQ_HZ}
              onFrameCreated={(frame) => {
                setSpectrogramFrames((prev) => [...prev, frame]);
                setActiveFrameId(frame.id);
              }}
            />
            <p className="text-[10px] text-muted-foreground mt-2">
              Drag to mark a time-frequency region. Resolved bounds are logged to the console.
            </p>
          </CardContent>
        </Card>
      )}

      {spectrogramFrames.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Apply Mutation to Selected Region</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap gap-2">
              {spectrogramFrames.map((frame) => (
                <button
                  key={frame.id}
                  type="button"
                  onClick={() => setActiveFrameId(frame.id)}
                  className={`text-[10px] px-2 py-1 rounded border transition-colors ${
                    activeFrameId === frame.id
                      ? 'bg-blue-600 text-white border-blue-600'
                      : 'bg-muted text-muted-foreground border-border hover:border-blue-400'
                  }`}
                >
                  {(frame.startTimeMs / 1000).toFixed(2)}s–{(frame.endTimeMs / 1000).toFixed(2)}s · {Math.round(frame.startFreqHz)}–{Math.round(frame.endFreqHz)}Hz
                </button>
              ))}
            </div>

            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                size="sm"
                variant={frameMutationType === 'time_freq_mask' ? 'default' : 'outline'}
                onClick={() => setFrameMutationType('time_freq_mask')}
              >
                <VolumeX className="h-3.5 w-3.5 mr-1" /> Localized Mute
              </Button>
              <Button
                type="button"
                size="sm"
                variant={frameMutationType === 'band_pass_filter' ? 'default' : 'outline'}
                onClick={() => setFrameMutationType('band_pass_filter')}
              >
                <Filter className="h-3.5 w-3.5 mr-1" /> Frequency Filter Band
              </Button>
              <Button
                type="button"
                size="sm"
                variant={frameMutationType === 'noise' ? 'default' : 'outline'}
                onClick={() => setFrameMutationType('noise')}
              >
                <Volume2 className="h-3.5 w-3.5 mr-1" /> Gaussian White Noise
              </Button>
            </div>

            {frameMutationType === 'noise' && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs">Noise Level</span>
                  <Badge variant="outline" className="text-xs border-blue-300 text-blue-700">{frameNoiseLevel[0]}%</Badge>
                </div>
                <Slider value={frameNoiseLevel} onValueChange={setFrameNoiseLevel} max={50} step={1} className="w-full [&_[role=slider]]:border-blue-500 [&_[role=slider]]:bg-blue-600" />
              </div>
            )}

            <Button
              onClick={handleApplyFrameMutation}
              disabled={isProcessing || !activeFrame}
              className="w-full h-9 bg-blue-600 hover:bg-blue-700 text-white font-medium"
              size="sm"
            >
              <Zap className="h-3.5 w-3.5 mr-1" />
              {isProcessing ? "Processing..." : "Apply Mutation"}
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
