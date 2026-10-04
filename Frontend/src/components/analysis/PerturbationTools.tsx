"use client"

import React, { useState, useEffect, useCallback, useMemo, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Checkbox } from "@/components/ui/checkbox"
import { Volume2, VolumeX, Filter, Plus, Play, Zap, XCircle } from "lucide-react"
import { WaveformViewer, WaveformSelection } from "../audio/WaveformViewer"
import { API_BASE } from '@/lib/api'
import { useTaskStatus } from '@/hooks/useTaskStatus'
import { GlobalTaskProgress } from '../layout/GlobalTaskProgress'
import { isUploadedAudio } from "@/lib/audioSelection";
import { axisFractionToHz, hzToAxisFraction } from "@/lib/melScale";
import { getModelTaskFamily } from "@/lib/modelTask";
import { cropSpectrogramToMaxHz, frequencyTicks, renderSpectrogramImage, timeTicks } from "@/lib/spectrogramImage";

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
  /**
   * The selected clip's acoustic profile. Supplies the spectrogram the region
   * selector is drawn over, and the clip's duration and sample rate - which a
   * dataset row does not carry, so without it the selector never rendered for
   * anything but an upload.
   */
  acousticProfile?: {
    sample_rate?: number;
    duration_s?: number;
    spectrogram?: number[][];
  } | null;
}

/**
 * The body of POST /api/inference/mutation for the selected clip.
 *
 * `dataset` is sent only for a dataset row. It used to be sent for uploads and
 * live recordings as well, and the backend then looked the upload up inside
 * that corpus and reported it missing.
 */
const buildMutationRequest = (
  selectedFile: UploadedFile,
  perturbations: Array<{ type: string; params: Record<string, any>; region?: Record<string, number> }>,
  dataset?: string,
  originalDataset?: string,
) => {
  const isUploaded = isUploadedAudio(selectedFile, dataset);
  return {
    audio_ref: isUploaded ? selectedFile.file_path : selectedFile.filename,
    mutation: isUploaded
      ? { perturbations, is_uploaded: true }
      : { perturbations, is_uploaded: false, dataset: originalDataset || dataset },
  };
};

const MODEL_FAMILY_TASK = { ASR: 'asr', SER: 'ser', DEEPFAKE: 'add' } as const;

/** The one line a re-run on the mutated clip is summarised as. */
const summarisePrediction = (aggregated: any): string => {
  const tasks = aggregated?.tasks ?? {};
  if (tasks.asr?.transcript) return tasks.asr.transcript;
  if (tasks.ser?.predicted_emotion) return tasks.ser.predicted_emotion;
  if (tasks.add) return tasks.add.predicted_label || tasks.add.label || '';
  return '';
};

const getAudioUrl = (selectedFile: UploadedFile, dataset?: string, originalDataset?: string): string => {
  const sfAny = selectedFile as any;
  const isUploadedFile = isUploadedAudio(selectedFile, dataset);
  
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
// frequency (sample_rate / 2). The frequency axis is the same mel scale the
// spectrogram behind the selection is drawn on (lib/melScale), so a box drawn
// over a feature resolves to that feature's band.
const pixelXToTimeMs = (x: number, width: number, durationSec: number) =>
  width > 0 ? (x / width) * durationSec * 1000 : 0;

const pixelYToFreqHz = (y: number, height: number, maxFreqHz: number) =>
  height > 0 ? axisFractionToHz((height - y) / height, maxFreqHz) : 0;

// Signal -> pixel, used to redraw persisted frames and grid labels.
const timeMsToPixelX = (timeMs: number, width: number, durationSec: number) =>
  durationSec > 0 ? (timeMs / 1000 / durationSec) * width : 0;

const freqHzToPixelY = (hz: number, height: number, maxFreqHz: number) =>
  height - hzToAxisFraction(hz, maxFreqHz) * height;

interface SpectrogramGridSelectorProps {
  durationSec: number;
  maxFreqHz?: number;
  height?: number;
  /** `[mel_bin][frame]`, 0..1. Drawn behind the grid when present. */
  spectrogram?: number[][] | null;
  /**
   * The one region to display. Passing it (including `null`) makes the
   * selector controlled, so it can show a region chosen on the waveform;
   * leaving it out keeps every drawn frame on screen, as before.
   */
  selection?: SpectrogramBoundaryFrame | null;
  onFrameCreated?: (frame: SpectrogramBoundaryFrame) => void;
}

export const SpectrogramGridSelector: React.FC<SpectrogramGridSelectorProps> = ({
  durationSec,
  maxFreqHz = DEFAULT_MAX_FREQ_HZ,
  height = 160,
  spectrogram,
  selection,
  onFrameCreated,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const gridCanvasRef = useRef<HTMLCanvasElement>(null);
  const selectionCanvasRef = useRef<HTMLCanvasElement>(null);
  const dragStateRef = useRef<{ startX: number; startY: number; currentX: number; currentY: number; dragging: boolean } | null>(null);
  const rafIdRef = useRef<number | null>(null);
  const [frames, setFrames] = useState<SpectrogramBoundaryFrame[]>([]);
  const isControlled = selection !== undefined;
  const framesRef = useRef(frames);
  framesRef.current = !isControlled ? frames : selection ? [selection] : [];
  const onFrameCreatedRef = useRef(onFrameCreated);
  onFrameCreatedRef.current = onFrameCreated;
  // Rendered once per clip, not once per resize.
  const spectrogramImage = useMemo(() => renderSpectrogramImage(spectrogram), [spectrogram]);

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

    // The clip's own spectrogram, so there is something to select against.
    // This canvas used to be the grid alone: a region had to be drawn blind.
    if (spectrogramImage) ctx.drawImage(spectrogramImage, 0, 0, w, h);

    ctx.strokeStyle = 'rgba(226, 232, 240, 0.3)'; // slate-200 @ 30%
    ctx.fillStyle = 'rgba(248, 250, 252, 0.9)'; // slate-50 @ 90%
    ctx.font = '10px monospace';
    ctx.lineWidth = 1;
    // Labels sit on the image, so they carry their own contrast.
    ctx.shadowColor = 'rgba(0, 0, 0, 0.9)';
    ctx.shadowBlur = 3;

    // Frequency gridlines at round values, placed on the mel axis.
    for (const hz of frequencyTicks(maxFreqHz)) {
      const y = freqHzToPixelY(hz, h, maxFreqHz);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
      // 0 Hz shares the bottom-left corner with the first time label.
      if (hz > 0) ctx.fillText(`${Math.round(hz)} Hz`, 2, Math.max(9, y - 2));
    }

    // Time gridlines, on a step that keeps a long clip to a handful of labels.
    for (const t of timeTicks(durationSec, 8)) {
      const x = timeMsToPixelX(t * 1000, w, durationSec);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
      // Keep the last label inside the canvas instead of clipping it.
      const label = `${t}s`;
      const labelWidth = ctx.measureText(label).width;
      ctx.fillText(label, Math.min(x + 2, w - labelWidth - 2), h - 2);
    }
    ctx.restore();
  }, [durationSec, maxFreqHz, spectrogramImage]);

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

    // The selection sits on a spectrogram whose colours run from dark purple
    // to yellow, so no single hue stands out everywhere. A white box with a
    // dark outline around it does: it was green on green before, and could
    // not be seen over voiced speech.
    framesRef.current.forEach((frame) => {
      const x1 = timeMsToPixelX(frame.startTimeMs, w, durationSec);
      const x2 = timeMsToPixelX(frame.endTimeMs, w, durationSec);
      const y1 = freqHzToPixelY(frame.endFreqHz, h, maxFreqHz);
      const y2 = freqHzToPixelY(frame.startFreqHz, h, maxFreqHz);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.22)';
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.lineWidth = 4;
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.75)';
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ffffff';
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    });

    // Active drag, drawn on top while the mouse is still down.
    const drag = dragStateRef.current;
    if (drag) {
      const x1 = Math.min(drag.startX, drag.currentX);
      const x2 = Math.max(drag.startX, drag.currentX);
      const y1 = Math.min(drag.startY, drag.currentY);
      const y2 = Math.max(drag.startY, drag.currentY);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 4;
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.75)';
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ffffff';
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
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
        if (!isControlled) setFrames((prev) => [...prev, frame]);
        onFrameCreatedRef.current?.(frame);
      }
      dragStateRef.current = null;
    }

    drawSelections();
  }, [drawSelections, durationSec, maxFreqHz, handleWindowMouseMove, isControlled]);

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
  }, [frames, selection, drawSelections]);

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

/**
 * The region the perturbations will be applied to. There is one, shared by the
 * waveform and the spectrogram: drawn on the waveform it is a time span across
 * every frequency (`band` absent); drawn on the spectrogram it is a box.
 */
export interface MutationRegion {
  startTimeMs: number;
  endTimeMs: number;
  band?: { lowHz: number; highHz: number };
}

// The perturbation engine resamples every clip to 16 kHz before it touches it,
// so 8 kHz is the top of what a mutation can change. The selector stops there:
// a box drawn above it would select audio the engine has already discarded.
const ENGINE_MAX_FREQ_HZ = 8000;

type PerturbationKey = 'noise' | 'mute' | 'bandPass' | 'pitchShift' | 'timeStretch';

/** "0.50–1.50 s · 300–2500 Hz", or "· all frequencies" for a time-only span. */
const describeRegion = (region: MutationRegion): string => {
  const span = `${(region.startTimeMs / 1000).toFixed(2)}–${(region.endTimeMs / 1000).toFixed(2)} s`;
  return region.band
    ? `${span} · ${Math.round(region.band.lowHz)}–${Math.round(region.band.highHz)} Hz`
    : `${span} · all frequencies`;
};

export const PerturbationTools: React.FC<PerturbationToolsProps> = ({
  selectedFile,
  onPerturbationComplete,
  onPredictionRefresh,
  model,
  dataset,
  originalDataset,
  acousticProfile,
}) => {
  const [noiseLevel, setNoiseLevel] = useState([10])
  const [pitchShift, setPitchShift] = useState([2])
  const [timeStretch, setTimeStretch] = useState([110])

  const [selectedPerturbations, setSelectedPerturbations] = useState<Record<PerturbationKey, boolean>>({
    noise: false,
    mute: false,
    bandPass: false,
    pitchShift: false,
    timeStretch: false,
  })

  const [error, setError] = useState<string | null>(null)
  const [perturbationResult, setPerturbationResult] = useState<PerturbationResult | null>(null)

  // The one selection, whichever view it was drawn on.
  const [region, setRegion] = useState<MutationRegion | null>(null)
  // The waveform's own duration, for a clip whose acoustic profile has not
  // arrived (or failed): the waveform can still be selected on.
  const [waveformDurationSec, setWaveformDurationSec] = useState(0)

  // Duration and sample rate. The acoustic profile is the authority (measured
  // from the audio); a dataset row carries neither on its own.
  const clipDurationSec = acousticProfile?.duration_s || selectedFile?.duration || waveformDurationSec || 0;
  const clipSampleRate = acousticProfile?.sample_rate || selectedFile?.sample_rate;
  const fullMaxFreqHz = clipSampleRate ? clipSampleRate / 2 : DEFAULT_MAX_FREQ_HZ;
  const selectorMaxFreqHz = Math.min(fullMaxFreqHz, ENGINE_MAX_FREQ_HZ);
  const selectorSpectrogram = useMemo(
    () => cropSpectrogramToMaxHz(acousticProfile?.spectrogram, fullMaxFreqHz, selectorMaxFreqHz),
    [acousticProfile?.spectrogram, fullMaxFreqHz, selectorMaxFreqHz],
  );

  // FR12.2 — client-side region preview, before any network call.
  //
  // AudioContext previously appeared only in a test mock: the suite asserted
  // against a feature that had never been built, which is how this stayed
  // invisible through several reviews. Decoding happens once per clip and is
  // cached; nothing here touches the backend.
  const audioCtxRef = useRef<AudioContext | null>(null);
  const bufferRef = useRef<{ url: string; buffer: AudioBuffer } | null>(null);
  const sourceRef = useRef<AudioBufferSourceNode | null>(null);
  const [previewing, setPreviewing] = useState<'region' | 'muted' | null>(null);

  const stopPreview = useCallback(() => {
    try {
      sourceRef.current?.stop();
    } catch {
      // already stopped
    }
    sourceRef.current = null;
    setPreviewing(null);
  }, []);

  useEffect(() => () => {
    stopPreview();
    audioCtxRef.current?.close().catch(() => undefined);
    audioCtxRef.current = null;
  }, [stopPreview]);

  const loadBuffer = useCallback(async (url: string) => {
    if (bufferRef.current?.url === url) return bufferRef.current.buffer;
    const Ctor = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = audioCtxRef.current ?? new Ctor();
    audioCtxRef.current = ctx;
    const bytes = await (await fetch(url)).arrayBuffer();
    const buffer = await ctx.decodeAudioData(bytes);
    bufferRef.current = { url, buffer };
    return buffer;
  }, []);

  /** Play only the selected time span, or the clip with that span silenced. */
  const previewRegion = useCallback(async (target: MutationRegion, mode: 'region' | 'muted') => {
    const url = getAudioUrl(selectedFile, dataset, originalDataset);
    if (!url) return;
    stopPreview();
    const ctx = audioCtxRef.current ?? new (window.AudioContext)();
    audioCtxRef.current = ctx;
    const buffer = await loadBuffer(url);

    const start = Math.max(0, target.startTimeMs / 1000);
    const end = Math.min(buffer.duration, target.endTimeMs / 1000);
    const source = ctx.createBufferSource();

    if (mode === 'region') {
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.start(0, start, Math.max(0.01, end - start));
    } else {
      // Copy, then zero the selected span: the counterfactual the mutation
      // will produce, auditioned locally first.
      const muted = ctx.createBuffer(buffer.numberOfChannels, buffer.length, buffer.sampleRate);
      for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
        const data = Float32Array.from(buffer.getChannelData(ch));
        data.fill(0, Math.floor(start * buffer.sampleRate), Math.floor(end * buffer.sampleRate));
        muted.copyToChannel(data, ch);
      }
      source.buffer = muted;
      source.connect(ctx.destination);
      source.start();
    }

    source.onended = () => setPreviewing(null);
    sourceRef.current = source;
    setPreviewing(mode);
  }, [selectedFile, dataset, originalDataset, loadBuffer, stopPreview]);

  // RQ Task IDs
  const [mutationTaskId, setMutationTaskId] = useState<string | null>(null)
  const [inferenceTaskId, setInferenceTaskId] = useState<string | null>(null)

  // Track mutation job state
  const { state: mutationState, result: mutationResult, error: mutationError } = useTaskStatus(mutationTaskId)
  // Track inference job state
  const { state: inferenceState, result: inferenceResult, error: inferenceError } = useTaskStatus(inferenceTaskId)

  useEffect(() => {
    setPerturbationResult(null);
    setError(null);
    setMutationTaskId(null);
    setInferenceTaskId(null);
    setRegion(null);
    setWaveformDurationSec(0);
  }, [selectedFile]);

  const handlePerturbationToggle = (perturbationType: PerturbationKey) => {
    setSelectedPerturbations(prev => ({ ...prev, [perturbationType]: !prev[perturbationType] }));
  }

  // A drag on the waveform: a time span, across every frequency.
  const handleWaveformSelection = useCallback((selection: WaveformSelection | null) => {
    if (!selection) {
      setRegion(null);
      return;
    }
    if (!(clipDurationSec > 0) || !(selection.containerWidth > 0)) return;
    const toMs = (x: number) => (x / selection.containerWidth) * clipDurationSec * 1000;
    setRegion({ startTimeMs: toMs(selection.startX), endTimeMs: toMs(selection.endX) });
  }, [clipDurationSec]);

  // A box on the spectrogram: a time span and a frequency band.
  const handleSpectrogramFrame = useCallback((frame: SpectrogramBoundaryFrame) => {
    setRegion({
      startTimeMs: frame.startTimeMs,
      endTimeMs: frame.endTimeMs,
      band: { lowHz: frame.startFreqHz, highHz: frame.endFreqHz },
    });
  }, []);

  // The same region, in each view's own terms.
  const waveformSelectionRange = region && clipDurationSec > 0
    ? { start: region.startTimeMs / 1000 / clipDurationSec, end: region.endTimeMs / 1000 / clipDurationSec }
    : null;
  const spectrogramSelection: SpectrogramBoundaryFrame | null = region
    ? {
        id: 'active-region',
        startTimeMs: region.startTimeMs,
        endTimeMs: region.endTimeMs,
        startFreqHz: region.band?.lowHz ?? 0,
        endFreqHz: region.band?.highHz ?? selectorMaxFreqHz,
      }
    : null;

  // Mute and band-pass only mean something on a region: muting a whole clip
  // leaves nothing to analyse, and a band-pass needs a band to keep.
  const available: Record<PerturbationKey, boolean> = {
    noise: true,
    mute: !!region,
    bandPass: !!region?.band,
    pitchShift: true,
    timeStretch: !model?.includes('whisper'),
  };
  const active = (key: PerturbationKey) => selectedPerturbations[key] && available[key];
  const anyActive = (Object.keys(available) as PerturbationKey[]).some(active);

  // Effect: When mutation job succeeds, trigger inference
  useEffect(() => {
    if (mutationState === 'SUCCESS' && mutationResult) {
      // Adapt result to expected shape
      const perturbedData = mutationResult as PerturbationResult;

      // The job finishing is not the mutation succeeding: the engine reports
      // a clip it could not read, or a perturbation it could not apply, in the
      // result. Both used to be treated as a derived clip.
      if (!perturbedData.success || !perturbedData.perturbed_file) {
        setError(perturbedData.error || "The mutation could not be applied to this clip.");
        setMutationTaskId(null);
        return;
      }
      const notApplied = (perturbedData.applied_perturbations || []).filter((p) => p.status !== 'applied');
      if (notApplied.length > 0) {
        setError(
          `Not applied: ${notApplied.map((p) => `${p.type.replace(/_/g, ' ')}${p.error ? ` (${p.error})` : ''}`).join(', ')}`,
        );
        if (notApplied.length === (perturbedData.applied_perturbations || []).length) {
          setMutationTaskId(null);
          return;
        }
      }

      setPerturbationResult(perturbedData);
      if (onPerturbationComplete) onPerturbationComplete(perturbedData);

      // Start inference on the perturbed file, with the selected model's task.
      // This was ASR for Whisper and SER for everything else, so a mutation on
      // a deepfake detector re-ran an emotion model.
      const task = MODEL_FAMILY_TASK[getModelTaskFamily(model || 'whisper-base')];
      const runInference = async () => {
        try {
          const response = await fetch(`${API_BASE}/api/inference/multitask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              audio_ref: perturbedData.perturbed_file,
              tasks: [task],
              model_ids: model ? { [task]: model } : {},
            })
          });
          if (response.ok) {
            const data = await response.json();
            setInferenceTaskId(data.job_id);
          } else {
            setError("Failed to enqueue inference job for perturbed audio.");
            setMutationTaskId(null);
          }
        } catch (err) {
          setError("Error triggering inference job.");
          setMutationTaskId(null);
        }
      };
      runInference();
    } else if (mutationState === 'FAILURE') {
      setError(mutationError || "Perturbation task failed in worker.");
      // Release the controls. The id was left set, so one failed job disabled
      // the Apply button until another clip was selected.
      setMutationTaskId(null);
    }
  }, [mutationState, mutationResult, mutationError]);

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
        onPredictionRefresh(perturbedFile, summarisePrediction(inferenceResult));
      }
      setInferenceTaskId(null);
      setMutationTaskId(null);
    } else if (inferenceState === 'FAILURE') {
      setError(inferenceError || "Inference task failed in worker.");
      setInferenceTaskId(null);
      setMutationTaskId(null);
    }
  }, [inferenceState, inferenceResult, inferenceError, perturbationResult]);

  /**
   * The checked perturbations, each scoped to the selected region when there
   * is one. Without a region they act on the whole clip.
   *
   * Order matters: the mute goes first so that noise added to the same region
   * is not silenced again, and the time stretch goes last because it changes
   * the clip's length and would move the region under everything after it.
   */
  const buildPerturbations = () => {
    const span = region ? { t_start_ms: region.startTimeMs, t_end_ms: region.endTimeMs } : null;
    const box = span && region?.band
      ? { ...span, f_low_hz: region.band.lowHz, f_high_hz: region.band.highHz }
      : span;
    const scoped = (scope: Record<string, number> | null) => (scope ? { region: scope } : {});

    const perturbations: Array<{ type: string; params: Record<string, any>; region?: Record<string, number> }> = [];
    if (active('mute') && span) {
      perturbations.push({
        type: "time_freq_mask",
        params: { ...span, f_low_hz: region?.band?.lowHz ?? 0, f_high_hz: region?.band?.highHz ?? selectorMaxFreqHz },
      });
    }
    if (active('bandPass') && region?.band) {
      perturbations.push({
        type: "band_pass_filter",
        params: { f_low_hz: region.band.lowHz, f_high_hz: region.band.highHz },
        ...scoped(box),
      });
    }
    if (active('noise')) perturbations.push({ type: "noise", params: { noise_level: noiseLevel[0] / 100.0 }, ...scoped(box) });
    // Pitch and tempo act on the whole spectrum, so they take the time span only.
    if (active('pitchShift')) perturbations.push({ type: "pitch_shift", params: { pitch_shift_semitones: pitchShift[0] }, ...scoped(span) });
    if (active('timeStretch')) perturbations.push({ type: "time_stretch", params: { stretch_factor: timeStretch[0] / 100.0 }, ...scoped(span) });
    return perturbations;
  };

  const handleApply = async () => {
    if (!selectedFile) { setError("No file selected"); return; }
    if (!anyActive) { setError("Please select at least one perturbation type"); return; }

    setError(null);

    try {
      // Enqueue mutation job via RQ
      const response = await fetch(`${API_BASE}/api/inference/mutation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // The session cookie: a custom dataset is resolved per session.
        credentials: "include",
        body: JSON.stringify(buildMutationRequest(selectedFile, buildPerturbations(), dataset, originalDataset)),
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
  const checkboxClass = "border-blue-400 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600";
  const sliderClass = "w-full [&_[role=slider]]:border-blue-500 [&_[role=slider]]:bg-blue-600";
  const timeOnlyNote = region?.band
    ? <p className="text-[10px] text-muted-foreground pl-6">Acts on the whole spectrum, so it uses the region's time span only.</p>
    : null;

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

      {/* 1. Region: the waveform and the spectrogram are two views of one selection. */}
      {selectedFile && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Region</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-xs font-medium flex items-center gap-2">Original Audio <Badge variant="outline" className="text-[10px]">O</Badge></div>
            <WaveformViewer
              audioUrl={getAudioUrl(selectedFile, dataset, originalDataset)}
              onReady={(wavesurfer) => setWaveformDurationSec(wavesurfer.getDuration() || 0)}
              onSelectionChange={handleWaveformSelection}
              selectionRange={waveformSelectionRange}
            />
            {clipDurationSec > 0 && (
              // Inset to match the waveform card's padding and border, so the
              // two time axes line up and a selection sits at the same x in both.
              <div className="px-[13px]">
                <SpectrogramGridSelector
                  key={selectedFile.file_id}
                  durationSec={clipDurationSec}
                  maxFreqHz={selectorMaxFreqHz}
                  spectrogram={selectorSpectrogram}
                  selection={spectrogramSelection}
                  onFrameCreated={handleSpectrogramFrame}
                />
              </div>
            )}

            {region ? (
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <Badge variant="outline" className="text-[11px] tabular-nums border-blue-300 text-blue-700">
                  {describeRegion(region)}
                </Badge>
                <Button type="button" size="sm" variant="outline" onClick={() => void previewRegion(region, 'region')}>
                  Preview region{previewing === 'region' ? '…' : ''}
                </Button>
                <Button type="button" size="sm" variant="outline" onClick={() => void previewRegion(region, 'muted')}>
                  Preview muted{previewing === 'muted' ? '…' : ''}
                </Button>
                {previewing && (
                  <Button type="button" size="sm" variant="ghost" onClick={stopPreview}>
                    Stop
                  </Button>
                )}
                <Button type="button" size="sm" variant="ghost" onClick={() => { stopPreview(); setRegion(null); }}>
                  Clear
                </Button>
              </div>
            ) : (
              <p className="text-[10px] text-muted-foreground">
                Drag on the waveform to select a time span, or on the spectrogram to select a time and frequency box. With no selection, perturbations apply to the whole clip.
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {/* 2. Perturbation configuration */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Perturbation Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Noise */}
          <div className="space-y-3 p-3 border rounded-lg">
            <div className="flex items-center space-x-2">
              <Checkbox id="noise-checkbox" checked={selectedPerturbations.noise} onCheckedChange={() => handlePerturbationToggle('noise')} className={checkboxClass} />
              <Volume2 className="h-4 w-4 text-blue-600" />
              <label htmlFor="noise-checkbox" className="text-sm font-medium">Add Gaussian Noise</label>
            </div>
            {selectedPerturbations.noise && (
              <div className="space-y-2 pl-6">
                <div className="flex items-center justify-between">
                  <span className="text-xs">Noise Level</span>
                  <Badge variant="outline" className="text-xs border-blue-300 text-blue-700">{noiseLevel[0]}%</Badge>
                </div>
                <Slider value={noiseLevel} onValueChange={setNoiseLevel} max={50} step={1} className={sliderClass} />
              </div>
            )}
          </div>

          {/* Mute */}
          <div className="space-y-2 p-3 border rounded-lg">
            <div className="flex items-center space-x-2">
              <Checkbox id="mute-checkbox" disabled={!available.mute} checked={active('mute')} onCheckedChange={() => handlePerturbationToggle('mute')} className={checkboxClass} />
              <VolumeX className="h-4 w-4 text-blue-600" />
              <label htmlFor="mute-checkbox" className={`text-sm font-medium ${available.mute ? '' : 'text-muted-foreground'}`}>Mute Region</label>
            </div>
            {!available.mute && <p className="text-[10px] text-muted-foreground pl-6">Select a region first.</p>}
          </div>

          {/* Band-pass */}
          <div className="space-y-2 p-3 border rounded-lg">
            <div className="flex items-center space-x-2">
              <Checkbox id="bandpass-checkbox" disabled={!available.bandPass} checked={active('bandPass')} onCheckedChange={() => handlePerturbationToggle('bandPass')} className={checkboxClass} />
              <Filter className="h-4 w-4 text-blue-600" />
              <label htmlFor="bandpass-checkbox" className={`text-sm font-medium ${available.bandPass ? '' : 'text-muted-foreground'}`}>Band-Pass Filter</label>
            </div>
            <p className="text-[10px] text-muted-foreground pl-6">
              {available.bandPass
                ? "Keeps only the selected frequency band for the region's duration."
                : "Draw the region on the spectrogram to choose a frequency band."}
            </p>
          </div>

          {/* Pitch Shift */}
          <div className="space-y-3 p-3 border rounded-lg">
            <div className="flex items-center space-x-2">
              <Checkbox id="pitch-checkbox" checked={selectedPerturbations.pitchShift} onCheckedChange={() => handlePerturbationToggle('pitchShift')} className={checkboxClass} />
              <Plus className="h-4 w-4 text-blue-600" />
              <label htmlFor="pitch-checkbox" className="text-sm font-medium">Apply Pitch Shift</label>
            </div>
            {selectedPerturbations.pitchShift && (
              <>
                <div className="space-y-2 pl-6">
                  <div className="flex items-center justify-between">
                    <span className="text-xs">Pitch Shift</span>
                    <Badge variant="outline" className="text-xs border-blue-300 text-blue-700">{pitchShift[0] > 0 ? "+" : ""}{pitchShift[0]} semitones</Badge>
                  </div>
                  <Slider value={pitchShift} onValueChange={setPitchShift} min={-6} max={6} step={1} className={sliderClass} />
                </div>
                {timeOnlyNote}
              </>
            )}
          </div>

          {/* Time Stretch - Hidden for Whisper */}
          {available.timeStretch && (
            <div className="space-y-3 p-3 border rounded-lg">
              <div className="flex items-center space-x-2">
                <Checkbox id="time-checkbox" checked={selectedPerturbations.timeStretch} onCheckedChange={() => handlePerturbationToggle('timeStretch')} className={checkboxClass} />
                <Play className="h-4 w-4 text-blue-600" />
                <label htmlFor="time-checkbox" className="text-sm font-medium">Apply Time Stretch</label>
              </div>
              {selectedPerturbations.timeStretch && (
                <>
                  <div className="space-y-2 pl-6">
                    <div className="flex items-center justify-between">
                      <span className="text-xs">Time Stretch</span>
                      <Badge variant="outline" className="text-xs border-blue-300 text-blue-700">{timeStretch[0]}%</Badge>
                    </div>
                    <Slider value={timeStretch} onValueChange={setTimeStretch} min={50} max={200} step={5} className={sliderClass} />
                  </div>
                  {timeOnlyNote}
                  {region && <p className="text-[10px] text-muted-foreground pl-6">Changes the clip's length, so the result no longer lines up in time with the original.</p>}
                </>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* 3. Apply */}
      <Card>
        <CardContent className="pt-4">
          <Button onClick={handleApply} disabled={isProcessing || !selectedFile || !anyActive} className="w-full h-10 bg-blue-600 hover:bg-blue-700 text-white font-medium shadow-md" size="lg">
            <Zap className="h-4 w-4 mr-2" />
            {isProcessing ? "Processing..." : region ? "Apply to selected region" : "Apply to whole clip"}
          </Button>
        </CardContent>
      </Card>

      {/* 4. Result */}
      {perturbationResult && perturbationResult.success && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Result</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-xs font-medium flex items-center gap-2">Perturbed Audio <Badge variant="secondary" className="text-[10px]">P</Badge></div>
            <WaveformViewer audioUrl={getPerturbedAudioUrl(perturbationResult.perturbed_file)} />
          </CardContent>
        </Card>
      )}
    </div>
  )
}
