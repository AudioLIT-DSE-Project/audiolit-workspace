import React, { useEffect, useRef } from 'react';
import { usePlayback } from '@/contexts/PlaybackContext';

export type XAIMethod = 'gradcam' | 'integrated_gradients' | 'lime' | 'shap';

export interface XAIResult {
  method: XAIMethod;
  matrix?: number[][]; // [mel_bins][time_frames]
  saliency_matrix?: number[][]; // API alias from /saliency/generate
  max_val?: number;
}

export interface F0Point {
  time_ms: number;
  // null on unvoiced frames. pYIN reports no pitch for silence and noise, and
  // drawing through those gaps would invent a pitch track the model never saw.
  freq_hz: number | null;
}

interface XAIOverlayCanvasProps {
  audioDuration: number; // in seconds
  baseSpectrogram?: number[][];
  waveformData?: number[]; // Normalized amplitude values (0.0 to 1.0)
  xaiResults?: XAIResult[];
  f0Data?: F0Point[];
  activeMethod?: XAIMethod;
  // Upper bound of the spectrogram's mel axis. librosa defaults fmax to sr/2,
  // so a fixed 8000 puts the F0 line at the wrong height on any other rate.
  maxFreqHz?: number;
  /** Overlay transparency, 0-1 (FR8.4 requires it be adjustable). */
  overlayOpacity?: number;
  width?: number;
  height?: number;
}

// Viridis, 32 control points, linearly interpolated (FR8.4).
//
// The previous ramp was blue->cyan->green->yellow->red - the jet family. Jet's
// luminance is non-monotonic, so it invents banding that is not in the data,
// loses ordering in greyscale print, and is not colourblind-safe. The SRS names
// "perceptually uniform, accessible" explicitly, so this is a stated
// requirement rather than a preference.
const VIRIDIS: [number, number, number][] = [
  [68, 1, 84], [71, 13, 96], [72, 24, 106], [72, 35, 116], [71, 45, 123],
  [69, 55, 129], [66, 64, 134], [62, 73, 137], [59, 82, 139], [55, 91, 141],
  [51, 99, 141], [47, 107, 142], [44, 114, 142], [41, 122, 142], [38, 130, 142],
  [35, 137, 142], [33, 145, 140], [31, 152, 139], [31, 160, 136], [34, 167, 133],
  [40, 174, 128], [51, 182, 122], [64, 189, 114], [80, 196, 105], [99, 203, 95],
  [119, 209, 83], [141, 215, 68], [164, 220, 53], [187, 225, 39], [210, 229, 34],
  [232, 233, 39], [253, 231, 37],
];

const getHeatmapColor = (value: number): [number, number, number] => {
  const v = Math.max(0, Math.min(1, value));
  const pos = v * (VIRIDIS.length - 1);
  const i = Math.floor(pos);
  const j = Math.min(i + 1, VIRIDIS.length - 1);
  const t = pos - i;
  return [
    VIRIDIS[i][0] + (VIRIDIS[j][0] - VIRIDIS[i][0]) * t,
    VIRIDIS[i][1] + (VIRIDIS[j][1] - VIRIDIS[i][1]) * t,
    VIRIDIS[i][2] + (VIRIDIS[j][2] - VIRIDIS[i][2]) * t,
  ];
};

/** Relative luminance, for the monotonicity test FR8.4 actually requires. */
export const heatmapLuminance = (v: number): number => {
  const [r, g, b] = getHeatmapColor(v);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};

export const XAIOverlayCanvas: React.FC<XAIOverlayCanvasProps> = ({
  audioDuration,
  baseSpectrogram,
  waveformData = [],
  xaiResults = [],
  f0Data = [],
  activeMethod = 'gradcam',
  maxFreqHz = 8000,
  overlayOpacity = 0.7,
  width = 800,
  height = 400,
}) => {
  // FR10.2: one shared playhead across the player, profiler and this overlay.
  const { currentTime, duration: playDuration, seek } = usePlayback();
  const baseCanvasRef = useRef<HTMLCanvasElement>(null);
  const waveCanvasRef = useRef<HTMLCanvasElement>(null);
  const f0CanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRefs = useRef<Record<XAIMethod, HTMLCanvasElement | null>>({
    gradcam: null,
    integrated_gradients: null,
    lime: null,
    shap: null,
  });

  const mapTimeToX = (timeMs: number) => (timeMs / 1000 / audioDuration) * width;
  const mapHzToY = (hz: number, maxFreq = maxFreqHz) => {
    const mel = 2595 * Math.log10(1 + hz / 500);
    const maxMel = 2595 * Math.log10(1 + maxFreq / 500);
    return height - (mel / maxMel) * height;
  };

  // 1. Render Base Spectrogram
  useEffect(() => {
    if (!baseCanvasRef.current || !baseSpectrogram) return;
    const ctx = baseCanvasRef.current.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    
    const melBins = baseSpectrogram.length;
    const timeFrames = baseSpectrogram[0]?.length || 0;
    if (timeFrames === 0) return;
    
    const imgData = ctx.createImageData(timeFrames, melBins);
    for (let y = 0; y < melBins; y++) {
      for (let x = 0; x < timeFrames; x++) {
        const val = baseSpectrogram[y][x];
        const idx = (y * timeFrames + x) * 4;
        const c = Math.floor(val * 255);
        imgData.data[idx] = c;
        imgData.data[idx + 1] = c;
        imgData.data[idx + 2] = c;
        imgData.data[idx + 3] = 255;
      }
    }
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = timeFrames;
    tempCanvas.height = melBins;
    tempCanvas.getContext('2d')!.putImageData(imgData, 0, 0);
    ctx.drawImage(tempCanvas, 0, 0, width, height);
  }, [baseSpectrogram, width, height]);

  // 2. Render Waveform Overlay (Raw Amplitude Map)
  useEffect(() => {
    if (!waveCanvasRef.current) return;
    const ctx = waveCanvasRef.current.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    if (waveformData.length === 0) return;

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    
    const step = width / waveformData.length;
    waveformData.forEach((amp, i) => {
      const x = i * step;
      const y = height / 2 - (amp * (height / 2)); 
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }, [waveformData, width, height]);

  // 3. Render XAI Overlays (putImageData for sub-30ms performance)
  useEffect(() => {
    xaiResults.forEach((res) => {
      const canvas = overlayRefs.current[res.method];
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, width, height);

      const targetMatrix = res.saliency_matrix || res.matrix;
      if (!targetMatrix || targetMatrix.length === 0) return;

      const melBins = targetMatrix.length;
      const timeFrames = targetMatrix[0]?.length || 0;
      if (timeFrames === 0) return;
      
      let maxVal = res.max_val;
      if (maxVal === undefined || maxVal <= 0) {
        let currentMax = 0;
        for (let y = 0; y < melBins; y++) {
          for (let x = 0; x < timeFrames; x++) {
            if (targetMatrix[y][x] > currentMax) currentMax = targetMatrix[y][x];
          }
        }
        maxVal = currentMax > 0 ? currentMax : 1.0;
      }

      const imgData = ctx.createImageData(timeFrames, melBins);

      for (let y = 0; y < melBins; y++) {
        for (let x = 0; x < timeFrames; x++) {
          const normVal = targetMatrix[y][x] / maxVal;
          const [r, g, b] = getHeatmapColor(normVal);
          const idx = (y * timeFrames + x) * 4;
          imgData.data[idx] = r;
          imgData.data[idx + 1] = g;
          imgData.data[idx + 2] = b;
          imgData.data[idx + 3] = Math.floor(normVal * 200); 
        }
      }

      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = timeFrames;
      tempCanvas.height = melBins;
      tempCanvas.getContext('2d')!.putImageData(imgData, 0, 0);
      ctx.drawImage(tempCanvas, 0, 0, width, height);
    });
  }, [xaiResults, width, height]);

  // 4. Render F0 Pitch Trajectory
  useEffect(() => {
    if (!f0CanvasRef.current) return;
    const ctx = f0CanvasRef.current.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    if (f0Data.length === 0) return;

    ctx.strokeStyle = '#00FFFF';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#000000';
    ctx.shadowBlur = 4;
    ctx.beginPath();

    // Pen up on every unvoiced frame, so the contour shows where pitch was
    // actually measured rather than a continuous line across silence.
    let penDown = false;
    f0Data.forEach((point) => {
      if (point.freq_hz === null || point.freq_hz === undefined || !isFinite(point.freq_hz)) {
        penDown = false;
        return;
      }
      const x = mapTimeToX(point.time_ms);
      const y = mapHzToY(point.freq_hz);
      if (!penDown) {
        ctx.moveTo(x, y);
        penDown = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  }, [f0Data, width, height, audioDuration, maxFreqHz]);

  return (
    <div className="relative bg-black rounded-lg overflow-hidden border border-border" style={{ width, height }}>
      <canvas ref={baseCanvasRef} width={width} height={height} className="absolute top-0 left-0" />
      <canvas ref={waveCanvasRef} width={width} height={height} className="absolute top-0 left-0 pointer-events-none" />
      {(['gradcam', 'integrated_gradients', 'lime', 'shap'] as XAIMethod[]).map((method) => (
        <canvas
          key={method}
          ref={(el) => (overlayRefs.current[method] = el)}
          width={width}
          height={height}
          className="absolute top-0 left-0 transition-opacity duration-150 ease-out"
          style={{ opacity: activeMethod === method ? overlayOpacity : 0, mixBlendMode: 'screen' }}
        />
      ))}
      <canvas ref={f0CanvasRef} width={width} height={height} className="absolute top-0 left-0 pointer-events-none" />
      {/* Playhead. Click anywhere on the canvas to seek (FR10.2). */}
      <div
        className="absolute inset-0 cursor-crosshair"
        onClick={(e) => {
          const total = playDuration || audioDuration;
          if (!total) return;
          const rect = e.currentTarget.getBoundingClientRect();
          seek(((e.clientX - rect.left) / rect.width) * total);
        }}
      >
        {(playDuration || audioDuration) > 0 && (
          <div
            className="absolute top-0 bottom-0 w-px bg-white/90 pointer-events-none"
            style={{ left: `${Math.min(100, (currentTime / (playDuration || audioDuration)) * 100)}%` }}
          />
        )}
      </div>
      {/* Colourbar. A heatmap without a scale is not readable (FR8.4). */}
      <div className="absolute bottom-2 right-2 flex items-center gap-2 rounded bg-black/60 px-2 py-1">
        <span className="text-[10px] text-white/80">low</span>
        <div
          className="h-2 w-24 rounded-sm"
          style={{
            background: `linear-gradient(to right, ${[0, 0.25, 0.5, 0.75, 1]
              .map((v) => {
                const [r, g, b] = getHeatmapColor(v);
                return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
              })
              .join(', ')})`,
          }}
        />
        <span className="text-[10px] text-white/80">high</span>
      </div>
    </div>
  );
};
