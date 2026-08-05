import React, { useEffect, useRef, useState } from 'react';

export type XAIMethod = 'saliency' | 'shap' | 'lime' | 'ig';

export interface XAIResult {
  method: XAIMethod;
  matrix: number[][]; // [mel_bins][time_frames]
  max_val?: number;
}

export interface F0Point {
  time_ms: number;
  freq_hz: number;
}

interface XAIOverlayCanvasProps {
  audioDuration: number; // in seconds
  baseSpectrogram?: number[][]; // [mel_bins][time_frames]
  xaiResults?: XAIResult[];
  f0Data?: F0Point[];
  activeMethod?: XAIMethod;
  width?: number;
  height?: number;
}

// Color palette mapping for heatmaps (Red -> Yellow -> Green -> Blue)
const getHeatmapColor = (value: number): [number, number, number] => {
  const v = Math.max(0, Math.min(1, value));
  if (v < 0.25) return [0, 0, 255 * (v * 4)]; // Blue to Cyan
  if (v < 0.5) return [0, 255 * ((v - 0.25) * 4), 255]; // Cyan to Green
  if (v < 0.75) return [255 * ((v - 0.5) * 4), 255, 255 - (255 * ((v - 0.5) * 4))]; // Green to Yellow
  return [255, 255 - (255 * ((v - 0.75) * 4)), 0]; // Yellow to Red
};

export const XAIOverlayCanvas: React.FC<XAIOverlayCanvasProps> = ({
  audioDuration,
  baseSpectrogram,
  xaiResults = [],
  f0Data = [],
  activeMethod = 'saliency',
  width = 800,
  height = 400,
}) => {
  const baseCanvasRef = useRef<HTMLCanvasElement>(null);
  const f0CanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRefs = useRef<Record<XAIMethod, HTMLCanvasElement | null>>({
    saliency: null,
    shap: null,
    lime: null,
    ig: null,
  });

  // Coordinate Map: Time (ms) -> Pixel X
  const mapTimeToX = (timeMs: number) => (timeMs / 1000 / audioDuration) * width;
  // Coordinate Map: Frequency (Hz) -> Pixel Y (Logarithmic scale for spectrograms)
  const mapHzToY = (hz: number, maxFreq = 8000) => {
    const mel = 2595 * Math.log10(1 + hz / 500);
    const maxMel = 2595 * Math.log10(1 + maxFreq / 500);
    return height - (mel / maxMel) * height;
  };

  // 1. Render Base Spectrogram / Amplitude Map
  useEffect(() => {
    if (!baseCanvasRef.current || !baseSpectrogram) return;
    const ctx = baseCanvasRef.current.getContext('2d');
    if (!ctx) return;

    const t0 = performance.now();
    ctx.clearRect(0, 0, width, height);
    
    const melBins = baseSpectrogram.length;
    const timeFrames = baseSpectrogram[0]?.length || 0;
    const imgData = ctx.createImageData(timeFrames, melBins);
    
    for (let y = 0; y < melBins; y++) {
      for (let x = 0; x < timeFrames; x++) {
        const val = baseSpectrogram[y][x];
        const idx = (y * timeFrames + x) * 4;
        // Grayscale mapping for base spectrogram
        const c = Math.floor(val * 255);
        imgData.data[idx] = c;
        imgData.data[idx + 1] = c;
        imgData.data[idx + 2] = c;
        imgData.data[idx + 3] = 255;
      }
    }
    
    // Use a temporary canvas to scale the image data to actual width/height
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = timeFrames;
    tempCanvas.height = melBins;
    tempCanvas.getContext('2d')!.putImageData(imgData, 0, 0);
    ctx.drawImage(tempCanvas, 0, 0, width, height);
    
    const dt = performance.now() - t0;
    console.log(`[XAI Canvas] Base spectrogram rendered in ${dt.toFixed(2)}ms`);
  }, [baseSpectrogram, width, height]);

  // 2. Render XAI Overlays (putImageData for sub-30ms performance)
  useEffect(() => {
    xaiResults.forEach((res) => {
      const canvas = overlayRefs.current[res.method];
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const t0 = performance.now();
      ctx.clearRect(0, 0, width, height);

      const melBins = res.matrix.length;
      const timeFrames = res.matrix[0]?.length || 0;
      const maxVal = res.max_val || 1.0;
      const imgData = ctx.createImageData(timeFrames, melBins);

      for (let y = 0; y < melBins; y++) {
        for (let x = 0; x < timeFrames; x++) {
          const normVal = res.matrix[y][x] / maxVal;
          const [r, g, b] = getHeatmapColor(normVal);
          const idx = (y * timeFrames + x) * 4;
          imgData.data[idx] = r;
          imgData.data[idx + 1] = g;
          imgData.data[idx + 2] = b;
          // Alpha based on intensity, allowing base spectrogram to show through
          imgData.data[idx + 3] = Math.floor(normVal * 200); 
        }
      }

      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = timeFrames;
      tempCanvas.height = melBins;
      tempCanvas.getContext('2d')!.putImageData(imgData, 0, 0);
      ctx.drawImage(tempCanvas, 0, 0, width, height);

      const dt = performance.now() - t0;
      console.log(`[XAI Canvas] ${res.method} overlay rendered in ${dt.toFixed(2)}ms`);
    });
  }, [xaiResults, width, height]);

  // 3. Render High-Contrast F0 Pitch Trajectory
  useEffect(() => {
    if (!f0CanvasRef.current) return;
    const ctx = f0CanvasRef.current.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);
    if (f0Data.length === 0) return;

    ctx.strokeStyle = '#00FFFF'; // Bright Cyan for high contrast
    ctx.lineWidth = 2;
    ctx.shadowColor = '#000000';
    ctx.shadowBlur = 4;
    ctx.beginPath();

    f0Data.forEach((point, idx) => {
      const x = mapTimeToX(point.time_ms);
      const y = mapHzToY(point.freq_hz);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    ctx.stroke();
  }, [f0Data, width, height, audioDuration]);

  return (
    <div className="relative bg-black rounded-lg overflow-hidden border border-border" style={{ width, height }}>
      {/* Layer 0: Base Spectrogram */}
      <canvas
        ref={baseCanvasRef}
        width={width}
        height={height}
        className="absolute top-0 left-0"
      />
      
      {/* Layer 1..N: XAI Overlays (Dynamic Opacity Swapping) */}
      {(['saliency', 'shap', 'lime', 'ig'] as XAIMethod[]).map((method) => (
        <canvas
          key={method}
          ref={(el) => (overlayRefs.current[method] = el)}
          width={width}
          height={height}
          className="absolute top-0 left-0 transition-opacity duration-150 ease-out"
          style={{
            opacity: activeMethod === method ? 1 : 0,
            mixBlendMode: 'screen', // Blends heatmap cleanly over the spectrogram
          }}
        />
      ))}

      {/* Layer Top: F0 Pitch Trajectory */}
      <canvas
        ref={f0CanvasRef}
        width={width}
        height={height}
        className="absolute top-0 left-0 pointer-events-none"
      />
    </div>
  );
};
