import React, { useEffect, useRef } from 'react';

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
  baseSpectrogram?: number[][];
  waveformData?: number[]; // Normalized amplitude values (0.0 to 1.0)
  xaiResults?: XAIResult[];
  f0Data?: F0Point[];
  activeMethod?: XAIMethod;
  width?: number;
  height?: number;
}

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
  waveformData = [],
  xaiResults = [],
  f0Data = [],
  activeMethod = 'saliency',
  width = 800,
  height = 400,
}) => {
  const baseCanvasRef = useRef<HTMLCanvasElement>(null);
  const waveCanvasRef = useRef<HTMLCanvasElement>(null);
  const f0CanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRefs = useRef<Record<XAIMethod, HTMLCanvasElement | null>>({
    saliency: null,
    shap: null,
    lime: null,
    ig: null,
  });

  const mapTimeToX = (timeMs: number) => (timeMs / 1000 / audioDuration) * width;
  const mapHzToY = (hz: number, maxFreq = 8000) => {
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

      const melBins = res.matrix.length;
      const timeFrames = res.matrix[0]?.length || 0;
      if (timeFrames === 0) return;
      
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
      <canvas ref={baseCanvasRef} width={width} height={height} className="absolute top-0 left-0" />
      <canvas ref={waveCanvasRef} width={width} height={height} className="absolute top-0 left-0 pointer-events-none" />
      {(['saliency', 'shap', 'lime', 'ig'] as XAIMethod[]).map((method) => (
        <canvas
          key={method}
          ref={(el) => (overlayRefs.current[method] = el)}
          width={width}
          height={height}
          className="absolute top-0 left-0 transition-opacity duration-150 ease-out"
          style={{ opacity: activeMethod === method ? 0.85 : 0, mixBlendMode: 'screen' }}
        />
      ))}
      <canvas ref={f0CanvasRef} width={width} height={height} className="absolute top-0 left-0 pointer-events-none" />
    </div>
  );
};
