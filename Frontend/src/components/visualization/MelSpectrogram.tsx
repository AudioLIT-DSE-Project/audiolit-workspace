import React, { useEffect, useRef } from "react";
import { usePlayback } from "@/contexts/PlaybackContext";
import { getHeatmapColor } from "@/lib/heatmap";
import { hzToAxisFraction } from "@/lib/melScale";
import { formatHz, frequencyTicks, renderSpectrogramImage, timeTicks } from "@/lib/spectrogramImage";

interface MelSpectrogramProps {
  /** `[mel_bin][frame]`, values 0..1, lowest band first (librosa order). */
  spectrogram: number[][];
  durationSec: number;
  /** Top of the mel axis: half the sample rate the spectrogram was computed at. */
  maxFreqHz: number;
  height?: number;
  /** Width reserved for the Hz labels; set it to line up with a chart below. */
  axisWidth?: number;
  /** Right-hand inset, for the same reason. */
  rightInset?: number;
}

const COLOURBAR = `linear-gradient(to right, ${[0, 0.25, 0.5, 0.75, 1]
  .map((v) => {
    const [r, g, b] = getHeatmapColor(v);
    return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
  })
  .join(", ")})`;

// The log-mel spectrogram on its own (FR10.1), with the playhead every other
// time-aligned view shares (FR10.2). Until this existed the spectrogram was
// only ever a grey underlay behind an attribution heatmap.
export const MelSpectrogram: React.FC<MelSpectrogramProps> = ({
  spectrogram,
  durationSec,
  maxFreqHz,
  height = 160,
  axisWidth = 40,
  rightInset = 0,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { currentTime, seek } = usePlayback();

  useEffect(() => {
    const canvas = canvasRef.current;
    const image = renderSpectrogramImage(spectrogram);
    if (!canvas || !image) return;
    // One canvas pixel per spectrogram cell; CSS stretches it to the plot area.
    canvas.width = image.width;
    canvas.height = image.height;
    canvas.getContext("2d")?.drawImage(image, 0, 0);
  }, [spectrogram]);

  const playheadPercent =
    durationSec > 0 ? Math.min(100, Math.max(0, (currentTime / durationSec) * 100)) : 0;

  return (
    <div
      role="img"
      aria-label={`Log-mel spectrogram, ${durationSec.toFixed(1)} seconds, 0 to ${Math.round(maxFreqHz)} hertz`}
    >
      <div className="flex" style={{ paddingRight: rightInset }}>
        <div className="relative shrink-0" style={{ width: axisWidth, height }}>
          {frequencyTicks(maxFreqHz).map((hz) => (
            <span
              key={hz}
              className="absolute right-1.5 -translate-y-1/2 text-[10px] leading-none text-muted-foreground tabular-nums"
              style={{ top: `${(1 - hzToAxisFraction(hz, maxFreqHz)) * 100}%` }}
            >
              {formatHz(hz)}
            </span>
          ))}
        </div>
        <div
          className="relative flex-1 min-w-0 overflow-hidden rounded-sm border border-border cursor-crosshair"
          style={{ height }}
          onClick={(e) => {
            if (!(durationSec > 0)) return;
            const rect = e.currentTarget.getBoundingClientRect();
            if (rect.width > 0) seek(((e.clientX - rect.left) / rect.width) * durationSec);
          }}
        >
          <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
          {durationSec > 0 && (
            <div
              className="absolute top-0 bottom-0 w-px bg-white/90 pointer-events-none"
              style={{ left: `${playheadPercent}%` }}
            />
          )}
        </div>
      </div>

      <div className="relative h-4 mt-0.5" style={{ marginLeft: axisWidth, marginRight: rightInset }}>
        {timeTicks(durationSec).map((t) => (
          <span
            key={t}
            className="absolute -translate-x-1/2 text-[10px] leading-none text-muted-foreground tabular-nums"
            style={{ left: `${(t / durationSec) * 100}%` }}
          >
            {t}s
          </span>
        ))}
      </div>

      <div
        className="flex items-center justify-end gap-1.5 mt-1 text-[10px] text-muted-foreground"
        style={{ paddingRight: rightInset }}
      >
        <span>Hz (mel scale)</span>
        <span aria-hidden="true">·</span>
        <span>quiet</span>
        <div className="h-2 w-20 rounded-sm" style={{ background: COLOURBAR }} aria-hidden="true" />
        <span>loud</span>
      </div>
    </div>
  );
};
