import { getHeatmapColor } from "@/lib/heatmap";
import { hzToAxisFraction } from "@/lib/melScale";

export type SpectrogramPalette = "viridis" | "grey";

// A canvas dimension above ~32k pixels is silently refused by browsers, and a
// 15-minute clip has more frames than that. Columns are pooled down to this.
const MAX_COLUMNS = 4096;

/**
 * Render a `[mel_bin][frame]` matrix of 0..1 values to an offscreen canvas, one
 * pixel per cell, ready to be scaled onto a visible one with `drawImage`.
 *
 * Row 0 of the matrix is the lowest mel band (librosa's order) and row 0 of a
 * canvas is the top, so the rows are flipped here: low frequencies end up at
 * the bottom, where every axis and overlay in the app assumes they are.
 *
 * Returns null when there is nothing to draw or no 2-D context is available.
 */
export const renderSpectrogramImage = (
  matrix: number[][] | null | undefined,
  palette: SpectrogramPalette = "viridis",
): HTMLCanvasElement | null => {
  const melBins = matrix?.length ?? 0;
  const frames = matrix?.[0]?.length ?? 0;
  if (!matrix || melBins === 0 || frames === 0) return null;

  const stride = Math.ceil(frames / MAX_COLUMNS);
  const columns = Math.ceil(frames / stride);

  const canvas = document.createElement("canvas");
  canvas.width = columns;
  canvas.height = melBins;
  const ctx = canvas.getContext("2d");
  const image = ctx?.createImageData(columns, melBins);
  if (!ctx || !image?.data) return null;

  for (let y = 0; y < melBins; y++) {
    const row = matrix[melBins - 1 - y];
    for (let x = 0; x < columns; x++) {
      // Peak over the pooled frames, so a short transient is not averaged away.
      let value = 0;
      const end = Math.min(frames, (x + 1) * stride);
      for (let f = x * stride; f < end; f++) {
        if (row[f] > value) value = row[f];
      }
      const idx = (y * columns + x) * 4;
      if (palette === "grey") {
        const c = Math.floor(value * 255);
        image.data[idx] = c;
        image.data[idx + 1] = c;
        image.data[idx + 2] = c;
      } else {
        const [r, g, b] = getHeatmapColor(value);
        image.data[idx] = r;
        image.data[idx + 1] = g;
        image.data[idx + 2] = b;
      }
      image.data[idx + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);
  return canvas;
};

/** Hz gridlines that fall inside 0..maxHz, spaced to read well on a mel axis. */
export const frequencyTicks = (maxHz: number): number[] =>
  [0, 500, 1000, 2000, 4000, 8000, 16000].filter((hz) => hz <= maxHz);

/** Evenly spaced time ticks, in seconds, on a step that is easy to read. */
export const timeTicks = (durationSec: number, target = 6): number[] => {
  if (!(durationSec > 0)) return [];
  const step = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300].find(
    (candidate) => durationSec / candidate <= target,
  ) ?? 600;
  const ticks: number[] = [];
  for (let t = 0; t <= durationSec + 1e-9; t += step) ticks.push(Number(t.toFixed(2)));
  return ticks;
};

export const formatHz = (hz: number): string => (hz >= 1000 ? `${hz / 1000}k` : `${hz}`);

/**
 * The bottom of a mel spectrogram, up to `capHz`.
 *
 * Mel bands are evenly spaced on the mel axis, so the bands below `capHz` are
 * the first `fraction * bands` rows. Used where only part of the spectrum can
 * be acted on: the perturbation engine works on audio resampled to 16 kHz, so
 * nothing above 8 kHz exists for it to change.
 */
export const cropSpectrogramToMaxHz = (
  matrix: number[][] | null | undefined,
  fullMaxHz: number,
  capHz: number,
): number[][] | null => {
  if (!matrix || matrix.length === 0) return null;
  if (!(capHz < fullMaxHz)) return matrix;
  const rows = Math.max(1, Math.round(hzToAxisFraction(capHz, fullMaxHz) * matrix.length));
  return matrix.slice(0, rows);
};
