import { axisFractionToHz, hzToAxisFraction, hzToMel, melToHz } from "@/lib/melScale";
import { cropSpectrogramToMaxHz, frequencyTicks, renderSpectrogramImage, timeTicks } from "@/lib/spectrogramImage";

describe("melScale (librosa's default, Slaney)", () => {
  it("matches librosa.hz_to_mel reference values", () => {
    // librosa.hz_to_mel([0, 440, 1000, 4000, 8000])
    expect(hzToMel(0)).toBeCloseTo(0, 6);
    expect(hzToMel(440)).toBeCloseTo(6.6, 6);
    expect(hzToMel(1000)).toBeCloseTo(15, 6);
    expect(hzToMel(4000)).toBeCloseTo(35.163, 2);
    expect(hzToMel(8000)).toBeCloseTo(45.245, 2);
  });

  it("round-trips across the linear and the logarithmic region", () => {
    for (const hz of [0, 120, 999, 1000, 1001, 3400, 8000, 22050]) {
      expect(melToHz(hzToMel(hz))).toBeCloseTo(hz, 6);
    }
  });

  it("puts 0 Hz at the bottom of the axis and the ceiling at the top", () => {
    expect(hzToAxisFraction(0, 8000)).toBe(0);
    expect(hzToAxisFraction(8000, 8000)).toBe(1);
    expect(hzToAxisFraction(1000, 8000)).toBeCloseTo(15 / 45.245, 3);
  });

  it("inverts pixel fraction back to the same frequency", () => {
    for (const hz of [200, 1000, 2500, 7000]) {
      expect(axisFractionToHz(hzToAxisFraction(hz, 8000), 8000)).toBeCloseTo(hz, 6);
    }
  });

  it("clamps out-of-range input instead of extrapolating", () => {
    expect(hzToAxisFraction(20000, 8000)).toBe(1);
    expect(axisFractionToHz(-0.5, 8000)).toBe(0);
    expect(axisFractionToHz(1.5, 8000)).toBeCloseTo(8000, 6);
  });
});

describe("spectrogram image", () => {
  const putImageData = jest.fn();

  beforeEach(() => {
    putImageData.mockReset();
    jest.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(
      () =>
        ({
          createImageData: (w: number, h: number) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
          putImageData,
        }) as unknown as CanvasRenderingContext2D,
    );
  });

  afterEach(() => jest.restoreAllMocks());

  it("draws the lowest mel band on the bottom row", () => {
    // Two bands, one frame: band 0 (low) silent, band 1 (high) at full scale.
    const canvas = renderSpectrogramImage([[0], [1]], "grey");
    expect(canvas).not.toBeNull();
    const { data } = putImageData.mock.calls[0][0] as ImageData;
    expect(data[0]).toBe(255); // top row is the high band
    expect(data[4]).toBe(0); // bottom row is the low band
  });

  it("pools columns so a long clip stays inside canvas size limits", () => {
    const frames = 10000;
    const row = new Array(frames).fill(0);
    row[5001] = 1; // a single-frame transient must survive the pooling
    const canvas = renderSpectrogramImage([row], "grey") as HTMLCanvasElement;
    expect(canvas.width).toBeLessThanOrEqual(4096);
    const { data } = putImageData.mock.calls[0][0] as ImageData;
    expect(Array.from(data).some((v, i) => i % 4 === 0 && v === 255)).toBe(true);
  });

  it("returns null for an empty matrix", () => {
    expect(renderSpectrogramImage([])).toBeNull();
    expect(renderSpectrogramImage(undefined)).toBeNull();
  });

  it("only offers frequency ticks that exist on the axis", () => {
    expect(frequencyTicks(8000)).toEqual([0, 500, 1000, 2000, 4000, 8000]);
    expect(frequencyTicks(4000)).not.toContain(8000);
  });

  it("keeps a long clip to a handful of time ticks", () => {
    expect(timeTicks(5)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(timeTicks(600).length).toBeLessThanOrEqual(7);
    expect(timeTicks(0)).toEqual([]);
  });
});

describe("cropSpectrogramToMaxHz", () => {
  const matrix = Array.from({ length: 128 }, (_, band) => [band]);

  it("keeps the bands below the cap, lowest first", () => {
    // 8 kHz sits at mel 45.25 of 62.6 on a 24 kHz axis: 72% of the bands.
    const cropped = cropSpectrogramToMaxHz(matrix, 24000, 8000) as number[][];
    expect(cropped.length).toBe(Math.round(hzToAxisFraction(8000, 24000) * 128));
    expect(cropped.length).toBeLessThan(128);
    expect(cropped[0]).toEqual([0]);
  });

  it("leaves a spectrogram that already stops at the cap untouched", () => {
    expect(cropSpectrogramToMaxHz(matrix, 8000, 8000)).toBe(matrix);
  });

  it("returns null when there is no spectrogram", () => {
    expect(cropSpectrogramToMaxHz(undefined, 24000, 8000)).toBeNull();
  });
});
