/**
 * LIT-177 — 2D Spectrogram Grid Selector & Coordinate Resolution Handler.
 *
 * DoD: "Verification logs display correctly calculated timestamps and
 * frequency bounds ... immediately after a bounding box selection is drawn
 * on the spectrogram canvas view." This exercises the actual pixel -> time/
 * frequency resolution a mouse drag produces, not just the math functions in
 * isolation - LIT-164/177 were marked Done with no test file covering this
 * component at all.
 */
import React from 'react';
import { render, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { jest } from '@jest/globals';
import { SpectrogramGridSelector, SpectrogramBoundaryFrame } from '../components/analysis/PerturbationTools';

const CONTAINER_WIDTH = 800;
const CONTAINER_HEIGHT = 160;

beforeAll(() => {
  // jsdom's layout engine doesn't compute real geometry; the component reads
  // container.getBoundingClientRect() to translate pixels to signal units.
  jest.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue({
    width: CONTAINER_WIDTH,
    height: CONTAINER_HEIGHT,
    top: 0,
    left: 0,
    bottom: CONTAINER_HEIGHT,
    right: CONTAINER_WIDTH,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  } as DOMRect);
});

const drag = (container: HTMLElement, from: { x: number; y: number }, to: { x: number; y: number }) => {
  fireEvent.mouseDown(container, { button: 0, clientX: from.x, clientY: from.y });
  fireEvent.mouseMove(window, { clientX: to.x, clientY: to.y });
  fireEvent.mouseUp(window, { clientX: to.x, clientY: to.y });
};

describe('SpectrogramGridSelector (LIT-177)', () => {
  it('resolves a drag into millisecond/Hz bounds and reports them via onFrameCreated', () => {
    const onFrameCreated = jest.fn();
    const { container } = render(
      <SpectrogramGridSelector durationSec={10} maxFreqHz={8000} height={CONTAINER_HEIGHT} onFrameCreated={onFrameCreated} />
    );
    const selectorDiv = container.firstElementChild as HTMLElement;

    // x: 100 -> 300 px of an 800px/10s track = 1.25s -> 3.75s.
    // y: 40 -> 120 px, higher on screen (y=40) is higher frequency.
    drag(selectorDiv, { x: 100, y: 120 }, { x: 300, y: 40 });

    expect(onFrameCreated).toHaveBeenCalledTimes(1);
    const frame = onFrameCreated.mock.calls[0][0] as SpectrogramBoundaryFrame;

    expect(frame.startTimeMs).toBeCloseTo(1250, 0);
    expect(frame.endTimeMs).toBeCloseTo(3750, 0);
    expect(frame.endTimeMs).toBeGreaterThan(frame.startTimeMs);

    // The pixel closer to the top of the canvas (y=40) must resolve to the
    // higher frequency bound - inverting this silently would mute the wrong
    // half of the spectrum for every "Localized Mute"/"Frequency Filter Band"
    // mutation built on top of this component.
    expect(frame.endFreqHz).toBeGreaterThan(frame.startFreqHz);
    expect(frame.startFreqHz).toBeGreaterThanOrEqual(0);
    expect(frame.endFreqHz).toBeLessThanOrEqual(8000);
  });

  it('ignores a drag shorter than the drag threshold (no accidental frame from a click)', () => {
    const onFrameCreated = jest.fn();
    const { container } = render(
      <SpectrogramGridSelector durationSec={10} maxFreqHz={8000} height={CONTAINER_HEIGHT} onFrameCreated={onFrameCreated} />
    );
    const selectorDiv = container.firstElementChild as HTMLElement;

    drag(selectorDiv, { x: 100, y: 100 }, { x: 101, y: 101 });

    expect(onFrameCreated).not.toHaveBeenCalled();
  });

  it('supports multiple sequential selections, each resolved independently', () => {
    const onFrameCreated = jest.fn();
    const { container } = render(
      <SpectrogramGridSelector durationSec={10} maxFreqHz={8000} height={CONTAINER_HEIGHT} onFrameCreated={onFrameCreated} />
    );
    const selectorDiv = container.firstElementChild as HTMLElement;

    drag(selectorDiv, { x: 0, y: 160 }, { x: 200, y: 100 });
    drag(selectorDiv, { x: 400, y: 80 }, { x: 600, y: 0 });

    expect(onFrameCreated).toHaveBeenCalledTimes(2);
    const [first, second] = onFrameCreated.mock.calls.map((c) => c[0] as SpectrogramBoundaryFrame);
    expect(second.startTimeMs).toBeGreaterThan(first.endTimeMs);
  });
});
