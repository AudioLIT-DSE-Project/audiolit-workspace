/**
 * Web Audio node lifetime under peak load (LIT-170).
 *
 * WaveformViewer owns one WaveSurfer instance (and the underlying Web Audio
 * nodes it creates) per mount, torn down in its effect cleanup. Nothing
 * previously exercised that lifecycle beyond a single mount/unmount, so a
 * regression here (a missing destroy(), a leaked instance) would only ever
 * surface as slow browser memory growth after many clip switches — never as
 * a failing test. This drives many create/destroy cycles, sequential and
 * concurrent, and asserts every created node is destroyed exactly once.
 */
import React from 'react';
import { render } from '@testing-library/react';
import WaveSurfer from 'wavesurfer.js';
import { WaveformViewer } from '@/components/audio/WaveformViewer';

jest.mock('wavesurfer.js', () => ({
  __esModule: true,
  default: { create: jest.fn() },
}));

const createMockInstance = () => ({
  on: jest.fn(),
  destroy: jest.fn(),
  load: jest.fn(),
  play: jest.fn(),
  pause: jest.fn(),
  seekTo: jest.fn(),
  getDuration: jest.fn(() => 0),
  getCurrentTime: jest.fn(() => 0),
});

const createMock = WaveSurfer.create as jest.Mock;

beforeEach(() => {
  createMock.mockReset();
  createMock.mockImplementation(() => createMockInstance());
});

describe('WaveformViewer Web Audio node lifetime under peak load', () => {
  it('destroys the underlying node exactly once per sequential mount/unmount cycle', () => {
    const CYCLES = 50;

    for (let i = 0; i < CYCLES; i++) {
      const { unmount } = render(<WaveformViewer />);
      unmount();
    }

    expect(createMock).toHaveBeenCalledTimes(CYCLES);
    const destroyCounts = createMock.mock.results.map(
      (r) => (r.value as ReturnType<typeof createMockInstance>).destroy.mock.calls.length,
    );
    expect(destroyCounts).toEqual(new Array(CYCLES).fill(1));
  });

  it('destroys every node when many instances are mounted simultaneously (multi-panel peak load)', () => {
    const CONCURRENT = 20;
    const renders = Array.from({ length: CONCURRENT }, () => render(<WaveformViewer />));

    expect(createMock).toHaveBeenCalledTimes(CONCURRENT);

    renders.forEach(({ unmount }) => unmount());

    const destroyCounts = createMock.mock.results.map(
      (r) => (r.value as ReturnType<typeof createMockInstance>).destroy.mock.calls.length,
    );
    expect(destroyCounts.every((count) => count === 1)).toBe(true);
  });

  it('never leaves more than one live (undestroyed) node at a time across rapid remounts', () => {
    let live = 0;
    let maxLive = 0;
    createMock.mockImplementation(() => {
      live += 1;
      maxLive = Math.max(maxLive, live);
      const instance = createMockInstance();
      instance.destroy.mockImplementation(() => {
        live -= 1;
      });
      return instance;
    });

    for (let i = 0; i < 30; i++) {
      const { unmount } = render(<WaveformViewer />);
      unmount();
    }

    expect(maxLive).toBe(1);
    expect(live).toBe(0);
  });
});
