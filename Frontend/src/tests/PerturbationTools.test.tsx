/**
 * LIT-164/178 — Frontend Mutation Event Trigger & Asynchronous State
 * Dispatcher, plus the FR12.2 Web Audio preview and non-destructive
 * before/after display. LIT-164 and its children (LIT-176/177/178) were all
 * marked Done with zero frontend test coverage; this closes that gap against
 * the actual acceptance criteria: "2D spectrogram bbox selection -> time-
 * frequency units -> inherited perturbation engine ... Web Audio preview;
 * non-destructive before/after."
 */
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { jest } from '@jest/globals';

jest.mock('../components/audio/WaveformViewer', () => ({
  // WaveformViewer's own drag-selection (LIT-176) has its own coverage
  // elsewhere; here we only need to know *which* clip PerturbationTools
  // asked to display, for the before/after (non-destructive) assertion.
  WaveformViewer: ({ audioUrl }: { audioUrl?: string }) => (
    <div data-testid="waveform" data-audio-url={audioUrl} />
  ),
}));

const mockUseTaskStatus = jest.fn();
jest.mock('../hooks/useTaskStatus', () => ({
  useTaskStatus: (taskId: string | null) => mockUseTaskStatus(taskId),
}));

import { PerturbationTools } from '../components/analysis/PerturbationTools';

const SELECTED_FILE = {
  file_id: 'clip-1',
  filename: 'clip-1.wav',
  file_path: 'uploads/clip-1.wav',
  message: 'File uploaded successfully',
  duration: 5,
  sample_rate: 16000,
};

const CONTAINER_RECT = {
  width: 800,
  height: 160,
  top: 0,
  left: 0,
  bottom: 160,
  right: 800,
  x: 0,
  y: 0,
  toJSON: () => ({}),
} as DOMRect;

const drag = (container: HTMLElement, from: { x: number; y: number }, to: { x: number; y: number }) => {
  fireEvent.mouseDown(container, { button: 0, clientX: from.x, clientY: from.y });
  fireEvent.mouseMove(window, { clientX: to.x, clientY: to.y });
  fireEvent.mouseUp(window, { clientX: to.x, clientY: to.y });
};

/** Draw one spectrogram region so the mutation-trigger UI (LIT-178) mounts. */
const createRegion = (container: HTMLElement) => {
  const selectorDiv = container.querySelector('.cursor-crosshair') as HTMLElement;
  drag(selectorDiv, { x: 100, y: 120 }, { x: 300, y: 40 });
};

describe('PerturbationTools (LIT-164/178)', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUseTaskStatus.mockReturnValue({ state: 'QUEUED', result: null, error: null });
    jest.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue(CONTAINER_RECT);
    (global as any).fetch = jest.fn(async () => ({
      ok: true,
      json: async () => ({ job_id: 'job-123', websocket_url: 'ws://x', schema_version: '1', family_jobs: {} }),
    }));
  });

  it('sends the free-form perturbation payload to /api/inference/mutation', async () => {
    render(<PerturbationTools selectedFile={SELECTED_FILE} />);

    fireEvent.click(screen.getByRole('checkbox', { name: /Add Gaussian Noise/i }));
    fireEvent.click(screen.getByRole('button', { name: /Apply Perturbations/i }));

    await waitFor(() => expect(global.fetch).toHaveBeenCalled());

    const [url, options] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe('http://localhost:8000/api/inference/mutation');
    const body = JSON.parse((options as RequestInit).body as string);
    expect(body.audio_ref).toBe('uploads/clip-1.wav');
    expect(body.mutation.perturbations).toEqual([
      { type: 'noise', params: { noise_level: 0.1 } },
    ]);
  });

  it('scopes a region mutation to the drawn spectrogram frame\'s time/frequency bounds', async () => {
    const { container } = render(<PerturbationTools selectedFile={SELECTED_FILE} />);

    createRegion(container);

    // "Localized Mute" (time_freq_mask) is the default frameMutationType.
    fireEvent.click(screen.getByRole('button', { name: /Apply Mutation/i }));

    await waitFor(() => expect(global.fetch).toHaveBeenCalled());

    const [, options] = (global.fetch as jest.Mock).mock.calls[0];
    const body = JSON.parse((options as RequestInit).body as string);
    const [perturbation] = body.mutation.perturbations;

    // SELECTED_FILE.duration is 5s: x 100->300px of an 800px canvas maps to
    // 0.625s -> 1.875s (same pixel-to-time formula verified precisely in
    // SpectrogramGridSelector.test.tsx).
    expect(perturbation.type).toBe('time_freq_mask');
    expect(perturbation.params.t_start_ms).toBeCloseTo(625, 0);
    expect(perturbation.params.t_end_ms).toBeCloseTo(1875, 0);
    expect(perturbation.params.f_high_hz).toBeGreaterThan(perturbation.params.f_low_hz);
  });

  it('sends a band_pass_filter payload when that mutation type is selected', async () => {
    const { container } = render(<PerturbationTools selectedFile={SELECTED_FILE} />);

    createRegion(container);
    fireEvent.click(screen.getByRole('button', { name: /Frequency Filter Band/i }));
    fireEvent.click(screen.getByRole('button', { name: /Apply Mutation/i }));

    await waitFor(() => expect(global.fetch).toHaveBeenCalled());

    const [, options] = (global.fetch as jest.Mock).mock.calls[0];
    const body = JSON.parse((options as RequestInit).body as string);
    expect(body.mutation.perturbations[0].type).toBe('band_pass_filter');
  });

  it('previews the selected region via Web Audio before any network call (FR12.2)', async () => {
    const start = jest.fn();
    const connect = jest.fn();
    const createBufferSource = jest.fn(() => ({ connect, start, buffer: null, onended: null }));
    const decodeAudioData = jest.fn(async () => ({
      duration: 5,
      numberOfChannels: 1,
      length: 80000,
      sampleRate: 16000,
      getChannelData: () => new Float32Array(80000),
    }));
    (global as any).AudioContext = jest.fn().mockImplementation(() => ({
      createBufferSource,
      decodeAudioData,
      destination: {},
      close: jest.fn(async () => undefined),
    }));
    (global as any).fetch = jest.fn().mockImplementation((url: string) => {
      if (typeof url === 'string' && url.includes('/api/inference/mutation')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ job_id: 'job-123', websocket_url: 'ws://x', schema_version: '1', family_jobs: {} }),
        });
      }
      // Audio file fetch for decodeAudioData.
      return Promise.resolve({ arrayBuffer: async () => new ArrayBuffer(8) });
    });

    const { container } = render(<PerturbationTools selectedFile={SELECTED_FILE} />);
    createRegion(container);

    fireEvent.click(screen.getByRole('button', { name: /Preview region/i }));

    await waitFor(() => expect(decodeAudioData).toHaveBeenCalled());
    expect(createBufferSource).toHaveBeenCalled();
    expect(start).toHaveBeenCalled();
    // The preview must never touch the mutation endpoint - it's a purely
    // client-side audition of the counterfactual before committing to it.
    const mutationCalls = (global.fetch as jest.Mock).mock.calls.filter(([u]) =>
      typeof u === 'string' && u.includes('/api/inference/mutation')
    );
    expect(mutationCalls).toHaveLength(0);
  });

  it('shows both original and perturbed waveforms once a mutation succeeds (non-destructive before/after)', async () => {
    mockUseTaskStatus.mockImplementation((taskId: string | null) => {
      if (taskId === 'job-123') {
        return {
          state: 'SUCCESS',
          result: {
            perturbed_file: 'uploads/clip-1_perturbed_abc123.wav',
            filename: 'clip-1_perturbed_abc123.wav',
            duration_ms: 5000,
            sample_rate: 16000,
            applied_perturbations: [{ type: 'noise', params: {}, status: 'applied' }],
            success: true,
          },
        };
      }
      return { state: 'QUEUED', result: null, error: null };
    });

    render(<PerturbationTools selectedFile={SELECTED_FILE} />);

    fireEvent.click(screen.getByRole('checkbox', { name: /Add Gaussian Noise/i }));
    fireEvent.click(screen.getByRole('button', { name: /Apply Perturbations/i }));

    await waitFor(() => expect(global.fetch).toHaveBeenCalled());

    // Drive the job id the mocked fetch resolved with into useTaskStatus's
    // return by re-rendering isn't necessary - PerturbationTools re-renders
    // itself once setMutationTaskId('job-123') runs, and the mock above keys
    // off that exact id.
    await waitFor(() => {
      expect(screen.getByText('Original Audio')).toBeInTheDocument();
      expect(screen.getByText('Perturbed Audio')).toBeInTheDocument();
    });

    const waveforms = screen.getAllByTestId('waveform');
    expect(waveforms).toHaveLength(2);
    expect(waveforms[0]).toHaveAttribute('data-audio-url', expect.stringContaining('clip-1'));
    expect(waveforms[1]).toHaveAttribute('data-audio-url', expect.stringContaining('clip-1_perturbed_abc123.wav'));
  });
});
