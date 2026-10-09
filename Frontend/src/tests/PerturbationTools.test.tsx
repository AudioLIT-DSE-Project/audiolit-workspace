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
  //
  // It also stands in for the waveform's half of the shared region: the range
  // it is asked to display is exposed as a data attribute, and a button fires
  // the selection a 200 -> 400 px drag on an 800 px waveform would report.
  WaveformViewer: ({
    audioUrl,
    selectionRange,
    onSelectionChange,
  }: {
    audioUrl?: string;
    selectionRange?: { start: number; end: number } | null;
    onSelectionChange?: (selection: { startX: number; endX: number; containerWidth: number } | null) => void;
  }) => (
    <div
      data-testid="waveform"
      data-audio-url={audioUrl}
      data-selection={selectionRange ? `${selectionRange.start}-${selectionRange.end}` : ''}
    >
      {onSelectionChange && (
        <>
          <button type="button" onClick={() => onSelectionChange({ startX: 200, endX: 400, containerWidth: 800 })}>
            drag-waveform
          </button>
          <button type="button" onClick={() => onSelectionChange(null)}>click-waveform</button>
        </>
      )}
    </div>
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
    fireEvent.click(screen.getByRole('button', { name: /Apply to whole clip/i }));

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

    fireEvent.click(screen.getByRole('checkbox', { name: /Mute Region/i }));
    fireEvent.click(screen.getByRole('button', { name: /Apply to selected region/i }));

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
    fireEvent.click(screen.getByRole('checkbox', { name: /Band-Pass Filter/i }));
    fireEvent.click(screen.getByRole('button', { name: /Apply to selected region/i }));

    await waitFor(() => expect(global.fetch).toHaveBeenCalled());

    const [, options] = (global.fetch as jest.Mock).mock.calls[0];
    const body = JSON.parse((options as RequestInit).body as string);
    const [bandPass] = body.mutation.perturbations;
    expect(bandPass.type).toBe('band_pass_filter');
    // The band to keep, and the region it is kept for.
    expect(bandPass.params.f_high_hz).toBeGreaterThan(bandPass.params.f_low_hz);
    expect(bandPass.region.t_start_ms).toBeCloseTo(625, 0);
    expect(bandPass.region.t_end_ms).toBeCloseTo(1875, 0);
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
    fireEvent.click(screen.getByRole('button', { name: /Apply to whole clip/i }));

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

describe('PerturbationTools request and failure handling', () => {
  const DATASET_ROW = {
    file_id: 'sample-000001.mp3',
    filename: 'sample-000001.mp3',
    file_path: 'cv-valid-dev/sample-000001.mp3',
    message: 'Selected from dataset',
  };
  const PROFILE = { sample_rate: 16000, duration_s: 4, spectrogram: [[0, 0.5], [1, 0.25]] };

  const applyNoise = async () => {
    fireEvent.click(screen.getByRole('checkbox', { name: /Add Gaussian Noise/i }));
    fireEvent.click(screen.getByRole('button', { name: /Apply to whole clip/i }));
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
  };

  const requestBody = (call = 0) =>
    JSON.parse(((global.fetch as jest.Mock).mock.calls[call][1] as RequestInit).body as string);

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseTaskStatus.mockReturnValue({ state: 'QUEUED', result: null, error: null });
    jest.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue(CONTAINER_RECT);
    (global as any).fetch = jest.fn(async () => ({
      ok: true,
      json: async () => ({ job_id: 'job-123', websocket_url: 'ws://x', schema_version: '1', family_jobs: {} }),
    }));
  });

  it('does not send the active corpus with an upload', async () => {
    // A live recording selected while a corpus is active. The corpus used to
    // go along, and the backend looked for the upload inside it.
    render(<PerturbationTools selectedFile={SELECTED_FILE} dataset="common-voice" originalDataset="common-voice" />);
    await applyNoise();

    const body = requestBody();
    expect(body.audio_ref).toBe('uploads/clip-1.wav');
    expect(body.mutation.is_uploaded).toBe(true);
    expect(body.mutation).not.toHaveProperty('dataset');
  });

  it('sends a dataset row by filename with its corpus', async () => {
    render(<PerturbationTools selectedFile={DATASET_ROW} dataset="common-voice" originalDataset="common-voice" />);
    await applyNoise();

    const body = requestBody();
    expect(body.audio_ref).toBe('sample-000001.mp3');
    expect(body.mutation.is_uploaded).toBe(false);
    expect(body.mutation.dataset).toBe('common-voice');
  });

  it('offers the region selector for a dataset row once its acoustic profile is known', () => {
    // A dataset row has no duration of its own, so the selector was hidden.
    const { container, rerender } = render(<PerturbationTools selectedFile={DATASET_ROW} dataset="common-voice" />);
    expect(container.querySelector('.cursor-crosshair')).toBeNull();

    rerender(<PerturbationTools selectedFile={DATASET_ROW} dataset="common-voice" acousticProfile={PROFILE} />);
    expect(container.querySelector('.cursor-crosshair')).not.toBeNull();

    // 4 s clip on an 800 px canvas: 100 -> 300 px is 0.5 s -> 1.5 s.
    createRegion(container);
    expect(screen.getByText(/0\.50–1\.50 s · \d+–\d+ Hz/)).toBeInTheDocument();
  });

  it('reports a mutation the engine could not apply, and releases the controls', async () => {
    mockUseTaskStatus.mockImplementation((taskId: string | null) =>
      taskId === 'job-123'
        ? {
            state: 'SUCCESS',
            result: { success: false, error: 'Dataset file not found: clip-1.wav', perturbed_file: '', applied_perturbations: [] },
            error: null,
          }
        : { state: 'QUEUED', result: null, error: null },
    );
    render(<PerturbationTools selectedFile={SELECTED_FILE} />);
    await applyNoise();

    await waitFor(() => expect(screen.getByText('Dataset file not found: clip-1.wav')).toBeInTheDocument());
    expect(screen.queryByText('Perturbed Audio')).not.toBeInTheDocument();
    // Only the mutation was requested: no inference on a clip that does not exist.
    expect((global.fetch as jest.Mock).mock.calls).toHaveLength(1);
    await waitFor(() => expect(screen.getByRole('button', { name: /Apply to whole clip/i })).toBeEnabled());
  });

  it('shows the worker error when the job fails, and releases the controls', async () => {
    mockUseTaskStatus.mockImplementation((taskId: string | null) =>
      taskId === 'job-123'
        ? { state: 'FAILURE', result: null, error: 'RuntimeError: out of memory' }
        : { state: 'QUEUED', result: null, error: null },
    );
    render(<PerturbationTools selectedFile={SELECTED_FILE} />);
    await applyNoise();

    await waitFor(() => expect(screen.getByText('RuntimeError: out of memory')).toBeInTheDocument());
    await waitFor(() => expect(screen.getByRole('button', { name: /Apply to whole clip/i })).toBeEnabled());
  });

  it("re-runs the selected model's own task on the mutated clip", async () => {
    mockUseTaskStatus.mockImplementation((taskId: string | null) =>
      taskId === 'job-123'
        ? {
            state: 'SUCCESS',
            result: {
              success: true,
              perturbed_file: 'uploads/clip-1_perturbed_abc123.wav',
              filename: 'clip-1_perturbed_abc123.wav',
              duration_ms: 5000,
              sample_rate: 16000,
              applied_perturbations: [{ type: 'noise', params: {}, status: 'applied' }],
            },
            error: null,
          }
        : { state: 'QUEUED', result: null, error: null },
    );
    render(<PerturbationTools selectedFile={SELECTED_FILE} model="melody-machine" />);
    await applyNoise();

    await waitFor(() => expect((global.fetch as jest.Mock).mock.calls.length).toBeGreaterThanOrEqual(2));
    const [url] = (global.fetch as jest.Mock).mock.calls[1];
    expect(url).toBe('http://localhost:8000/api/inference/multitask');
    expect(requestBody(1)).toEqual({
      audio_ref: 'uploads/clip-1_perturbed_abc123.wav',
      tasks: ['add'],
      model_ids: { add: 'melody-machine' },
    });
  });
});

describe('PerturbationTools shared region', () => {
  const requestBody = (call = 0) =>
    JSON.parse(((global.fetch as jest.Mock).mock.calls[call][1] as RequestInit).body as string);

  const apply = async (label: RegExp) => {
    fireEvent.click(screen.getByRole('button', { name: label }));
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    return requestBody().mutation.perturbations;
  };

  const originalWaveform = () => screen.getAllByTestId('waveform')[0];

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseTaskStatus.mockReturnValue({ state: 'QUEUED', result: null, error: null });
    jest.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue(CONTAINER_RECT);
    (global as any).fetch = jest.fn(async () => ({
      ok: true,
      json: async () => ({ job_id: 'job-123', websocket_url: 'ws://x', schema_version: '1', family_jobs: {} }),
    }));
  });

  it('applies to the whole clip when nothing is selected', async () => {
    render(<PerturbationTools selectedFile={SELECTED_FILE} />);
    expect(screen.queryByRole('button', { name: /Apply to selected region/i })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('checkbox', { name: /Add Gaussian Noise/i }));
    const [noise] = await apply(/Apply to whole clip/i);
    expect(noise).toEqual({ type: 'noise', params: { noise_level: 0.1 } });
  });

  it('shows a region drawn on the spectrogram on the waveform too', () => {
    const { container } = render(<PerturbationTools selectedFile={SELECTED_FILE} />);
    expect(originalWaveform()).toHaveAttribute('data-selection', '');

    // 100 -> 300 px of 800: the same fractions of the waveform's width.
    createRegion(container);
    expect(originalWaveform()).toHaveAttribute('data-selection', '0.125-0.375');
    expect(screen.getByText(/0\.63–1\.88 s · \d+–\d+ Hz/)).toBeInTheDocument();
  });

  it('turns a waveform drag into a time span across all frequencies', async () => {
    render(<PerturbationTools selectedFile={SELECTED_FILE} />);

    // 200 -> 400 px of 800 on a 5 s clip: 1.25 s -> 2.5 s.
    fireEvent.click(screen.getByRole('button', { name: 'drag-waveform' }));
    expect(screen.getByText('1.25–2.50 s · all frequencies')).toBeInTheDocument();
    expect(originalWaveform()).toHaveAttribute('data-selection', '0.25-0.5');

    fireEvent.click(screen.getByRole('checkbox', { name: /Add Gaussian Noise/i }));
    const [noise] = await apply(/Apply to selected region/i);
    expect(noise.region).toEqual({ t_start_ms: 1250, t_end_ms: 2500 });
  });

  it('confines noise to the box drawn on the spectrogram, in time and frequency', async () => {
    const { container } = render(<PerturbationTools selectedFile={SELECTED_FILE} />);
    createRegion(container);

    fireEvent.click(screen.getByRole('checkbox', { name: /Add Gaussian Noise/i }));
    const [noise] = await apply(/Apply to selected region/i);
    expect(noise.type).toBe('noise');
    expect(noise.region.t_start_ms).toBeCloseTo(625, 0);
    expect(noise.region.t_end_ms).toBeCloseTo(1875, 0);
    expect(noise.region.f_high_hz).toBeGreaterThan(noise.region.f_low_hz);
  });

  it('gives pitch shift the time span only, even for a box', async () => {
    const { container } = render(<PerturbationTools selectedFile={SELECTED_FILE} />);
    createRegion(container);

    fireEvent.click(screen.getByRole('checkbox', { name: /Apply Pitch Shift/i }));
    const [pitch] = await apply(/Apply to selected region/i);
    expect(pitch.type).toBe('pitch_shift');
    expect(Object.keys(pitch.region).sort()).toEqual(['t_end_ms', 't_start_ms']);
  });

  it('replaces the region with the latest drag, whichever view it came from', () => {
    const { container } = render(<PerturbationTools selectedFile={SELECTED_FILE} />);
    createRegion(container);
    fireEvent.click(screen.getByRole('button', { name: 'drag-waveform' }));

    // The box from the spectrogram is gone; the waveform's span replaced it.
    expect(screen.getByText('1.25–2.50 s · all frequencies')).toBeInTheDocument();
    expect(screen.queryByText(/0\.63–1\.88 s/)).not.toBeInTheDocument();
  });

  it('only offers mute and band-pass when the selection supports them', () => {
    const { container } = render(<PerturbationTools selectedFile={SELECTED_FILE} />);
    expect(screen.getByRole('checkbox', { name: /Mute Region/i })).toBeDisabled();
    expect(screen.getByRole('checkbox', { name: /Band-Pass Filter/i })).toBeDisabled();

    // A waveform span has no frequency band: mute yes, band-pass no.
    fireEvent.click(screen.getByRole('button', { name: 'drag-waveform' }));
    expect(screen.getByRole('checkbox', { name: /Mute Region/i })).toBeEnabled();
    expect(screen.getByRole('checkbox', { name: /Band-Pass Filter/i })).toBeDisabled();

    createRegion(container);
    expect(screen.getByRole('checkbox', { name: /Band-Pass Filter/i })).toBeEnabled();
  });

  it('goes back to the whole clip when the region is cleared', async () => {
    const { container } = render(<PerturbationTools selectedFile={SELECTED_FILE} />);
    createRegion(container);
    fireEvent.click(screen.getByRole('checkbox', { name: /Mute Region/i }));
    fireEvent.click(screen.getByRole('checkbox', { name: /Add Gaussian Noise/i }));

    fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
    expect(originalWaveform()).toHaveAttribute('data-selection', '');

    // The mute was checked, but with no region it is not sent.
    const perturbations = await apply(/Apply to whole clip/i);
    expect(perturbations).toEqual([{ type: 'noise', params: { noise_level: 0.1 } }]);
  });

  it('has no separate region-mutation section any more', () => {
    const { container } = render(<PerturbationTools selectedFile={SELECTED_FILE} />);
    createRegion(container);
    expect(screen.queryByText('Apply Mutation to Selected Region')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^Apply Mutation$/i })).not.toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: /^Apply to /i })).toHaveLength(1);
  });
});
