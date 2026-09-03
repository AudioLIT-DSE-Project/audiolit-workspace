/**
 * Real-component boundary tests for WaveformViewer (LIT-187).
 *
 * The pre-existing src/tests/ui-components.test.tsx exercises inline mock
 * components, not the actual app -- this file tests the real
 * WaveformViewer.tsx directly. jsdom doesn't implement canvas 2D rendering,
 * so WaveSurfer itself is mocked (as its own author's test file already does
 * for a hypothetical WaveformViewer); this exercises the component's real
 * lifecycle wiring around that mock: event registration, cleanup on unmount,
 * and state-driven rendering.
 */

import React from "react";
import { render, screen, cleanup, act } from "@testing-library/react";
import "@testing-library/jest-dom";
import WaveSurfer from "wavesurfer.js";
import { WaveformViewer } from "./WaveformViewer";

type WaveSurferHandlers = Record<string, (...args: unknown[]) => void>;

interface MockWaveSurferInstance {
  on: jest.Mock;
  load: jest.Mock;
  play: jest.Mock;
  pause: jest.Mock;
  destroy: jest.Mock;
  getDuration: jest.Mock;
  getCurrentTime: jest.Mock;
}

// jest.mock factories are hoisted above this file's own const declarations,
// so the mock instance has to be built entirely inside the factory closure
// rather than referencing an outer variable (a TDZ ReferenceError otherwise)
// -- pulled back out via the mocked import below instead.
jest.mock("wavesurfer.js", () => {
  const instance = {
    on: jest.fn(),
    load: jest.fn(),
    play: jest.fn(),
    pause: jest.fn(),
    destroy: jest.fn(),
    getDuration: jest.fn(() => 5),
    getCurrentTime: jest.fn(() => 0),
  };
  return {
    __esModule: true,
    default: { create: jest.fn(() => instance), __mockInstance: instance },
  };
});

const mockCreate = WaveSurfer.create as jest.Mock;
const mockInstance = (WaveSurfer as unknown as { __mockInstance: MockWaveSurferInstance }).__mockInstance;

/** Pulls the handler registered for a given wavesurfer.on(event, handler) call. */
const registeredHandlers = (): WaveSurferHandlers => {
  const handlers: WaveSurferHandlers = {};
  for (const [event, handler] of mockInstance.on.mock.calls as [string, (...a: unknown[]) => void][]) {
    handlers[event] = handler;
  }
  return handlers;
};

beforeEach(() => {
  jest.clearAllMocks();
  // fetch is used by the audio-url-change effect (HEAD probe); resolve it
  // successfully by default so isLoading settles and doesn't block assertions
  // that don't care about the loading state itself.
  global.fetch = jest.fn().mockResolvedValue({ ok: true, status: 200, statusText: "OK" }) as unknown as typeof fetch;
});

afterEach(() => {
  cleanup();
});

describe("WaveformViewer", () => {
  test("shows the empty state when no audioUrl is provided", () => {
    render(<WaveformViewer />);
    expect(screen.getByText(/No audio file selected/)).toBeInTheDocument();
  });

  test("creates exactly one WaveSurfer instance on mount", () => {
    render(<WaveformViewer audioUrl="/test-audio.wav" />);
    expect(mockCreate).toHaveBeenCalledTimes(1);
  });

  test("destroys the WaveSurfer instance on unmount", () => {
    const { unmount } = render(<WaveformViewer audioUrl="/test-audio.wav" />);
    expect(mockInstance.destroy).not.toHaveBeenCalled();
    unmount();
    expect(mockInstance.destroy).toHaveBeenCalledTimes(1);
  });

  test("registers ready/error/loading/audioprocess handlers", () => {
    render(<WaveformViewer audioUrl="/test-audio.wav" />);
    const handlers = registeredHandlers();
    expect(Object.keys(handlers)).toEqual(
      expect.arrayContaining(["ready", "error", "loading", "audioprocess"])
    );
  });

  test("onReady fires when the mocked wavesurfer emits 'ready'", () => {
    const onReady = jest.fn();
    render(<WaveformViewer audioUrl="/test-audio.wav" onReady={onReady} />);
    registeredHandlers().ready();
    expect(onReady).toHaveBeenCalledWith(mockInstance);
  });

  test("an emitted 'error' event renders the error UI, not a silent failure", () => {
    render(<WaveformViewer audioUrl="/test-audio.wav" />);
    // fireEvent/user-event wrap state updates in act() automatically; a
    // directly-invoked handler like this one doesn't, so without act() here
    // the setError/setIsLoading updates inside it aren't guaranteed to have
    // flushed before the assertion below reads the DOM.
    act(() => {
      registeredHandlers().error({ message: "decode failed" });
    });
    expect(screen.getByText(/Error loading audio/)).toBeInTheDocument();
    expect(screen.getByText(/decode failed/)).toBeInTheDocument();
  });

  test("isPlaying=true calls play(), isPlaying=false calls pause()", () => {
    const { rerender } = render(<WaveformViewer audioUrl="/test-audio.wav" isPlaying={false} />);
    expect(mockInstance.pause).toHaveBeenCalled();

    rerender(<WaveformViewer audioUrl="/test-audio.wav" isPlaying={true} />);
    expect(mockInstance.play).toHaveBeenCalled();
  });

  test("onProgress fires with the mocked duration on an 'interaction' event", () => {
    const onProgress = jest.fn();
    render(<WaveformViewer audioUrl="/test-audio.wav" onProgress={onProgress} />);
    const handlers = registeredHandlers();
    handlers["interaction"]();
    expect(onProgress).toHaveBeenCalledWith(0, 5);
  });
});
