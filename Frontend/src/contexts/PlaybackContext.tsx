import { createContext, useCallback, useContext, useMemo, useRef, useState, ReactNode } from "react";

/**
 * FR10.2 — one playhead, shared by the player, the acoustic profiler and the
 * XAI overlay.
 *
 * The profiler's stated purpose is relating what a model attends to against the
 * physical signal. Without a shared time cursor that comparison is done by eye
 * across two independent axes, which is precisely the manual alignment the pane
 * exists to remove.
 *
 * `WaveformViewer` owns the wavesurfer instance and publishes here; every other
 * time-aligned view subscribes. Seeking is registered by the owner so a click on
 * any chart can drive the audio, not only the other way round.
 */
interface PlaybackState {
  currentTime: number;
  duration: number;
  /** Seek the audio. No-op until a player registers a handler. */
  seek: (seconds: number) => void;
  publish: (currentTime: number, duration: number) => void;
  registerSeek: (handler: ((seconds: number) => void) | null) => void;
}

const PlaybackContext = createContext<PlaybackState>({
  currentTime: 0,
  duration: 0,
  seek: () => {},
  publish: () => {},
  registerSeek: () => {},
});

export const PlaybackProvider = ({ children }: { children: ReactNode }) => {
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const seekRef = useRef<((seconds: number) => void) | null>(null);

  const publish = useCallback((t: number, d: number) => {
    setCurrentTime(t);
    if (d && Number.isFinite(d)) setDuration(d);
  }, []);

  const registerSeek = useCallback((handler: ((seconds: number) => void) | null) => {
    seekRef.current = handler;
  }, []);

  const seek = useCallback((seconds: number) => {
    seekRef.current?.(Math.max(0, seconds));
  }, []);

  const value = useMemo(
    () => ({ currentTime, duration, seek, publish, registerSeek }),
    [currentTime, duration, seek, publish, registerSeek],
  );

  return <PlaybackContext.Provider value={value}>{children}</PlaybackContext.Provider>;
};

export const usePlayback = () => useContext(PlaybackContext);
