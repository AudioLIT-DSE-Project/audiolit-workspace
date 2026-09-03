import { useCallback, useEffect, useRef, useState } from "react";
import { usePlayback } from '@/contexts/PlaybackContext';
import { Card } from "@/components/ui/card";
import WaveSurfer from "wavesurfer.js";

// LIT-176: minimum pixel travel before a mousedown/mouseup pair counts as a
// drag selection rather than a click (which WaveSurfer's own seek handler
// already owns).
const DRAG_THRESHOLD_PX = 4;

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

export interface WaveformSelection {
  startX: number;
  endX: number;
  containerWidth: number;
}

interface WaveformViewerProps {
  audioUrl?: string;
  isPlaying?: boolean;
  onReady?: (wavesurfer: WaveSurfer) => void;
  onProgress?: (currentTime: number, duration: number) => void;
  onSelectionChange?: (selection: WaveformSelection | null) => void;
}

export const WaveformViewer = ({ audioUrl, isPlaying, onReady, onProgress, onSelectionChange }: WaveformViewerProps) => {
  // FR10.2: this component owns the wavesurfer instance, so it is the single
  // source of playback time for every other time-aligned view.
  const { publish, registerSeek } = usePlayback();
  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const currentBlobUrlRef = useRef<string | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const dragStateRef = useRef<{ startX: number; currentX: number; dragging: boolean } | null>(null);
  const rafIdRef = useRef<number | null>(null);
  const onSelectionChangeRef = useRef(onSelectionChange);
  onSelectionChangeRef.current = onSelectionChange;

  // Initialize WaveSurfer instance
  useEffect(() => {
    if (!waveformRef.current) return;
    
    // Test browser FLAC support
    const audio = new Audio();
    const flacSupport = audio.canPlayType('audio/flac');
    console.log('Browser FLAC support:', flacSupport);
    console.log('Browser supports:', {
      'audio/wav': audio.canPlayType('audio/wav'),
      'audio/mp3': audio.canPlayType('audio/mp3'),
      'audio/mpeg': audio.canPlayType('audio/mpeg'),
      'audio/flac': audio.canPlayType('audio/flac'),
      'audio/ogg': audio.canPlayType('audio/ogg')
    });
    
    // Create WaveSurfer instance
    const wavesurfer = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#3b82f6', // Blue-500 for waveform
      progressColor: '#1d4ed8', // Blue-700 for progress
      cursorColor: '#60a5fa', // Blue-400 for cursor
      height: 80,
      normalize: true,
      barWidth: 2,
      barRadius: 1,
      cursorWidth: 1,
      hideScrollbar: true,
    });

    wavesurferRef.current = wavesurfer;
    registerSeek((seconds: number) => {
      const total = wavesurfer.getDuration();
      if (total > 0) wavesurfer.seekTo(Math.min(1, Math.max(0, seconds / total)));
    });

    // Set up event listeners
    wavesurfer.on('ready', () => {
      setIsLoading(false);
      setError(null);
      if (onReady) {
        onReady(wavesurfer);
      }
    });

    wavesurfer.on('audioprocess', (currentTime) => {
      publish(currentTime, wavesurfer.getDuration());
      if (onProgress) {
        onProgress(currentTime, wavesurfer.getDuration());
      }
    });

    wavesurfer.on('interaction' as any, () => {
      const currentTime = wavesurfer.getCurrentTime();
      const duration = wavesurfer.getDuration();
      publish(currentTime, duration || 0);
      if (onProgress) {
        onProgress(currentTime, duration || 0);
      }
    });

    wavesurfer.on('error', (err) => {
      console.error('WaveSurfer error:', err);
      console.error('Error details:', {
        message: err.message,
        stack: err.stack,
        audioUrl: audioUrl
      });
      setError(`Failed to load audio file: ${err.message || 'Unknown error'}`);
      setIsLoading(false);
    });

    wavesurfer.on('loading', (progress) => {
      if (progress < 100) {
        setIsLoading(true);
      }
    });

    // Cleanup function
    return () => {
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
        wavesurferRef.current = null;
      }
      // Clean up blob URLs to prevent memory leaks
      if (currentBlobUrlRef.current) {
        URL.revokeObjectURL(currentBlobUrlRef.current);
        currentBlobUrlRef.current = null;
      }
    };
  }, []); // Only run once when component mounts

  // Draw (or clear) the semi-transparent selection box on the overlay canvas.
  // Pure ref reads/writes so it can be driven from a rAF loop without
  // triggering React re-renders on every animation frame.
  const drawSelection = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const drag = dragStateRef.current;
    if (!drag) return;

    const dpr = window.devicePixelRatio || 1;
    const x1 = Math.min(drag.startX, drag.currentX) * dpr;
    const x2 = Math.max(drag.startX, drag.currentX) * dpr;

    ctx.fillStyle = 'rgba(59, 130, 246, 0.25)'; // blue-500 @ 25% — semi-transparent fill
    ctx.fillRect(x1, 0, x2 - x1, canvas.height);
    ctx.strokeStyle = 'rgba(29, 78, 216, 0.9)'; // blue-700 — high-contrast border
    ctx.lineWidth = 2 * dpr;
    ctx.strokeRect(x1, 0, x2 - x1, canvas.height);
  }, []);

  const renderSelectionFrame = useCallback(() => {
    drawSelection();
    if (dragStateRef.current?.dragging) {
      rafIdRef.current = requestAnimationFrame(renderSelectionFrame);
    }
  }, [drawSelection]);

  const handleWindowMouseMove = useCallback((event: MouseEvent) => {
    const container = waveformRef.current;
    const drag = dragStateRef.current;
    if (!container || !drag) return;

    const rect = container.getBoundingClientRect();
    drag.currentX = clamp(event.clientX - rect.left, 0, rect.width);
  }, []);

  const handleWindowMouseUp = useCallback(() => {
    window.removeEventListener('mousemove', handleWindowMouseMove);
    window.removeEventListener('mouseup', handleWindowMouseUp);
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }

    const drag = dragStateRef.current;
    if (drag) {
      drag.dragging = false;
      const startX = Math.min(drag.startX, drag.currentX);
      const endX = Math.max(drag.startX, drag.currentX);

      if (endX - startX >= DRAG_THRESHOLD_PX) {
        const containerWidth = waveformRef.current?.clientWidth ?? 0;
        onSelectionChangeRef.current?.({ startX, endX, containerWidth });
      } else {
        // Too small to be a real drag — treat as a click-through and clear.
        dragStateRef.current = null;
        onSelectionChangeRef.current?.(null);
      }
    }

    drawSelection();
  }, [drawSelection, handleWindowMouseMove]);

  const handleMouseDown = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    // Left button only, and only once a waveform is actually loaded —
    // dragging over a loading/error/empty state has nothing to select.
    if (event.button !== 0 || isLoading || !!error || !audioUrl) return;

    const container = waveformRef.current;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    const x = clamp(event.clientX - rect.left, 0, rect.width);
    dragStateRef.current = { startX: x, currentX: x, dragging: true };

    window.addEventListener('mousemove', handleWindowMouseMove);
    window.addEventListener('mouseup', handleWindowMouseUp);
    rafIdRef.current = requestAnimationFrame(renderSelectionFrame);
  }, [isLoading, error, audioUrl, handleWindowMouseMove, handleWindowMouseUp, renderSelectionFrame]);

  // Unmount safety net in case a drag is still in progress.
  useEffect(() => {
    return () => {
      window.removeEventListener('mousemove', handleWindowMouseMove);
      window.removeEventListener('mouseup', handleWindowMouseUp);
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
      }
    };
  }, [handleWindowMouseMove, handleWindowMouseUp]);

  // Keep the overlay canvas's backing store sized (and DPR-scaled) to match
  // the waveform container, independent of WaveSurfer's own canvas.
  useEffect(() => {
    const container = waveformRef.current;
    const canvas = overlayCanvasRef.current;
    if (!container || !canvas) return;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const { clientWidth, clientHeight } = container;
      canvas.width = Math.max(1, Math.round(clientWidth * dpr));
      canvas.height = Math.max(1, Math.round(clientHeight * dpr));
      canvas.style.width = `${clientWidth}px`;
      canvas.style.height = `${clientHeight}px`;
      drawSelection();
    };

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    return () => observer.disconnect();
  }, [drawSelection]);

  // Handle audio URL changes
  useEffect(() => {
    if (!wavesurferRef.current || !audioUrl) {
      return;
    }

    setIsLoading(true);
    setError(null);
    
    // First, test if the URL is accessible with proper credentials for cross-origin
    const fetchOptions: RequestInit = { 
      method: 'HEAD',
      credentials: 'include',  // Include credentials for CORS
      mode: 'cors',           // Explicit CORS mode
      headers: {
        'Accept': 'audio/*',
      }
    };
    
    fetch(audioUrl, fetchOptions)
      .then(response => {
        if (!response.ok) {
          throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }
        
        // If HEAD request succeeds, try to load with WaveSurfer
        try {
          // For cross-origin requests (like ngrok) or custom datasets, try to preload audio data
          if (audioUrl.includes('ngrok') || audioUrl.includes('colab') || audioUrl.includes('https://') || audioUrl.includes('custom%3A')) {
            // First try to fetch audio data
            fetch(audioUrl, {
              credentials: 'include',
              mode: 'cors'
            }).then(audioResponse => {
              if (audioResponse.ok) {
                return audioResponse.blob();
              }
              throw new Error(`Failed to fetch audio data: ${audioResponse.status} ${audioResponse.statusText}`);
            }).then(blob => {
              // Clean up previous blob URL if exists
              if (currentBlobUrlRef.current) {
                URL.revokeObjectURL(currentBlobUrlRef.current);
              }
              const audioBlob = URL.createObjectURL(blob);
              currentBlobUrlRef.current = audioBlob;
              wavesurferRef.current?.load(audioBlob);
            }).catch(blobErr => {
              console.warn('Blob loading failed, trying direct URL:', blobErr);
              // Fallback to direct URL loading
              wavesurferRef.current?.load(audioUrl);
            });
          } else {
            // Local files can be loaded directly
            wavesurferRef.current?.load(audioUrl);
          }
        } catch (err) {
          console.error('WaveSurfer load error:', err);
          setError(`WaveSurfer failed to load: ${err?.message || 'Unknown error'}`);
          setIsLoading(false);
        }
      })
      .catch(err => {
        console.error('Audio URL accessibility test failed:', err);
        setError(`Cannot access audio file: ${err.message}`);
        setIsLoading(false);
        
        // Try with GET request as fallback
        const getFallbackOptions: RequestInit = {
          method: 'GET',
          credentials: 'include',
          mode: 'cors',
          headers: {
            'Accept': 'audio/*',
          }
        };
        
        fetch(audioUrl, getFallbackOptions)
          .then(response => {
            console.log('Audio URL GET fallback response:', response.status);
            if (response.ok) {
              setError('File accessible, trying blob loading for cross-origin compatibility');
              // Try blob loading for cross-origin
              try {
                if (audioUrl.includes('ngrok') || audioUrl.includes('colab') || audioUrl.includes('https://')) {
                  response.blob().then(blob => {
                    // Clean up previous blob URL if exists
                    if (currentBlobUrlRef.current) {
                      URL.revokeObjectURL(currentBlobUrlRef.current);
                    }
                    const audioBlob = URL.createObjectURL(blob);
                    currentBlobUrlRef.current = audioBlob;
                    wavesurferRef.current?.load(audioBlob);
                  }).catch(blobErr => {
                    setError(`Blob loading failed: ${blobErr?.message || 'Unknown error'}`);
                  });
                } else {
                  wavesurferRef.current?.load(audioUrl);
                }
              } catch (wsErr: any) {
                setError(`WaveSurfer error: ${wsErr?.message || 'Unknown error'}`);
              }
            }
          })
          .catch(() => {
            setError('Audio file completely inaccessible');
          });
      });
  }, [audioUrl]);

  // Handle play/pause state
  useEffect(() => {
    if (!wavesurferRef.current) return;
    
    try {
      if (isPlaying) {
        wavesurferRef.current.play();
      } else {
        wavesurferRef.current.pause();
      }
    } catch (err) {
      console.error('Error controlling playback:', err);
    }
  }, [isPlaying]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Card className="p-3 border-gray-200 bg-white shadow-sm">
      <div
        ref={waveformRef}
        className="h-20 bg-white rounded relative min-h-[80px] border border-gray-200 select-none"
        onMouseDown={handleMouseDown}
      >
        <canvas
          ref={overlayCanvasRef}
          className="absolute inset-0 z-[5] pointer-events-none"
        />
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center z-10 bg-white/90">
            <div className="text-xs text-blue-600 flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
              Loading waveform...
            </div>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center z-10 bg-white/90">
            <div className="text-xs text-red-500 text-center px-2">
              <div className="font-medium">⚠️ Error loading audio</div>
              <div className="mt-1">{error}</div>
              {audioUrl && (
                <div className="mt-2">
                  <button 
                    onClick={() => {
                      setError(null);
                      setIsLoading(true);
                      if (wavesurferRef.current && audioUrl) {
                        wavesurferRef.current.load(audioUrl);
                      }
                    }}
                    className="px-2 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700"
                  >
                    Retry
                  </button>
                </div>
              )}
              {audioUrl && (
                <div className="mt-2 text-xs text-blue-600 break-all">
                  URL: {audioUrl}
                </div>
              )}
            </div>
          </div>
        )}
        {!audioUrl && !isLoading && !error && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="text-xs text-gray-500 text-center">
              <div>🎵 No audio file selected</div>
              <div className="mt-1">Upload and select a file to view waveform</div>
            </div>
          </div>
        )}
      </div>
      
      {/* Time markers and controls */}
      <div className="flex justify-between text-xs text-gray-600 mt-2">
        <span>0:00</span>
        <span className="text-center flex-1 italic">
          {audioUrl ? 'Click waveform to seek' : 'Waveform will appear here'}
        </span>
        <span>{wavesurferRef.current ? formatTime(wavesurferRef.current.getDuration() || 0) : '0:00'}</span>
      </div>
    
    </Card>
  );
};