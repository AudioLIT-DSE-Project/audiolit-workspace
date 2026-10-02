import { useEffect, useRef, useState } from 'react';
import { API_BASE } from '@/lib/api';

export type TaskState = 'QUEUED' | 'PROCESSING' | 'RETRYING' | 'SUCCESS' | 'FAILURE' | 'UNKNOWN';

/**
 * WebSocket origin for the backend, derived from the same API_BASE every other
 * component uses. Deriving it from window.location instead pointed the socket at
 * the Vite dev server (port 8080, no proxy) rather than the API on 8000, and
 * forced ws:// on an https:// page, where the browser blocks it as mixed
 * content. http -> ws, https -> wss.
 */
const wsOrigin = (): string => {
  const base = new URL(API_BASE, window.location.origin);
  base.protocol = base.protocol === 'https:' ? 'wss:' : 'ws:';
  return base.origin;
};

interface UseTaskStatusResult {
  state: TaskState;
  result: any;
  error: string | null;
}

const KNOWN_STATES: readonly TaskState[] = ['QUEUED', 'PROCESSING', 'RETRYING', 'SUCCESS', 'FAILURE', 'UNKNOWN'];

/**
 * Workers also publish finer-grained stages ("asr.running", "aggregated"...).
 * They all mean the job is under way; passing them through as the state left
 * every consumer that switches on TaskState showing its "unknown" branch.
 */
const toTaskState = (value: unknown): TaskState =>
  KNOWN_STATES.includes(value as TaskState) ? (value as TaskState) : 'PROCESSING';

export const useTaskStatus = (taskId: string | null): UseTaskStatusResult => {
  const [state, setState] = useState<TaskState>('QUEUED');
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const pollIntervalRef = useRef<NodeJS.Timeout>();
  const retryCountRef = useRef<number>(0);
  const isManualClose = useRef(false);

  useEffect(() => {
    if (!taskId) return;

    // Set on cleanup, so a late response for a previous task id cannot write
    // its outcome into the state of the task that replaced it.
    let cancelled = false;
    isManualClose.current = false;
    retryCountRef.current = 0;
    setState('QUEUED');
    // Clear the previous task's outcome, or it renders briefly under the new id.
    setResult(null);
    setError(null);

    /** One read of the job's final state from the polling route. */
    const fetchOutcome = async (): Promise<boolean> => {
      try {
        const res = await fetch(`${API_BASE}/api/tasks/${taskId}/status`);
        const data = await res.json();
        if (cancelled) return true;
        const polled = toTaskState(data.state);
        if (polled === 'SUCCESS') {
          setResult(data.result ?? null);
          setState('SUCCESS');
          return true;
        }
        if (polled === 'FAILURE') {
          setError(data.error || 'Task failed');
          setState('FAILURE');
          return true;
        }
        setState(polled);
      } catch (e) {
        console.error('[Polling] Failed to fetch status', e);
      }
      return false;
    };

    const connectWs = () => {
      const ws = new WebSocket(`${wsOrigin()}/api/ws/tasks/${taskId}`);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log(`[WS] Connected for task ${taskId}`);
        retryCountRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const rawState = data.state || data.stage;
          if (!rawState) return;
          const currentState = toTaskState(rawState);

          if (currentState === 'SUCCESS') {
            isManualClose.current = true;
            const taskResult = data.payload?.result;
            if (taskResult === undefined || taskResult === null) {
              // The result is `payload.result` and nothing else. This used to
              // fall back to `data.payload`, which for a live SUCCESS event is
              // the worker's `{duration_s}` - so consumers were handed timing
              // metadata as their result. A SUCCESS without one is fetched.
              void fetchOutcome().then((finished) => {
                if (!finished && !cancelled) startPolling();
              });
              return;
            }
            // Result before state: consumers act on `SUCCESS && result`.
            setResult(taskResult);
            setState('SUCCESS');
          } else if (currentState === 'FAILURE') {
            isManualClose.current = true;
            setError(data.payload?.error || 'Task failed');
            setState('FAILURE');
          } else {
            setState(currentState);
          }
        } catch (e) {
          console.error('[WS] Failed to parse message', e);
        }
      };

      ws.onerror = () => {
        console.warn(`[WS] Error for task ${taskId}`);
      };

      ws.onclose = () => {
        if (isManualClose.current) return;
        
        retryCountRef.current += 1;
        console.warn(`[WS] Disconnected. Retry attempt: ${retryCountRef.current}`);

        if (retryCountRef.current > 3) {
          console.warn(`[WS] Max retries reached. Falling back to HTTP polling.`);
          startPolling();
        } else {
          const delay = Math.pow(2, retryCountRef.current) * 1000;
          reconnectTimeoutRef.current = setTimeout(connectWs, delay);
        }
      };
    };

    const startPolling = () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
      
      pollIntervalRef.current = setInterval(async () => {
        const finished = await fetchOutcome();
        if (finished && pollIntervalRef.current) clearInterval(pollIntervalRef.current);
      }, 2000);
    };

    connectWs();

    return () => {
      cancelled = true;
      isManualClose.current = true;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [taskId]);

  return { state, result, error };
};
