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

    isManualClose.current = false;
    retryCountRef.current = 0;
    setState('QUEUED');
    // Clear the previous task's outcome, or it renders briefly under the new id.
    setResult(null);
    setError(null);

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
          const currentState = data.state || data.stage;
          
          if (currentState) {
            setState(currentState as TaskState);
            
            if (currentState === 'SUCCESS') {
              setResult(data.payload?.result || data.payload);
              isManualClose.current = true;
            } else if (currentState === 'FAILURE') {
              setError(data.payload?.error || 'Task failed');
              isManualClose.current = true;
            }
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
        try {
          const res = await fetch(`${API_BASE}/api/tasks/${taskId}/status`);
          const data = await res.json();
          setState(data.state as TaskState);
          
          if (data.state === 'SUCCESS' || data.state === 'FAILURE') {
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
            if (data.error) setError(data.error);
            if (data.result) setResult(data.result);
          }
        } catch (e) {
          console.error('[Polling] Failed to fetch status', e);
        }
      }, 2000);
    };

    connectWs();

    return () => {
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
