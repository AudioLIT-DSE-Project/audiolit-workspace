import { useEffect, useRef, useState } from 'react';

export type TaskState = 'QUEUED' | 'PROCESSING' | 'RETRYING' | 'SUCCESS' | 'FAILURE' | 'UNKNOWN';

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

    const connectWs = () => {
      const wsHost = window.location.hostname;
      const wsPort = window.location.port || '8000';
      const wsUrl = `ws://${wsHost}:${wsPort}/api/ws/tasks/${taskId}`;
      
      const ws = new WebSocket(wsUrl);
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
          const res = await fetch(`/api/tasks/${taskId}/status`);
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
