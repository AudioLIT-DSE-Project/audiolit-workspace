import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { CircleDot, Loader2 } from "lucide-react";
import { API_BASE } from "@/lib/api";

interface StatusBarProps {
  activeTaskId?: string | null;
  taskState?: string | null;
}

// SRS §3.9.1 status bar: "Task Queue/Cache Status, GPU/CPU status, Processing
// Progress". GPU/CPU is intentionally omitted - GET /health only reports
// Redis connectivity, and a fabricated reading would be worse than none.
export const StatusBar: React.FC<StatusBarProps> = ({ activeTaskId, taskState }) => {
  const [redisOk, setRedisOk] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    const checkHealth = () => {
      fetch(`${API_BASE}/health`)
        .then((r) => r.json().then((body) => ({ ok: r.ok, body })))
        .then(({ ok, body }) => {
          if (!cancelled) setRedisOk(ok && !!body?.redis);
        })
        .catch(() => {
          if (!cancelled) setRedisOk(false);
        });
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const isProcessing = !!activeTaskId && taskState !== 'SUCCESS' && taskState !== 'FAILURE';

  return (
    <div className="h-6 bg-panel-header border-t border-border px-4 flex items-center justify-between text-[10px] text-muted-foreground shrink-0">
      <div className="flex items-center gap-3">
        <span className="flex items-center gap-1">
          <CircleDot className={`h-2.5 w-2.5 ${redisOk ? 'text-emerald-500' : redisOk === false ? 'text-destructive' : 'text-muted-foreground'}`} />
          Cache/Queue: {redisOk === null ? 'checking...' : redisOk ? 'connected' : 'unreachable'}
        </span>
      </div>
      <div className="flex items-center gap-2">
        {isProcessing ? (
          <span className="flex items-center gap-1">
            <Loader2 className="h-2.5 w-2.5 animate-spin" />
            Job {activeTaskId?.slice(0, 8)} · {taskState || 'PENDING'}
          </span>
        ) : (
          <span>Idle</span>
        )}
      </div>
    </div>
  );
};
