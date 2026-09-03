import React from "react";
import { WarmupProgress } from "./WarmupModal";
import { Button } from "@/components/ui/button";
import { Flame, Maximize2, XCircle, CheckCircle2, AlertCircle, Clock } from "lucide-react";

interface WarmupStatusBannerProps {
  warmupJobId: string | null;
  warmupProgress: WarmupProgress | null;
  dataset: string;
  isMinimized: boolean;
  onExpand: () => void;
  onCancel: () => void;
  onDismiss: () => void;
}

export const WarmupStatusBanner: React.FC<WarmupStatusBannerProps> = ({
  warmupJobId,
  warmupProgress,
  dataset,
  isMinimized,
  onExpand,
  onCancel,
  onDismiss,
}) => {
  if (!warmupJobId || !isMinimized) return null;

  const isRunning = warmupProgress?.status === "running";
  const isCompleted = warmupProgress?.status === "completed";
  const isCancelled = warmupProgress?.status === "cancelled" || warmupProgress?.status === "cancelling";

  return (
    <div className="fixed bottom-4 right-4 z-50 max-w-md w-full bg-card/95 backdrop-blur-md border border-border shadow-2xl rounded-xl p-4 transition-all duration-300 animate-in fade-in slide-in-from-bottom-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2.5">
          {isRunning && (
            <div className="p-2 rounded-lg bg-amber-500/10 text-amber-500 animate-pulse">
              <Flame className="h-5 w-5" />
            </div>
          )}
          {isCompleted && (
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-500">
              <CheckCircle2 className="h-5 w-5" />
            </div>
          )}
          {isCancelled && (
            <div className="p-2 rounded-lg bg-amber-500/10 text-amber-500">
              <XCircle className="h-5 w-5" />
            </div>
          )}

          <div>
            <div className="text-xs font-semibold flex items-center gap-1.5">
              <Flame className="h-3.5 w-3.5 text-amber-500" />
              {isRunning && "Dataset Warmup Running"}
              {isCompleted && "Dataset Warmup Complete"}
              {isCancelled && "Warmup Cancelled"}
            </div>
            <p className="text-[11px] text-muted-foreground font-mono mt-0.5 truncate max-w-[240px]">
              <span className="uppercase font-semibold text-foreground">{dataset}</span> · {warmupProgress?.completed || 0}/{warmupProgress?.total || 0} ({warmupProgress?.percent || 0}%)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          <Button
            variant="outline"
            size="sm"
            onClick={onExpand}
            className="h-7 text-xs px-2.5 gap-1 border-border text-foreground hover:bg-muted"
            title="Expand Warmup Progress Modal"
          >
            <Maximize2 className="h-3.5 w-3.5" />
            Expand
          </Button>

          {isRunning && (
            <Button
              variant="destructive"
              size="sm"
              onClick={onCancel}
              className="h-7 text-xs px-2 bg-destructive/90 hover:bg-destructive shadow-sm"
              title="Stop Dataset Warmup"
            >
              <XCircle className="h-3.5 w-3.5" />
              Stop
            </Button>
          )}

          {(isCompleted || isCancelled) && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onDismiss}
              className="h-7 text-xs px-2 text-muted-foreground hover:text-foreground"
            >
              Dismiss
            </Button>
          )}
        </div>
      </div>

      {isRunning && (
        <div className="mt-3 space-y-1.5">
          <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
            <div
              className="h-full bg-amber-500 transition-all duration-500 rounded-full"
              style={{ width: `${Math.min(100, Math.max(0, warmupProgress?.percent || 0))}%` }}
            />
          </div>

          <div className="flex justify-between items-center text-[10px] text-muted-foreground">
            <span className="truncate max-w-[220px]">
              {warmupProgress?.active_subtask || "Processing audio clips..."}
            </span>
            {warmupProgress?.eta_formatted && (
              <span className="font-mono text-amber-600 dark:text-amber-400 font-medium flex items-center gap-1 shrink-0">
                <Clock className="h-3 w-3" />
                ~{warmupProgress.eta_formatted}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
