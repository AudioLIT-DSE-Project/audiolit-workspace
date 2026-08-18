import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Flame, ShieldAlert, Cpu, CheckCircle2, XCircle, Minimize2, Trash2 } from "lucide-react";

export interface WarmupProgress {
  completed: number;
  total: number;
  current_file: string;
  active_subtask?: string;
  status: string;
  percent: number;
  eta_seconds?: number;
  eta_formatted?: string;
}

interface WarmupModalProps {
  isOpen: boolean;
  onClose: () => void;
  dataset: string;
  model: string;
  warmupJobId: string | null;
  warmupProgress: WarmupProgress | null;
  isStarting: boolean;
  onStartWarmup: () => void;
  onCancelWarmup: () => void;
  onMinimize: () => void;
  onClearCache?: () => void;
}

export const WarmupModal: React.FC<WarmupModalProps> = ({
  isOpen,
  onClose,
  dataset,
  model,
  warmupJobId,
  warmupProgress,
  isStarting,
  onStartWarmup,
  onCancelWarmup,
  onMinimize,
  onClearCache,
}) => {
  const isRunning = !!warmupJobId && warmupProgress?.status === "running";
  const isCompleted = warmupProgress?.status === "completed";
  const isCancelled = warmupProgress?.status === "cancelled" || warmupProgress?.status === "cancelling";

  return (
    <Dialog open={isOpen} onOpenChange={(open) => { if (!open && !isRunning) onClose(); }}>
      <DialogContent className="max-w-md bg-background border-border shadow-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-base">
            <Flame className="h-5 w-5 text-amber-500" />
            {isRunning
              ? "Dataset Warmup in Progress"
              : isCompleted
              ? "Warmup Complete"
              : isCancelled
              ? "Warmup Cancelled"
              : "Confirm Dataset Warmup"}
          </DialogTitle>
          <DialogDescription className="text-xs text-muted-foreground pt-1">
            {!warmupJobId
              ? "Pre-compute and cache XAI saliency maps, acoustic profiles, and predictions for the entire dataset."
              : "Background evaluation runner is processing dataset samples."}
          </DialogDescription>
        </DialogHeader>

        {/* Confirmation View (Before Starting) */}
        {!warmupJobId && (
          <div className="space-y-3 py-2 text-xs">
            <div className="bg-muted/40 p-3 rounded-md space-y-1.5 border border-border">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Target Dataset:</span>
                <span className="font-semibold text-foreground uppercase">{dataset || "Selected Dataset"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Target Model:</span>
                <span className="font-semibold text-foreground">{model || "Whisper Base"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Pipelines Included:</span>
                <span className="font-medium text-foreground">ASR, SER, Acoustic, Saliency</span>
              </div>
            </div>

            <div className="flex items-start gap-2 bg-amber-50 dark:bg-amber-950/30 text-amber-800 dark:text-amber-300 p-2.5 rounded-md border border-amber-200 dark:border-amber-800 text-[11px]">
              <Cpu className="h-4 w-4 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">CPU Thermal & Hardware Safety</p>
                <p className="opacity-90">Inserts 100ms cooling yields between files to prevent CPU overheating during long runs.</p>
              </div>
            </div>

            <div className="flex items-start gap-2 bg-blue-50 dark:bg-blue-950/30 text-blue-800 dark:text-blue-300 p-2.5 rounded-md border border-blue-200 dark:border-blue-800 text-[11px]">
              <ShieldAlert className="h-4 w-4 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Cancellable & Cache Persistent</p>
                <p className="opacity-90">You can cancel at any time. All samples processed up to cancellation remain 100% saved in Redis.</p>
              </div>
            </div>
          </div>
        )}

        {/* Active Progress View */}
        {warmupJobId && (
          <div className="space-y-4 py-2 text-xs">
            <div className="space-y-2">
              <div className="flex justify-between items-center font-medium">
                <span className="flex items-center gap-1.5">
                  {isRunning && <span className="animate-spin text-sky-500">⏳</span>}
                  {isCompleted && <CheckCircle2 className="h-4 w-4 text-emerald-500" />}
                  {isCancelled && <XCircle className="h-4 w-4 text-amber-500" />}
                  <span className="capitalize">{warmupProgress?.status || "Running"}</span>
                </span>
                <span className="font-mono text-xs">{warmupProgress?.completed || 0} / {warmupProgress?.total || 100} ({warmupProgress?.percent || 0}%)</span>
              </div>
              <Progress value={warmupProgress?.percent || 0} className="h-2" />
            </div>

            <div className="bg-muted/40 p-2.5 rounded-md border border-border space-y-1.5 text-[11px]">
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Current File:</span>
                <span className="font-mono truncate max-w-[220px] text-foreground font-medium">{warmupProgress?.current_file || "Processing..."}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Active Pipeline:</span>
                <span className="font-semibold text-amber-600 dark:text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
                  {warmupProgress?.active_subtask || "Processing..."}
                </span>
              </div>
              {isRunning && warmupProgress?.eta_formatted && (
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Estimated Time Remaining:</span>
                  <span className="font-mono text-amber-600 dark:text-amber-400 font-semibold bg-amber-500/10 px-1.5 py-0.5 rounded">
                    ~{warmupProgress.eta_formatted}
                  </span>
                </div>
              )}
              <div className="flex justify-between text-muted-foreground pt-0.5">
                <span>Thermal Protection:</span>
                <span className="text-emerald-600 dark:text-emerald-400 font-medium">100ms Safety Cooldown Active</span>
              </div>
            </div>
          </div>
        )}

        <DialogFooter className="flex items-center justify-between sm:justify-between gap-2 pt-2 border-t border-border">
          {!warmupJobId ? (
            <>
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="sm" onClick={onClose} className="text-xs h-8">
                  Cancel
                </Button>
                {onClearCache && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={onClearCache}
                    className="text-xs h-8 text-destructive border-destructive/30 hover:bg-destructive/10 gap-1"
                    title="Clear cached ML predictions, acoustic profiles, and saliency heatmaps"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    Clear Cache
                  </Button>
                )}
              </div>
              <Button
                variant="default"
                size="sm"
                onClick={onStartWarmup}
                disabled={isStarting || !dataset}
                className="text-xs h-8 bg-amber-600 hover:bg-amber-700 text-white gap-1.5"
              >
                <Flame className="h-3.5 w-3.5" />
                {isStarting ? "Starting..." : "Start Warmup"}
              </Button>
            </>
          ) : (
            <>
              <Button
                variant="outline"
                size="sm"
                onClick={onMinimize}
                className="text-xs h-8 gap-1 text-muted-foreground"
                title="Minimize modal to run in background while browsing"
              >
                <Minimize2 className="h-3.5 w-3.5" />
                Minimize Progress
              </Button>
              {isRunning && (
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={onCancelWarmup}
                  className="text-xs h-8"
                >
                  Cancel Warmup
                </Button>
              )}
              {(isCompleted || isCancelled) && (
                <Button variant="default" size="sm" onClick={onClose} className="text-xs h-8">
                  Done
                </Button>
              )}
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
