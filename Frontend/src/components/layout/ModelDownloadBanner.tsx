import React from "react";
import { useModelRegistry } from "@/context/ModelRegistryContext";
import { Button } from "@/components/ui/button";
import { Loader2, XCircle, CheckCircle2, AlertCircle, Sparkles } from "lucide-react";

export const ModelDownloadBanner: React.FC = () => {
  const { status, activeModelId, resolvedModel, error, cancelResolution, clearState } = useModelRegistry();

  if (status === "idle") return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 max-w-md w-full bg-card/95 backdrop-blur-md border border-border shadow-2xl rounded-xl p-4 transition-all duration-300 animate-in fade-in slide-in-from-bottom-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2.5">
          {status === "downloading" && (
            <div className="p-2 rounded-lg bg-primary/10 text-primary animate-pulse">
              <Loader2 className="h-5 w-5 animate-spin" />
            </div>
          )}
          {status === "resolved" && (
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-500">
              <CheckCircle2 className="h-5 w-5" />
            </div>
          )}
          {status === "cancelled" && (
            <div className="p-2 rounded-lg bg-amber-500/10 text-amber-500">
              <XCircle className="h-5 w-5" />
            </div>
          )}
          {status === "error" && (
            <div className="p-2 rounded-lg bg-destructive/10 text-destructive">
              <AlertCircle className="h-5 w-5" />
            </div>
          )}

          <div>
            <div className="text-xs font-semibold flex items-center gap-1.5">
              <Sparkles className="h-3.5 w-3.5 text-primary" />
              {status === "downloading" && "Downloading Custom Model..."}
              {status === "resolved" && "Custom Model Ready"}
              {status === "cancelled" && "Download Cancelled"}
              {status === "error" && "Resolution Failed"}
            </div>
            <p className="text-[11px] text-muted-foreground font-mono mt-0.5 truncate max-w-[240px]">
              {activeModelId || resolvedModel?.model_id}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          {status === "downloading" && (
            <Button
              variant="destructive"
              size="sm"
              onClick={cancelResolution}
              className="h-7 text-xs px-2.5 bg-destructive/90 hover:bg-destructive shadow-sm"
            >
              <XCircle className="h-3.5 w-3.5 mr-1" />
              Stop / Cancel
            </Button>
          )}

          {(status === "resolved" || status === "cancelled" || status === "error") && (
            <Button
              variant="ghost"
              size="sm"
              onClick={clearState}
              className="h-7 text-xs px-2 text-muted-foreground hover:text-foreground"
            >
              Dismiss
            </Button>
          )}
        </div>
      </div>

      {status === "downloading" && (
        <div className="mt-3">
          <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
            <div className="h-full bg-primary animate-pulse w-3/4 rounded-full transition-all duration-500" />
          </div>
          <p className="text-[10px] text-muted-foreground mt-1.5">
            Safetensors weight download running in background. You may close dialogs; progress continues.
          </p>
        </div>
      )}

      {status === "error" && error && (
        <p className="text-[11px] text-destructive mt-2 bg-destructive/5 p-1.5 rounded border border-destructive/20">
          {error.message}
        </p>
      )}
    </div>
  );
};
