import React, { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, CheckCircle2, Sparkles } from "lucide-react";
import { API_BASE } from "@/lib/api";

interface ResolvedModel {
  model_id: string;
  revision: string;
  family: string;
  weights_sha256: string;
  available_layers: string[];
}

interface ResolveError {
  code: string;
  message: string;
}

// SRS Use Case 5: "An unsafe file, an unsupported model, or a download
// problem each produce a distinct, clear message" - human-readable labels
// for the typed error codes POST /models/resolve returns (LIT-231).
const ERROR_LABELS: Record<string, string> = {
  UNSUPPORTED_ARCHITECTURE: "Unsupported model architecture",
  UNSAFE_ARTIFACT: "Unsafe or missing safetensors weights",
  HUB_UNAVAILABLE: "Hugging Face Hub is unreachable",
};

export const HFModelSelector: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [modelId, setModelId] = useState("");
  const [revision, setRevision] = useState("main");
  const [loading, setLoading] = useState(false);
  const [resolved, setResolved] = useState<ResolvedModel | null>(null);
  const [error, setError] = useState<ResolveError | null>(null);

  const handleResolve = async () => {
    if (!modelId.trim()) return;
    setLoading(true);
    setError(null);
    setResolved(null);

    try {
      const response = await fetch(`${API_BASE}/models/resolve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId.trim(), revision: revision.trim() || "main" }),
      });

      const body = await response.json();
      if (!response.ok) {
        const detail = body?.detail;
        if (detail && typeof detail === "object" && detail.code) {
          setError({ code: detail.code, message: detail.message || "Failed to resolve model" });
        } else {
          setError({ code: "ERROR", message: typeof detail === "string" ? detail : "Failed to resolve model" });
        }
        return;
      }
      setResolved(body as ResolvedModel);
    } catch (err) {
      setError({ code: "ERROR", message: err instanceof Error ? err.message : "Failed to resolve model" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="h-7 text-xs">
          <Sparkles className="h-3.5 w-3.5 mr-1.5" />
          Custom Model
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Add a Hugging Face Model</DialogTitle>
          <DialogDescription>
            Resolve a Whisper or Wav2Vec2-family model from the Hugging Face Hub. The registry
            downloads it safely, verifies its weights, and lists the layers available for
            attribution.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          <div className="space-y-1.5">
            <Label htmlFor="hf-model-id" className="text-xs">Model ID</Label>
            <Input
              id="hf-model-id"
              placeholder="e.g. openai/whisper-base"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="h-8 text-xs"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="hf-revision" className="text-xs">Revision (optional)</Label>
            <Input
              id="hf-revision"
              placeholder="main"
              value={revision}
              onChange={(e) => setRevision(e.target.value)}
              className="h-8 text-xs"
            />
          </div>

          <Button
            onClick={handleResolve}
            disabled={loading || !modelId.trim()}
            size="sm"
            className="w-full h-8 text-xs"
          >
            {loading ? "Resolving..." : "Resolve Model"}
          </Button>

          {error && (
            <div className="flex items-start gap-2 p-2.5 rounded border border-destructive/30 bg-destructive/5 text-xs">
              <AlertCircle className="h-3.5 w-3.5 text-destructive mt-0.5 shrink-0" />
              <div>
                <div className="font-medium text-destructive">
                  {ERROR_LABELS[error.code] || error.code}
                </div>
                <div className="text-muted-foreground mt-0.5">{error.message}</div>
              </div>
            </div>
          )}

          {resolved && (
            <div className="p-2.5 rounded border border-primary/30 bg-primary/5 text-xs space-y-1.5">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-3.5 w-3.5 text-primary" />
                <span className="font-medium">Model ready</span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                <Badge variant="outline" className="text-[10px]">family: {resolved.family}</Badge>
                <Badge variant="outline" className="text-[10px]">
                  revision: {resolved.revision.slice(0, 8)}
                </Badge>
                <Badge variant="outline" className="text-[10px]">
                  hooks: {resolved.available_layers.length} layers
                </Badge>
              </div>
              <div className="text-muted-foreground font-mono text-[10px] break-all">
                sha256:{resolved.weights_sha256.slice(0, 16)}...
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};
