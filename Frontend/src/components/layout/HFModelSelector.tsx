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
import { AlertCircle, CheckCircle2, Sparkles, Loader2, XCircle } from "lucide-react";
import { useModelRegistry } from "@/context/ModelRegistryContext";

const ERROR_LABELS: Record<string, string> = {
  UNSUPPORTED_ARCHITECTURE: "Unsupported model architecture",
  UNSAFE_ARTIFACT: "Unsafe or missing safetensors weights",
  HUB_UNAVAILABLE: "Hugging Face Hub is unreachable",
  CANCELLED: "Resolution cancelled",
};

interface HFModelSelectorProps {
  onModelResolved?: (modelId: string) => void;
}

export const HFModelSelector: React.FC<HFModelSelectorProps> = ({ onModelResolved }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [inputModelId, setInputModelId] = useState("");
  const [inputRevision, setInputRevision] = useState("main");

  const { status, resolvedModel, error, resolveModel, cancelResolution } = useModelRegistry();

  const handleResolve = async () => {
    if (!inputModelId.trim()) return;
    const res = await resolveModel(inputModelId.trim(), inputRevision.trim() || "main");
    if (res && onModelResolved) {
      onModelResolved(res.model_id);
    }
  };

  const handleUseModel = () => {
    if (resolvedModel && onModelResolved) {
      onModelResolved(resolvedModel.model_id);
      setIsOpen(false);
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
              value={inputModelId}
              onChange={(e) => setInputModelId(e.target.value)}
              disabled={status === "downloading"}
              className="h-8 text-xs"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="hf-revision" className="text-xs">Revision (optional)</Label>
            <Input
              id="hf-revision"
              placeholder="main"
              value={inputRevision}
              onChange={(e) => setInputRevision(e.target.value)}
              disabled={status === "downloading"}
              className="h-8 text-xs"
            />
          </div>

          <div className="flex items-center gap-2">
            <Button
              onClick={handleResolve}
              disabled={status === "downloading" || !inputModelId.trim()}
              size="sm"
              className="flex-1 h-8 text-xs"
            >
              {status === "downloading" ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                  Resolving...
                </>
              ) : (
                "Resolve Model"
              )}
            </Button>

            {status === "downloading" && (
              <Button
                onClick={cancelResolution}
                variant="destructive"
                size="sm"
                className="h-8 text-xs px-3 bg-destructive/90 hover:bg-destructive"
              >
                <XCircle className="h-3.5 w-3.5 mr-1" />
                Stop
              </Button>
            )}
          </div>

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

          {status === "resolved" && resolvedModel && (
            <div className="p-2.5 rounded border border-primary/30 bg-primary/5 text-xs space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-primary" />
                  <span className="font-medium">Model ready</span>
                </div>
                <Button size="sm" variant="default" className="h-6 text-[11px] px-2" onClick={handleUseModel}>
                  Use Model
                </Button>
              </div>
              <div className="flex flex-wrap gap-1.5">
                <Badge variant="outline" className="text-[10px]">family: {resolvedModel.family}</Badge>
                <Badge variant="outline" className="text-[10px]">
                  revision: {resolvedModel.revision.slice(0, 8)}
                </Badge>
                <Badge variant="outline" className="text-[10px]">
                  hooks: {resolvedModel.available_layers.length} layers
                </Badge>
              </div>
              <div className="text-muted-foreground font-mono text-[10px] break-all">
                sha256:{resolvedModel.weights_sha256.slice(0, 16)}...
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};
