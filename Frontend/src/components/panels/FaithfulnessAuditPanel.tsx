import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { HelpCircle, PlayCircle, Loader2 } from "lucide-react";
import { API_BASE } from "@/lib/api";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
}

interface FaithfulnessAuditPanelProps {
  selectedFile?: UploadedFile | null;
  selectedEmbeddingFile?: string | null;
  model?: string;
  dataset?: string;
  originalDataset?: string;
}

interface DegradationStep {
  k_percent: number;
  masked_confidence: number;
  confidence_drop: number;
  degradation_ratio: number;
}

interface FaithfulnessResult {
  target_class: string;
  baseline_confidence: number;
  degradation_curve: DegradationStep[];
  audc: number;
  mean_degradation_score: number;
  degradation_trend: string;
  audit_verdict: string;
}

const getRequestRef = (
  selectedFile: UploadedFile | null | undefined,
  selectedEmbeddingFile: string | null | undefined,
  dataset: string | undefined,
  originalDataset: string | undefined,
) => {
  if (selectedFile) {
    const isUploadedFile = selectedFile.file_path && (
      selectedFile.file_path.includes('uploads/') ||
      selectedFile.file_path.startsWith('uploads/') ||
      selectedFile.message === "Perturbed file" ||
      selectedFile.message === "File uploaded successfully" ||
      selectedFile.message === "File uploaded and processed successfully"
    ) && !selectedFile.message.includes("Selected from");

    if (isUploadedFile) return { file_path: selectedFile.file_path };
    const datasetToUse = originalDataset && originalDataset !== "custom" ? originalDataset : dataset;
    return { dataset: datasetToUse, dataset_file: selectedFile.filename };
  }
  if (selectedEmbeddingFile && dataset) {
    const datasetToUse = originalDataset && originalDataset !== "custom" ? originalDataset : dataset;
    return { dataset: datasetToUse, dataset_file: selectedEmbeddingFile };
  }
  return null;
};

// Faithfulness Audit / Deletion Score (SRS §2.2, FR16). Only meaningful for
// SER/ADD (perturbation_service.evaluate_downstream_degradation only
// supports "ser"/"add" model types - not ASR/Whisper).
export const FaithfulnessAuditPanel: React.FC<FaithfulnessAuditPanelProps> = ({
  selectedFile,
  selectedEmbeddingFile,
  model,
  dataset,
  originalDataset,
}) => {
  const [method, setMethod] = useState("gradcam");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<FaithfulnessResult | null>(null);

  const isSerModel = model === "wav2vec2";
  const ref = getRequestRef(selectedFile, selectedEmbeddingFile, dataset, originalDataset);

  const handleRun = async () => {
    if (!ref || !isSerModel) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch(`${API_BASE}/evaluation/faithfulness`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, method, ...ref }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || `Faithfulness audit failed: ${response.status}`);
      }
      setResult(await response.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Faithfulness audit failed");
    } finally {
      setLoading(false);
    }
  };

  const chartData = (result?.degradation_curve || []).map((s) => ({
    k: `${s.k_percent}%`,
    degradation_ratio: s.degradation_ratio,
  }));

  return (
    <TooltipProvider>
      <div className="p-3 space-y-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs flex items-center gap-1.5">
              Faithfulness Audit (Deletion Score)
              <Tooltip>
                <TooltipTrigger><HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" /></TooltipTrigger>
                <TooltipContent className="space-y-1">
                  <p className="text-xs">Masks the highest-saliency regions of the audio at increasing thresholds,</p>
                  <p className="text-xs">re-runs the model on each masked clip, and measures the confidence drop -</p>
                  <p className="text-xs">a real, measured degradation curve, not a simulated estimate.</p>
                </TooltipContent>
              </Tooltip>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {!isSerModel && (
              <div className="text-xs text-muted-foreground">
                Faithfulness auditing is available for the Wav2Vec2 (SER/ADD) model only.
              </div>
            )}

            {isSerModel && (
              <div className="flex items-center gap-2">
                <Select value={method} onValueChange={setMethod}>
                  <SelectTrigger className="w-28 h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="gradcam">Grad-CAM</SelectItem>
                    <SelectItem value="integrated_gradients">Integrated Gradients</SelectItem>
                    <SelectItem value="lime">LIME</SelectItem>
                    <SelectItem value="shap">SHAP</SelectItem>
                  </SelectContent>
                </Select>
                <Button onClick={handleRun} disabled={loading || !ref} size="sm" className="h-8 text-xs">
                  <PlayCircle className="h-3.5 w-3.5 mr-1.5" />
                  {loading ? "Auditing..." : "Run Audit"}
                </Button>
              </div>
            )}

            {loading && <div className="flex items-center gap-2 text-xs text-muted-foreground py-2"><Loader2 className="h-3.5 w-3.5 animate-spin" />Masking and re-running inference at each threshold...</div>}
            {error && <div className="text-xs text-destructive">{error}</div>}

            {result && (
              <>
                <div className="flex flex-wrap gap-1.5">
                  <Badge variant="outline" className="text-[10px]">baseline confidence: {result.baseline_confidence}</Badge>
                  <Badge variant="outline" className="text-[10px]">AUDC: {result.audc}</Badge>
                  <Badge variant={result.audit_verdict === "faithful" ? "default" : "destructive"} className="text-[10px]">
                    {result.audit_verdict}
                  </Badge>
                  <Badge variant="outline" className="text-[10px]">{result.degradation_trend.replace('_', ' ')}</Badge>
                </div>
                <ResponsiveContainer width="100%" height={140}>
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                    <XAxis dataKey="k" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} domain={[0, 1]} />
                    <RechartsTooltip contentStyle={{ fontSize: 11 }} formatter={(v: number) => v.toFixed(3)} />
                    <Bar dataKey="degradation_ratio" fill="hsl(var(--saliency-high))" />
                  </BarChart>
                </ResponsiveContainer>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </TooltipProvider>
  );
};
