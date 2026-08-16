import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Cell } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { HelpCircle, PlayCircle } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { useTaskStatus } from "@/hooks/useTaskStatus";
import { GlobalTaskProgress } from "../layout/GlobalTaskProgress";

// Built-in Toolbar model keys map to the real HF ids the group-wise WER
// diagnostic actually transcribes with (model_loader_service.py's
// get_whisper_base_models/get_whisper_large_models).
const WHISPER_MODEL_IDS: Record<string, string> = {
  "whisper-base": "openai/whisper-base",
  "whisper-large": "openai/whisper-large-v3",
};

interface CohortSummary {
  accent: string;
  sample_count: number;
  scored_count: number;
  mean_wer: number | null;
  median_wer: number | null;
  stdev_wer: number | null;
  min_wer: number | null;
  max_wer: number | null;
}

interface AccentBiasReport {
  corpus: string;
  model_id: string;
  cohorts: CohortSummary[];
}

interface AccentBiasPanelProps {
  model?: string;
}

// Group-wise WER accent-bias diagnostic (SRS Use Case 6, FR15).
export const AccentBiasPanel: React.FC<AccentBiasPanelProps> = ({ model }) => {
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { state, result } = useTaskStatus(jobId);
  const report = state === 'SUCCESS' ? (typeof result === 'string' ? JSON.parse(result) : result) as AccentBiasReport : null;

  const whisperModelId = WHISPER_MODEL_IDS[model || ""] || null;
  const isRunning = jobId !== null && state !== 'SUCCESS' && state !== 'FAILURE';

  const handleRun = async () => {
    if (!whisperModelId) return;
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/evaluation/accent-bias`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: whisperModelId, corpus: "l2-arctic" }),
      });
      if (!response.ok) throw new Error(`Failed to start diagnostic: ${response.status}`);
      const data = await response.json();
      setJobId(data.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start diagnostic");
    }
  };

  const chartData = (report?.cohorts || [])
    .filter((c) => c.mean_wer !== null)
    .map((c) => ({ accent: c.accent, mean_wer: c.mean_wer as number }));

  const worstWer = chartData.length > 0 ? Math.max(...chartData.map((c) => c.mean_wer)) : 0;
  const bestWer = chartData.length > 0 ? Math.min(...chartData.map((c) => c.mean_wer)) : 0;

  return (
    <TooltipProvider>
      <div className="p-3 space-y-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs flex items-center gap-1.5">
              Accent Bias Dashboard
              <Tooltip>
                <TooltipTrigger><HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" /></TooltipTrigger>
                <TooltipContent className="space-y-1">
                  <p className="text-xs">Runs the selected Whisper model over the L2-ARCTIC non-native reading corpus,</p>
                  <p className="text-xs">grouped by accent (L1), and ranks cohorts by mean Word Error Rate.</p>
                </TooltipContent>
              </Tooltip>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {!whisperModelId && (
              <div className="text-xs text-muted-foreground">
                Select a Whisper model (Whisper Base or Whisper Large) to run the accent-bias diagnostic.
              </div>
            )}

            {whisperModelId && (
              <Button onClick={handleRun} disabled={isRunning} size="sm" className="h-8 text-xs">
                <PlayCircle className="h-3.5 w-3.5 mr-1.5" />
                {isRunning ? "Running diagnostic..." : "Run Accent-Bias Diagnostic (10 samples/cohort)"}
              </Button>
            )}

            {isRunning && <GlobalTaskProgress taskId={jobId} onComplete={() => {}} />}
            {error && <div className="text-xs text-destructive">{error}</div>}
            {state === 'FAILURE' && <div className="text-xs text-destructive">Diagnostic failed in the worker.</div>}

            {report && chartData.length > 0 && (
              <>
                <ResponsiveContainer width="100%" height={Math.max(120, chartData.length * 28)}>
                  <BarChart data={chartData} layout="vertical" margin={{ left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                    <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10 }} />
                    <YAxis type="category" dataKey="accent" tick={{ fontSize: 10 }} width={70} />
                    <RechartsTooltip contentStyle={{ fontSize: 11 }} formatter={(v: number) => v.toFixed(3)} />
                    <Bar dataKey="mean_wer">
                      {chartData.map((entry, i) => (
                        <Cell
                          key={i}
                          fill={
                            entry.mean_wer === worstWer
                              ? "hsl(var(--destructive))"
                              : entry.mean_wer === bestWer
                              ? "hsl(var(--saliency-low))"
                              : "hsl(var(--primary))"
                          }
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div className="flex flex-wrap gap-1.5">
                  <Badge variant="outline" className="text-[10px]">corpus: {report.corpus}</Badge>
                  <Badge variant="outline" className="text-[10px]">cohorts: {chartData.length}</Badge>
                  <Badge variant="destructive" className="text-[10px]">
                    worst: {chartData.find((c) => c.mean_wer === worstWer)?.accent} ({worstWer.toFixed(3)})
                  </Badge>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </TooltipProvider>
  );
};
