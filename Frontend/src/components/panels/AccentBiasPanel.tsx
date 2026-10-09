import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Cell } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { HelpCircle, PlayCircle } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { useTaskStatus } from "@/hooks/useTaskStatus";
import { GlobalTaskProgress } from "../layout/GlobalTaskProgress";
import { getModelTaskFamily } from "@/lib/modelTask";

// Built-in Toolbar model keys map to the real HF ids the group-wise WER
// diagnostic actually transcribes with (model_loader_service.py's
// get_whisper_base_models).
const WHISPER_MODEL_IDS: Record<string, string> = {
  "whisper-base": "openai/whisper-base",
};

/**
 * The Hugging Face id to transcribe with, or null when the selected model
 * cannot produce a transcript. A custom model is listed in the toolbar under
 * its own Hugging Face id, so a Whisper checkpoint added that way is passed
 * through as it is.
 */
const resolveWhisperModelId = (model?: string): string | null => {
  if (!model) return null;
  if (WHISPER_MODEL_IDS[model]) return WHISPER_MODEL_IDS[model];
  return model.includes("/") && model.toLowerCase().includes("whisper") ? model : null;
};

/**
 * What the diagnostic measures on each dataset it can run on. A dataset is
 * here only if it labels its speakers with something to group by; the rest of
 * the corpora have nothing to compare.
 *
 * - `wer`: word error rate per group, with a transcribing (Whisper) model.
 * - `emotion`: emotion-recognition accuracy per group, with the SER model.
 */
interface BiasDataset {
  label: string;
  measure: "wer" | "emotion";
  groupings: Array<{ value: string; label: string }>;
  samplesPerGroup: number;
  description: string;
}

const BIAS_DATASETS: Record<string, BiasDataset> = {
  "l2-arctic": {
    label: "L2-ARCTIC",
    measure: "wer",
    groupings: [{ value: "accent", label: "accent" }],
    samplesPerGroup: 10,
    description:
      "Transcribes non-native English speakers who all read the same sentences, grouped by first language, and ranks the groups by mean word error rate.",
  },
  "crema-d": {
    label: "CREMA-D",
    measure: "emotion",
    groupings: [
      { value: "race", label: "race" },
      { value: "sex", label: "sex" },
      { value: "ethnicity", label: "ethnicity" },
    ],
    samplesPerGroup: 25,
    description:
      "Classifies the emotion of acted clips, grouped by the actor's race, sex or ethnicity, and ranks the groups by accuracy.",
  },
  esd: {
    label: "ESD",
    measure: "emotion",
    groupings: [{ value: "language", label: "language" }],
    samplesPerGroup: 25,
    description:
      "Classifies the emotion of clips from Mandarin and English speakers and compares accuracy between the two languages.",
  },
};

/** One bar of the chart, whichever measure produced it. */
interface CohortRow {
  name: string;
  value: number;
}

interface BiasReport {
  corpus: string;
  model_id: string;
  metric?: string;
  group_by?: string;
  cohorts: Array<Record<string, unknown>>;
}

/** The report's cohorts as chart rows; cohorts with no scored clip are left out. */
const toRows = (report: BiasReport, measure: BiasDataset["measure"]): CohortRow[] =>
  report.cohorts
    .map((c) =>
      measure === "wer"
        ? { name: String(c.accent ?? ""), value: c.mean_wer as number | null }
        : { name: String(c.group ?? ""), value: c.accuracy as number | null },
    )
    .filter((row): row is CohortRow => typeof row.value === "number");

interface AccentBiasPanelProps {
  model?: string;
  /** The dataset selected in the toolbar; the diagnostic follows it. */
  dataset?: string;
}

// Group-wise bias diagnostic (SRS Use Case 6, FR15): word error rate per accent
// on L2-ARCTIC, emotion accuracy per speaker group on CREMA-D and ESD.
export const AccentBiasPanel: React.FC<AccentBiasPanelProps> = ({ model, dataset }) => {
  const config = dataset ? BIAS_DATASETS[dataset.toLowerCase()] : undefined;
  const [groupBy, setGroupBy] = useState<string>(config?.groupings[0].value ?? "");
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { state, result, error: taskError } = useTaskStatus(jobId);

  // A report belongs to the dataset, model and grouping it was run for.
  useEffect(() => {
    setJobId(null);
    setError(null);
    setGroupBy(config?.groupings[0].value ?? "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, model]);

  if (!config) {
    return (
      <TooltipProvider>
        <div className="p-3 space-y-3">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs">Accent Bias Dashboard</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xs text-muted-foreground">
                Bias profiling is available for the L2-ARCTIC, CREMA-D and ESD datasets only. Select one of them in the toolbar to run it.
              </div>
            </CardContent>
          </Card>
        </div>
      </TooltipProvider>
    );
  }

  const isWer = config.measure === "wer";
  // The model that can produce this dataset's measure, or null.
  const modelId = !model
    ? null
    : isWer
      ? resolveWhisperModelId(model)
      : getModelTaskFamily(model) === "SER" ? model : null;

  // `jobId` guards against the hook's last result outliving a reset.
  const parsed = jobId && state === 'SUCCESS' ? (typeof result === 'string' ? JSON.parse(result) : result) : null;
  const report = parsed && Array.isArray(parsed.cohorts) ? (parsed as BiasReport) : null;
  const isRunning = jobId !== null && state !== 'SUCCESS' && state !== 'FAILURE';

  const handleRun = async () => {
    if (!modelId || !dataset) return;
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/evaluation/accent-bias`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          isWer
            ? { model_id: modelId, corpus: dataset.toLowerCase() }
            : { model_id: modelId, corpus: dataset.toLowerCase(), group_by: groupBy, samples_per_cohort: config.samplesPerGroup },
        ),
      });
      if (!response.ok) {
        const detail = await response.json().then((body) => body?.detail).catch(() => null);
        throw new Error(detail || `Failed to start diagnostic: ${response.status}`);
      }
      const data = await response.json();
      setJobId(data.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start diagnostic");
    }
  };

  const chartData = report ? toRows(report, config.measure) : [];
  const values = chartData.map((c) => c.value);
  // "Worst" is the highest error rate, or the lowest accuracy.
  const worst = values.length === 0 ? 0 : isWer ? Math.max(...values) : Math.min(...values);
  const best = values.length === 0 ? 0 : isWer ? Math.min(...values) : Math.max(...values);
  // WER is not capped at 1: insertions can outnumber the reference words. A
  // fixed 0..1 axis clipped exactly the cohorts the chart exists to expose.
  const axisMax = isWer ? Math.max(1, Math.ceil(worst * 10) / 10) : 1;
  const format = (v: number) => (isWer ? v.toFixed(3) : `${(v * 100).toFixed(1)}%`);
  const groupLabel = config.groupings.find((g) => g.value === (report?.group_by ?? groupBy))?.label ?? "group";

  return (
    <TooltipProvider>
      <div className="p-3 space-y-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs flex items-center gap-1.5">
              Accent Bias Dashboard
              <Badge variant="outline" className="text-[10px]">
                {config.label} · {isWer ? "word error rate" : "emotion accuracy"}
              </Badge>
              <Tooltip>
                <TooltipTrigger aria-label={config.description}>
                    <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" aria-hidden="true" />
                  </TooltipTrigger>
                <TooltipContent className="max-w-[260px]">
                  <p className="text-xs">{config.description}</p>
                </TooltipContent>
              </Tooltip>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {!modelId && (
              <div className="text-xs text-muted-foreground">
                {isWer
                  ? "Select a Whisper model to run the accent-bias diagnostic. It measures word error rate, so it needs a model that transcribes."
                  : `Select the Wav2Vec2 (SER) model to run the bias diagnostic on ${config.label}. It measures emotion accuracy, so it needs a model that classifies emotion.`}
              </div>
            )}

            {modelId && (
              <div className="flex flex-wrap items-center gap-2">
                {config.groupings.length > 1 && (
                  <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    Group by
                    <select
                      value={groupBy}
                      disabled={isRunning}
                      onChange={(e) => { setGroupBy(e.target.value); setJobId(null); }}
                      className="h-8 rounded-md border border-input bg-background px-2 text-xs text-foreground"
                    >
                      {config.groupings.map((g) => (
                        <option key={g.value} value={g.value}>{g.label}</option>
                      ))}
                    </select>
                  </label>
                )}
                <Button onClick={handleRun} disabled={isRunning} size="sm" className="h-8 text-xs">
                  <PlayCircle className="h-3.5 w-3.5 mr-1.5" />
                  {isRunning
                    ? "Running diagnostic..."
                    : isWer
                      ? `Run Accent-Bias Diagnostic (${config.samplesPerGroup} samples/cohort)`
                      : `Run Bias Diagnostic (up to ${config.samplesPerGroup} samples/group)`}
                </Button>
              </div>
            )}

            {isRunning && <GlobalTaskProgress taskId={jobId} onComplete={() => {}} />}
            {error && <div className="text-xs text-destructive">{error}</div>}
            {jobId && state === 'FAILURE' && (
              <div className="text-xs text-destructive">Diagnostic failed: {taskError || "the worker reported an error."}</div>
            )}
            {report && chartData.length === 0 && (
              <div className="text-xs text-muted-foreground">
                The diagnostic finished but no group could be scored. Check that the {config.label} corpus is provisioned under Backend/data.
              </div>
            )}

            {report && chartData.length > 0 && (
              <>
                <div className="text-[11px] text-muted-foreground">
                  {isWer ? "Mean word error rate" : "Emotion accuracy"} by {groupLabel}. {isWer ? "Longer bars are worse." : "Shorter bars are worse."}
                </div>
                <ResponsiveContainer width="100%" height={Math.max(120, chartData.length * 28)}>
                  <BarChart data={chartData} layout="vertical" margin={{ left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                    <XAxis type="number" domain={[0, axisMax]} tick={{ fontSize: 10 }} tickFormatter={(v: number) => (isWer ? String(v) : `${Math.round(v * 100)}%`)} />
                    <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={110} />
                    <RechartsTooltip contentStyle={{ fontSize: 11 }} formatter={(v: number) => format(v)} />
                    <Bar dataKey="value" name={isWer ? "mean WER" : "accuracy"}>
                      {chartData.map((entry, i) => (
                        <Cell
                          key={i}
                          fill={
                            entry.value === worst
                              ? "hsl(var(--destructive))"
                              : entry.value === best
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
                    worst: {chartData.find((c) => c.value === worst)?.name} ({format(worst)})
                  </Badge>
                  <Badge variant="outline" className="text-[10px]">
                    best: {chartData.find((c) => c.value === best)?.name} ({format(best)})
                  </Badge>
                  <Badge variant="outline" className="text-[10px]">
                    disparity: {format(Math.abs(worst - best))}
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
