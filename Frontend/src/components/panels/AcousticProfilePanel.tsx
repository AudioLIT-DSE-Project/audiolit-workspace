import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { HelpCircle, Loader2 } from "lucide-react";
import { API_BASE } from "@/lib/api";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
}

interface AcousticProfilePanelProps {
  selectedFile?: UploadedFile | null;
  selectedEmbeddingFile?: string | null;
  dataset?: string;
  originalDataset?: string;
}

interface TimelinePoint {
  t_ms: number;
  f0_hz: number | null;
  rms: number;
}

interface AcousticProfile {
  sample_rate: number;
  duration_s: number;
  timeline: TimelinePoint[];
}

const getRequestBody = (
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

// STFT/pYIN F0 + RMS Acoustic Wave Profiler (SRS §2.2/§3.9.1, FR10).
export const AcousticProfilePanel: React.FC<AcousticProfilePanelProps> = ({
  selectedFile,
  selectedEmbeddingFile,
  dataset,
  originalDataset,
}) => {
  const [profile, setProfile] = useState<AcousticProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const body = getRequestBody(selectedFile, selectedEmbeddingFile, dataset, originalDataset);
    if (!body) {
      setProfile(null);
      setError(null);
      return;
    }

    const abortController = new AbortController();
    setLoading(true);
    setError(null);

    fetch(`${API_BASE}/acoustic/profile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify(body),
      signal: abortController.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Failed to fetch acoustic profile: ${response.status}`);
        return response.json();
      })
      .then((data: AcousticProfile) => setProfile(data))
      .catch((err) => {
        if (err.name === "AbortError") return;
        setError(err instanceof Error ? err.message : "Failed to fetch acoustic profile");
      })
      .finally(() => setLoading(false));

    return () => abortController.abort();
  }, [selectedFile, selectedEmbeddingFile, dataset, originalDataset]);

  const chartData = (profile?.timeline || []).map((p) => ({
    t: +(p.t_ms / 1000).toFixed(2),
    f0: p.f0_hz,
    rms: p.rms,
  }));

  return (
    <TooltipProvider>
      <div className="p-3 space-y-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs flex items-center gap-1.5">
              Pitch Contour (pYIN F0)
              <Tooltip>
                <TooltipTrigger><HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" /></TooltipTrigger>
                <TooltipContent>Fundamental frequency over time. Gaps mean unvoiced/silent frames.</TooltipContent>
              </Tooltip>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading && <div className="flex items-center gap-2 text-xs text-muted-foreground py-6 justify-center"><Loader2 className="h-3.5 w-3.5 animate-spin" />Extracting acoustic profile...</div>}
            {error && <div className="text-xs text-destructive py-2">{error}</div>}
            {!loading && !error && chartData.length > 0 && (
              <ResponsiveContainer width="100%" height={140}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="t" tick={{ fontSize: 10 }} unit="s" />
                  <YAxis tick={{ fontSize: 10 }} unit="Hz" domain={['auto', 'auto']} />
                  <RechartsTooltip contentStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="f0" stroke="hsl(var(--waveform-primary))" dot={false} strokeWidth={1.5} connectNulls={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
            {!loading && !error && chartData.length === 0 && (
              <div className="text-xs text-muted-foreground py-6 text-center">Select a file to see its pitch contour.</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs flex items-center gap-1.5">
              RMS Energy Envelope
              <Tooltip>
                <TooltipTrigger><HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" /></TooltipTrigger>
                <TooltipContent>Localized loudness/intensity over time.</TooltipContent>
              </Tooltip>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {!loading && !error && chartData.length > 0 && (
              <ResponsiveContainer width="100%" height={100}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="t" tick={{ fontSize: 10 }} unit="s" />
                  <YAxis tick={{ fontSize: 10 }} />
                  <RechartsTooltip contentStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="rms" stroke="hsl(var(--waveform-secondary))" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>
    </TooltipProvider>
  );
};
