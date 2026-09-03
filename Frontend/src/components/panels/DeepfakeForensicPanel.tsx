import { useCallback, useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer,
  Tooltip as RechartsTooltip, XAxis, YAxis,
} from "recharts";
import { API_BASE } from "@/lib/api";
import { usePlayback } from "@/contexts/PlaybackContext";

interface TimelineWindow {
  start_s: number;
  end_s: number;
  synthetic_probability: number;
  confidence: number;
  predicted_label: string;
}

interface DeepfakeForensicPanelProps {
  selectedFile?: { filename?: string; file_path?: string } | null;
  dataset?: string | null;
  /** Clip-level verdict (FR7.1), shown beside the timeline. */
  clipVerdict?: { predicted_label?: string; synthetic_probability?: number } | null;
}

/**
 * FR7.2 — where in the clip the detector suspects synthesis.
 *
 * FR7.1 (a clip-level bona-fide/synthetic probability) has worked for a while.
 * The forensic view that makes it interpretable did not exist: the `timeline`
 * field was declared on the results schema and never populated, and no
 * component rendered an ADD panel at all.
 */
export const DeepfakeForensicPanel = ({
  selectedFile, dataset, clipVerdict,
}: DeepfakeForensicPanelProps) => {
  const [timeline, setTimeline] = useState<TimelineWindow[] | null>(null);
  const [needsScan, setNeedsScan] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const { currentTime, seek } = usePlayback();

  const requestBody = useCallback((cacheOnly: boolean) => (
    dataset && !dataset.startsWith("custom:")
      ? { dataset, dataset_file: selectedFile?.filename, cache_only: cacheOnly }
      : { file_path: selectedFile?.file_path, cache_only: cacheOnly }
  ), [dataset, selectedFile?.filename, selectedFile?.file_path]);

  const load = useCallback(async (cacheOnly: boolean) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/inferences/deepfake-timeline`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(requestBody(cacheOnly)),
      });
      if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
      const data = await res.json();
      setNeedsScan(Boolean(data.needs_scan));
      setTimeline(data.needs_scan ? null : data.timeline || []);
    } catch (e) {
      // No placeholder curve (A2): an absent timeline is honest, an invented
      // one would be a forensic claim the model never made.
      setError(String((e as Error).message || e));
      setTimeline(null);
    } finally {
      setLoading(false);
    }
  }, [requestBody]);

  useEffect(() => {
    if (!(selectedFile?.filename || selectedFile?.file_path)) {
      setTimeline(null);
      setNeedsScan(false);
      return;
    }
    // Cache only on selection; a real scan is one model call per window and
    // must be asked for.
    void load(true);
  }, [selectedFile?.filename, selectedFile?.file_path, dataset, load]);

  const chartData = (timeline || []).map((w) => ({
    t: Number(((w.start_s + w.end_s) / 2).toFixed(2)),
    synthetic: w.synthetic_probability,
  }));

  const peak = timeline?.length
    ? timeline.reduce((a, b) => (b.synthetic_probability > a.synthetic_probability ? b : a))
    : null;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          Deepfake Forensics
          <Badge variant="outline" className="text-[10px]">ADD</Badge>
          {clipVerdict?.predicted_label && (
            <Badge
              variant={clipVerdict.predicted_label === "spoof" ? "destructive" : "default"}
              className="text-[10px]"
            >
              {clipVerdict.predicted_label}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {loading && <p className="text-xs text-muted-foreground">Scanning the clip…</p>}
        {error && !loading && (
          <p className="text-xs text-red-500">Timeline unavailable: {error}</p>
        )}
        {!loading && !error && needsScan && (
          <div className="flex items-center gap-2">
            <p className="text-xs text-muted-foreground">Not scanned yet.</p>
            <button
              type="button"
              onClick={() => void load(false)}
              className="text-xs px-2 py-1 rounded border border-border hover:bg-accent"
            >
              Scan clip
            </button>
          </div>
        )}
        {!loading && !error && !needsScan && timeline && timeline.length === 0 && (
          <p className="text-xs text-muted-foreground">No windows returned for this clip.</p>
        )}
        {!loading && !error && chartData.length > 0 && (
          <>
            <div className="h-[140px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={chartData}
                  onClick={(e: { activeLabel?: string | number }) => {
                    const t = Number(e?.activeLabel);
                    if (Number.isFinite(t)) seek(t);
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="t" tick={{ fontSize: 10 }} unit="s" />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} width={30} />
                  <RechartsTooltip
                    formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, "P(synthetic)"]}
                    labelFormatter={(t) => `${t}s`}
                  />
                  <Area
                    type="monotone"
                    dataKey="synthetic"
                    stroke="hsl(var(--destructive))"
                    fill="hsl(var(--destructive))"
                    fillOpacity={0.25}
                    isAnimationActive={false}
                  />
                  {/* Shared playhead (FR10.2) */}
                  <ReferenceLine x={Number(currentTime.toFixed(2))} stroke="hsl(var(--foreground))" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            {peak && (
              <p className="mt-1 text-[11px] text-muted-foreground">
                Peak suspicion {(peak.synthetic_probability * 100).toFixed(1)}% at{" "}
                {peak.start_s.toFixed(2)}s–{peak.end_s.toFixed(2)}s
              </p>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
};
