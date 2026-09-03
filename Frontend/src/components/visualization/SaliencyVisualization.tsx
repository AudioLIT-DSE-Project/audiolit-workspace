import React, { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Download, Loader2 } from "lucide-react";
import { API_BASE } from '@/lib/api';
import { XAIOverlayCanvas, XAIResult } from './XAIOverlayCanvas';

interface SaliencySegment {
  start_time: number;
  end_time: number;
  saliency: number;
  intensity: number;
  word?: string;
}

interface SaliencyData {
  model: string;
  method: string;
  segments: SaliencySegment[];
  total_duration: number;
  emotion?: string;
  series?: number[];
  base_spectrogram?: number[][];
  saliency_matrix?: number[][];
}

interface SaliencyVisualizationProps {
  selectedFile?: any;
  model?: string;
  dataset?: string;
  originalDataset?: string;
}

export const SaliencyVisualization = ({ selectedFile, model, dataset, originalDataset }: SaliencyVisualizationProps) => {
  const [saliencyData, setSaliencyData] = useState<SaliencyData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedMethod, setSelectedMethod] = useState("gradcam");
  
  // State for 2D coordinate mapping (DoD: Translator Engine)
  const [baseSpectrogram, setBaseSpectrogram] = useState<number[][] | null>(null);
  const [xaiResults, setXaiResults] = useState<XAIResult[]>([]);

  // --- Translator Engine ---
  // Transforms 1D time-aligned attribution weights into an explicit 2D 
  // coordinate array matching the spectrogram layout dimensions.
  const translateTo2DCoordinateGrid = useCallback((
    series: number[], 
    targetFreqBins: number, 
    targetTimeBins: number
  ): number[][] => {
    if (!series || series.length === 0) return Array(targetFreqBins).fill(0).map(() => Array(targetTimeBins).fill(0));
    
    // 1. Interpolate 1D series to match target time bins
    const interpolatedTime: number[] = [];
    for (let i = 0; i < targetTimeBins; i++) {
      const idx = (i / targetTimeBins) * (series.length - 1);
      const low = Math.floor(idx);
      const high = Math.ceil(idx);
      const frac = idx - low;
      const val = series[low] * (1 - frac) + (series[high] || 0) * frac;
      interpolatedTime.push(val);
    }
    
    // 2. Broadcast across frequency bins to form 2D coordinate grid
    const grid2D: number[][] = [];
    for (let f = 0; f < targetFreqBins; f++) {
      grid2D.push([...interpolatedTime]);
    }
    
    return grid2D;
  }, []);

  const fetchSaliencyData = async () => {
    if (!selectedFile || !model) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const backendModel = model;
      const datasetToUse = originalDataset && originalDataset !== "custom" ? originalDataset : dataset;
      const fileIdentifier = datasetToUse && datasetToUse !== 'custom' 
        ? `${datasetToUse}_${selectedFile?.filename || selectedFile}` 
        : `custom_${selectedFile?.file_path || selectedFile?.file_id}`;
      
      const requestBody: any = {
        model: backendModel,
        method: selectedMethod,
        no_cache: true,
        _file_id: fileIdentifier,
      };

      const isUploadedFile = typeof selectedFile === 'object' && selectedFile.file_path && (
        selectedFile.file_path.includes('uploads/') || 
        selectedFile.file_path.startsWith('uploads/') ||
        selectedFile.message === "Perturbed file" ||
        selectedFile.message === "File uploaded successfully" ||
        selectedFile.message === "File uploaded and processed successfully"
      ) && selectedFile.message !== "Selected from embeddings" && selectedFile.message !== "Selected from dataset";

      if (isUploadedFile) {
        requestBody.file_path = selectedFile.file_path;
      } else if (datasetToUse && datasetToUse !== 'custom') {
        const dsFilename = typeof selectedFile === 'string' 
          ? selectedFile 
          : (selectedFile?.filename as string | undefined);
        if (!dsFilename) throw new Error('No dataset file selected.');
        requestBody.dataset = datasetToUse;
        requestBody.dataset_file = dsFilename;
      } else {
        if (typeof selectedFile === 'object' && (selectedFile.file_path || selectedFile.file_id)) {
          if (!selectedFile.file_path) throw new Error('Selected upload has no file_path.');
          requestBody.file_path = selectedFile.file_path;
        } else {
          throw new Error('Invalid file selection or missing file information.');
        }
      }

      const response = await fetch(`${API_BASE}/saliency/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: 'include',
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        let detail = '';
        try {
          const err = await response.json();
          detail = err?.detail || '';
        } catch {
          // response body isn't JSON — fall back to the status-based message below
        }
        throw new Error(detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setSaliencyData(data);

      // Process data into 2D coordinate grids for the XAI Canvas
      const series = data.series || [];
      const targetFreqBins = 128;
      const targetTimeBins = series.length > 0 ? series.length : 100;

      // Extract or generate base spectrogram
      let spectGrid = data.base_spectrogram;
      if (!spectGrid && series.length > 0) {
        spectGrid = translateTo2DCoordinateGrid(series, targetFreqBins, targetTimeBins);
      }
      setBaseSpectrogram(spectGrid || null);

      // Translate series to 2D coordinate grid for XAI overlay
      let xaiMatrix = data.saliency_matrix;
      if (!xaiMatrix && series.length > 0) {
        xaiMatrix = translateTo2DCoordinateGrid(series, targetFreqBins, targetTimeBins);
      }

      if (xaiMatrix) {
        const result: XAIResult = {
          method: selectedMethod as any,
          matrix: xaiMatrix,
          max_val: 1.0
        };
        setXaiResults([result]);
      } else {
        setXaiResults([]);
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch saliency data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSaliencyData();
  }, [selectedFile, model, selectedMethod]);

  const audioDuration = saliencyData?.total_duration || (typeof selectedFile === 'object' ? selectedFile?.duration || 10 : 10);

  return (
    <Card className="w-full h-full flex flex-col bg-panel-background">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Saliency Overlay</CardTitle>
          <div className="flex items-center gap-2">
            <Select value={selectedMethod} onValueChange={setSelectedMethod}>
              <SelectTrigger className="w-24 h-6 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="gradcam">Grad-CAM</SelectItem>
                <SelectItem value="integrated_gradients">Integrated Gradients</SelectItem>
                <SelectItem value="lime">LIME</SelectItem>
                <SelectItem value="shap">SHAP</SelectItem>
              </SelectContent>
            </Select>
            <Button size="sm" variant="outline" className="h-6" onClick={fetchSaliencyData} disabled={loading}>
              {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <Download className="h-3 w-3" />}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex-1 p-3 space-y-3 flex flex-col">
        {error && (
          <div className="text-xs text-destructive bg-destructive/10 p-2 rounded">
            Error: {error}
          </div>
        )}
        
        {loading && (
          <div className="flex-1 bg-muted/30 rounded flex items-center justify-center">
            <Loader2 className="h-4 w-4 animate-spin mr-2" />
            <span className="text-xs text-muted-foreground">Calculating 2D saliency map...</span>
          </div>
        )}
        
        {!loading && !error && saliencyData && (
          <>
            {/* 2D Spectrogram Attribution Canvas */}
            {baseSpectrogram && xaiResults.length > 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center">
                <XAIOverlayCanvas
                  audioDuration={audioDuration}
                  baseSpectrogram={baseSpectrogram}
                  xaiResults={xaiResults}
                  activeMethod={selectedMethod as any}
                  width={800}
                  height={300}
                />
              </div>
            ) : (
              <div className="flex-1 bg-muted/30 rounded flex items-center justify-center text-xs text-muted-foreground">
                No 2D saliency data available.
              </div>
            )}

            {/* Top Salient Segments List */}
            <div className="text-xs space-y-2 mt-4">
              <div className="font-medium">Top Salient Segments:</div>
              {saliencyData.segments
                .sort((a, b) => b.intensity - a.intensity)
                .slice(0, 5)
                .map((segment, idx) => (
                  <div key={idx} className="flex items-center justify-between p-2 bg-muted/50 rounded">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-[10px]">
                        {segment.start_time.toFixed(1)}-{segment.end_time.toFixed(1)}s
                      </Badge>
                      <span className="font-mono">{segment.word || 'segment'}</span>
                    </div>
                    <span className="text-muted-foreground">{(segment.intensity * 100).toFixed(0)}%</span>
                  </div>
                ))}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};
