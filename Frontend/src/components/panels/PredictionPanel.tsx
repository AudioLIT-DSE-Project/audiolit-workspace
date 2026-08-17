import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { SaliencyVisualization } from "../visualization/SaliencyVisualization";
import { AttentionVisualization } from "../visualization/AttentionVisualization";
import { PerturbationTools } from "../analysis/PerturbationTools";
import { AcousticProfilePanel } from "./AcousticProfilePanel";
import { AccentBiasPanel } from "./AccentBiasPanel";
import { FaithfulnessAuditPanel } from "./FaithfulnessAuditPanel";
import { XAIOverlayCanvas, XAIMethod, XAIResult, F0Point } from "../visualization/XAIOverlayCanvas";
import { useState, useEffect } from "react";
import { AlertTriangle, ShieldCheck, ChevronDown, ChevronUp } from "lucide-react";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
}

interface PerturbationResult {
  perturbed_file: string;
  filename: string;
  duration_ms: number;
  sample_rate: number;
  applied_perturbations: Array<{
    type: string;
    params: Record<string, any>;
    status: string;
    error?: string;
  }>;
  success: boolean;
  error?: string;
}

// Interfaces for the Unified RQ Fan-in Result
export interface ASRToken {
  text: string;
  start: number;
  end: number;
}

export interface UnifiedTaskResult {
  tasks: {
    asr?: {
      transcript: string;
      tokens?: ASRToken[];
    };
    ser?: {
      predicted_emotion: string;
      probabilities: Record<string, number>;
    };
    add?: {
      label: 'bona-fide' | 'spoof' | 'synthetic';
      confidence: number;
      synthetic_probability?: number;
    };
    xai?: XAIResult;
    acoustic?: {
      spectrogram?: number[][];
      f0?: F0Point[];
      waveform?: number[]; // Added waveform array
    };
  };
  cache_key?: string;
}

interface PredictionPanelProps {
  selectedFile?: UploadedFile | null;
  selectedEmbeddingFile?: string | null;
  model?: string;
  dataset?: string;
  originalDataset?: string;
  onPerturbationComplete?: (result: PerturbationResult) => void;
  onPredictionRefresh?: (file: UploadedFile, prediction: string) => void;
  onPredictionUpdate?: (fileId: string, prediction: string) => void;
  unifiedResult?: UnifiedTaskResult | null; 
  audioDuration?: number; 
}

export const PredictionPanel = ({ 
  selectedFile, 
  selectedEmbeddingFile, 
  model, 
  dataset, 
  originalDataset, 
  onPerturbationComplete, 
  onPredictionRefresh, 
  onPredictionUpdate,
  unifiedResult,
  audioDuration = 10.0
}: PredictionPanelProps) => {
  const [originalFile, setOriginalFile] = useState<UploadedFile | null>(selectedFile || null);
  const [perturbedFile, setPerturbedFile] = useState<UploadedFile | null>(null);
  const [isLoadingPerturbed, setIsLoadingPerturbed] = useState(false);
  const [hoveredToken, setHoveredToken] = useState<ASRToken | null>(null);

  // Progressive disclosure (SRS §3.6.6): Accent Bias / Faithfulness stay
  // hidden behind this toggle so the default view isn't overloaded.
  const [showAdvanced, setShowAdvanced] = useState(false);

  // State for dynamic XAI canvas layer opacity swapping
  const [activeXAIMethod, setActiveXAIMethod] = useState<XAIMethod>('saliency');

  // Handle perturbation completion
  const handlePerturbationComplete = async (result: PerturbationResult) => {
    if (!result.success) {
      console.error("Perturbation failed:", result.error);
      return;
    }

    const perturbedFileObj: UploadedFile = {
      file_id: result.filename,
      filename: result.filename,
      file_path: result.perturbed_file,
      message: "Perturbed audio",
      duration: result.duration_ms / 1000,
      sample_rate: result.sample_rate
    };
    
    setPerturbedFile(perturbedFileObj);
    
    if (onPerturbationComplete) {
      onPerturbationComplete(result);
    }
  };

  const hasAttention = !!model && model.includes('whisper');
  const addResult = unifiedResult?.tasks?.add;
  const serResult = unifiedResult?.tasks?.ser;
  const asrResult = unifiedResult?.tasks?.asr;
  
  // Extract XAI and Acoustic data from unified result
  const xaiResult = unifiedResult?.tasks?.xai;
  const spectrogramData = unifiedResult?.tasks?.acoustic?.spectrogram;
  const waveformData = unifiedResult?.tasks?.acoustic?.waveform || []; // Extract waveform
  const f0Data = unifiedResult?.tasks?.acoustic?.f0 || [];

  return (
    <div className="h-full bg-panel-background border-t border-border flex flex-col">
      
      {/* 1. High-Visibility Deepfake (ADD) Warning Banner (Main Viewport Layout) */}
      {addResult && (() => {
        const isSpoof = addResult.label === 'spoof' || addResult.label === 'synthetic' || (typeof addResult.synthetic_probability === 'number' && addResult.synthetic_probability > 0.5);
        return (
        <div className={`p-3 flex items-center justify-between transition-all duration-500 border-b-2 ${
          isSpoof
            ? 'bg-red-50 border-red-500 text-red-700 dark:bg-red-950/50 dark:text-red-400'
            : 'bg-green-50 border-green-500 text-green-700 dark:bg-green-950/50 dark:text-green-400'
        }`}>
          <div className="flex items-center gap-3">
            {isSpoof ? (
              <AlertTriangle className="h-6 w-6" />
            ) : (
              <ShieldCheck className="h-6 w-6" />
            )}
            <div>
              <h3 className="text-sm font-bold tracking-tight">
                {isSpoof ? 'Deepfake Detected' : 'Bona-fide Audio'}
              </h3>
              <p className="text-[10px] opacity-80">
                Binary classification (ASVspoof 2021 DF) - No multi-class fingerprinting (SRS §4.4)
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-xl font-bold">
              {(addResult.confidence * 100).toFixed(1)}%
            </div>
            <div className="text-[10px] opacity-80">Confidence</div>
          </div>
        </div>
        );
      })()}

      <Tabs defaultValue="analytics" className="h-full flex flex-col">
        <div className="bg-panel-header border-b border-border px-3 py-2 flex items-center gap-2">
          <TabsList
            className={`h-7 grid flex-1 ${
              hasAttention
                ? (showAdvanced ? 'grid-cols-7' : 'grid-cols-4')
                : (showAdvanced ? 'grid-cols-6' : 'grid-cols-3')
            } bg-muted`}
          >
            <TabsTrigger value="analytics" className="text-xs">Analytics</TabsTrigger>
            <TabsTrigger value="saliency" className="text-xs">Saliency</TabsTrigger>
            <TabsTrigger value="acoustic" className="text-xs">Acoustic</TabsTrigger>
            <TabsTrigger value="perturbation" className="text-xs">Perturbation</TabsTrigger>
            {showAdvanced && hasAttention && <TabsTrigger value="attention" className="text-xs">Attention</TabsTrigger>}
            {showAdvanced && <TabsTrigger value="accent-bias" className="text-xs">Accent Bias</TabsTrigger>}
            {showAdvanced && <TabsTrigger value="faithfulness" className="text-xs">Faithfulness</TabsTrigger>}
          </TabsList>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 px-2 text-[10px] shrink-0"
            onClick={() => setShowAdvanced((v) => !v)}
          >
            Advanced
            {showAdvanced ? <ChevronUp className="h-3 w-3 ml-1" /> : <ChevronDown className="h-3 w-3 ml-1" />}
          </Button>
        </div>

        <div className="flex-1 overflow-auto bg-background">
          {/* Analytics Tab Content for Unified RQ Results */}
          <TabsContent value="analytics" forceMount className="m-0 h-full p-3 space-y-4 data-[state=inactive]:hidden">
            
            {/* Interactive ASR Token Timeline */}
            {asrResult && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    Transcription Timeline
                    <Badge variant="outline" className="text-[10px]">ASR</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="text-sm p-3 bg-muted/30 rounded-md leading-relaxed flex flex-wrap gap-1">
                    {asrResult.tokens && asrResult.tokens.length > 0 ? (
                      asrResult.tokens.map((token, idx) => (
                        <span
                          key={idx}
                          onMouseEnter={() => setHoveredToken(token)}
                          onMouseLeave={() => setHoveredToken(null)}
                          className={`cursor-pointer px-1 rounded transition-colors duration-150 ${
                            hoveredToken?.text === token.text
                              ? 'bg-blue-200 dark:bg-blue-800 text-blue-900 dark:text-blue-100'
                              : 'hover:bg-muted'
                          }`}
                        >
                          {token.text}
                        </span>
                      ))
                    ) : (
                      <span className="text-muted-foreground italic">{asrResult.transcript || "No transcript available"}</span>
                    )}
                  </div>

                  {asrResult.tokens && asrResult.tokens.length > 0 && (
                    <div className="relative h-8 w-full bg-gray-100 dark:bg-gray-800 rounded-md overflow-hidden">
                      {asrResult.tokens.map((token, idx) => {
                        const left = (token.start / audioDuration) * 100;
                        const width = ((token.end - token.start) / audioDuration) * 100;
                        return (
                          <div
                            key={idx}
                            onMouseEnter={() => setHoveredToken(token)}
                            onMouseLeave={() => setHoveredToken(null)}
                            className={`absolute h-full bg-blue-400 dark:bg-blue-600 opacity-70 hover:opacity-100 hover:bg-blue-600 dark:hover:bg-blue-400 transition-all flex items-center justify-center ${
                              hoveredToken?.start === token.start ? 'ring-2 ring-blue-500 z-10' : ''
                            }`}
                            style={{
                              left: `${left}%`,
                              width: `${Math.max(width, 0.5)}%`,
                            }}
                            title={`${token.text} (${token.start.toFixed(2)}s - ${token.end.toFixed(2)}s)`}
                          />
                        );
                      })}
                      <div className="absolute bottom-0 left-0 text-[9px] text-gray-500 p-0.5">0s</div>
                      <div className="absolute bottom-0 right-0 text-[9px] text-gray-500 p-0.5">{audioDuration.toFixed(1)}s</div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Speech Emotion Recognition (SER) Analytics Table */}
            {serResult ? (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    Emotion Analytics
                    <Badge variant="outline" className="text-[10px]">SER</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {Object.entries(serResult.probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([emotion, probability]) => {
                      const isPredicted = emotion === serResult.predicted_emotion;
                      return (
                        <div key={emotion} className="flex items-center justify-between text-xs">
                          <div className="flex items-center gap-2 w-24">
                            <span className={`capitalize ${isPredicted ? 'font-bold text-blue-600' : ''}`}>
                              {emotion}
                            </span>
                            {isPredicted && (
                              <Badge variant="default" className="text-[9px] h-4 px-1">Predicted</Badge>
                            )}
                          </div>
                          <div className="flex items-center gap-2 flex-1 max-w-[200px]">
                            <Progress value={probability * 100} className={`h-2 ${isPredicted ? 'bg-blue-200' : ''}`} />
                            <span className="text-muted-foreground min-w-[2.5rem] text-right">
                              {(probability * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      );
                    })}
                </CardContent>
              </Card>
            ) : (
              !asrResult && (
                <Card className="w-full h-full flex items-center justify-center min-h-[200px] bg-muted/20">
                  <CardContent className="text-center text-muted-foreground">
                    <p className="text-sm">Awaiting multi-task inference results...</p>
                  </CardContent>
                </Card>
              )
            )}

          </TabsContent>

          {/* Saliency Tab with integrated XAI Canvas */}
          <TabsContent value="saliency" forceMount className="m-0 h-full data-[state=inactive]:hidden">
            <div className="p-3 space-y-4">
              {/* Dynamic XAI Method Toggle Buttons */}
              <div className="flex gap-2 mb-2">
                {(['saliency', 'shap', 'lime', 'ig'] as XAIMethod[]).map(m => (
                  <button 
                    key={m} 
                    onClick={() => setActiveXAIMethod(m)}
                    className={`px-3 py-1 text-xs rounded-md border transition-colors ${
                      activeXAIMethod === m 
                        ? 'bg-primary text-primary-foreground border-primary' 
                        : 'bg-muted text-muted-foreground border-border hover:bg-accent'
                    }`}
                  >
                    {m.toUpperCase()}
                  </button>
                ))}
              </div>

              {/* New High-Performance XAI Overlay Canvas */}
              {spectrogramData || xaiResult ? (
                <XAIOverlayCanvas
                  audioDuration={audioDuration}
                  baseSpectrogram={spectrogramData}
                  waveformData={waveformData} // Pass waveform data to canvas
                  xaiResults={xaiResult ? [xaiResult] : []}
                  f0Data={f0Data}
                  activeMethod={activeXAIMethod}
                  width={800}
                  height={400}
                />
              ) : (
                <Card className="w-full h-[400px] flex items-center justify-center bg-muted/20">
                  <CardContent className="text-center text-muted-foreground">
                    <p className="text-sm">Awaiting XAI & Spectrogram data from worker...</p>
                  </CardContent>
                </Card>
              )}

              {/* Existing inherited Saliency Visualization */}
              <SaliencyVisualization
                selectedFile={selectedFile || selectedEmbeddingFile}
                model={model}
                dataset={dataset}
              />
            </div>
          </TabsContent>

          <TabsContent value="acoustic" forceMount className="m-0 h-full data-[state=inactive]:hidden">
            <AcousticProfilePanel
              selectedFile={selectedFile}
              selectedEmbeddingFile={selectedEmbeddingFile}
              dataset={dataset}
              originalDataset={originalDataset}
            />
          </TabsContent>

          {showAdvanced && hasAttention && (
            <TabsContent value="attention" forceMount className="m-0 h-full data-[state=inactive]:hidden">
              <div className="p-3">
                <AttentionVisualization
                  selectedFile={selectedFile || selectedEmbeddingFile}
                  model={model}
                  dataset={dataset}
                />
              </div>
            </TabsContent>
          )}

          {showAdvanced && (
            <TabsContent value="accent-bias" forceMount className="m-0 h-full data-[state=inactive]:hidden">
              <AccentBiasPanel model={model} />
            </TabsContent>
          )}

          {showAdvanced && (
            <TabsContent value="faithfulness" forceMount className="m-0 h-full data-[state=inactive]:hidden">
              <FaithfulnessAuditPanel
                selectedFile={selectedFile}
                selectedEmbeddingFile={selectedEmbeddingFile}
                model={model}
                dataset={dataset}
                originalDataset={originalDataset}
              />
            </TabsContent>
          )}

          <TabsContent value="perturbation" forceMount className="m-0 h-full data-[state=inactive]:hidden">
            <div className="p-3">
              <PerturbationTools
                selectedFile={selectedFile}
                onPerturbationComplete={handlePerturbationComplete}
                onPredictionRefresh={onPredictionRefresh}
                model={model}
                dataset={dataset}
                originalDataset={originalDataset}
              />
            </div>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
};
