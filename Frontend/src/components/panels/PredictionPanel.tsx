import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { SaliencyVisualization } from "../visualization/SaliencyVisualization";
import { AttentionVisualization } from "../visualization/AttentionVisualization";
import { PerturbationTools } from "../analysis/PerturbationTools";
import { useState, useEffect } from "react";
import { API_BASE } from '@/lib/api';
import { AlertTriangle, ShieldCheck } from "lucide-react";

// ... [Keep your existing UploadedFile, Wav2Vec2Prediction, WhisperPrediction, PerturbationResult interfaces here] ...

// New interfaces for the Unified RQ Fan-in Result
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
      label: 'bona-fide' | 'synthetic';
      confidence: number;
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
  unifiedResult?: UnifiedTaskResult | null; // New prop for RQ result
  audioDuration?: number; // New prop for timeline scaling
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
  const [wav2vecPrediction, setWav2vecPrediction] = useState<Wav2Vec2Prediction | null>(null);
  const [whisperPrediction, setWhisperPrediction] = useState<WhisperPrediction | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [perturbedPredictions, setPerturbedPredictions] = useState<Wav2Vec2Prediction | WhisperPrediction | null>(null);
  const [originalFile, setOriginalFile] = useState<UploadedFile | null>(selectedFile || null);
  const [perturbedFile, setPerturbedFile] = useState<UploadedFile | null>(null);
  const [isLoadingPerturbed, setIsLoadingPerturbed] = useState(false);
  const [hoveredToken, setHoveredToken] = useState<ASRToken | null>(null);

  // ... [Keep your existing handlePerturbationComplete, runInferenceOnPerturbed, and useEffect hooks for fetching wav2vec/whisper predictions here] ...
  // (I am omitting them to save space, but do NOT delete them from your file)

  const hasAttention = !!model && model.includes('whisper');
  const addResult = unifiedResult?.tasks?.add;
  const serResult = unifiedResult?.tasks?.ser;
  const asrResult = unifiedResult?.tasks?.asr;

  return (
    <div className="h-full bg-panel-background border-t border-border flex flex-col">
      
      {/* 1. High-Visibility Deepfake (ADD) Warning Banner (Main Viewport Layout) */}
      {addResult && (
        <div className={`p-3 flex items-center justify-between transition-all duration-500 border-b-2 ${
          addResult.label === 'synthetic'
            ? 'bg-red-50 border-red-500 text-red-700 dark:bg-red-950/50 dark:text-red-400'
            : 'bg-green-50 border-green-500 text-green-700 dark:bg-green-950/50 dark:text-green-400'
        }`}>
          <div className="flex items-center gap-3">
            {addResult.label === 'synthetic' ? (
              <AlertTriangle className="h-6 w-6" />
            ) : (
              <ShieldCheck className="h-6 w-6" />
            )}
            <div>
              <h3 className="text-sm font-bold tracking-tight">
                {addResult.label === 'synthetic' ? 'Deepfake Detected' : 'Bona-fide Audio'}
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
      )}

      <Tabs defaultValue="analytics" className="h-full flex flex-col">
        <div className="bg-panel-header border-b border-border px-3 py-2">
          <TabsList className={`h-7 grid w-full ${hasAttention ? 'grid-cols-4' : 'grid-cols-3'} bg-muted`}>
            {/* New Analytics Tab for Unified RQ Results */}
            <TabsTrigger value="analytics" className="text-xs">Analytics</TabsTrigger>
            <TabsTrigger value="saliency" className="text-xs">Saliency</TabsTrigger>
            {hasAttention && <TabsTrigger value="attention" className="text-xs">Attention</TabsTrigger>}
            <TabsTrigger value="perturbation" className="text-xs">Perturbation</TabsTrigger>
          </TabsList>
        </div>

        <div className="flex-1 overflow-auto bg-background">
          {/* New Analytics Tab Content */}
          <TabsContent value="analytics" className="m-0 h-full p-3 space-y-4">
            
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

          {/* Existing Tabs */}
          <TabsContent value="saliency" className="m-0 h-full">
            <div className="p-3">
              <SaliencyVisualization
                selectedFile={selectedFile || selectedEmbeddingFile}
                model={model}
                dataset={dataset}
              />
            </div>
          </TabsContent>

          {hasAttention && (
            <TabsContent value="attention" className="m-0 h-full">
              <div className="p-3">
                <AttentionVisualization
                  selectedFile={selectedFile || selectedEmbeddingFile}
                  model={model}
                  dataset={dataset}
                />
              </div>
            </TabsContent>
          )}

          <TabsContent value="perturbation" className="m-0 h-full">
            <div className="p-3">
              <PerturbationTools
                selectedFile={selectedFile}
                onPerturbationComplete={onPerturbationComplete}
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
