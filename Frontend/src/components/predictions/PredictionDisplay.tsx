import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { GlobalTaskProgress } from "../layout/GlobalTaskProgress"; // Import the progress component

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
}

interface Wav2Vec2Prediction {
  predicted_emotion: string;
  probabilities: Record<string, number>;
  confidence: number;
  ground_truth_emotion?: string;
}

interface WhisperPrediction {
  predicted_transcript: string;
  ground_truth: string;
  accuracy_percentage: number | null;
  word_error_rate: number | null;
  character_error_rate: number | null;
  levenshtein_distance: number | null;
  exact_match: number | null;
  character_similarity: number | null;
  word_count_predicted: number;
  word_count_truth: number;
}

interface AddPrediction {
  predicted_label: string; // "bona-fide" | "spoof"
  synthetic_probability: number;
  confidence: number;
  probabilities: Record<string, number>;
}

const ADD_MODEL_KEYS = ["melody-machine", "wav2vec2-add"];
const ADD_MODEL_LABELS: Record<string, string> = {
  "melody-machine": "MelodyMachine",
  "wav2vec2-add": "Wav2Vec2 XLSR",
};

interface PredictionDisplayProps {
  selectedFile?: UploadedFile | null;
  selectedEmbeddingFile?: string | null;
  model?: string;
  wav2vecPrediction?: Wav2Vec2Prediction | null;
  whisperPrediction?: WhisperPrediction | null;
  addPrediction?: AddPrediction | null;
  perturbedPredictions?: Wav2Vec2Prediction | WhisperPrediction | null;
  isLoading?: boolean;
  isLoadingPerturbed?: boolean;
  error?: string | null;
  showPerturbed?: boolean;
  activeTaskId?: string | null; // Add taskId prop to hook into WebSocket
}

export const PredictionDisplay = ({
  selectedFile,
  selectedEmbeddingFile,
  model,
  wav2vecPrediction,
  whisperPrediction,
  addPrediction,
  perturbedPredictions,
  isLoading,
  isLoadingPerturbed,
  error,
  showPerturbed = false,
  activeTaskId = null
}: PredictionDisplayProps) => {
  const isAddModel = !!model && ADD_MODEL_KEYS.includes(model);
  if (!selectedFile && !selectedEmbeddingFile) {
    return (
      <Card>
        <CardContent className="p-3 text-center text-muted-foreground">
          <div className="text-xs">No file selected</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="bg-panel-header">
        <CardTitle className="text-xs">
          {model === "wav2vec2" ? "Classification Results" : model?.includes("whisper") ? "Transcription Results" : isAddModel ? "Deepfake Detection Results" : "Prediction Results"}
          {model === "wav2vec2" && (
            <Badge variant="outline" className="ml-1.5 text-[10px] bg-primary/10 text-primary border-primary/20">Wav2Vec2 Emotion</Badge>
          )}
          {model?.includes("whisper") && (
            <Badge variant="outline" className="ml-1.5 text-[10px] bg-primary/10 text-primary border-primary/20">Whisper Base</Badge>
          )}
          {isAddModel && (
            <Badge variant="outline" className="ml-1.5 text-[10px] bg-primary/10 text-primary border-primary/20">{ADD_MODEL_LABELS[model!]}</Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1.5">
        
        {/* Replace static spinner with real-time WebSocket progress bar */}
        {(isLoading || isLoadingPerturbed) && activeTaskId && (
          <div className="py-4">
            <GlobalTaskProgress taskId={activeTaskId} onComplete={() => {}} />
          </div>
        )}

        {error && (
          <div className="text-xs text-destructive p-2 bg-destructive/5 rounded-sm border border-destructive/20">
            Error: {error}
          </div>
        )}
        
        {model === "wav2vec2" && wav2vecPrediction && wav2vecPrediction.probabilities && !isLoading ? (
          // Wav2Vec2 display logic remains identical
          <div className="space-y-3">
            {!showPerturbed ? (
              <div className="space-y-2">
                <div className="text-xs-tight font-medium flex items-center gap-2">
                  Original Audio Prediction
                  <span className="text-xs-tight text-gray-500 border border-gray-300 px-1 rounded">Original</span>
                </div>
                {Object.entries(wav2vecPrediction.probabilities)
                  .sort(([,a], [,b]) => b - a)
                  .map(([emotion, probability]) => {
                    const isPredicted = emotion === wav2vecPrediction.predicted_emotion;
                    return (
                      <div key={emotion} className="flex items-center justify-between text-xs-tight">
                        <div className="flex items-center gap-2">
                          <span className="capitalize">{emotion}</span>
                          {isPredicted && <span className="text-xs-tight text-gray-600 font-medium">Predicted</span>}
                        </div>
                        <div className="flex items-center gap-2 flex-1 max-w-[120px]">
                          <Progress value={probability * 100} className="h-2" />
                          <span className="text-muted-foreground min-w-[2rem]">{(probability * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    );
                  })}
              </div>
            ) : (
              <div className="space-y-2">
                <div className="text-xs-tight font-medium flex items-center gap-2">
                  Perturbed Audio Prediction
                  <span className="text-xs-tight text-gray-500 border border-gray-300 px-1 rounded">Perturbed</span>
                </div>
                {!isLoadingPerturbed && perturbedPredictions && (perturbedPredictions as Wav2Vec2Prediction).probabilities ? (
                  <div className="space-y-2">
                    {Object.entries((perturbedPredictions as Wav2Vec2Prediction).probabilities)
                      .sort(([,a], [,b]) => b - a)
                      .map(([emotion, probability]) => {
                        const isPredicted = emotion === (perturbedPredictions as Wav2Vec2Prediction).predicted_emotion;
                        const originalProb = wav2vecPrediction.probabilities[emotion] || 0;
                        const change = (probability - originalProb) * 100;
                        const isSignificantChange = Math.abs(change) > 1;
                        return (
                          <div key={emotion} className="flex items-center justify-between text-xs-tight">
                            <div className="flex items-center gap-2">
                              <span className="capitalize">{emotion}</span>
                              {isPredicted && <span className="text-xs-tight text-gray-700 font-medium">Predicted</span>}
                            </div>
                            <div className="flex items-center gap-2 flex-1 max-w-[140px]">
                              <Progress value={probability * 100} className="h-2" />
                              <span className="text-muted-foreground min-w-[2rem]">{(probability * 100).toFixed(1)}%</span>
                              <span className={`text-[10px] min-w-[3rem] font-medium ${!isSignificantChange ? "text-muted-foreground" : change > 0 ? "text-green-600" : "text-red-600"}`}>
                                {change > 0 ? "+" : ""}{change.toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    <div className="mt-3 p-2 bg-blue-50 rounded border border-blue-200">
                      <div className="text-xs font-medium text-blue-800">Prediction Change</div>
                      <div className="text-xs text-blue-700 mt-1">
                        Original: <span className="font-medium">{wav2vecPrediction.predicted_emotion}</span> → Perturbed: <span className="font-medium">{(perturbedPredictions as Wav2Vec2Prediction).predicted_emotion}</span>
                        {wav2vecPrediction.predicted_emotion !== (perturbedPredictions as Wav2Vec2Prediction).predicted_emotion && <span className="text-red-600 font-medium ml-2">Changed!</span>}
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>
            )}
          </div>
        ) : model?.includes("whisper") && whisperPrediction && !isLoading ? (
          // Whisper display logic remains identical
          <div className="space-y-3">
            {!showPerturbed ? (
              <div className="space-y-3">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="text-xs font-semibold flex items-center gap-2">
                      Original Transcription Metrics
                      <span className="text-xs-tight text-blue-600 border border-blue-300 px-1 rounded">Original</span>
                    </div>
                    {whisperPrediction.ground_truth && whisperPrediction.ground_truth.trim() !== "" ? (
                      whisperPrediction.accuracy_percentage !== null && whisperPrediction.word_error_rate !== null ? (
                        <div className="mt-2 grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                          <div className="p-2 bg-gray-50 rounded border text-gray-700"><div className="text-[10px] text-gray-500">WER</div><div className="font-medium">{whisperPrediction.word_error_rate.toFixed(3)}</div></div>
                          <div className="p-2 bg-gray-50 rounded border text-gray-700"><div className="text-[10px] text-gray-500">CER</div><div className="font-medium">{whisperPrediction.character_error_rate.toFixed(3)}</div></div>
                          <div className="p-2 bg-gray-50 rounded border text-gray-700"><div className="text-[10px] text-gray-500">Accuracy</div><div className="font-medium">{whisperPrediction.accuracy_percentage.toFixed(1)}%</div></div>
                        </div>
                      ) : <div className="mt-2 p-3 bg-blue-50 rounded border border-blue-200 text-xs text-blue-700"><div className="font-medium">Ground Truth Available</div><div className="mt-1">Accuracy metrics are being calculated...</div></div>
                    ) : <div className="mt-2 p-3 bg-yellow-50 rounded border border-yellow-200 text-xs text-yellow-700"><div className="font-medium">No Ground Truth Available</div></div>}
                  </div>
                </div>
                <div className="w-full">
                  <div className="text-xs font-medium mb-2">Predicted Transcript</div>
                  <div className="text-xs p-4 bg-green-50 rounded-lg border border-green-200 font-mono whitespace-pre-wrap leading-relaxed">
                    {whisperPrediction.predicted_transcript ? `"${whisperPrediction.predicted_transcript}"` : <span className="italic text-gray-400">No prediction available</span>}
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                 <div className="text-xs font-semibold flex items-center gap-2">Perturbed Transcription Results</div>
                 {!isLoadingPerturbed && perturbedPredictions ? (
                   <div className="space-y-4">
                     <div className="w-full">
                       <div className="text-xs font-medium mb-2">Perturbed Transcript</div>
                       <div className="text-xs p-4 bg-blue-50 rounded-lg border border-blue-200 font-mono whitespace-pre-wrap leading-relaxed">
                         {(perturbedPredictions as WhisperPrediction).predicted_transcript ? `"${(perturbedPredictions as WhisperPrediction).predicted_transcript}"` : <span className="italic text-gray-400">No prediction available</span>}
                       </div>
                     </div>
                   </div>
                 ) : null}
              </div>
            )}
          </div>
        ) : isAddModel && addPrediction && addPrediction.probabilities && !isLoading ? (
          <div className="space-y-3">
            {!showPerturbed ? (
              <div className="space-y-2">
                <div className="text-xs-tight font-medium flex items-center gap-2">
                  Original Audio Prediction
                  <span className="text-xs-tight text-gray-500 border border-gray-300 px-1 rounded">Original</span>
                </div>
                {Object.entries(addPrediction.probabilities)
                  .sort(([, a], [, b]) => b - a)
                  .map(([label, probability]) => {
                    const isPredicted = label === addPrediction.predicted_label;
                    return (
                      <div key={label} className="flex items-center justify-between text-xs-tight">
                        <div className="flex items-center gap-2">
                          <span className="capitalize">{label}</span>
                          {isPredicted && <span className="text-xs-tight text-gray-600 font-medium">Predicted</span>}
                        </div>
                        <div className="flex items-center gap-2 flex-1 max-w-[120px]">
                          <Progress value={probability * 100} className="h-2" />
                          <span className="text-muted-foreground min-w-[2rem]">{(probability * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    );
                  })}
              </div>
            ) : (
              <div className="text-xs text-muted-foreground p-2 bg-gray-50 rounded border border-gray-200">
                Perturbed-audio re-inference is not yet available for deepfake-detection models.
              </div>
            )}
          </div>
        ) : isAddModel && !isLoading ? (
          <div className="text-xs text-muted-foreground p-2">No prediction yet.</div>
        ) : null}
      </CardContent>
    </Card>
  );
};
