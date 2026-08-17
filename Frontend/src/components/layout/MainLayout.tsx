import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { Toolbar, SelectedTasks } from "./Toolbar";
import { StatusBar } from "./StatusBar";
import { useState, useEffect, useCallback, useRef } from 'react';
import { useTaskStatus } from '@/hooks/useTaskStatus';
import { GlobalTaskProgress } from "./GlobalTaskProgress";
import { EmbeddingPanel } from "../panels/EmbeddingPanel";
import { AudioDatasetPanel } from "../panels/AudioDatasetPanel";
import { DatapointEditorPanel } from "../panels/DatapointEditorPanel";
import { PredictionPanel, UnifiedTaskResult } from "../panels/PredictionPanel";
import { EmbeddingProvider } from "../../contexts/EmbeddingContext";
import { API_BASE } from '@/lib/api';

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
  prediction?: string;
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

export const MainLayout = () => {
  const [apiData, setApiData] = useState<unknown>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<UploadedFile | null>(null);
  const [model, setModel] = useState("whisper-base");
  const [dataset, setDataset] = useState("common-voice");
  // SRS §3.9.1 sidebar "Task Selection (ASR/SER/ADD)" - which analyses run on
  // upload. Defaults to all-on, matching the previous hardcoded behavior.
  const [selectedTasks, setSelectedTasks] = useState<SelectedTasks>({ asr: true, ser: true, add: true });
  const [batchInferenceStatus, setBatchInferenceStatus] = useState<'idle' | 'running' | 'done'>('idle');
  const [availableFiles, setAvailableFiles] = useState<string[]>([]);
  const [selectedEmbeddingFile, setSelectedEmbeddingFile] = useState<string | null>(null);
  const [perturbationResult, setPerturbationResult] = useState<any>(null);
  
  // Prediction state
  const [wav2vecPrediction, setWav2vecPrediction] = useState<Wav2Vec2Prediction | null>(null);
  const [whisperPrediction, setWhisperPrediction] = useState<WhisperPrediction | null>(null);
  const [isLoadingPredictions, setIsLoadingPredictions] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [perturbedPredictions, setPerturbedPredictions] = useState<Wav2Vec2Prediction | WhisperPrediction | null>(null);
  const [isLoadingPerturbed, setIsLoadingPerturbed] = useState(false);

  // Refs to track ongoing requests and prevent duplicates
  const wav2vecRequestRef = useRef<AbortController | null>(null);
  const whisperRequestRef = useRef<AbortController | null>(null);
  
  // RQ Task State (WebSocket listener)
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const { state, result } = useTaskStatus(activeTaskId);

  // Clear selected file, embedding file, and predictions when dataset changes
  useEffect(() => {
    setSelectedFile(null);
    setSelectedEmbeddingFile(null);
    setAvailableFiles([]);
    setWav2vecPrediction(null);
    setWhisperPrediction(null);
    setPredictionError(null);
    setPerturbationResult(null);
  }, [dataset]);

  // Clear perturbation result and predictions when selected file changes
  useEffect(() => {
    setPerturbationResult(null);
    setWav2vecPrediction(null);
    setWhisperPrediction(null);
    setPerturbedPredictions(null);
    setPredictionError(null);
  }, [selectedFile, selectedEmbeddingFile]);

  // Fetch perturbed predictions when perturbation result is available
  useEffect(() => {
    const fetchPerturbedPredictions = async () => {
      if (!perturbationResult?.success || !model) {
        setPerturbedPredictions(null);
        return;
      }

      setIsLoadingPerturbed(true);
      setPredictionError(null);

      try {
        const requestBody: any = {
          file_path: perturbationResult.perturbed_file
        };

        let endpoint: string;
        if (model === "wav2vec2") {
          endpoint = `${API_BASE}/inferences/wav2vec2-detailed`;
          requestBody.include_attention = false; // Disable attention for better performance
        } else if (model?.includes("whisper")) {
          endpoint = `${API_BASE}/inferences/whisper-accuracy`;
          requestBody.model = model;
        } else {
          return; // Unsupported model
        }

        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: 'include',
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) throw new Error(`Failed to fetch perturbed prediction: ${response.status}`);
        const prediction = await response.json();
        setPerturbedPredictions(prediction);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Unknown error";
        setPredictionError(errorMessage);
        console.error("Error fetching perturbed predictions:", err);
      } finally {
        setIsLoadingPerturbed(false);
      }
    };

    fetchPerturbedPredictions();
  }, [perturbationResult, model]);

  // Fetch wav2vec prediction for dataset-browsing selections only. Uploaded
  // files are covered by the async multitask job (startMultiTaskInference)
  // via `unifiedResult` instead - LIT-232 removed the redundant, racing fetch
  // this effect used to also make for uploads (it never had an async
  // equivalent for dataset browsing, so that path stays as-is).
  useEffect(() => {
    const fetchWav2vecPrediction = async () => {
      const isUploadedFile = !!selectedFile?.file_path && (
        selectedFile.file_path.includes('uploads/') ||
        selectedFile.file_path.startsWith('uploads/') ||
        selectedFile.message === "Perturbed file" ||
        selectedFile.message === "File uploaded successfully" ||
        selectedFile.message === "File uploaded and processed successfully"
      ) && !selectedFile.message.includes("Selected from");

      if (model !== "wav2vec2" || (!selectedFile && !selectedEmbeddingFile) || isUploadedFile) {
        setWav2vecPrediction(null);
        setPredictionError(null);
        setIsLoadingPredictions(false);
        return;
      }

      if (wav2vecRequestRef.current) wav2vecRequestRef.current.abort();
      const abortController = new AbortController();
      wav2vecRequestRef.current = abortController;

      setIsLoadingPredictions(true);
      setPredictionError(null);

      try {
        const requestBody: any = {};
        if (selectedFile) {
          requestBody.dataset = dataset;
          requestBody.dataset_file = selectedFile.filename;
        } else if (selectedEmbeddingFile && dataset) {
          requestBody.dataset = dataset;
          requestBody.dataset_file = selectedEmbeddingFile;
        }
        requestBody.include_attention = false;

        const response = await fetch(`${API_BASE}/inferences/wav2vec2-detailed`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: 'include',
          body: JSON.stringify(requestBody),
          signal: abortController.signal
        });

        if (!response.ok) throw new Error(`Failed to fetch prediction: ${response.status}`);
        const prediction = await response.json();
        setWav2vecPrediction(prediction);
      } catch (err) {
        if (err.name === 'AbortError') return;
        const errorMessage = err instanceof Error ? err.message : "Unknown error";
        setPredictionError(errorMessage);
        console.error("Error fetching wav2vec2 prediction:", err);
      } finally {
        setIsLoadingPredictions(false);
        if (wav2vecRequestRef.current === abortController) wav2vecRequestRef.current = null;
      }
    };

    fetchWav2vecPrediction();
    return () => { if (wav2vecRequestRef.current) { wav2vecRequestRef.current.abort(); wav2vecRequestRef.current = null; } };
  }, [selectedFile, selectedEmbeddingFile, model, dataset]);

  // Fetch whisper prediction for dataset-browsing selections only (built-in
  // and custom datasets alike). Uploaded files are covered by the async
  // multitask job (startMultiTaskInference) via `unifiedResult` instead -
  // LIT-232 removed the redundant, racing fetch this effect used to also
  // make for uploads.
  useEffect(() => {
    const fetchWhisperPrediction = async () => {
      const isUploadedFile = !!selectedFile?.file_path && (
        selectedFile.file_path.includes('uploads/') ||
        selectedFile.file_path.startsWith('uploads/') ||
        selectedFile.message === "Perturbed file" ||
        selectedFile.message === "File uploaded successfully" ||
        selectedFile.message === "File uploaded and processed successfully"
      ) && !selectedFile.message.includes("Selected from");

      if (!model?.includes("whisper") || (!selectedFile && !selectedEmbeddingFile) || isUploadedFile) {
        setWhisperPrediction(null);
        setPredictionError(null);
        setIsLoadingPredictions(false);
        return;
      }

      if (whisperRequestRef.current) whisperRequestRef.current.abort();
      const abortController = new AbortController();
      whisperRequestRef.current = abortController;

      setIsLoadingPredictions(true);
      setPredictionError(null);

      try {
        const requestBody: any = { model: model };
        const isCustomDataset = dataset?.startsWith('custom:');

        if (selectedFile) {
          requestBody.dataset = dataset;
          requestBody.dataset_file = selectedFile.filename;
        } else if (selectedEmbeddingFile && dataset) {
          requestBody.dataset = dataset;
          requestBody.dataset_file = selectedEmbeddingFile;
        }

        const endpoint = isCustomDataset ? `${API_BASE}/inferences/run` : `${API_BASE}/inferences/whisper-accuracy`;

        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: 'include',
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) throw new Error(`Failed to fetch whisper prediction: ${response.status}`);
        const prediction = await response.json();

        let whisperPrediction: WhisperPrediction;
        if (isCustomDataset) {
          whisperPrediction = {
            predicted_transcript: typeof prediction === 'string' ? prediction : prediction?.text || JSON.stringify(prediction),
            ground_truth: "", accuracy_percentage: null, word_error_rate: null, character_error_rate: null,
            levenshtein_distance: null, exact_match: null, character_similarity: null,
            word_count_predicted: 0, word_count_truth: 0
          };
        } else {
          whisperPrediction = {
            predicted_transcript: prediction.predicted_transcript || "", ground_truth: prediction.ground_truth || "",
            accuracy_percentage: prediction.accuracy_percentage !== null ? prediction.accuracy_percentage : null,
            word_error_rate: prediction.word_error_rate !== null ? prediction.word_error_rate : null,
            character_error_rate: prediction.character_error_rate !== null ? prediction.character_error_rate : null,
            levenshtein_distance: prediction.levenshtein_distance !== null ? prediction.levenshtein_distance : null,
            exact_match: prediction.exact_match !== null ? prediction.exact_match : null,
            character_similarity: prediction.character_similarity !== null ? prediction.character_similarity : null,
            word_count_predicted: prediction.word_count_predicted || 0, word_count_truth: prediction.word_count_truth || 0
          };
        }
        setWhisperPrediction(whisperPrediction);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Unknown error";
        setPredictionError(errorMessage);
        console.error("Error fetching whisper prediction:", err);
      } finally {
        setIsLoadingPredictions(false);
        if (whisperRequestRef.current === abortController) whisperRequestRef.current = null;
      }
    };

    fetchWhisperPrediction();
    return () => { if (whisperRequestRef.current) { whisperRequestRef.current.abort(); whisperRequestRef.current = null; } };
  }, [selectedFile, selectedEmbeddingFile, model, dataset]);
  
  const effectiveDataset = (() => {
    if (dataset.startsWith('custom:')) return dataset;
    if (uploadedFiles && uploadedFiles.length > 0) return "custom";
    return dataset;
  })();

  const [predictionMap, setPredictionMap] = useState<Record<string, string>>({});

  const handlePredictionUpdate = (fileId: string, prediction: string) => {
    setPredictionMap(prev => ({ ...prev, [fileId]: prediction }));
  };

  // Hook up upload action to the RQ multi-task endpoint
  const startMultiTaskInference = async (file: UploadedFile) => {
    // Only ASR/SER/ADD are real multitask fan-out families (task_orchestrator's
    // _TASK_FUNCS has no XAI entry - attribution is a separate job kind,
    // enqueued via POST /api/inference/attribution, SAD Use Case 3). Including
    // "xai" here used to make every upload 500 with a backend KeyError before
    // the job could even start - fixed as part of LIT-233's task selector.
    const tasks = (Object.keys(selectedTasks) as Array<keyof SelectedTasks>).filter(
      (task) => selectedTasks[task]
    );
    if (tasks.length === 0) return;

    try {
      const response = await fetch(`${API_BASE}/api/inference/multitask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio_ref: file.file_path,
          tasks
        }),
      });
      if (response.ok) {
        const data = await response.json();
        setActiveTaskId(data.job_id); // Start WebSocket polling
      }
    } catch (error) {
      console.error("Failed to start multi-task inference:", error);
    }
  };

  const handleUploadSuccess = (uploadResponse: UploadedFile) => {
    setUploadedFiles(prev => [...prev, uploadResponse]);
    setSelectedFile(uploadResponse);
    startMultiTaskInference(uploadResponse); // Trigger RQ Job
  };

  const handleFileSelection = (file: UploadedFile) => {
    setSelectedFile(file);
    setSelectedEmbeddingFile(file.filename);
  };

  const handleEmbeddingSelection = (filename: string) => {
    setSelectedEmbeddingFile(filename);
    const matchingUploadedFile = uploadedFiles.find(f => f.filename === filename);
    if (matchingUploadedFile) { setSelectedFile(matchingUploadedFile); return; }
    const fileLike: UploadedFile = { file_id: filename, filename: filename, file_path: filename, message: "Selected from embeddings" };
    setSelectedFile(fileLike);
  };

  const handlePerturbationComplete = (result: any) => {
    setPerturbationResult(result);
    setPerturbedPredictions(null);
  };

  const handlePredictionRefresh = (file: UploadedFile, prediction: string) => {
    if (file.message === "Perturbed file") {
      setUploadedFiles(prevFiles => {
        const existingFile = prevFiles.find(f => f.file_id === file.file_id);
        if (existingFile) return prevFiles.map(f => f.file_id === file.file_id ? { ...f, prediction: prediction } : f);
        else return [...prevFiles, { ...file, prediction: prediction }];
      });
      setPredictionMap(prev => ({ ...prev, [file.filename]: prediction }));
    }
    if (selectedFile && selectedFile.file_id === file.file_id) setSelectedFile(prev => prev ? { ...prev, prediction: prediction } : null);
  };

  const handleBatchInferenceStart = useCallback(() => setBatchInferenceStatus('running'), []);
  const handleBatchInferenceComplete = useCallback(() => setBatchInferenceStatus('done'), []);

  useEffect(() => { setPredictionMap({}); setBatchInferenceStatus('idle'); }, [model, dataset]);

  // Mode 1: Speculative Idle XAI Prefetching (3-second idle timer)
  useEffect(() => {
    if (!selectedFile && !selectedEmbeddingFile) return;

    const abortController = new AbortController();
    const idleTimer = setTimeout(() => {
      const isCustomDataset = dataset?.startsWith('custom:');
      const filename = selectedFile?.filename || selectedEmbeddingFile;
      if (!filename) return;

      const requestBody = isCustomDataset
        ? { file_path: selectedFile?.file_path }
        : { dataset: dataset, dataset_file: filename };

      // Prefetch Acoustic Profile in background silently
      fetch(`${API_BASE}/acoustic/profile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(requestBody),
        signal: abortController.signal,
      }).catch(() => {});

      // Prefetch Saliency Map in background silently
      const saliencyBody = {
        model: model,
        dataset: dataset,
        dataset_file: filename,
        method: "gradcam",
      };
      fetch(`${API_BASE}/saliency/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(saliencyBody),
        signal: abortController.signal,
      }).catch(() => {});
    }, 3000);

    return () => {
      clearTimeout(idleTimer);
      abortController.abort();
    };
  }, [selectedFile, selectedEmbeddingFile, dataset, model]);


  const handleBatchInference = async (selectedModel: string, selectedDataset: string) => {
    if (selectedDataset === 'custom') return;
    setPredictionMap({});
    setBatchInferenceStatus('running');
    try { setBatchInferenceStatus('done'); } catch (error) { console.error('Batch inference failed:', error); setBatchInferenceStatus('idle'); }
  };

  return (
    <EmbeddingProvider>
      <div className="h-screen flex flex-col bg-background">
        <Toolbar
          apiData={apiData} setApiData={setApiData} selectedFile={selectedFile} uploadedFiles={uploadedFiles}
          onFileSelect={setSelectedFile} model={model} setModel={setModel} dataset={dataset} setDataset={setDataset}
          onBatchInference={handleBatchInference}
          selectedTasks={selectedTasks} setSelectedTasks={setSelectedTasks}
        />
        <div className="flex-1 overflow-hidden bg-background">
          <PanelGroup direction="horizontal" className="h-full">
            <Panel defaultSize={25} minSize={20}>
              <EmbeddingPanel model={model} dataset={dataset} availableFiles={availableFiles} selectedFile={selectedEmbeddingFile} onFileSelect={handleEmbeddingSelection} />
            </Panel>
            <PanelResizeHandle className="w-1 bg-border hover:bg-primary/20 transition-colors" />
            
            <Panel defaultSize={50} minSize={30}>
              <PanelGroup direction="vertical">
                <Panel defaultSize={70} minSize={40}>
                  {/* Unified Workflow: Loading screen resolves into visualizations */}
                  <div className="h-full flex flex-col">
                    {activeTaskId && state !== 'SUCCESS' && state !== 'FAILURE' && (
                      <div className="p-4">
                        <GlobalTaskProgress taskId={activeTaskId} onComplete={() => {}} />
                      </div>
                    )}
                    <div className="flex-1 overflow-hidden">
                      <PredictionPanel 
                        selectedFile={selectedFile}
                        selectedEmbeddingFile={selectedEmbeddingFile}
                        model={model}
                        dataset={effectiveDataset}
                        originalDataset={dataset}
                        onPerturbationComplete={handlePerturbationComplete}
                        onPredictionRefresh={handlePredictionRefresh}
                        onPredictionUpdate={handlePredictionUpdate}
                        unifiedResult={state === 'SUCCESS' ? (typeof result === 'string' ? JSON.parse(result) : result) as UnifiedTaskResult : null}
                        audioDuration={selectedFile?.duration || 10.0}
                      />
                    </div>
                  </div>
                </Panel>
                <PanelResizeHandle className="h-1 bg-border hover:bg-primary/20 transition-colors" />
                <Panel defaultSize={30} minSize={20}>
                  <AudioDatasetPanel
                    apiData={apiData} uploadedFiles={uploadedFiles} selectedFile={selectedFile} onFileSelect={handleFileSelection}
                    onUploadSuccess={handleUploadSuccess} model={model} dataset={effectiveDataset} originalDataset={dataset}
                    batchInferenceStatus={batchInferenceStatus} onBatchInferenceStart={handleBatchInferenceStart}
                    onBatchInferenceComplete={handleBatchInferenceComplete} onAvailableFilesChange={setAvailableFiles}
                    onPredictionUpdate={handlePredictionUpdate} predictionMap={predictionMap}
                  />
                </Panel>
              </PanelGroup>
            </Panel>

            <PanelResizeHandle className="w-1 bg-border hover:bg-primary/20 transition-colors" />
            <Panel defaultSize={25} minSize={20}>
              <DatapointEditorPanel 
                selectedFile={selectedFile} selectedEmbeddingFile={selectedEmbeddingFile} dataset={effectiveDataset}
                originalDataset={dataset} perturbationResult={perturbationResult} predictionMap={predictionMap}
                model={model} wav2vecPrediction={wav2vecPrediction} whisperPrediction={whisperPrediction}
                perturbedPredictions={perturbedPredictions} isLoadingPredictions={isLoadingPredictions}
                isLoadingPerturbed={isLoadingPerturbed} predictionError={predictionError}
              />
            </Panel>
          </PanelGroup>
        </div>
        <StatusBar activeTaskId={activeTaskId} taskState={state} />
      </div>
    </EmbeddingProvider>
  );
};
