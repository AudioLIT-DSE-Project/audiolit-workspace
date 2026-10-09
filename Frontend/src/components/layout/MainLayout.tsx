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
import { toast } from "sonner";
import { WarmupModal, WarmupProgress } from "../dataset/WarmupModal";
import { WarmupStatusBanner } from "../dataset/WarmupStatusBanner";
import { QuickStartDialog } from "./QuickStartDialog";
import { readQuickStartDismissed } from "./quickStartStorage";
import {
  readActiveWarmupJobId,
  writeActiveWarmupJobId,
  clearActiveWarmupJobId,
  isTerminalWarmupStatus,
} from "@/lib/warmupJob";
import { isUploadedAudio } from "@/lib/audioSelection";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
  prediction?: string;
  ground_truth?: string;
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
  const [addPrediction, setAddPrediction] = useState<AddPrediction | null>(null);
  const [isLoadingPredictions, setIsLoadingPredictions] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [perturbedPredictions, setPerturbedPredictions] = useState<Wav2Vec2Prediction | WhisperPrediction | null>(null);
  const [isLoadingPerturbed, setIsLoadingPerturbed] = useState(false);
  const [activeInferenceCount, setActiveInferenceCount] = useState(0);

  // Refs to track ongoing requests and prevent duplicates
  const wav2vecRequestRef = useRef<AbortController | null>(null);
  const whisperRequestRef = useRef<AbortController | null>(null);
  const addRequestRef = useRef<AbortController | null>(null);
  
  // RQ Task State (WebSocket listener)
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const { state, result } = useTaskStatus(activeTaskId);

  // Global Warmup Runner State
  const [isQuickStartOpen, setIsQuickStartOpen] = useState(false);
  // Auto-open once per browser, gated by localStorage - never runs a second
  // time in the same session so it doesn't re-fight a user who reopened it
  // manually via the toolbar and then dismissed it.
  useEffect(() => {
    if (!readQuickStartDismissed()) setIsQuickStartOpen(true);
  }, []);

  const [isWarmupModalOpen, setIsWarmupModalOpen] = useState(false);
  const [warmupJobId, setWarmupJobId] = useState<string | null>(null);
  const [warmupProgress, setWarmupProgress] = useState<WarmupProgress | null>(null);
  const [isStartingWarmup, setIsStartingWarmup] = useState(false);
  const [isWarmupMinimized, setIsWarmupMinimized] = useState(false);
  // Dataset the running job belongs to, which is not necessarily the one
  // currently selected in the UI when we reattach to a job after a reload.
  const [warmupDataset, setWarmupDataset] = useState<string | null>(null);

  // Reattach to a warmup that is still running.
  //
  // The job id used to live only in this component's state, so a reload, a
  // navigation, or the browser discarding a backgrounded tab dropped it. The
  // RQ job kept running (24h job timeout) with no banner and - because cancel
  // is addressed by job id - no way to stop it. Two recovery paths, in order:
  // the id we persisted locally, then the server's own list of live jobs,
  // which also covers a cleared storage, a different browser, or a job another
  // tab started.
  useEffect(() => {
    let cancelled = false;

    const reattach = async () => {
      const stored = readActiveWarmupJobId();
      if (stored) {
        try {
          const res = await fetch(`${API_BASE}/api/inference/progress/${stored}`);
          if (res.ok) {
            const data = await res.json();
            if (!cancelled && (data.status === "running" || data.status === "cancelling")) {
              setWarmupJobId(stored);
              setWarmupProgress(data);
              if (data.dataset) setWarmupDataset(data.dataset);
              return;
            }
          }
        } catch {
          /* fall through to server discovery */
        }
        // Stored id is finished, unknown or unreachable - stop carrying it.
        clearActiveWarmupJobId();
      }

      try {
        const res = await fetch(`${API_BASE}/api/inference/warmup/active`);
        if (!res.ok) return;
        const data = await res.json();
        const job = data?.jobs?.[0];
        if (!cancelled && job?.job_id) {
          setWarmupJobId(job.job_id);
          setWarmupProgress(job);
          if (job.dataset) setWarmupDataset(job.dataset);
          writeActiveWarmupJobId(job.job_id);
        }
      } catch (err) {
        console.error("Failed to look up active warmups:", err);
      }
    };

    reattach();
    return () => { cancelled = true; };
  }, []);

  // Poll for Warmup Progress
  useEffect(() => {
    if (!warmupJobId) return;

    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/api/inference/progress/${warmupJobId}`);
        if (response.ok) {
          const data = await response.json();
          setWarmupProgress(data);
          if (data.dataset) setWarmupDataset(data.dataset);
          if (isTerminalWarmupStatus(data.status)) {
            clearInterval(interval);
            // Terminal: stop advertising this id so the next mount does not
            // try to reattach to a finished run.
            clearActiveWarmupJobId();
          }
          if (data.status === 'not_found') {
            // The progress record expired or was flushed; nothing to track.
            clearInterval(interval);
            clearActiveWarmupJobId();
          }
        }
      } catch (err) {
        console.error("Failed to poll warmup progress:", err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [warmupJobId]);

  const handleStartWarmup = async () => {
    if (!dataset) return;
    setIsStartingWarmup(true);
    try {
      const response = await fetch(`${API_BASE}/api/inference/batch-warmup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset: dataset,
          model: model,
          tasks: ["asr", "ser", "acoustic", "saliency"],
          cooldown_ms: 100
        }),
      });
      if (response.ok) {
        const data = await response.json();
        setWarmupJobId(data.job_id);
        setWarmupDataset(dataset);
        // Persisted immediately: if the tab is reloaded or discarded a second
        // later, this is what lets the banner and the cancel button come back.
        writeActiveWarmupJobId(data.job_id);
      }
    } catch (err) {
      console.error("Failed to start batch warmup:", err);
    } finally {
      setIsStartingWarmup(false);
    }
  };

  const handleCancelWarmup = async () => {
    if (!warmupJobId) return;
    try {
      const response = await fetch(`${API_BASE}/api/inference/cancel/${warmupJobId}`, {
        method: "POST",
      });
      if (!response.ok) throw new Error(`Cancel failed: ${response.status}`);
      // The backend reports the state the run is actually in: "cancelling"
      // while a live worker finishes its current step, or "cancelled" at once
      // when the run was still queued or its worker had died (previously the
      // UI assumed "cancelling" and the next poll flipped it back to running).
      const data = await response.json();
      const status: string = data.status === 'not_found' ? 'cancelled' : data.status;
      setWarmupProgress(prev => prev ? { ...prev, ...data, status } : data);
      // The id stays persisted while "cancelling", so a reload in that window
      // can still find the run; it is only dropped once the run is terminal.
      if (isTerminalWarmupStatus(status)) clearActiveWarmupJobId();
    } catch (err) {
      console.error("Failed to cancel warmup:", err);
    }
  };

  const handleClearCache = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/cache/clear`, {
        method: "POST",
      });
      if (response.ok) {
        setPredictionMap({});
        setWav2vecPrediction(null);
        setWhisperPrediction(null);
        setPerturbedPredictions(null);
        alert("Cache cleared successfully! All cached ML predictions, acoustic profiles, and saliency maps have been reset.");
      }
    } catch (err) {
      console.error("Failed to clear cache:", err);
    }
  };

  // Clear selected file, embedding file, and predictions when dataset changes
  useEffect(() => {
    setSelectedFile(null);
    setSelectedEmbeddingFile(null);
    setAvailableFiles([]);
    setWav2vecPrediction(null);
    setWhisperPrediction(null);
    setAddPrediction(null);
    setPredictionError(null);
    setPerturbationResult(null);
  }, [dataset]);

  // Clear perturbation result and predictions when selected file changes
  useEffect(() => {
    setPerturbationResult(null);
    setWav2vecPrediction(null);
    setWhisperPrediction(null);
    setAddPrediction(null);
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
      const isUploadedFile = isUploadedAudio(selectedFile, dataset);

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
      const isUploadedFile = isUploadedAudio(selectedFile, dataset);

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
          // LIT-247 follow-up: custom datasets can now carry ground truth
          // (uploaded via the Ground Truth CSV tab), but /inferences/run
          // doesn't compute WER/accuracy for them - selectedFile.ground_truth
          // (populated by AudioDatasetPanel's row selection) is the only
          // source for it here, so metrics stay null while the text itself
          // still displays instead of "No Ground Truth Available".
          whisperPrediction = {
            predicted_transcript: typeof prediction === 'string' ? prediction : prediction?.text || JSON.stringify(prediction),
            ground_truth: selectedFile?.ground_truth || "", accuracy_percentage: null, word_error_rate: null, character_error_rate: null,
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

  // Fetch deepfake (ADD) prediction for dataset-browsing selections only,
  // mirroring the wav2vec2/whisper effects above. /inferences/run is a
  // generic dict dispatch (inference_service.MODEL_FUNCTIONS), so this works
  // for both selectable ADD checkpoints (melody-machine, wav2vec2-add) once
  // they're registered there.
  useEffect(() => {
    const fetchAddPrediction = async () => {
      const isUploadedFile = isUploadedAudio(selectedFile, dataset);

      if (!ADD_MODEL_KEYS.includes(model) || (!selectedFile && !selectedEmbeddingFile) || isUploadedFile) {
        setAddPrediction(null);
        setPredictionError(null);
        setIsLoadingPredictions(false);
        return;
      }

      if (addRequestRef.current) addRequestRef.current.abort();
      const abortController = new AbortController();
      addRequestRef.current = abortController;

      setIsLoadingPredictions(true);
      setPredictionError(null);

      try {
        const requestBody: any = { model };
        if (selectedFile) {
          requestBody.dataset = dataset;
          requestBody.dataset_file = selectedFile.filename;
        } else if (selectedEmbeddingFile && dataset) {
          requestBody.dataset = dataset;
          requestBody.dataset_file = selectedEmbeddingFile;
        }

        const response = await fetch(`${API_BASE}/inferences/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: 'include',
          body: JSON.stringify(requestBody),
          signal: abortController.signal
        });

        if (!response.ok) throw new Error(`Failed to fetch prediction: ${response.status}`);
        const prediction = await response.json();
        setAddPrediction(prediction);
      } catch (err) {
        if (err.name === 'AbortError') return;
        const errorMessage = err instanceof Error ? err.message : "Unknown error";
        setPredictionError(errorMessage);
        console.error("Error fetching deepfake prediction:", err);
      } finally {
        setIsLoadingPredictions(false);
        if (addRequestRef.current === abortController) addRequestRef.current = null;
      }
    };

    fetchAddPrediction();
    return () => { if (addRequestRef.current) { addRequestRef.current.abort(); addRequestRef.current = null; } };
  }, [selectedFile, selectedEmbeddingFile, model, dataset]);

  const effectiveDataset = dataset;

  const [predictionMap, setPredictionMap] = useState<Record<string, string>>({});

  // Stable identity. AudioDatasetPanel's cached-prediction check lists this
  // callback as an effect dependency and calls it with each cached result. As
  // a plain function it was new on every render, so the check re-ran after its
  // own state update and never stopped: about 30 POST /inferences/batch-check a
  // second for as long as a dataset with cached predictions was on screen.
  const handlePredictionUpdate = useCallback((fileId: string, prediction: string) => {
    setPredictionMap(prev => ({ ...prev, [fileId]: prediction }));
  }, []);

  const handleUploadSuccess = (uploadResponse: UploadedFile) => {
    setUploadedFiles(prev => [uploadResponse, ...prev.filter(f => f.file_id !== uploadResponse.file_id)]);
    setSelectedFile(uploadResponse);
  };

  const handleDeleteLiveRecording = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(f => f.file_id !== fileId));
    if (selectedFile?.file_id === fileId) {
      setSelectedFile(null);
    }
    toast.success("Live recording removed from session");
  };

  const handleSaveLiveToCustom = async (file: UploadedFile) => {
    try {
      const datasetName = window.prompt(
        `Save live recording "${file.filename}" to a custom dataset (enter dataset name):`,
        "my_recordings"
      );
      if (!datasetName || !datasetName.trim()) return;

      const trimmedName = datasetName.trim().toLowerCase().replace(/[^a-z0-9_-]/g, "_");
      
      // Ensure custom dataset exists
      await fetch(`${API_BASE}/dataset/create`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ dataset_name: trimmedName }),
        credentials: "include"
      }).catch(() => {});

      toast.success(`Saved "${file.filename}" to custom dataset "${trimmedName}"`);
    } catch (err) {
      console.error("Failed to save live recording to custom dataset:", err);
      toast.error("Failed to save live recording to custom dataset");
    }
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
      const sfAny = selectedFile as any;
      const isUploadedFile = isUploadedAudio(selectedFile, dataset);
      const filename = selectedFile?.filename || selectedEmbeddingFile;
      if (!filename && !selectedFile?.file_path) return;

      const requestBody = isUploadedFile
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
      const saliencyBody = isUploadedFile
        ? { model: model, file_path: selectedFile?.file_path, method: "gradcam" }
        : { model: model, dataset: dataset, dataset_file: filename, method: "gradcam" };

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
          onFileSelect={setSelectedFile} onUploadSuccess={handleUploadSuccess} model={model} setModel={setModel} dataset={dataset} setDataset={setDataset}
          onBatchInference={handleBatchInference}
          selectedTasks={selectedTasks} setSelectedTasks={setSelectedTasks}
          onWarmupClick={() => { setIsWarmupMinimized(false); setIsWarmupModalOpen(true); }}
          warmupJobId={warmupJobId}
          onQuickStartClick={() => setIsQuickStartOpen(true)}
        />

        <QuickStartDialog open={isQuickStartOpen} onOpenChange={setIsQuickStartOpen} />
        
        {/* Global Dataset Warmup Modal (Confirmation & Active Progress) */}
        <WarmupModal
          isOpen={isWarmupModalOpen && !isWarmupMinimized}
          onClose={() => setIsWarmupModalOpen(false)}
          dataset={effectiveDataset || dataset}
          model={model}
          warmupJobId={warmupJobId}
          warmupProgress={warmupProgress}
          isStarting={isStartingWarmup}
          onStartWarmup={handleStartWarmup}
          onCancelWarmup={handleCancelWarmup}
          onMinimize={() => setIsWarmupMinimized(true)}
          onClearCache={handleClearCache}
        />

        {/* Floating Bottom-Right Status Banner when Warmup Modal is Minimized or Running in Background */}
        <WarmupStatusBanner
          warmupJobId={warmupJobId}
          warmupProgress={warmupProgress}
          dataset={warmupDataset || effectiveDataset || dataset}
          isMinimized={isWarmupMinimized || !isWarmupModalOpen}
          onExpand={() => { setIsWarmupMinimized(false); setIsWarmupModalOpen(true); }}
          onCancel={handleCancelWarmup}
          onDismiss={() => {
            // Hides the banner. Deliberately does NOT cancel the run, and
            // deliberately does not forget the id while the run is still
            // going: a reload re-surfaces it, because a job burning CPU for
            // hours should not be silently dismissable.
            setWarmupJobId(null);
            setWarmupProgress(null);
            setWarmupDataset(null);
          }}
        />
        {/* The workbench panels are the page's main content. Without a
            main landmark a screen reader user has no way to skip the
            toolbar and jump straight to the work area, which axe reports
            as landmark-one-main. */}
        <main className="flex-1 overflow-hidden bg-background">
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
                        whisperPrediction={whisperPrediction}
                        wav2vecPrediction={wav2vecPrediction}
                        addPrediction={addPrediction}
                      />
                    </div>
                  </div>
                </Panel>
                <PanelResizeHandle className="h-1 bg-border hover:bg-primary/20 transition-colors" />
                <Panel defaultSize={30} minSize={20}>
                  <AudioDatasetPanel
                    apiData={apiData} uploadedFiles={uploadedFiles} selectedFile={selectedFile} onFileSelect={handleFileSelection}
                    onUploadSuccess={handleUploadSuccess} onDeleteLiveRecording={handleDeleteLiveRecording} onSaveLiveToCustom={handleSaveLiveToCustom} model={model} dataset={effectiveDataset} originalDataset={dataset}
                    batchInferenceStatus={batchInferenceStatus} onBatchInferenceStart={handleBatchInferenceStart}
                    onBatchInferenceComplete={handleBatchInferenceComplete} onAvailableFilesChange={setAvailableFiles}
                    onPredictionUpdate={handlePredictionUpdate} predictionMap={predictionMap}
                    onActiveInferenceCountChange={setActiveInferenceCount}
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
                addPrediction={addPrediction}
                perturbedPredictions={perturbedPredictions} isLoadingPredictions={isLoadingPredictions}
                isLoadingPerturbed={isLoadingPerturbed} predictionError={predictionError}
              />
            </Panel>
          </PanelGroup>
        </main>
        <StatusBar activeTaskId={activeTaskId} taskState={state} activeInferenceCount={activeInferenceCount} />
      </div>
    </EmbeddingProvider>
  );
};
