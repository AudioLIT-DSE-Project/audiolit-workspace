import { useState, useRef, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { Upload, Search, Play, Pause, Square, RefreshCw, HelpCircle } from "lucide-react";
import { AudioUploader } from "../audio/AudioUploader";
import { AudioDataTable } from "../audio/AudioDataTable";
import { DatasetLicenseNotice } from "../dataset/DatasetLicenseNotice";
import { toast } from "sonner";
import { API_BASE } from '@/lib/api';

interface DatasetLicenseInfo {
  license: string | null;
  task_family: string | null;
  non_commercial: boolean;
}

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
  prediction?:string
  ground_truth?: string;
}

// Same generic key list AudioDataTable.tsx's "Ground Truth" column reads,
// so a row selected here carries the same ground truth the table displays
// for it (LIT-247 follow-up: custom datasets' Original Transcription
// Metrics panel was hardcoding "" instead of reading this).
const GROUND_TRUTH_KEYS = ["sentence", "transcript", "text", "statement", "emotion", "label", "ground_truth", "target"];

// LIT-249: /inferences/run returns a bare string for Whisper (ASR) but a
// structured object for everything else - deepfake (ADD) predictions come
// back as {predicted_label, confidence, probabilities, ...}. The dataset
// table only ever needs a short display value per cell (the full object,
// with its probability breakdown, is fetched separately for the sidebar's
// "Deepfake Detection Results" card by MainLayout's fetchAddPrediction
// effect) - stringifying the whole object here used to dump raw JSON into
// the table cell instead of just the label.
const extractPredictionDisplayText = (prediction: unknown): string => {
  if (typeof prediction === 'string') return prediction;
  if (prediction && typeof prediction === 'object') {
    const p = prediction as Record<string, unknown>;
    const candidate = p.text ?? p.predicted_transcript ?? p.predicted_emotion ?? p.predicted_label ?? p.prediction;
    if (typeof candidate === 'string') return candidate;
  }
  return JSON.stringify(prediction);
};

interface AudioDatasetPanelProps {
  apiData?: unknown;
  model: string | null;
  dataset: string;
  originalDataset?: string;
  uploadedFiles?: UploadedFile[];
  selectedFile?: UploadedFile | null;
  onFileSelect?: (file: UploadedFile) => void;
  onUploadSuccess?: (uploadResponse: UploadedFile) => void;
  batchInferenceStatus?: 'idle' | 'running' | 'done';
  onBatchInferenceStart?: () => void;
  onBatchInferenceComplete?: () => void;
  onAvailableFilesChange?: (files: string[]) => void;
  onPredictionUpdate?: (fileId: string, prediction: string) => void;
  predictionMap?: Record<string, string>;
}

export const AudioDatasetPanel = ({ 
  apiData, 
  model,
  dataset,
  originalDataset,
  selectedFile, 
  onFileSelect, 
  onUploadSuccess,
  batchInferenceStatus,
  onBatchInferenceStart,
  onBatchInferenceComplete,
  onAvailableFilesChange,
  onPredictionUpdate,
  predictionMap: externalPredictionMap
}: AudioDatasetPanelProps) => {
  const [selectedRow, setSelectedRow] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [datasetMetadata, setDatasetMetadata] = useState<Record<string, string | number>[]>([]);
  // Use external predictionMap from parent
  const predictionMap = externalPredictionMap || {};
  const [inferenceStatus, setInferenceStatus] = useState<Record<string, 'idle' | 'loading' | 'done' | 'error'>>({});
  
  // Batch inference state
  const [currentInferenceIndex, setCurrentInferenceIndex] = useState(0);
  const [batchInferenceQueue, setBatchInferenceQueue] = useState<string[]>([]);
  const [isInferenceComplete, setIsInferenceComplete] = useState(false);
  const [currentModelDataset, setCurrentModelDataset] = useState<string>("");
  const abortControllerRef = useRef<AbortController | null>(null);

  // FR2.3 / SAD C5 — licence metadata for the built-in benchmark corpora, so a
  // non-commercial dataset (RAVDESS, L2-ARCTIC, ESD, ASVspoof 2021 DF) can show
  // a notice on load instead of the licence info being retained server-side
  // but never surfaced to the user.
  const [datasetLicenses, setDatasetLicenses] = useState<Record<string, DatasetLicenseInfo>>({});
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/datasets/list`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!cancelled && data?.licenses) setDatasetLicenses(data.licenses);
      })
      .catch(() => {
        // Non-fatal: the dataset table still works without licence info.
      });
    return () => {
      cancelled = true;
    };
  }, []);
  const currentLicenseInfo = datasetLicenses[dataset];

  // Sync selectedRow when selectedFile changes from external selection (e.g., embeddings)
  useEffect(() => {
    if (selectedFile) {
      // For uploaded files, use file_id
      if (uploadedFiles.some(f => f.file_id === selectedFile.file_id)) {
        setSelectedRow(selectedFile.file_id);
        return;
      }
      
      // For dataset files, find matching row by filename
      if (datasetMetadata.length > 0) {
        const matchingRow = datasetMetadata.find(row => {
          const pathVal = (row["path"] || row["filepath"] || row["file"] || row["filename"]) as string;
          const filename = pathVal ? (pathVal.split("/").pop() || pathVal.split("\\").pop() || pathVal) : String(row["id"]);
          return filename === selectedFile.filename;
        });
        
        if (matchingRow) {
          const rowId = String(matchingRow["id"] || matchingRow["path"] || matchingRow["filepath"] || matchingRow["file"] || matchingRow["filename"]);
          setSelectedRow(rowId);
        }
      }
    }
  }, [selectedFile, uploadedFiles, datasetMetadata]);

  // Stable handlers to prevent downstream re-renders
  const handleRowSelect = useCallback((id: string) => {
    setSelectedRow(id);
    
    // When a row is selected, just propagate the file selection for UI/audio playback
    // No inference should be triggered here
    if (!onFileSelect) {
      return;
    }
    
    // When showing combined data (uploaded + dataset files), check if it's an uploaded file first
    if (dataset === "custom") {
      const uploadedFile = uploadedFiles?.find(f => f.file_id === id);
      if (uploadedFile) {
        onFileSelect(uploadedFile);
        return;
      }
      // If not an uploaded file, treat it as a dataset file (fall through to dataset logic)
    }

    const findMatch = () => {
      for (const row of datasetMetadata) {
        const rowId = row["id"]; 
        const path = row["path"] || row["filepath"] || row["file"] || row["filename"];
        if (typeof rowId === "string" && rowId === id) return row;
        if (typeof path === "string" && (path === id || path.endsWith(`/${id}`) || path.endsWith(`\\${id}`))) return row;
      }
      return null;
    };

    const match = findMatch();
    if (!match) return;

    const pathVal = (match["path"] || match["filepath"] || match["file"] || match["filename"]) as string | undefined;
    const filename = pathVal ? (pathVal.split("/").pop() || pathVal.split("\\").pop() || String(id)) : String(id);
    const groundTruth = GROUND_TRUTH_KEYS.map((k) => match[k]).find((v) => typeof v === "string" && v.trim() !== "") as string | undefined;

    const fileLike: UploadedFile = {
      file_id: String(id),
      filename,
      file_path: pathVal || filename,
      message: dataset.startsWith('custom:') ? "Selected from custom dataset" : "Selected from dataset", // This indicates it's a dataset file
      ground_truth: groundTruth,
    };

    // Just select the file for UI purposes, no inference
    onFileSelect(fileLike);
  }, [dataset, datasetMetadata, onFileSelect]);

  const handleFilePlay = useCallback((file: UploadedFile) => {
    if (onFileSelect) {
      onFileSelect(file);
    }
  }, [onFileSelect]);

  // No need for local prediction update handling since we use external predictionMap

  const [isInferencingActive, setIsInferencingActive] = useState(false);

  const handleVisibleRowIdsChange = useCallback((ids: string[]) => {
    // This is now just for pagination, no inference triggering
  }, []);

  // Abort ongoing requests and reset inference state whenever model or dataset changes
  useEffect(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    setIsInferencingActive(false);
    setIsInferenceComplete(false);
    setCurrentInferenceIndex(0);
    setBatchInferenceQueue([]);
    setInferenceStatus({});
  }, [model, dataset, originalDataset]);

  // Explicit handler triggered ONLY when user clicks "Get Inferences"
  const handleStartBatchInference = useCallback(async () => {
    if (!model || datasetMetadata.length === 0) return;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    setIsInferencingActive(true);
    setIsInferenceComplete(false);
    setCurrentInferenceIndex(0);
    
    if (onBatchInferenceStart) onBatchInferenceStart();

    try {
      const filenames = datasetMetadata.map(row => {
        const pathVal = (row["path"] || row["filepath"] || row["file"] || row["filename"]) as string;
        return pathVal ? (pathVal.split("/").pop() || pathVal.split("\\").pop() || pathVal) : String(row["id"] || "unknown");
      });

      const response = await fetch(`${API_BASE}/inferences/batch-check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ model, dataset, files: filenames }),
        signal: abortControllerRef.current?.signal,
      });

      if (response.ok) {
        const { cached_results, missing_files } = await response.json();
        const newPredictionMap: Record<string, string> = {};
        const newInferenceStatus: Record<string, 'idle' | 'loading' | 'done' | 'error'> = {};

        datasetMetadata.forEach((row, index) => {
          const fileId = String(row["id"] || row["path"] || row["filepath"] || row["file"] || row["filename"] || index);
          const pathVal = (row["path"] || row["filepath"] || row["file"] || row["filename"]) as string;
          const filename = pathVal ? (pathVal.split("/").pop() || pathVal.split("\\").pop() || pathVal) : fileId;

          if (cached_results[filename]) {
            newPredictionMap[fileId] = cached_results[filename];
            newInferenceStatus[fileId] = 'done';
            if (onPredictionUpdate) onPredictionUpdate(fileId, cached_results[filename]);
          } else {
            newInferenceStatus[fileId] = 'idle';
          }
        });

        setInferenceStatus(newInferenceStatus);

        if (missing_files.length === 0) {
          setIsInferenceComplete(true);
          setIsInferencingActive(false);
          if (onBatchInferenceComplete) onBatchInferenceComplete();
          toast.success("All predictions loaded!");
          return;
        }

        const fileIds = datasetMetadata
          .filter((row, index) => {
            const pathVal = (row["path"] || row["filepath"] || row["file"] || row["filename"]) as string;
            const filename = pathVal ? (pathVal.split("/").pop() || pathVal.split("\\").pop() || pathVal) : String(row["id"] || index);
            return missing_files.includes(filename);
          })
          .map((row, index) => String(row["id"] || row["path"] || row["filepath"] || row["file"] || row["filename"] || index));

        setBatchInferenceQueue(fileIds);
      }
    } catch (err: any) {
      if (err.name === 'AbortError') return;
      console.error("Batch check failed:", err);
      const fileIds = datasetMetadata.map((row, index) => String(row["id"] || row["path"] || row["filepath"] || row["file"] || row["filename"] || index));
      setBatchInferenceQueue(fileIds);
    }
  }, [model, dataset, datasetMetadata, onBatchInferenceStart, onBatchInferenceComplete, onPredictionUpdate]);

  const handleStopBatchInference = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setIsInferencingActive(false);
    setBatchInferenceQueue([]);
    toast.info("Inference stopped by user.");
  }, []);

  // LIT-248: re-run inference for exactly one row on demand, independent of
  // (and safe to call alongside) the batch queue above. force_refresh: true
  // bypasses /inferences/run's cache - without it, a deterministic model on
  // an unchanged file would just hand back the same cached prediction and
  // the button would appear to do nothing.
  const handleRegenerateRow = useCallback(async (fileId: string) => {
    if (!model) return;
    const currentRow = datasetMetadata.find(row => {
      const id = row["id"] || row["path"] || row["filepath"] || row["file"] || row["filename"];
      return String(id) === fileId;
    });
    if (!currentRow) return;

    const pathVal = (currentRow["path"] || currentRow["filepath"] || currentRow["file"] || currentRow["filename"]) as string;
    const filename = pathVal ? (pathVal.split("/").pop() || pathVal.split("\\").pop() || fileId) : fileId;

    setInferenceStatus(prev => ({ ...prev, [fileId]: 'loading' }));

    try {
      const response = await fetch(`${API_BASE}/inferences/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ model, dataset, dataset_file: filename, force_refresh: true }),
        signal: abortControllerRef.current?.signal,
      });

      if (!response.ok) throw new Error(`API error: ${response.status}`);

      const prediction = await response.json();
      const predictionText = extractPredictionDisplayText(prediction);

      if (onPredictionUpdate) onPredictionUpdate(fileId, predictionText);
      setInferenceStatus(prev => ({ ...prev, [fileId]: 'done' }));
      toast.success(`Regenerated prediction for ${filename}`);
    } catch (error: any) {
      if (error.name === 'AbortError') return;
      console.error(`Regenerate failed for ${fileId}:`, error);
      setInferenceStatus(prev => ({ ...prev, [fileId]: 'error' }));
      toast.error(`Failed to regenerate prediction for ${filename}`);
    }
  }, [model, dataset, datasetMetadata, onPredictionUpdate]);

  // Process batch inference queue when active
  useEffect(() => {
    if (!isInferencingActive || batchInferenceQueue.length === 0) return;
    if (currentInferenceIndex >= batchInferenceQueue.length) {
      console.log('Batch inference completed');
      setIsInferenceComplete(true);
      setIsInferencingActive(false);
      if (onBatchInferenceComplete) {
        onBatchInferenceComplete();
      }
      toast.success("Batch inference complete!");
      return;
    }

    const currentFileId = batchInferenceQueue[currentInferenceIndex];
    const currentRow = datasetMetadata.find(row => {
      const id = row["id"] || row["path"] || row["filepath"] || row["file"] || row["filename"];
      return String(id) === currentFileId;
    });

    if (!currentRow) {
      setCurrentInferenceIndex(prev => prev + 1);
      return;
    }

    const runInference = async () => {
      try {
        setInferenceStatus(prev => ({ ...prev, [currentFileId]: 'loading' }));
        
        const pathVal = (currentRow["path"] || currentRow["filepath"] || currentRow["file"] || currentRow["filename"]) as string;
        const filename = pathVal ? (pathVal.split("/").pop() || pathVal.split("\\").pop() || currentFileId) : currentFileId;

        const requestBody = {
          model,
          dataset,
          dataset_file: filename
        };

        const response = await fetch(`${API_BASE}/inferences/run`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify(requestBody),
          signal: abortControllerRef.current?.signal,
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const prediction = await response.json();
        const predictionText = extractPredictionDisplayText(prediction);

        if (onPredictionUpdate) {
          onPredictionUpdate(currentFileId, predictionText);
        }
        setInferenceStatus(prev => ({ ...prev, [currentFileId]: 'done' }));
        
      } catch (error: any) {
        if (error.name === 'AbortError') return;
        console.error(`Inference failed for ${currentFileId}:`, error);
        setInferenceStatus(prev => ({ ...prev, [currentFileId]: 'error' }));
      }
      
      setCurrentInferenceIndex(prev => prev + 1);
    };

    const timeoutId = setTimeout(runInference, 100);
    return () => clearTimeout(timeoutId);
  }, [isInferencingActive, batchInferenceQueue, currentInferenceIndex, datasetMetadata, model, dataset, onBatchInferenceComplete, onPredictionUpdate]);

  // Cleanup on unmount or when dataset changes
  // Reload function to refresh dataset metadata
  const handleReloadDataset = useCallback(async () => {
    const datasetToUse = originalDataset || dataset;
    if (!datasetToUse || datasetToUse === "custom") {
      setDatasetMetadata([]);
      return;
    }
    
    try {
      const res = await fetch(`${API_BASE}/${datasetToUse}/metadata`, { credentials: 'include' });
      if (!res.ok) throw new Error(`Failed to fetch metadata: ${res.status}`);
      const data = await res.json();
      if (Array.isArray(data)) {
        setDatasetMetadata(data as Record<string, string | number>[]);
        
        // Extract filenames for embeddings
        const filenames = data.map((row: Record<string, string | number>) => {
          const pathVal = row["path"] || row["filepath"] || row["file"] || row["filename"];
          const filename = typeof pathVal === 'string' ? 
            (pathVal.split("/").pop() || pathVal.split("\\").pop() || pathVal) : 
            String(pathVal);
          return filename;
        });
        
        onAvailableFilesChange?.(filenames);
        toast.success("Dataset reloaded successfully");
      } else {
        setDatasetMetadata([]);
        onAvailableFilesChange?.([]);
      }
    } catch (error) {
      console.error('Failed to reload dataset:', error);
      toast.error("Failed to reload dataset");
    }
  }, [dataset, originalDataset, onAvailableFilesChange]);

  useEffect(() => {
    abortControllerRef.current = new AbortController();
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [model, dataset]);

  // Fetch dataset metadata when originalDataset changes 
  useEffect(() => {
    const datasetToUse = originalDataset || dataset;
    setDatasetMetadata([]);
    onAvailableFilesChange?.([]);
    
    // Skip legacy "custom" (individual uploaded files)
    if (datasetToUse === "custom") {
      return;
    }
    
    // Handle both global datasets and custom datasets
    const isCustomDataset = datasetToUse.startsWith('custom:');
    if (!datasetToUse) {
      setDatasetMetadata([]);
      return;
    }
    
    const ac = new AbortController();
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/${datasetToUse}/metadata`, { signal: ac.signal, credentials: 'include' });
        if (!res.ok) throw new Error(`Failed to fetch metadata: ${res.status}`);
        const data = await res.json();
        if (Array.isArray(data)) {
          setDatasetMetadata(data as Record<string, string | number>[]);
          
          // Extract filenames for embeddings
          const filenames = data.map((row: Record<string, string | number>) => {
            const pathVal = row["path"] || row["filepath"] || row["file"] || row["filename"];
            const filename = typeof pathVal === 'string' ? 
              (pathVal.split("/").pop() || pathVal.split("\\").pop() || pathVal) : 
              String(pathVal);
            return filename;
          });
          
          onAvailableFilesChange?.(filenames);
        } else {
          setDatasetMetadata([]);
          onAvailableFilesChange?.([]);
        }
      } catch (e) {
        const name = (e as { name?: string } | null)?.name;
        if (name !== 'AbortError') console.error(e);
      }
    })();
    return () => ac.abort();
  }, [originalDataset, dataset]);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        // Check both MIME type and file extension for better .flac support
        const allowedExtensions = ['.wav', '.mp3', '.m4a', '.flac'];
        const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
        const isValidFile = file.type.startsWith('audio/') || allowedExtensions.includes(fileExtension);
        
        if (isValidFile) {
          try {
            await uploadFile(file, model ?? "");
          } catch (error) {
            console.error('Upload error:', error);
          }
        } else {
          toast.error(`Invalid file type: ${file.name}. Supported formats: WAV, MP3, M4A, FLAC`);
        }
      }
    }
    // Reset the input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const uploadFile = async (file: File, model: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        credentials: 'include',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await response.json();
      setUploadedFiles(prevFiles => [...prevFiles, data]);
      toast.success(`Uploaded: ${file.name}`);
      
      if (onUploadSuccess) {
        onUploadSuccess(data);
      }
      
      return data;
    } catch (error) {
      console.error('Upload error:', error);
      const msg = error instanceof Error ? error.message : 'Unknown error';
      toast.error(`Failed to upload ${file.name}: ${msg}`);
      throw error;
    }
  };

  return (
    <TooltipProvider>
      <div className="h-full bg-panel-background flex flex-col">
        <div className="bg-panel-header p-3 border-b border-border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <h3 className="font-semibold text-foreground text-sm">Audio Dataset</h3>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" />
                </TooltipTrigger>
                <TooltipContent className="text-xs space-y-1">
                  <p>Browse and manage audio files in your selected dataset.</p>
                  <p>Upload new files or select from existing datasets.</p>
                </TooltipContent>
              </Tooltip>
            </div>
            <div className="flex items-center gap-1.5">
              <Badge variant="outline" className="text-[10px] bg-muted">
                {uploadedFiles ? `${uploadedFiles.length} uploaded` : "0 files"}
              </Badge>
              {isInferencingActive && (
                <Badge variant="outline" className="text-[10px] bg-primary/10 text-primary border-primary/20 animate-pulse">
                  Inferencing... {currentInferenceIndex + 1}/{batchInferenceQueue.length || datasetMetadata.length}
                </Badge>
              )}
              {isInferenceComplete && !isInferencingActive && (
                <Badge variant="outline" className="text-[10px] bg-emerald-500/10 text-emerald-600 border-emerald-500/20">
                  ✓ Inference Complete
                </Badge>
              )}
              
              {dataset !== "custom" && (
                isInferencingActive ? (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button size="sm" variant="destructive" className="h-7 text-xs font-medium gap-1" onClick={handleStopBatchInference}>
                        <Square className="h-3 w-3 fill-current" />
                        Stop
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Stop running batch inference</p>
                    </TooltipContent>
                  </Tooltip>
                ) : (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        size="sm"
                        variant="default"
                        className="h-7 text-xs font-medium gap-1 bg-primary text-primary-foreground hover:bg-primary/90"
                        onClick={handleStartBatchInference}
                        disabled={datasetMetadata.length === 0 || !model}
                      >
                        <Play className="h-3 w-3 fill-current" />
                        Get Inferences
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Run model inference on all files in this dataset</p>
                    </TooltipContent>
                  </Tooltip>
                )
              )}

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button size="sm" variant="secondary" className="h-7 text-xs" onClick={handleUploadClick}>
                    <Upload className="h-3 w-3 mr-1" />
                    Upload
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Upload audio files (.wav, .mp3, .m4a, .flac)</p>
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button size="sm" variant="secondary" className="h-7 w-7 p-0" onClick={handleReloadDataset} title="Reload dataset">
                    <RefreshCw className="h-3 w-3" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Reload dataset metadata and refresh the file list</p>
                </TooltipContent>
              </Tooltip>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*,.flac,.wav,.mp3,.m4a"
              multiple
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>
        </div>

        {/* Live Batch Inference Progress Bar Banner */}
        {isInferencingActive && batchInferenceQueue.length > 0 && (
          <div className="bg-primary/10 border-b border-primary/20 px-3 py-1.5 flex items-center justify-between text-xs text-primary font-medium animate-in fade-in duration-200">
            <div className="flex items-center gap-2 truncate">
              <RefreshCw className="h-3.5 w-3.5 animate-spin shrink-0" />
              <span className="truncate">
                Inferencing sample {currentInferenceIndex + 1} of {batchInferenceQueue.length}...
              </span>
            </div>
            <span className="font-mono text-[11px] font-semibold shrink-0 ml-2">
              {Math.round(((currentInferenceIndex + 1) / batchInferenceQueue.length) * 100)}%
            </span>
          </div>
        )}

        {/* FR2.3 / SAD C5 — licence notice for non-commercial corpora */}
        {currentLicenseInfo?.non_commercial && currentLicenseInfo.license && (
          <DatasetLicenseNotice datasetName={dataset} license={currentLicenseInfo.license} />
        )}

        {/* Search bar */}
        <div className="px-3 pt-2.5 pb-1">
          <div className="relative border border-gray-200 rounded-lg px-2 py-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-3 w-3 text-muted-foreground" />
            <Tooltip>
              <TooltipTrigger asChild>
                <Input
                  placeholder="Search audio files..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9 h-6 text-xs bg-transparent border-0 focus:ring-0 rounded-md"
                />
              </TooltipTrigger>
              <TooltipContent>
                <p>Search by filename or any metadata field</p>
              </TooltipContent>
            </Tooltip>
          </div>
        </div>
      </div>
      
      <div className="flex-1 overflow-hidden px-3 pb-3">
        <Card className="h-full rounded-lg">
          <CardContent className="p-0 h-full">
            <AudioDataTable 
              selectedRow={selectedRow}
              onRowSelect={handleRowSelect}
              searchQuery={searchQuery}
              apiData={apiData}
              model={model ?? ""}
              dataset={dataset}
              datasetMetadata={datasetMetadata}
              uploadedFiles={uploadedFiles}
              onFilePlay={handleFilePlay}
              predictionMap={predictionMap}
              inferenceStatus={inferenceStatus}
              onVisibleRowIdsChange={handleVisibleRowIdsChange}
              onRegenerateRow={handleRegenerateRow}
            />
          </CardContent>
        </Card>
      </div>
      
      {/* Upload overlay */}
      <AudioUploader onUploadSuccess={onUploadSuccess} model={model} />
    </div>
    </TooltipProvider>
  );
};