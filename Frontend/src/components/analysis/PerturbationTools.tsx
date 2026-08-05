"use client"

import React, { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Checkbox } from "@/components/ui/checkbox"
import { RangeSlider } from "@/components/ui/range-slider"
import { Volume2, Scissors, Plus, Play, Zap, XCircle } from "lucide-react"
import { WaveformViewer } from "../audio/WaveformViewer"
import { API_BASE } from '@/lib/api'
import { useTaskStatus } from '@/hooks/useTaskStatus'
import { GlobalTaskProgress } from '../layout/GlobalTaskProgress'

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

interface PerturbationToolsProps {
  selectedFile: UploadedFile | null;
  onPerturbationComplete?: (result: PerturbationResult) => void;
  onPredictionRefresh?: (file: UploadedFile, prediction: string) => void;
  model?: string;
  dataset?: string;
  originalDataset?: string;
}

const getAudioUrl = (selectedFile: UploadedFile, dataset?: string, originalDataset?: string): string => {
  const isUploadedFile = selectedFile.file_path && (
    selectedFile.file_path.includes('uploads/') || 
    selectedFile.file_path.includes('uploads\\') ||
    selectedFile.message === "Perturbed file" ||
    selectedFile.message === "File uploaded successfully" ||
    selectedFile.message === "File uploaded and processed successfully"
  ) && selectedFile.message !== "Selected from dataset";
  
  if (isUploadedFile) {
    return `${API_BASE}/upload/file/${selectedFile.file_id}`;
  } else {
    const datasetToUse = originalDataset && originalDataset !== "custom" ? originalDataset : dataset;
    if (datasetToUse && datasetToUse !== "custom") {
      const filename = encodeURIComponent(selectedFile.filename);
      return `${API_BASE}/${encodeURIComponent(datasetToUse)}/file/${filename}`;
    } else {
      return `${API_BASE}/upload/file/${selectedFile.file_id}`;
    }
  }
};

const getPerturbedAudioUrl = (perturbedFilePath: string): string => {
  const filename = perturbedFilePath.split('/').pop() || perturbedFilePath.split('\\').pop();
  return `${API_BASE}/upload/file/${filename}`;
};

export const PerturbationTools: React.FC<PerturbationToolsProps> = ({
  selectedFile,
  onPerturbationComplete,
  onPredictionRefresh,
  model,
  dataset,
  originalDataset,
}) => {
  const [noiseLevel, setNoiseLevel] = useState([10])
  const [maskRange, setMaskRange] = useState<[number, number]>([20, 40])
  const [pitchShift, setPitchShift] = useState([2])
  const [timeStretch, setTimeStretch] = useState([110])
  
  const [selectedPerturbations, setSelectedPerturbations] = useState({
    noise: false,
    timeMasking: false,
    pitchShift: false,
    timeStretch: false,
  })
  
  const [error, setError] = useState<string | null>(null)
  const [perturbationResult, setPerturbationResult] = useState<PerturbationResult | null>(null)
  
  // RQ Task IDs
  const [mutationTaskId, setMutationTaskId] = useState<string | null>(null)
  const [inferenceTaskId, setInferenceTaskId] = useState<string | null>(null)

  // Track mutation job state
  const { state: mutationState, result: mutationResult } = useTaskStatus(mutationTaskId)
  // Track inference job state
  const { state: inferenceState, result: inferenceResult } = useTaskStatus(inferenceTaskId)

  useEffect(() => {
    setPerturbationResult(null);
    setError(null);
    setMutationTaskId(null);
    setInferenceTaskId(null);
  }, [selectedFile]);

  const handlePerturbationToggle = (perturbationType: keyof typeof selectedPerturbations) => {
    setSelectedPerturbations(prev => ({ ...prev, [perturbationType]: !prev[perturbationType] }));
  }

  // Effect: When mutation job succeeds, trigger inference
  useEffect(() => {
    if (mutationState === 'SUCCESS' && mutationResult) {
      // Adapt result to expected shape
      const perturbedData = mutationResult as any;
      setPerturbationResult(perturbedData);
      if (onPerturbationComplete) onPerturbationComplete(perturbedData);

      // Start inference on the perturbed file
      const runInference = async () => {
        try {
          const response = await fetch(`${API_BASE}/api/inference/multitask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              audio_ref: perturbedData.perturbed_file,
              tasks: model?.includes('whisper') ? ['asr'] : ['ser']
            })
          });
          if (response.ok) {
            const data = await response.json();
            setInferenceTaskId(data.job_id);
          } else {
            setError("Failed to enqueue inference job for perturbed audio.");
          }
        } catch (err) {
          setError("Error triggering inference job.");
        }
      };
      runInference();
    } else if (mutationState === 'FAILURE') {
      setError("Perturbation task failed in worker.");
    }
  }, [mutationState, mutationResult]);

  // Effect: When inference job succeeds, notify parent
  useEffect(() => {
    if (inferenceState === 'SUCCESS' && inferenceResult && perturbationResult) {
      const perturbedFile: UploadedFile = {
        file_id: perturbationResult.filename,
        filename: perturbationResult.filename,
        file_path: perturbationResult.perturbed_file,
        message: "Perturbed file",
        duration: perturbationResult.duration_ms / 1000,
        sample_rate: perturbationResult.sample_rate
      };
      if (onPredictionRefresh) {
        onPredictionRefresh(perturbedFile, JSON.stringify(inferenceResult));
      }
      setInferenceTaskId(null);
      setMutationTaskId(null);
    } else if (inferenceState === 'FAILURE') {
      setError("Inference task failed in worker.");
      setInferenceTaskId(null);
      setMutationTaskId(null);
    }
  }, [inferenceState, inferenceResult, perturbationResult]);

  const handleAddPerturbations = async () => {
    if (!selectedFile) { setError("No file selected"); return; }
    if (!Object.values(selectedPerturbations).some(Boolean)) { setError("Please select at least one perturbation type"); return; }
    
    setError(null);

    try {
      const perturbations = [];
      if (selectedPerturbations.noise) perturbations.push({ type: "noise", params: { noise_level: noiseLevel[0] / 100.0 } });
      if (selectedPerturbations.timeMasking) perturbations.push({ type: "time_masking", params: { mask_start_percent: maskRange[0], mask_end_percent: maskRange[1] } });
      if (selectedPerturbations.pitchShift) perturbations.push({ type: "pitch_shift", params: { pitch_shift_semitones: pitchShift[0] } });
      if (selectedPerturbations.timeStretch) perturbations.push({ type: "time_stretch", params: { stretch_factor: timeStretch[0] / 100.0 } });

      const isUploadedFile = selectedFile.file_path && (
        selectedFile.file_path.includes('uploads/') || 
        selectedFile.file_path.startsWith('uploads/') ||
        selectedFile.message === "Perturbed file" ||
        selectedFile.message === "File uploaded successfully" ||
        selectedFile.message === "File uploaded and processed successfully"
      ) && selectedFile.message !== "Selected from dataset";

      // Enqueue mutation job via RQ
      const response = await fetch(`${API_BASE}/api/inference/mutation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio_ref: isUploadedFile ? selectedFile.file_path : selectedFile.filename,
          mutation: { 
            perturbations, 
            is_uploaded: isUploadedFile,
            dataset: originalDataset || dataset 
          }
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const result = await response.json();
      setMutationTaskId(result.job_id); // Start tracking via WebSocket
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error occurred";
      setError(errorMessage);
    }
  };

  const isProcessing = mutationTaskId !== null || inferenceTaskId !== null;

  return (
    <div className="space-y-4">
      {error && (
        <Card className="border-destructive/20 bg-destructive/5">
          <CardContent className="pt-4">
            <div className="flex items-center gap-2 text-destructive text-xs">
              <XCircle className="h-4 w-4" />
              {error}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Show dynamic progress bar when processing */}
      {isProcessing && (
        <GlobalTaskProgress 
          taskId={inferenceTaskId || mutationTaskId} 
          onComplete={() => {}} 
        />
      )}

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Perturbation Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Noise */}
          <div className="space-y-3 p-3 border rounded-lg">
            <div className="flex items-center space-x-2">
              <Checkbox id="noise-checkbox" checked={selectedPerturbations.noise} onCheckedChange={() => handlePerturbationToggle('noise')} className="border-blue-400 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600" />
              <Volume2 className="h-4 w-4 text-blue-600" />
              <label htmlFor="noise-checkbox" className="text-sm font-medium">Add Gaussian Noise</label>
            </div>
            {selectedPerturbations.noise && (
              <div className="space-y-2 pl-6">
                <div className="flex items-center justify-between">
                  <span className="text-xs">Noise Level</span>
                  <Badge variant="outline" className="text-xs border-blue-300 text-blue-700">{noiseLevel[0]}%</Badge>
                </div>
                <Slider value={noiseLevel} onValueChange={setNoiseLevel} max={50} step={1} className="w-full [&_[role=slider]]:border-blue-500 [&_[role=slider]]:bg-blue-600" />
              </div>
            )}
          </div>

          {/* Time Masking */}
          <div className="space-y-3 p-3 border rounded-lg">
            <div className="flex items-center space-x-2">
              <Checkbox id="masking-checkbox" checked={selectedPerturbations.timeMasking} onCheckedChange={() => handlePerturbationToggle('timeMasking')} className="border-blue-400 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600" />
              <Scissors className="h-4 w-4 text-blue-600" />
              <label htmlFor="masking-checkbox" className="text-sm font-medium">Apply Time Masking</label>
            </div>
            {selectedPerturbations.timeMasking && (
              <div className="space-y-3 pl-6">
                <div className="flex items-center justify-between">
                  <span className="text-xs">Mask Region</span>
                  <Badge variant="outline" className="text-xs border-blue-300 text-blue-700">{maskRange[0]}% - {maskRange[1]}%</Badge>
                </div>
                <RangeSlider value={maskRange} onValueChange={setMaskRange} min={0} max={100} step={1} className="w-full" formatLabel={(value) => `${value}%`} />
              </div>
            )}
          </div>

          {/* Pitch Shift */}
          <div className="space-y-3 p-3 border rounded-lg">
            <div className="flex items-center space-x-2">
              <Checkbox id="pitch-checkbox" checked={selectedPerturbations.pitchShift} onCheckedChange={() => handlePerturbationToggle('pitchShift')} className="border-blue-400 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600" />
              <Plus className="h-4 w-4 text-blue-600" />
              <label htmlFor="pitch-checkbox" className="text-sm font-medium">Apply Pitch Shift</label>
            </div>
            {selectedPerturbations.pitchShift && (
              <div className="space-y-2 pl-6">
                <div className="flex items-center justify-between">
                  <span className="text-xs">Pitch Shift</span>
                  <Badge variant="outline" className="text-xs border-blue-300 text-blue-700">{pitchShift[0] > 0 ? "+" : ""}{pitchShift[0]} semitones</Badge>
                </div>
                <Slider value={pitchShift} onValueChange={setPitchShift} min={-6} max={6} step={1} className="w-full [&_[role=slider]]:border-blue-500 [&_[role=slider]]:bg-blue-600" />
              </div>
            )}
          </div>

          {/* Time Stretch - Hidden for Whisper */}
          {!model?.includes('whisper') && (
            <div className="space-y-3 p-3 border rounded-lg">
              <div className="flex items-center space-x-2">
                <Checkbox id="time-checkbox" checked={selectedPerturbations.timeStretch} onCheckedChange={() => handlePerturbationToggle('timeStretch')} className="border-blue-400 data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600" />
                <Play className="h-4 w-4 text-blue-600" />
                <label htmlFor="time-checkbox" className="text-sm font-medium">Apply Time Stretch</label>
              </div>
              {selectedPerturbations.timeStretch && (
                <div className="space-y-2 pl-6">
                  <div className="flex items-center justify-between">
                    <span className="text-xs">Time Stretch</span>
                    <Badge variant="outline" className="text-xs border-blue-300 text-blue-700">{timeStretch[0]}%</Badge>
                  </div>
                  <Slider value={timeStretch} onValueChange={setTimeStretch} min={50} max={200} step={5} className="w-full [&_[role=slider]]:border-blue-500 [&_[role=slider]]:bg-blue-600" />
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="pt-4">
          <Button onClick={handleAddPerturbations} disabled={isProcessing || !selectedFile || !Object.values(selectedPerturbations).some(Boolean)} className="w-full h-10 bg-blue-600 hover:bg-blue-700 text-white font-medium shadow-md" size="lg">
            <Zap className="h-4 w-4 mr-2" />
            {isProcessing ? "Processing..." : "Apply Perturbations"}
          </Button>
        </CardContent>
      </Card>

      {(selectedFile || perturbationResult) && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Audio Waveforms</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {selectedFile && (
              <div className="space-y-2">
                <div className="text-xs font-medium flex items-center gap-2">Original Audio <Badge variant="outline" className="text-[10px]">O</Badge></div>
                <WaveformViewer audioUrl={getAudioUrl(selectedFile, dataset, originalDataset)} />
              </div>
            )}
            {perturbationResult && perturbationResult.success && (
              <div className="space-y-2">
                <div className="text-xs font-medium flex items-center gap-2">Perturbed Audio <Badge variant="secondary" className="text-[10px]">P</Badge></div>
                <WaveformViewer audioUrl={getPerturbedAudioUrl(perturbationResult.perturbed_file)} />
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
