import React, { useState, useRef, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import {
  Mic,
  Square,
  Play,
  Pause,
  Upload,
  Radio,
  Trash2,
  Loader2,
  CheckCircle2,
  AlertCircle,
  FileAudio,
  RotateCcw,
  Download,
  Volume2,
  VolumeX,
} from "lucide-react";
import { API_BASE } from "@/lib/api";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
}

interface SelectedTasks {
  asr: boolean;
  ser: boolean;
  add: boolean;
}

interface AudioUploadRecorderModalProps {
  isOpen: boolean;
  onClose: () => void;
  onFileUploaded: (file: UploadedFile) => void;
  defaultTasks?: SelectedTasks;
}

export const AudioUploadRecorderModal: React.FC<AudioUploadRecorderModalProps> = ({
  isOpen,
  onClose,
  onFileUploaded,
  defaultTasks = { asr: true, ser: true, add: true },
}) => {
  const [activeTab, setActiveTab] = useState<"upload" | "record">("upload");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);

  // Live Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioLevel, setAudioLevel] = useState(0);
  const [micError, setMicError] = useState<string | null>(null);

  // Recorded Audio Player Preview state
  const [isPlayingPreview, setIsPlayingPreview] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [totalDuration, setTotalDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);

  // Form submission state
  const [tasks, setTasks] = useState<SelectedTasks>(defaultTasks);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // MediaRecorder & Web Audio refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    return () => {
      stopRecordingCleanup();
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, []);

  const stopRecordingCleanup = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      try {
        mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
        mediaRecorderRef.current.stop();
      } catch (err) {
        console.error("Error stopping recorder tracks:", err);
      }
    }
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
      previewAudioRef.current = null;
    }
  };

  // Drag and Drop handlers
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("audio/") || /\.(wav|mp3|flac|m4a|ogg|aac|webm)$/i.test(file.name)) {
        setSelectedFile(file);
        setErrorMessage(null);
      } else {
        setErrorMessage("Please select a valid audio file (.wav, .mp3, .flac, .m4a, .ogg).");
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setErrorMessage(null);
    }
  };

  // Start Live Recording
  const startRecording = async () => {
    setMicError(null);
    setErrorMessage(null);
    setAudioBlob(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    audioChunksRef.current = [];
    setRecordingTime(0);
    setCurrentTime(0);
    setTotalDuration(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      // AudioContext for VU volume meter
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new AudioCtx();
      audioContextRef.current = audioCtx;
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 64;
      analyserRef.current = analyser;

      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);

      const updateMeter = () => {
        if (!analyserRef.current) return;
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((acc, val) => acc + val, 0) / dataArray.length;
        setAudioLevel(Math.min(100, Math.round((average / 128) * 100)));
        animFrameRef.current = requestAnimationFrame(updateMeter);
      };
      updateMeter();

      // Configure MediaRecorder
      let mimeType = "audio/webm";
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        if (MediaRecorder.isTypeSupported("audio/mp4")) mimeType = "audio/mp4";
        else if (MediaRecorder.isTypeSupported("audio/ogg")) mimeType = "audio/ogg";
        else mimeType = "";
      }

      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = () => {
        const finalBlob = new Blob(audioChunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
        setAudioBlob(finalBlob);
        const url = URL.createObjectURL(finalBlob);
        setAudioUrl(url);
        stream.getTracks().forEach((track) => track.stop());
      };

      recorder.start(200);
      setIsRecording(true);
      setIsPaused(false);

      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => {
          if (prev >= 900) { // 15 min cap per SR1
            stopRecording();
            return prev;
          }
          return prev + 1;
        });
      }, 1000);
    } catch (err) {
      console.error("Microphone access error:", err);
      setMicError("Microphone access denied or unavailable. Please check browser permissions.");
    }
  };

  const pauseRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      if (isPaused) {
        mediaRecorderRef.current.resume();
        setIsPaused(false);
        timerRef.current = setInterval(() => {
          setRecordingTime((prev) => prev + 1);
        }, 1000);
      } else {
        mediaRecorderRef.current.pause();
        setIsPaused(true);
        if (timerRef.current) clearInterval(timerRef.current);
      }
    }
  };

  const stopRecording = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close().catch(() => {});
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    setIsPaused(false);
    setAudioLevel(0);
  };

  const redoRecording = () => {
    discardRecording();
    startRecording();
  };

  const discardRecording = () => {
    stopRecordingCleanup();
    setIsRecording(false);
    setIsPaused(false);
    setRecordingTime(0);
    setAudioBlob(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    setAudioLevel(0);
    setIsPlayingPreview(false);
    setCurrentTime(0);
    setTotalDuration(0);
  };

  // Audio Preview Player logic
  useEffect(() => {
    if (!audioUrl) return;

    const audio = new Audio(audioUrl);
    previewAudioRef.current = audio;

    audio.onloadedmetadata = () => {
      setTotalDuration(audio.duration || recordingTime);
    };

    audio.ontimeupdate = () => {
      setCurrentTime(audio.currentTime);
    };

    audio.onended = () => {
      setIsPlayingPreview(false);
      setCurrentTime(0);
    };

    return () => {
      audio.pause();
      previewAudioRef.current = null;
    };
  }, [audioUrl, recordingTime]);

  const togglePreviewPlayback = () => {
    if (!previewAudioRef.current) return;
    if (isPlayingPreview) {
      previewAudioRef.current.pause();
      setIsPlayingPreview(false);
    } else {
      previewAudioRef.current.play().then(() => setIsPlayingPreview(true)).catch((err) => {
        console.error("Playback error:", err);
      });
    }
  };

  const handleSeek = (newValues: number[]) => {
    if (previewAudioRef.current && newValues.length > 0) {
      const seekTime = newValues[0];
      previewAudioRef.current.currentTime = seekTime;
      setCurrentTime(seekTime);
    }
  };

  const handleVolumeChange = (newValues: number[]) => {
    if (newValues.length > 0) {
      const vol = newValues[0];
      setVolume(vol);
      if (previewAudioRef.current) {
        previewAudioRef.current.volume = vol;
        setIsMuted(vol === 0);
      }
    }
  };

  const toggleMute = () => {
    if (previewAudioRef.current) {
      if (isMuted) {
        previewAudioRef.current.volume = volume || 1;
        setIsMuted(false);
      } else {
        previewAudioRef.current.volume = 0;
        setIsMuted(true);
      }
    }
  };

  const downloadRecording = () => {
    if (!audioUrl || !audioBlob) return;
    const a = document.createElement("a");
    a.href = audioUrl;
    a.download = `voice_recording_${Date.now()}.webm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  // Submit audio payload (File or Recorded Live Blob) to POST /upload
  const handleSubmit = async () => {
    const fileToUpload =
      activeTab === "upload"
        ? selectedFile
        : audioBlob
        ? new File([audioBlob], `live_recording_${Date.now()}.webm`, {
            type: audioBlob.type || "audio/webm",
          })
        : null;

    if (!fileToUpload) {
      setErrorMessage("Please select a file or record a voice input first.");
      return;
    }

    setIsSubmitting(true);
    setUploadProgress("Uploading and standardizing audio payload...");
    setErrorMessage(null);

    const formData = new FormData();
    formData.append("file", fileToUpload);
    formData.append("model", "whisper-base");

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
        credentials: "include",
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Upload failed with status ${response.status}`);
      }

      const data: UploadedFile = await response.json();
      setUploadProgress("Upload complete!");
      onFileUploaded(data);
      handleModalClose();
    } catch (err: unknown) {
      console.error("Upload error:", err);
      const msg = err instanceof Error ? err.message : "An unexpected error occurred during upload.";
      setErrorMessage(msg);
    } finally {
      setIsSubmitting(false);
      setUploadProgress(null);
    }
  };

  const handleModalClose = () => {
    stopRecordingCleanup();
    setSelectedFile(null);
    setAudioBlob(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    setIsRecording(false);
    setIsPaused(false);
    setRecordingTime(0);
    setIsPlayingPreview(false);
    setCurrentTime(0);
    setTotalDuration(0);
    setErrorMessage(null);
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleModalClose}>
      <DialogContent className="sm:max-w-[580px] bg-card border-border shadow-xl">
        <DialogHeader>
          <DialogTitle className="text-lg font-semibold flex items-center gap-2">
            <Upload className="h-5 w-5 text-primary" />
            Audio Data Input & Live Voice Recorder
          </DialogTitle>
          <DialogDescription className="text-xs text-muted-foreground">
            Upload custom audio files or capture your live voice input to evaluate against ASR, SER, and Deepfake models.
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={(val) => setActiveTab(val as "upload" | "record")} className="w-full mt-1">
          <TabsList className="grid w-full grid-cols-2 bg-muted/60 p-1">
            <TabsTrigger value="upload" className="text-xs flex items-center gap-1.5">
              <FileAudio className="h-3.5 w-3.5" />
              File Upload
            </TabsTrigger>
            <TabsTrigger value="record" className="text-xs flex items-center gap-1.5">
              <Radio className="h-3.5 w-3.5 text-rose-500 animate-pulse" />
              Live Voice Record
            </TabsTrigger>
          </TabsList>

          {/* TAB 1: FILE UPLOAD */}
          <TabsContent value="upload" className="mt-3 space-y-3">
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
                dragActive
                  ? "border-primary bg-primary/5"
                  : selectedFile
                  ? "border-emerald-500/50 bg-emerald-500/5"
                  : "border-border hover:border-primary/50 bg-muted/20"
              }`}
              onClick={() => document.getElementById("audio-file-input")?.click()}
            >
              <input
                id="audio-file-input"
                type="file"
                accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg,.webm"
                className="hidden"
                onChange={handleFileChange}
              />
              <FileAudio className="h-10 w-10 mx-auto mb-2 text-muted-foreground" />
              {selectedFile ? (
                <div className="space-y-1">
                  <p className="text-sm font-medium text-emerald-600 dark:text-emerald-400 flex items-center justify-center gap-1.5">
                    <CheckCircle2 className="h-4 w-4" />
                    {selectedFile.name}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • {selectedFile.type || "audio payload"}
                  </p>
                </div>
              ) : (
                <div>
                  <p className="text-sm font-medium text-foreground">
                    Drag and drop your audio file here, or <span className="text-primary underline">browse</span>
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Supports WAV, MP3, FLAC, M4A, OGG up to 100MB (Max 15 min per SR1)
                  </p>
                </div>
              )}
            </div>
          </TabsContent>

          {/* TAB 2: LIVE VOICE RECORDING */}
          <TabsContent value="record" className="mt-3 space-y-3">
            <div className="border border-border rounded-lg p-4 bg-muted/20 space-y-3">
              {micError ? (
                <div className="p-2.5 bg-destructive/10 text-destructive text-xs rounded-md flex items-center gap-2">
                  <AlertCircle className="h-4 w-4 flex-shrink-0" />
                  <span>{micError}</span>
                </div>
              ) : null}

              {/* Header & Status Indicator */}
              <div className="flex items-center justify-between">
                <Badge variant={isRecording ? "destructive" : audioBlob ? "default" : "outline"} className="text-xs">
                  {isRecording ? (isPaused ? "PAUSED" : "RECORDING LIVE") : audioBlob ? "RECORDED PREVIEW" : "READY TO RECORD"}
                </Badge>
                <span className="text-base font-mono font-bold tracking-wider text-foreground">
                  {formatTime(isRecording ? recordingTime : currentTime)} / {formatTime(totalDuration || 900)}
                </span>
              </div>

              {/* VU Audio Meter during Recording */}
              {isRecording ? (
                <div className="space-y-1">
                  <div className="flex justify-between text-[10px] text-muted-foreground">
                    <span>Input Volume Level</span>
                    <span>{audioLevel}%</span>
                  </div>
                  <div className="h-3 w-full bg-muted rounded-full overflow-hidden flex items-center px-1">
                    <div
                      className={`h-1.5 rounded-full transition-all duration-75 ${
                        audioLevel > 80 ? "bg-rose-500" : audioLevel > 40 ? "bg-amber-500" : "bg-primary"
                      }`}
                      style={{ width: `${!isPaused ? audioLevel : 0}%` }}
                    />
                  </div>
                </div>
              ) : null}

              {/* Recording Controls */}
              {!isRecording && !audioBlob ? (
                <div className="text-center py-4 space-y-3">
                  <Button onClick={startRecording} size="lg" className="bg-rose-600 hover:bg-rose-700 text-white font-medium rounded-full px-6">
                    <Mic className="h-5 w-5 mr-2" />
                    Start Live Recording
                  </Button>
                  <p className="text-xs text-muted-foreground">Click start and speak clearly into your microphone.</p>
                </div>
              ) : null}

              {isRecording ? (
                <div className="flex items-center justify-center gap-3 pt-1">
                  <Button onClick={pauseRecording} variant="outline" size="sm" className="text-xs">
                    {isPaused ? <Play className="h-3.5 w-3.5 mr-1" /> : <Pause className="h-3.5 w-3.5 mr-1" />}
                    {isPaused ? "Resume" : "Pause"}
                  </Button>
                  <Button onClick={stopRecording} variant="destructive" size="sm" className="text-xs">
                    <Square className="h-3.5 w-3.5 mr-1" />
                    Stop Recording
                  </Button>
                </div>
              ) : null}

              {/* Recorded Audio Preview & Scrubber Controls */}
              {audioBlob && !isRecording ? (
                <div className="space-y-3 bg-card p-3 rounded border border-border">
                  {/* Timeline Scrubber */}
                  <div className="space-y-1">
                    <Slider
                      value={[currentTime]}
                      max={totalDuration || 1}
                      step={0.1}
                      onValueChange={handleSeek}
                      className="cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-muted-foreground">
                      <span>{formatTime(currentTime)}</span>
                      <span>{formatTime(totalDuration)}</span>
                    </div>
                  </div>

                  {/* Playback Action Toolbar */}
                  <div className="flex items-center justify-between gap-2 pt-1 border-t border-border/50">
                    <div className="flex items-center gap-2">
                      <Button onClick={togglePreviewPlayback} variant="default" size="sm" className="h-8 text-xs">
                        {isPlayingPreview ? <Pause className="h-3.5 w-3.5 mr-1" /> : <Play className="h-3.5 w-3.5 mr-1" />}
                        {isPlayingPreview ? "Pause" : "Play Preview"}
                      </Button>

                      {/* Volume Slider */}
                      <div className="flex items-center gap-1.5 ml-2">
                        <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={toggleMute}>
                          {isMuted ? <VolumeX className="h-3.5 w-3.5 text-muted-foreground" /> : <Volume2 className="h-3.5 w-3.5 text-foreground" />}
                        </Button>
                        <Slider
                          value={[isMuted ? 0 : volume]}
                          max={1}
                          step={0.05}
                          onValueChange={handleVolumeChange}
                          className="w-16 cursor-pointer"
                        />
                      </div>
                    </div>

                    <div className="flex items-center gap-1.5">
                      <Button onClick={downloadRecording} variant="outline" size="sm" className="h-8 text-xs" title="Download Voice Input">
                        <Download className="h-3.5 w-3.5 mr-1" />
                        Download
                      </Button>
                      <Button onClick={redoRecording} variant="outline" size="sm" className="h-8 text-xs text-amber-600 dark:text-amber-400 hover:bg-amber-500/10">
                        <RotateCcw className="h-3.5 w-3.5 mr-1" />
                        Redo / Re-record
                      </Button>
                      <Button onClick={discardRecording} variant="ghost" size="sm" className="h-8 text-xs text-destructive hover:bg-destructive/10">
                        <Trash2 className="h-3.5 w-3.5 mr-1" />
                        Clear
                      </Button>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </TabsContent>
        </Tabs>

        {/* Task Selection Checkboxes */}
        <div className="space-y-2 pt-2 border-t border-border">
          <label className="text-xs font-medium text-foreground block">
            Select Analysis Tasks to Trigger on Upload:
          </label>
          <div className="flex items-center gap-4 text-xs">
            <label className="flex items-center gap-1.5 cursor-pointer">
              <Checkbox
                checked={tasks.asr}
                onCheckedChange={(c) => setTasks({ ...tasks, asr: !!c })}
                className="h-3.5 w-3.5"
              />
              <span>ASR (Transcription)</span>
            </label>
            <label className="flex items-center gap-1.5 cursor-pointer">
              <Checkbox
                checked={tasks.ser}
                onCheckedChange={(c) => setTasks({ ...tasks, ser: !!c })}
                className="h-3.5 w-3.5"
              />
              <span>SER (Emotion)</span>
            </label>
            <label className="flex items-center gap-1.5 cursor-pointer">
              <Checkbox
                checked={tasks.add}
                onCheckedChange={(c) => setTasks({ ...tasks, add: !!c })}
                className="h-3.5 w-3.5"
              />
              <span>ADD (Deepfake)</span>
            </label>
          </div>
        </div>

        {/* Error / Progress feedback */}
        {errorMessage ? (
          <p className="text-xs text-destructive bg-destructive/10 p-2.5 rounded flex items-center gap-1.5">
            <AlertCircle className="h-3.5 w-3.5 flex-shrink-0" />
            {errorMessage}
          </p>
        ) : null}

        {uploadProgress ? (
          <p className="text-xs text-primary bg-primary/10 p-2.5 rounded flex items-center gap-1.5">
            <Loader2 className="h-3.5 w-3.5 animate-spin flex-shrink-0" />
            {uploadProgress}
          </p>
        ) : null}

        {/* Footer Submit / Cancel Buttons */}
        <div className="flex justify-end gap-2 pt-2 border-t border-border">
          <Button variant="outline" size="sm" onClick={handleModalClose} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button
            size="sm"
            onClick={handleSubmit}
            disabled={isSubmitting || (activeTab === "upload" ? !selectedFile : !audioBlob)}
            className="bg-primary hover:bg-primary/90 text-primary-foreground font-medium"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                Uploading...
              </>
            ) : (
              <>
                <Upload className="h-3.5 w-3.5 mr-1.5" />
                Upload & Process Payload
              </>
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};
