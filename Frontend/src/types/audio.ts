export type ModelType = 'ASR' | 'SER' | 'ADD';

export interface AudioMetadata {
  fileName: string;
    duration: number;
      sampleRate: number;
        channels: number;
        }

        export interface AudioState {
          currentAudioFile: File | null;
            audioMetadata: AudioMetadata | null;
              isProcessing: boolean;
                activeModelSelection: ModelType;
                  saliencyMatrixData: number[][] | null;
                    taskStatus: 'idle' | 'loading' | 'analyzing' | 'complete' | 'error';
                    }

                    export interface AudioContextType extends AudioState {
                      setAudioFile: (file: File) => void;
                        setModelSelection: (model: ModelType) => void;
                          triggerAnalysis: () => void;
                            resetWorkspace: () => void;
                            }
                            