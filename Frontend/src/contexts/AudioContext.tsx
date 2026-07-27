import { createContext, useContext, useState, type ReactNode } from 'react';
import type { AudioContextType, ModelType, AudioMetadata } from '../types/audio';

const AudioContext = createContext<AudioContextType | undefined>(undefined);

export const AudioProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [currentAudioFile, setCurrentAudioFile] = useState<File | null>(null);
    const [audioMetadata, setAudioMetadata] = useState<AudioMetadata | null>(null);
      const [isProcessing, setIsProcessing] = useState<boolean>(false);
        const [activeModelSelection, setActiveModelSelection] = useState<ModelType>('ASR');
          const [saliencyMatrixData, setSaliencyMatrixData] = useState<number[][] | null>(null);
            const [taskStatus, setTaskStatus] = useState<'idle' | 'loading' | 'analyzing' | 'complete' | 'error'>('idle');

              const setAudioFile = (file: File) => {
                  setCurrentAudioFile(file);
                      setAudioMetadata({
                            fileName: file.name,
                                  duration: 12.4, // placeholder sample duration
                                        sampleRate: 16000,
                                              channels: 1,
                                                  });
                                                      setTaskStatus('idle');
                                                        };

                                                          const setModelSelection = (model: ModelType) => {
                                                              setActiveModelSelection(model);
                                                                };

                                                                  const triggerAnalysis = () => {
                                                                      if (!currentAudioFile) return;
                                                                          setIsProcessing(true);
                                                                              setTaskStatus('analyzing');
                                                                                  
                                                                                      // Simulate API diagnostic run
                                                                                          setTimeout(() => {
                                                                                                setSaliencyMatrixData([[0.1, 0.4, 0.9], [0.2, 0.7, 0.3]]);
                                                                                                      setIsProcessing(false);
                                                                                                            setTaskStatus('complete');
                                                                                                                }, 1500);
                                                                                                                  };

                                                                                                                    const resetWorkspace = () => {
                                                                                                                        setCurrentAudioFile(null);
                                                                                                                            setAudioMetadata(null);
                                                                                                                                setSaliencyMatrixData(null);
                                                                                                                                    setTaskStatus('idle');
                                                                                                                                      };

                                                                                                                                        return (
                                                                                                                                            <AudioContext.Provider
                                                                                                                                                  value={{
                                                                                                                                                          currentAudioFile,
                                                                                                                                                                  audioMetadata,
                                                                                                                                                                          isProcessing,
                                                                                                                                                                                  activeModelSelection,
                                                                                                                                                                                          saliencyMatrixData,
                                                                                                                                                                                                  taskStatus,
                                                                                                                                                                                                          setAudioFile,
                                                                                                                                                                                                                  setModelSelection,
                                                                                                                                                                                                                          triggerAnalysis,
                                                                                                                                                                                                                                  resetWorkspace,
                                                                                                                                                                                                                                        }}
                                                                                                                                                                                                                                            >
                                                                                                                                                                                                                                                  {children}
                                                                                                                                                                                                                                                      </AudioContext.Provider>
                                                                                                                                                                                                                                                        );
                                                                                                                                                                                                                                                        };

                                                                                                                                                                                                                                                        // eslint-disable-next-line react-refresh/only-export-components
                                                                                                                                                                                                                                                        export const useAudio = () => {
                                                                                                                                                                                                                                                          const context = useContext(AudioContext);
                                                                                                                                                                                                                                                            if (!context) throw new Error('useAudio must be used within an AudioProvider');
                                                                                                                                                                                                                                                              return context;
                                                                                                                                                                                                                                                              };
                                                                                                                                                                                                                                                              