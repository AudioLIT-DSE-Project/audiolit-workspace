import React from 'react';
import { useAudio } from '../../contexts/AudioContext';
import type { ModelType } from '../../types/audio';
import { Layers, Play, RefreshCw } from 'lucide-react';

export const Sidebar: React.FC = () => {
  const { activeModelSelection, setModelSelection, triggerAnalysis, isProcessing, currentAudioFile } = useAudio();

    const models: { id: ModelType; label: string }[] = [
        { id: 'ASR', label: 'ASR Transcript Diagnostics' },
            { id: 'SER', label: 'Emotion Recognition (SER)' },
                { id: 'ADD', label: 'Audio Deepfake Detection' },
                  ];

                    return (
                        <aside className="w-64 border-r border-border bg-surface p-4 flex flex-col justify-between h-full">
                              <div className="space-y-6">
                                      <div>
                                                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                                                            <Layers className="h-4 w-4" /> Diagnostic Model Target
                                                                      </h3>
                                                                                <div className="space-y-2">
                                                                                            {models.map((m) => (
                                                                                                          <button
                                                                                                                          key={m.id}
                                                                                                                                          onClick={() => setModelSelection(m.id)}
                                                                                                                                                          className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                                                                                                                                                                            activeModelSelection === m.id
                                                                                                                                                                                                ? 'bg-primary text-white'
                                                                                                                                                                                                                    : 'text-gray-300 hover:bg-border/50'
                                                                                                                                                                                                                                    }`}
                                                                                                                                                                                                                                                  >
                                                                                                                                                                                                                                                                  {m.label}
                                                                                                                                                                                                                                                                                </button>
                                                                                                                                                                                                                                                                                            ))}
                                                                                                                                                                                                                                                                                                      </div>
                                                                                                                                                                                                                                                                                                              </div>
                                                                                                                                                                                                                                                                                                                    </div>

                                                                                                                                                                                                                                                                                                                          <button
                                                                                                                                                                                                                                                                                                                                  onClick={triggerAnalysis}
                                                                                                                                                                                                                                                                                                                                          disabled={!currentAudioFile || isProcessing}
                                                                                                                                                                                                                                                                                                                                                  className="w-full py-2 px-4 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 text-white rounded-lg font-medium text-sm flex items-center justify-center gap-2 transition-colors"
                                                                                                                                                                                                                                                                                                                                                        >
                                                                                                                                                                                                                                                                                                                                                                {isProcessing ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                                                                                                                                                                                                                                                                                                                                                                        Run Diagnostics
                                                                                                                                                                                                                                                                                                                                                                              </button>
                                                                                                                                                                                                                                                                                                                                                                                  </aside>
                                                                                                                                                                                                                                                                                                                                                                                    );
                                                                                                                                                                                                                                                                                                                                                                                    };
                                                                                                                                                                                                                                                                                                                                                                                    