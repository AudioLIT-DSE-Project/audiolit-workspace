import React from 'react';
import { Activity, Cpu, Sliders } from 'lucide-react';

export const Navbar: React.FC = () => {
  return (
      <header className="h-14 border-b border-border bg-surface px-4 flex items-center justify-between">
            <div className="flex items-center space-x-3">
                    <Activity className="h-6 w-6 text-primary" />
                            <span className="font-bold text-lg tracking-wide text-white">AudioLIT <span className="text-xs text-indigo-400 font-normal">v1.0</span></span>
                                  </div>
                                        <div className="flex items-center space-x-4 text-sm text-gray-400">
                                                <span className="flex items-center gap-1"><Cpu className="h-4 w-4" /> ECHO 1.0 Engine</span>
                                                        <span className="flex items-center gap-1"><Sliders className="h-4 w-4" /> Multi-Pane Diagnostic</span>
                                                              </div>
                                                                  </header>
                                                                    );
                                                                    };
                                                                    