import React from 'react';
import { AudioProvider } from './contexts/AudioContext';
import { Navbar } from './components/layout/Navbar';
import { Sidebar } from './components/layout/Sidebar';
import { MainViewport } from './components/layout/MainViewport';

const App: React.FC = () => {
  return (
      <AudioProvider>
            <div className="flex flex-col h-screen w-screen overflow-hidden bg-background text-gray-100">
                    <Navbar />
                            <div className="flex flex-1 overflow-hidden">
                                      <Sidebar />
                                                <MainViewport />
                                                        </div>
                                                              </div>
                                                                  </AudioProvider>
                                                                    );
                                                                    };

                                                                    export default App;
                                                                    