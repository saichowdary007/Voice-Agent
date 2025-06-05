'use client';
/// <reference types="node" />

import React, { useState, useCallback } from 'react';
import dynamic from 'next/dynamic';

// Disable SSR for this client-only app
export const dynamicParams = false;

// Dynamically import VoiceAgent to avoid SSR issues
const VoiceAgent = dynamic(() => import('../components/VoiceAgent'), {
  ssr: false,
  loading: () => (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-400 mx-auto mb-4"></div>
        <p className="text-white/80 text-lg">Initializing Ultra-Fast Voice Agent...</p>
        <p className="text-white/60 text-sm mt-2">Target: &lt; 500ms response time</p>
      </div>
    </div>
  )
});

export default function HomePage() {
  const [errorMessage, setErrorMessage] = useState('');
  const [showError, setShowError] = useState(false);

  // Use useCallback to prevent function recreation on every render
  const handleError = useCallback((error: string) => {
    console.error('Voice Agent Error:', error);
    setErrorMessage(error);
    setShowError(true);
    
    // Auto-hide error after 5 seconds instead of unmounting the component
    setTimeout(() => {
      setShowError(false);
    }, 5000);
  }, []);

  const clearError = useCallback(() => {
    setShowError(false);
    setErrorMessage('');
  }, []);

  return (
    <main className="h-screen w-screen overflow-hidden bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative">
      {/* Always render VoiceAgent - don't unmount it */}
      <VoiceAgent key="voice-agent-singleton" onError={handleError} />
      
      {/* Show error overlay instead of replacing the component */}
      {showError && (
        <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="text-center max-w-md mx-auto p-6 bg-red-900/50 rounded-lg border border-red-500/50">
            <div className="text-red-400 text-4xl mb-4">⚠️</div>
            <h1 className="text-white text-xl font-bold mb-4">Connection Issue</h1>
            <p className="text-white/80 mb-6 text-sm">{errorMessage}</p>
            <div className="flex gap-3 justify-center">
              <button
                onClick={clearError}
                className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-medium transition-colors text-sm"
              >
                Dismiss
              </button>
              <button
                onClick={() => window.location.reload()}
                className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg font-medium transition-colors text-sm"
              >
                Reload Page
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
} 