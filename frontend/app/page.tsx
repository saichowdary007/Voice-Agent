'use client';
/// <reference types="node" />

import React, { useState } from 'react';
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
  const [isError, setIsError] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const handleError = (error: string) => {
    console.error('Voice Agent Error:', error);
    setIsError(true);
    setErrorMessage(error);
  };

  const clearError = () => {
    setIsError(false);
    setErrorMessage('');
  };

  if (isError) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-red-900 to-slate-900 flex items-center justify-center">
        <div className="text-center max-w-md mx-auto p-6">
          <div className="text-red-400 text-6xl mb-4">⚠️</div>
          <h1 className="text-white text-2xl font-bold mb-4">Voice Agent Error</h1>
          <p className="text-white/80 mb-6">{errorMessage}</p>
          <button
            onClick={clearError}
            className="bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <main className="h-screen w-screen overflow-hidden bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <VoiceAgent onError={handleError} />
    </main>
  );
} 