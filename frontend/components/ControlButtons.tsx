'use client';

import React from 'react';

interface ControlButtonsProps {
  isMuted: boolean;
  isConnected: boolean;
  isProcessing: boolean;
  onMuteToggle: () => void;
  onEndSession: () => void;
  className?: string;
}

export const ControlButtons: React.FC<ControlButtonsProps> = ({
  isMuted,
  isConnected,
  isProcessing,
  onMuteToggle,
  onEndSession,
  className = '',
}) => {
  return (
    <div className={`control-buttons ${className}`}>
      <button
        onClick={onMuteToggle}
        disabled={!isConnected}
        className={`relative z-50 px-6 py-3 rounded-full font-medium transition-all duration-200 shadow-lg backdrop-blur-sm border border-white/20 ${
          isMuted
            ? 'bg-red-500/90 hover:bg-red-600/90 text-white hover:scale-105 hover:shadow-red-500/25'
            : 'bg-green-500/90 hover:bg-green-600/90 text-white hover:scale-105 hover:shadow-green-500/25'
        } disabled:bg-gray-500/50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:shadow-none`}
        style={{ 
          pointerEvents: 'auto',
          touchAction: 'manipulation' 
        }}
      >
        <span className="flex items-center space-x-2">
          <span className="text-lg">{isMuted ? '🔇' : '🎤'}</span>
          <span className="text-sm font-semibold">
            {isMuted ? 'Unmute' : 'Mute'}
          </span>
        </span>
      </button>

      <button
        onClick={onEndSession}
        disabled={!isConnected || isProcessing}
        className="relative z-50 px-6 py-3 rounded-full font-medium bg-gray-600/90 hover:bg-gray-700/90 text-white transition-all duration-200 shadow-lg backdrop-blur-sm border border-white/20 hover:scale-105 hover:shadow-gray-500/25 disabled:bg-gray-500/50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:shadow-none"
        style={{ 
          pointerEvents: 'auto',
          touchAction: 'manipulation' 
        }}
      >
        <span className="flex items-center space-x-2">
          <span className="text-lg">📞</span>
          <span className="text-sm font-semibold">End Session</span>
        </span>
      </button>
    </div>
  );
}; 