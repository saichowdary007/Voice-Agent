'use client';

import React from 'react';

interface AudioVisualizerProps {
  audioLevel: number;
  isRecording: boolean;
  isMuted: boolean;
  className?: string;
}

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({
  audioLevel,
  isRecording,
  isMuted,
  className = '',
}) => {
  const bars = Array.from({ length: 20 }, (_, index) => {
    const threshold = index / 20;
    const isActive = audioLevel > threshold && isRecording && !isMuted;
    
    return (
      <div
        key={index}
        className={`w-1 rounded-full transition-all duration-100 ${
          isActive ? 'bg-green-500' : 'bg-gray-300'
        }`}
        style={{
          height: isActive ? `${Math.max(8, audioLevel * 40)}px` : '8px',
        }}
      />
    );
  });

  return (
    <div className={`audio-visualizer ${className}`}>
      <div className="flex items-end justify-center space-x-1 h-12 p-2 bg-gray-50 rounded-lg">
        {bars}
      </div>
      <div className="text-center mt-2 text-sm text-gray-600">
        {isMuted ? 'Muted' : isRecording ? 'Listening...' : 'Ready'}
      </div>
    </div>
  );
}; 