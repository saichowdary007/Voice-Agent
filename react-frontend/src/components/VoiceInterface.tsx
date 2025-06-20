import React, { useState, useEffect } from 'react';
import AudioVisualizer from './AudioVisualizer';
import { RippleButton } from './magicui/ripple-button';
import { useAuth } from '../contexts/AuthContext';

const VoiceInterface: React.FC = () => {
  const { logout } = useAuth();
  const [muted, setMuted] = useState(false);
  const [isUserSpeaking, setIsUserSpeaking] = useState(false);
  const [microphonePermission, setMicrophonePermission] = useState<'pending' | 'granted' | 'denied'>('pending');
  const [permissionError, setPermissionError] = useState<string | null>(null);

  const toggleMute = () => setMuted((prev) => !prev);

  const handleMicrophonePermission = (granted: boolean) => {
    setMicrophonePermission(granted ? 'granted' : 'denied');
    if (!granted) {
      setPermissionError('Microphone access denied. Please allow microphone access and refresh the page.');
    } else {
      setPermissionError(null);
    }
  };

  const handleVoiceActivity = (isActive: boolean) => {
    setIsUserSpeaking(isActive);
  };

  const requestMicrophonePermission = async () => {
    try {
      setMicrophonePermission('pending');
      setPermissionError(null);
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      // Stop the stream immediately, AudioVisualizer will create its own
      stream.getTracks().forEach(track => track.stop());
      
      setMicrophonePermission('granted');
      // Reload the page to reinitialize AudioVisualizer with permissions
      window.location.reload();
    } catch (error) {
      console.error('Failed to request microphone permission:', error);
      setMicrophonePermission('denied');
      setPermissionError('Failed to access microphone. Please check your browser settings.');
    }
  };

  return (
    <div className="relative w-screen h-screen bg-black overflow-hidden">
      {/* Visualizer fills full screen */}
      <AudioVisualizer 
        muted={muted} 
        isUserSpeaking={isUserSpeaking}
        onMicrophonePermission={handleMicrophonePermission}
        onVoiceActivity={handleVoiceActivity}
      />

      {/* Microphone Permission Banner */}
      {microphonePermission === 'denied' && (
        <div className="absolute top-0 left-0 right-0 bg-red-600/90 backdrop-blur-lg border-b border-red-500/50 px-4 py-3 z-50">
          <div className="text-center">
            <p className="text-white text-sm mb-2">
              {permissionError || 'Microphone access is required for voice interaction.'}
            </p>
            <RippleButton
              onClick={requestMicrophonePermission}
              className="px-4 py-2 bg-white text-red-600 hover:bg-gray-100 text-sm font-medium border-0"
              rippleColor="rgba(220, 38, 38, 0.3)"
            >
              Allow Microphone Access
            </RippleButton>
          </div>
        </div>
      )}

      {/* Microphone Permission Loading */}
      {microphonePermission === 'pending' && (
        <div className="absolute top-0 left-0 right-0 bg-yellow-600/90 backdrop-blur-lg border-b border-yellow-500/50 px-4 py-3 z-50">
          <div className="text-center">
            <p className="text-white text-sm">
              Requesting microphone access... Please allow when prompted.
            </p>
          </div>
        </div>
      )}

      {/* Voice Activity Indicator */}
      <div className="absolute top-4 left-4 pointer-events-none z-40">
        <div className={`flex items-center space-x-2 px-3 py-2 rounded-full transition-all duration-300 ${
          isUserSpeaking 
            ? 'bg-green-500/20 border border-green-400/50' 
            : 'bg-gray-800/20 border border-gray-600/30'
        }`}>
          <div className={`w-2 h-2 rounded-full transition-all duration-300 ${
            isUserSpeaking ? 'bg-green-400 animate-pulse' : 'bg-gray-500'
          }`} />
          <span className={`text-xs font-medium transition-colors duration-300 ${
            isUserSpeaking ? 'text-green-300' : 'text-gray-400'
          }`}>
            {isUserSpeaking ? 'Listening...' : 'Ready'}
          </span>
        </div>
      </div>

      {/* Microphone Status Indicator */}
      <div className="absolute top-4 right-4 pointer-events-none z-40">
        <div className={`flex items-center space-x-2 px-3 py-2 rounded-full transition-all duration-300 ${
          microphonePermission === 'granted'
            ? 'bg-blue-500/20 border border-blue-400/50'
            : microphonePermission === 'denied'
            ? 'bg-red-500/20 border border-red-400/50'
            : 'bg-yellow-500/20 border border-yellow-400/50'
        }`}>
          <div className={`w-2 h-2 rounded-full transition-all duration-300 ${
            microphonePermission === 'granted'
              ? 'bg-blue-400'
              : microphonePermission === 'denied'
              ? 'bg-red-400'
              : 'bg-yellow-400 animate-pulse'
          }`} />
          <span className={`text-xs font-medium transition-colors duration-300 ${
            microphonePermission === 'granted'
              ? 'text-blue-300'
              : microphonePermission === 'denied'
              ? 'text-red-300'
              : 'text-yellow-300'
          }`}>
            🎤 {microphonePermission === 'granted' ? 'Connected' : microphonePermission === 'denied' ? 'Blocked' : 'Pending'}
          </span>
        </div>
      </div>

      {/* Controls */}
      <div className="absolute bottom-8 left-0 right-0 flex items-center justify-between px-8 pointer-events-none">
        <RippleButton
          onClick={toggleMute}
          className={`pointer-events-auto px-6 py-3 text-white border transition-colors ${
            muted 
              ? 'bg-red-600/80 hover:bg-red-700/80 border-red-500/50' 
              : 'bg-white/10 hover:bg-white/20 border-white/20'
          }`}
          rippleColor="rgba(255,255,255,0.3)"
        >
          {muted ? '🔇 Unmute' : '🎤 Mute'}
        </RippleButton>

        <RippleButton
          onClick={logout}
          className="pointer-events-auto px-6 py-3 bg-red-600 hover:bg-red-700 text-white border border-red-800"
          rippleColor="rgba(255,255,255,0.3)"
        >
          End Session
        </RippleButton>
      </div>
    </div>
  );
};

export default VoiceInterface; 