import React, { useState, useEffect } from 'react';
import AudioVisualizer from './AudioVisualizer';
import { RippleButton } from './magicui/ripple-button';
import { useAuth } from '../contexts/AuthContext';
import { useWebSocket } from '../hooks/useWebSocket';

const VoiceInterface: React.FC = () => {
  const { logout, isAuthenticated } = useAuth();
  const [muted, setMuted] = useState(false);
  const [isUserSpeaking, setIsUserSpeaking] = useState(false);
  const [microphonePermission, setMicrophonePermission] = useState<'pending' | 'granted' | 'denied'>('pending');
  const [permissionError, setPermissionError] = useState<string | null>(null);
  const [isAIResponding, setIsAIResponding] = useState(false);
  const [lastAIResponse, setLastAIResponse] = useState<string>('');
  
  // WebSocket connection for voice agent
  const { 
    isConnected: wsConnected, 
    isConnecting: wsConnecting, 
    lastMessage,
    error: wsError,
    connectionStatus 
  } = useWebSocket({
    onMessage: (message) => {
      console.log('📩 WebSocket message received:', message);
      
      // Handle different message types from backend
      switch (message.type) {
        case 'text_response':
          setLastAIResponse(message.text || '');
          setIsAIResponding(false);
          break;
        case 'agent_text':
          setLastAIResponse(message.content || '');
          setIsAIResponding(false);
          break;
        case 'audio_response':
          // Audio playback is handled automatically in useWebSocket hook
          setIsAIResponding(false);
          break;
        case 'audio_processed':
          if (message.data === 'final') {
            setIsAIResponding(true);
          }
          break;
        case 'vad_status':
          // Backend VAD status (optional)
          break;
        case 'error':
          console.error('❌ Backend error:', (message as any).message || message.data);
          setIsAIResponding(false);
          break;
      }
    },
    onError: (error) => {
      console.error('❌ WebSocket error:', error);
    }
  }, isAuthenticated);

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

      {/* WebSocket Connection Status */}
      <div className="absolute top-4 right-4 pointer-events-none z-40">
        <div className={`flex items-center space-x-2 px-3 py-2 rounded-full transition-all duration-300 ${
          wsConnected
            ? 'bg-green-500/20 border border-green-400/50'
            : wsConnecting
            ? 'bg-yellow-500/20 border border-yellow-400/50'
            : 'bg-red-500/20 border border-red-400/50'
        }`}>
          <div className={`w-2 h-2 rounded-full transition-all duration-300 ${
            wsConnected
              ? 'bg-green-400'
              : wsConnecting
              ? 'bg-yellow-400 animate-pulse'
              : 'bg-red-400'
          }`} />
          <span className={`text-xs font-medium transition-colors duration-300 ${
            wsConnected
              ? 'text-green-300'
              : wsConnecting
              ? 'text-yellow-300'
              : 'text-red-300'
          }`}>
            🌐 {wsConnected ? 'Connected' : wsConnecting ? 'Connecting' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* AI Response Status */}
      {isAIResponding && (
        <div className="absolute top-16 right-4 pointer-events-none z-40">
          <div className="flex items-center space-x-2 px-3 py-2 rounded-full bg-purple-500/20 border border-purple-400/50 transition-all duration-300">
            <div className="w-2 h-2 rounded-full bg-purple-400 animate-pulse" />
            <span className="text-xs font-medium text-purple-300">
              🤖 AI Thinking...
            </span>
          </div>
        </div>
      )}

      {/* Microphone Permission Status */}
      {microphonePermission !== 'granted' && (
        <div className="absolute top-28 right-4 pointer-events-none z-40">
          <div className={`flex items-center space-x-2 px-3 py-2 rounded-full transition-all duration-300 ${
            microphonePermission === 'denied'
              ? 'bg-red-500/20 border border-red-400/50'
              : 'bg-yellow-500/20 border border-yellow-400/50'
          }`}>
            <div className={`w-2 h-2 rounded-full transition-all duration-300 ${
              microphonePermission === 'denied'
                ? 'bg-red-400'
                : 'bg-yellow-400 animate-pulse'
            }`} />
            <span className={`text-xs font-medium transition-colors duration-300 ${
              microphonePermission === 'denied'
                ? 'text-red-300'
                : 'text-yellow-300'
            }`}>
              🎤 {microphonePermission === 'denied' ? 'Mic Blocked' : 'Mic Pending'}
            </span>
          </div>
        </div>
      )}

      {/* AI Response Display */}
      {lastAIResponse && (
        <div className="absolute bottom-32 left-0 right-0 px-8 pointer-events-none z-40">
          <div className="max-w-2xl mx-auto">
            <div className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-lg px-6 py-4">
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 bg-purple-500/20 border border-purple-400/50 rounded-full flex items-center justify-center">
                    <span className="text-purple-300 text-sm">🤖</span>
                  </div>
                </div>
                <div className="flex-1">
                  <p className="text-white text-sm leading-relaxed">
                    {lastAIResponse}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

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

        {/* Connection Status Control */}
        <div className="pointer-events-auto flex items-center space-x-4">
          {wsError && (
            <div className="px-4 py-2 bg-red-600/80 border border-red-500/50 rounded-lg">
              <span className="text-white text-sm">❌ Connection Error</span>
            </div>
          )}
          
          <RippleButton
            onClick={logout}
            className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white border border-red-800"
            rippleColor="rgba(255,255,255,0.3)"
          >
            End Session
          </RippleButton>
        </div>
      </div>
    </div>
  );
};

export default VoiceInterface; 