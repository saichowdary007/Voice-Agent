'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Mic, MicOff, Square, AlertTriangle } from 'lucide-react';
import { MicVAD, utils as vadUtils } from '@ricky0123/vad-web';
import { AudioPlayer } from '../lib/audio';

interface VoiceAgentProps {
  onError?: (error: string) => void;
}

interface SessionState {
  isConnected: boolean;
  isRecording: boolean;
  isMuted: boolean;
  isProcessing: boolean;
  isSpeaking: boolean;
  sessionEnded: boolean;
  sessionId?: string;
  isReconnecting: boolean;
}

interface TranscriptState {
  partial: string;
  final: string[];
  aiResponse: string;
  isAiResponding: boolean;
}

export default function VoiceAgent({ onError }: VoiceAgentProps) {
  // WebSocket and Audio refs
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const vadRef = useRef<MicVAD | null>(null);
  const pcmPlayerNodeRef = useRef<AudioWorkletNode | null>(null);
  const keepAliveIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const keepAliveTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Add refs for persistent TTS audio player (replacing ttsAudioContextsRef)
  const isCleaningUpRef = useRef<boolean>(false);
  const ttsAudioPlayerRef = useRef<AudioPlayer | null>(null);
  const lastServerMessageTimeRef = useRef<number>(Date.now());

  // State
  const [session, setSession] = useState<SessionState>({
    isConnected: false,
    isRecording: false,
    isMuted: false,
    isProcessing: false,
    isSpeaking: false,
    sessionEnded: false,
    isReconnecting: false,
  });

  const [transcript, setTranscript] = useState<TranscriptState>({
    partial: '',
    final: [],
    aiResponse: '',
    isAiResponding: false,
  });

  const [audioLevel, setAudioLevel] = useState(0);
  const [latency, setLatency] = useState<number>(0);

  // WebSocket URL
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001/ws';

  // Initialize WebSocket connection
  const connectWebSocket = useCallback(() => {
    // Check if already connected or cleaning up
    if (wsRef.current?.readyState === WebSocket.OPEN || isCleaningUpRef.current) {
      return;
    }

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      lastServerMessageTimeRef.current = Date.now();

      ws.onopen = () => {
        console.log('WebSocket connected');
        setSession(prev => ({ ...prev, isConnected: true, sessionEnded: false, isReconnecting: false }));
        lastServerMessageTimeRef.current = Date.now();
        // Start keep-alive ping from client
        if (keepAliveIntervalRef.current) clearInterval(keepAliveIntervalRef.current);
        keepAliveIntervalRef.current = setInterval(() => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            const now = Date.now();
            if (now - lastServerMessageTimeRef.current > 10000) {
              console.warn('No server message for over 10 seconds. Closing connection.');
              wsRef.current.close(1000, "Keep-alive timeout"); 
              // Consider implementing a reconnect strategy here
            } else {
                // Send ping to server
                wsRef.current.send(JSON.stringify({ type: "ping", t: Date.now() }));
            }
          }
        }, 5000); // Send ping every 5 seconds
      };

      ws.onmessage = async (event) => {
        lastServerMessageTimeRef.current = Date.now();
        setSession(prev => ({ ...prev, isReconnecting: false }));
        try {
          const data = JSON.parse(event.data as string);
          if (!data || typeof data !== 'object') {
            console.error('Invalid message format - not an object:', data);
            return;
          }
          if (!data.type) {
            console.error('Message missing type field:', data);
            return;
          }
          await handleWebSocketMessage(data);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e, 'Raw data:', event.data);
        }
      };

      ws.onclose = (event: CloseEvent) => {
        console.log('WebSocket disconnected');
        setSession(prev => ({ 
          ...prev, 
          isConnected: false, 
          isRecording: false, 
          isProcessing: false, 
          isSpeaking: false 
        }));
        if (event.code !== 1000) {
          console.warn(`WebSocket closed with code: ${event.code}. Reason: ${event.reason}`);
          onError?.("Connection lost. Please try again."); // Display connection lost banner
          // Optionally, trigger reconnection logic here if desired
          // For example, attempt to reconnect after a delay:
          // setSession(prev => ({ ...prev, isReconnecting: true }));
          // setTimeout(() => connectWebSocket(), 5000); 
        }
        if (keepAliveIntervalRef.current) {
          clearInterval(keepAliveIntervalRef.current);
          keepAliveIntervalRef.current = null;
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        onError?.('WebSocket connection failed');
      };

    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      onError?.('Failed to connect to voice service');
    }
  }, [wsUrl, onError]);

  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback(async (data: any) => {
    switch (data.type) {
      case 'status':
        setSession(prev => ({ ...prev, sessionId: data.session_id, isProcessing: data.status === 'processing' }));
        if (data.status === 'processing') {
            setTranscript(prev => ({ ...prev, partial: '', aiResponse: '' })); // Clear previous messages on new processing
        }
        break;

      case 'recording_started':
        setSession(prev => ({ ...prev, isRecording: true }));
        break;

      case 'recording_stopped':
        setSession(prev => ({ ...prev, isRecording: false }));
        break;

      case 'pong':
        lastServerMessageTimeRef.current = Date.now(); // Clear keep-alive timeout
        // console.log("Pong received from server");
        break;

      case 'transcript':
        if (data.partial) {
          setTranscript(prev => ({ ...prev, partial: data.partial }));
        }
        if (data.final) {
          setTranscript(prev => ({ 
            ...prev, 
            final: [...prev.final, data.final],
            partial: '' 
          }));
          setSession(prev => ({ ...prev, isProcessing: true }));
        }
        break;

      case 'ai_response_start':
        setTranscript(prev => ({ ...prev, isAiResponding: true, aiResponse: '' }));
        break;

      case 'ai_response':
        if (data.complete) {
          setTranscript(prev => ({ ...prev, isAiResponding: false }));
          setSession(prev => ({ ...prev, isProcessing: false }));
        } else {
          setTranscript(prev => ({ ...prev, aiResponse: prev.aiResponse + data.token }));
        }
        break;

      case 'tts_start':
        setSession(prev => ({ ...prev, isSpeaking: true }));
        break;

      case 'audio_chunk':
        // Play TTS audio
        if (data.audio_data) {
          await playTTSAudio(data.audio_data);
        }
        break;

      case 'tts_complete':
        setSession(prev => ({ ...prev, isSpeaking: false }));
        break;

      case 'stop_audio':
        setSession(prev => ({ ...prev, isSpeaking: false }));
        break;

      case 'mute_status':
        setSession(prev => ({ ...prev, isMuted: data.muted }));
        break;

      case 'session_ended':
        // Use timeout to avoid direct call during message handling
        setTimeout(() => {
          if (!isCleaningUpRef.current) {
            handleEndSession();
          }
        }, 0);
        break;

      case 'session_metrics':
        if (data.metrics.total_latency) {
          setLatency(data.metrics.total_latency);
        }
        break;

      case 'error':
        console.error('Server error:', data.message);
        onError?.(data.message);
        break;

      default:
        console.log('Unknown message type:', data.type);
    }
  }, [onError]);

  // Initialize audio
  const initializeAudio = useCallback(async () => {
    if (isCleaningUpRef.current) {
      console.log('Cleanup in progress, cannot initialize audio');
      return;
    }

    try {
      console.log('Requesting microphone access...');
      
      // Enhanced audio constraints for better quality and compatibility
      const constraints: MediaStreamConstraints = {
        audio: {
          sampleRate: 16000,      // Match backend sample rate
          channelCount: 1,        // Mono audio for voice processing
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          // Additional constraints for better voice processing
          latency: 0.01,         // Low latency requirement
          volume: 1.0            // Full volume
        },
        video: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      
      console.log('Microphone access granted');
      console.log('Audio track settings:', stream.getAudioTracks()[0]?.getSettings());

      // Check if we need to clean up existing recorder
      if (mediaRecorderRef.current) {
        try {
          if (mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
          }
        } catch (recorderCleanupError) {
          console.warn('Error cleaning up existing MediaRecorder:', recorderCleanupError);
        }
        mediaRecorderRef.current = null;
      }

      // Set up MediaRecorder with optimized settings
      let mediaRecorder: MediaRecorder;
      try {
        // Try WebM/Opus first (best compression and quality for voice)
        mediaRecorder = new MediaRecorder(stream, {
          mimeType: 'audio/webm;codecs=opus',
          audioBitsPerSecond: 32000 // 32 kbps for voice
        });
        console.log('MediaRecorder initialized with WebM/Opus');
      } catch (webmError) {
        console.warn('WebM/Opus not supported, trying alternatives:', webmError);
        try {
          // Fallback to WebM without codec specification
          mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm',
            audioBitsPerSecond: 32000
          });
          console.log('MediaRecorder initialized with WebM (generic)');
        } catch (webmGenericError) {
          console.warn('WebM not supported, using default format:', webmGenericError);
          // Final fallback to default
          mediaRecorder = new MediaRecorder(stream);
          console.log('MediaRecorder initialized with default format');
        }
      }
      
      mediaRecorderRef.current = mediaRecorder;

      // Audio chunk buffering for improved streaming
      let audioChunks: Blob[] = [];
      let bufferTimer: number | null = null;
      const BUFFER_TIMEOUT = 120; // 120ms to match backend frame size

      const flushBuffer = () => {
        if (audioChunks.length > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
          const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
          
          audioBlob.arrayBuffer().then(buffer => {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(buffer);
              console.log(`Sent audio chunk: ${buffer.byteLength} bytes`);
            }
          }).catch(error => {
            console.error('Error converting audio blob to buffer:', error);
          });
          
          audioChunks = [];
        }
      };

      // Enhanced MediaRecorder event handlers
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          console.log(`Audio chunk received: ${event.data.size} bytes`);
          audioChunks.push(event.data);
          
          // Clear existing timer
          if (bufferTimer) {
            window.clearTimeout(bufferTimer);
          }
          
          // Set timer to flush buffer if no more chunks arrive soon
          bufferTimer = window.setTimeout(flushBuffer, BUFFER_TIMEOUT);
        }
      };

      mediaRecorder.onstart = () => {
        console.log('MediaRecorder started');
        setSession(prev => ({ ...prev, isRecording: true }));
      };

      mediaRecorder.onstop = () => {
        console.log('MediaRecorder stopped. Sending EOS to backend.');
        
        // Flush any remaining buffered audio
        if (bufferTimer) {
          window.clearTimeout(bufferTimer);
          bufferTimer = null;
        }
        flushBuffer();
        
        setSession(prev => ({ ...prev, isRecording: false }));
        
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: "eos" }));
        }
      };
      
      mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error:', event);
        onError?.('MediaRecorder error. Please check microphone.');
      };

      // Initialize VAD with optimized parameters for voice conversation
      console.log('Initializing VAD...');
      if (vadRef.current) {
        console.log('Destroying existing VAD instance...');
        try {
          vadRef.current.destroy();
        } catch (vadDestroyError) {
          console.warn('Error destroying existing VAD:', vadDestroyError);
        }
        vadRef.current = null;
      }

      let myVad: MicVAD | null = null;
      try {
        myVad = await MicVAD.new({
          stream: streamRef.current,
          onSpeechStart: () => {
            console.log("VAD: Speech started");
            setSession(prev => ({ ...prev, isSpeaking: true }));
            if (mediaRecorderRef.current?.state === 'inactive') {
              mediaRecorderRef.current.start(BUFFER_TIMEOUT); // Use consistent timing
            }
          },
          onSpeechEnd: (audio: Float32Array) => {
            console.log('VAD: Speech ended, audio length:', audio.length);
            setSession(prev => ({ ...prev, isSpeaking: false }));
            if (mediaRecorderRef.current?.state === 'recording') {
              console.log('VAD EOS: Stopping MediaRecorder');
              mediaRecorderRef.current.stop();
            }
          },
          // Optimized VAD parameters for responsive voice conversation
          minSpeechFrames: 3,              // Minimum frames to consider as speech start
          positiveSpeechThreshold: 0.7,    // Lower threshold for more sensitivity
          negativeSpeechThreshold: 0.35,   // Balanced threshold for speech end
          minSilenceFrames: 8,             // ~80ms of silence before ending speech
          redemptionFrames: 25,            // Frames to "forgive" brief silences
          preSpeechPadFrames: 1,           // Frames to include before speech starts
          onVADMisfire: () => {
            console.warn("VAD: Misfire detected - brief speech detection");
          },
        } as any); // Type assertion for extended VAD options
      } catch (vadCreationError) {
        console.error('Critical error during VAD initialization:', vadCreationError);
        let errorMessage = 'Failed to initialize Voice Activity Detection.';
        if (vadCreationError instanceof Error) {
          errorMessage += ` Details: ${vadCreationError.message}`;
          if (vadCreationError.message.toLowerCase().includes('onnx') || 
              vadCreationError.message.toLowerCase().includes('runtime')) {
            errorMessage += ' This might be due to ONNX runtime issues. Please refresh the page.';
          }
          if (vadCreationError.message.toLowerCase().includes('model') || 
              vadCreationError.message.toLowerCase().includes('silero')) {
            errorMessage += ' This might be due to VAD model loading issues.';
          }
        }
        onError?.(errorMessage);
        return;
      }

      if (!myVad) {
        console.error('VAD initialization returned null instance');
        onError?.('Failed to initialize Voice Activity Detection: Null instance returned.');
        return;
      }

      vadRef.current = myVad;
      
      try {
        vadRef.current.start();
        console.log('VAD initialized and started successfully');
      } catch (vadStartError) {
        console.error('Error starting VAD:', vadStartError);
        onError?.('Failed to start Voice Activity Detection after initialization.');
        if (vadRef.current) {
          vadRef.current.destroy();
          vadRef.current = null;
        }
        return;
      }

      // Set up enhanced audio level monitoring
      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      
      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;
      
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.8; // Smooth audio level changes
      source.connect(analyser);

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      
      const updateAudioLevel = () => {
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
          analyser.getByteFrequencyData(dataArray);
          const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
          setAudioLevel(Math.min(average / 255, 1.0)); // Ensure value is between 0 and 1
          requestAnimationFrame(updateAudioLevel);
        }
      };
      
      updateAudioLevel();

      console.log('Audio initialization completed successfully');

    } catch (error) {
      console.error('Failed to initialize audio:', error);
      
      // Provide more specific error messages
      if (error instanceof Error) {
        if (error.name === 'NotAllowedError') {
          onError?.('Microphone access denied. Please allow microphone access and refresh the page.');
        } else if (error.name === 'NotFoundError') {
          onError?.('No microphone found. Please connect a microphone and try again.');
        } else if (error.name === 'NotSupportedError') {
          onError?.('Audio recording not supported in this browser. Please use Chrome, Firefox, or Safari.');
        } else {
          onError?.(`Audio initialization failed: ${error.message}`);
        }
      } else {
        onError?.('Unknown error occurred during audio initialization.');
      }
    }
  }, [onError]);

  // Send WebSocket message
  const sendWebSocketMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  // Play TTS audio
  const playTTSAudio = useCallback(async (base64Audio: string) => {
    if (isCleaningUpRef.current || !session.isConnected || session.isMuted) return;

    try {
      // Initialize persistent TTS audio player if not exists
      if (!ttsAudioPlayerRef.current) {
        ttsAudioPlayerRef.current = new AudioPlayer();
      }

      // Decode base64 to binary data
      const binaryString = atob(base64Audio);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      // Play audio using the persistent audio player
      await ttsAudioPlayerRef.current.playAudio(bytes.buffer, () => {
        // Audio playback finished callback
        console.log('TTS audio chunk finished playing');
      });
      
    } catch (error) {
      console.error('Failed to play TTS audio:', error);
      // Try alternative method using HTML5 Audio
      try {
        const audio = new Audio(`data:audio/wav;base64,${base64Audio}`);
        await audio.play();
      } catch (fallbackError) {
        console.error('Fallback audio play failed:', fallbackError);
      }
    }
  }, []);

  // Toggle mute
  const toggleMute = useCallback(() => {
    const newMutedState = !session.isMuted;
    
    sendWebSocketMessage({
      type: 'mute',
      muted: newMutedState
    });

    if (newMutedState && mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.pause();
    } else if (!newMutedState && mediaRecorderRef.current?.state === 'paused') {
      mediaRecorderRef.current.resume();
    }
  }, [session.isMuted, sendWebSocketMessage]);

  // End session
  const endSession = useCallback(() => {
    console.log('Ending session...');
    setSession(prev => ({ ...prev, sessionEnded: true }));
    sendWebSocketMessage({ type: 'end_session' });
    
    // Delay the actual cleanup to allow the message to be sent
    setTimeout(() => {
      handleEndSession();
    }, 100);
  }, [sendWebSocketMessage]);

  // Handle end session
  const handleEndSession = useCallback(() => {
    // Prevent multiple simultaneous cleanup attempts
    if (isCleaningUpRef.current) {
      console.log('Cleanup already in progress, skipping...');
      return;
    }
    
    isCleaningUpRef.current = true;
    console.log('Cleaning up session...');
    
    try {
      // Clear any pending timeouts first
      if (keepAliveIntervalRef.current) {
        clearInterval(keepAliveIntervalRef.current);
        keepAliveIntervalRef.current = null;
      }
      if (keepAliveTimeoutRef.current) {
        clearTimeout(keepAliveTimeoutRef.current);
        keepAliveTimeoutRef.current = null;
      }

      // Stop VAD first to prevent new speech detection
      if (vadRef.current) {
        try {
          vadRef.current.destroy();
          console.log('VAD destroyed successfully');
        } catch (vadError) {
          console.error('Error destroying VAD:', vadError);
        }
        vadRef.current = null;
      }

      // Stop MediaRecorder
      if (mediaRecorderRef.current?.state !== 'inactive') {
        try {
          mediaRecorderRef.current?.stop();
          console.log('MediaRecorder stopped');
        } catch (recorderError) {
          console.error('Error stopping MediaRecorder:', recorderError);
        }
      }
      mediaRecorderRef.current = null;

      // Stop media stream tracks
      if (streamRef.current) {
        try {
          streamRef.current.getTracks().forEach(track => {
            track.stop();
            console.log(`Stopped track: ${track.kind}`);
          });
        } catch (streamError) {
          console.error('Error stopping stream tracks:', streamError);
        }
        streamRef.current = null;
      }
      
      // Disconnect audio nodes
      if (sourceRef.current) {
        try {
          sourceRef.current.disconnect();
          console.log('Audio source disconnected');
        } catch (sourceError) {
          console.error('Error disconnecting audio source:', sourceError);
        }
        sourceRef.current = null;
      }

      if (pcmPlayerNodeRef.current) {
        try {
          pcmPlayerNodeRef.current.disconnect();
          console.log('PCM player node disconnected');
        } catch (pcmError) {
          console.error('Error disconnecting PCM player node:', pcmError);
        }
        pcmPlayerNodeRef.current = null;
      }
      
      // Close AudioContext
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        try {
          audioContextRef.current.close().then(() => {
            console.log('AudioContext closed successfully');
          }).catch(closeError => {
            console.error('Error closing AudioContext:', closeError);
          });
        } catch (contextError) {
          console.error('Error initiating AudioContext close:', contextError);
        }
        audioContextRef.current = null;
      }
      
      // Clean up TTS audio player
      if (ttsAudioPlayerRef.current) {
        try {
          ttsAudioPlayerRef.current.cleanup();
          console.log('TTS audio player cleaned up');
        } catch (ttsError) {
          console.error('Error cleaning up TTS audio player:', ttsError);
        }
        ttsAudioPlayerRef.current = null;
      }

      // Close WebSocket
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        try {
          wsRef.current.close(1000, "Session ended gracefully");
          console.log('WebSocket closed');
        } catch (wsError) {
          console.error('Error closing WebSocket:', wsError);
        }
      }
      wsRef.current = null;

      // Reset state
      setSession({
        isConnected: false,
        isRecording: false,
        isMuted: false,
        isProcessing: false,
        isSpeaking: false,
        sessionEnded: true,
        isReconnecting: false,
      });

      setTranscript({
        partial: '',
        final: [],
        aiResponse: '',
        isAiResponding: false,
      });

      setAudioLevel(0);
      setLatency(0);
      
      console.log('Session cleanup completed successfully');
      
    } catch (error) {
      console.error('Error during cleanup:', error);
    } finally {
      // Reset cleanup flag after a delay to allow for any pending operations
      setTimeout(() => {
        isCleaningUpRef.current = false;
        console.log('Cleanup flag reset, ready for new session');
      }, 500); // Increased delay for better safety
    }
  }, []);

  // Start new session
  const startNewSession = useCallback(() => {
    console.log('Starting new session...');
    if (isCleaningUpRef.current) {
      console.log('Cleanup in progress, cannot start new session yet.');
      return;
    }
    
    isCleaningUpRef.current = false;
    
    setSession(prev => ({ 
      ...prev, 
      sessionEnded: false, 
      isProcessing: false, 
      isSpeaking: false,
      isReconnecting: false,
    }));
    setTranscript({
      partial: '',
      final: [],
      aiResponse: '',
      isAiResponding: false,
    });
    
    // Connect WebSocket
    connectWebSocket();
  }, [connectWebSocket]);

  // Initialize on mount
  useEffect(() => {
    connectWebSocket();
    
    // Return cleanup function that uses ref to avoid dependency issues
    return () => {
      if (!isCleaningUpRef.current) {
        handleEndSession();
      }
      if (vadRef.current) {
        vadRef.current.destroy();
        vadRef.current = null;
      }
    };
  }, []);

  // Start audio when connected
  useEffect(() => {
    if (session.isConnected && !streamRef.current && !vadRef.current) {
      initializeAudio();
    }
  }, [session.isConnected, initializeAudio]);

  // Keep-alive check
  useEffect(() => {
    if (keepAliveIntervalRef.current) {
      clearInterval(keepAliveIntervalRef.current);
    }
    keepAliveIntervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        const now = Date.now();
        if (now - lastServerMessageTimeRef.current > 2000 && now - lastServerMessageTimeRef.current <= 10000) {
          setSession(prev => ({ ...prev, isReconnecting: true }));
        } else if (now - lastServerMessageTimeRef.current > 10000) {
          console.warn('No server message for over 10 seconds. Potential connection loss.');
          setSession(prev => ({ ...prev, isConnected: false, isReconnecting: false }));
        }
      } else if (wsRef.current?.readyState === WebSocket.CLOSED || wsRef.current?.readyState === WebSocket.CLOSING) {
         setSession(prev => ({ ...prev, isConnected: false, isReconnecting: false }));
      }
    }, 1000);

    return () => {
      if (keepAliveIntervalRef.current) {
        clearInterval(keepAliveIntervalRef.current);
      }
    };
  }, [connectWebSocket]);

  // Get activity indicator
  const getActivityState = () => {
    if (session.isSpeaking) return 'speaking';
    if (session.isProcessing) return 'processing';
    if (session.isRecording && !session.isMuted) return 'listening';
    return 'idle';
  };

  const activityState = getActivityState();

  // UI elements
  const renderStatusIndicator = () => {
    if (session.isReconnecting) return <span className="text-yellow-500">Reconnecting...</span>;
    if (!session.isConnected) return <span className="text-red-500">Disconnected</span>;
    if (session.isProcessing) return <span className="text-blue-500">Processing...</span>; // TASK 2.c: Show spinner/text for processing
    if (session.isRecording) return <span className="text-green-500">Listening...</span>;
    return <span className="text-gray-500">Idle</span>;
  };

  return (
    <div className="h-screen w-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex flex-col">
      
      {/* Status Bar */}
      <div className="absolute top-4 left-4 right-4 flex justify-between items-center z-10">
        {/* Connection Status */}
        <div className="flex items-center space-x-2 bg-black/30 backdrop-blur-sm rounded-full px-4 py-2">
          <div className={`w-2 h-2 rounded-full ${
            session.isConnected ? 'bg-green-400' : 'bg-red-400'
          }`} />
          <span className="text-white/80 text-sm">
            {session.isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>

        {/* Latency Indicator */}
        {latency > 0 && (
          <div className="bg-black/30 backdrop-blur-sm rounded-full px-4 py-2">
            <span className="text-white/80 text-sm">
              {latency.toFixed(0)}ms
            </span>
          </div>
        )}
      </div>

      {/* Central Microphone Visual */}
      <div className="flex-1 flex items-center justify-center">
        <div className="relative">
          {/* Outer pulse ring for activity */}
          <div className={`absolute inset-0 rounded-full border-2 transition-all duration-300 ${
            activityState === 'listening' ? 'border-blue-400 animate-pulse scale-110' :
            activityState === 'processing' ? 'border-yellow-400 animate-pulse scale-105' :
            activityState === 'speaking' ? 'border-purple-400 animate-pulse scale-115' :
            'border-gray-600'
          }`} style={{ width: '200px', height: '200px' }} />
          
          {/* Audio level visualization */}
          <div 
            className={`absolute inset-0 rounded-full transition-all duration-100 ${
              activityState === 'listening' ? 'bg-blue-400/20' :
              activityState === 'processing' ? 'bg-yellow-400/20' :
              activityState === 'speaking' ? 'bg-purple-400/20' :
              'bg-gray-600/10'
            }`}
            style={{ 
              width: '200px', 
              height: '200px',
              transform: `scale(${1 + audioLevel * 0.3})`
            }}
          />
          
          {/* Microphone icon */}
          <div className="relative w-48 h-48 bg-black/40 backdrop-blur-sm rounded-full flex items-center justify-center">
            {session.isMuted ? (
              <MicOff className="w-16 h-16 text-red-400" />
            ) : (
              <Mic className={`w-16 h-16 transition-colors ${
                activityState === 'listening' ? 'text-blue-400' :
                activityState === 'processing' ? 'text-yellow-400' :
                activityState === 'speaking' ? 'text-purple-400' :
                'text-white/60'
              }`} />
            )}
          </div>
        </div>
      </div>

      {/* Live Transcript Bar */}
      <div className="mx-8 mb-32 min-h-[120px] bg-black/30 backdrop-blur-sm rounded-lg p-6 border border-white/10">
        {/* Partial transcript */}
        {transcript.partial && (
          <div className="text-blue-400 text-lg mb-2">
            <span className="text-white/60 text-sm">Listening: </span>
            {transcript.partial}
            <span className="animate-pulse">|</span>
          </div>
        )}
        
        {/* Final transcript */}
        {transcript.final.length > 0 && (
          <div className="text-white text-lg mb-2">
            <span className="text-white/60 text-sm">You: </span>
            {transcript.final[transcript.final.length - 1]}
          </div>
        )}
        
        {/* AI Response */}
        {(transcript.aiResponse || transcript.isAiResponding) && (
          <div className="text-purple-400 text-lg">
            <span className="text-white/60 text-sm">Assistant: </span>
            {transcript.aiResponse}
            {transcript.isAiResponding && <span className="animate-pulse">|</span>}
          </div>
        )}
        
        {/* Status text when idle */}
        {!transcript.partial && transcript.final.length === 0 && !transcript.aiResponse && (
          <div className="text-white/60 text-center py-8">
            {session.sessionEnded ? 
              'Session ended. Click "Start New Session" to begin again.' :
              session.isConnected ? 
                (session.isMuted ? 'Microphone muted - Click to unmute' : 'Start speaking to begin conversation') :
                'Connecting to voice service...'
            }
          </div>
        )}
      </div>

      {/* Control Buttons - Fixed at bottom */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 flex space-x-6">
        {session.sessionEnded ? (
          /* Start New Session Button */
          <button
            onClick={startNewSession}
            className="px-8 py-3 rounded-full bg-green-600 hover:bg-green-700 text-white flex items-center space-x-2 transition-all shadow-lg shadow-green-600/50"
          >
            <Mic className="w-5 h-5" />
            <span>Start New Session</span>
          </button>
        ) : (
          <>
            {/* Mute/Unmute Button */}
            <button
              onClick={toggleMute}
              disabled={!session.isConnected}
              className={`w-16 h-16 rounded-full flex items-center justify-center transition-all ${
                session.isMuted 
                  ? 'bg-red-500 hover:bg-red-600 text-white shadow-lg shadow-red-500/50' 
                  : 'bg-blue-500 hover:bg-blue-600 text-white shadow-lg shadow-blue-500/50'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {session.isMuted ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
            </button>

            {/* End Session Button */}
            <button
              onClick={endSession}
              disabled={!session.isConnected}
              className="w-16 h-16 rounded-full bg-red-600 hover:bg-red-700 text-white flex items-center justify-center transition-all shadow-lg shadow-red-600/50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Square className="w-6 h-6" />
            </button>
          </>
        )}
      </div>

      {/* Activity Indicator */}
      <div className="absolute bottom-4 right-4 bg-black/30 backdrop-blur-sm rounded-full px-4 py-2">
        <span className="text-white/80 text-sm capitalize">
          {activityState}
        </span>
      </div>

      {session.isReconnecting && (
        <div className="absolute top-0 left-0 right-0 bg-yellow-500 text-black text-center p-2 z-50 flex items-center justify-center">
          <AlertTriangle className="h-5 w-5 mr-2" />
          Reconnecting to server...
        </div>
      )}

      {session.isProcessing && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded-lg">
          <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
        </div>
      )}
    </div>
  );
} 