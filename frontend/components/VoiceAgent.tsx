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
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const vadRef = useRef<MicVAD | null>(null);
  const pcmPlayerNodeRef = useRef<AudioWorkletNode | null>(null); // May not be used if AudioPlayer handles all
  const keepAliveIntervalRef = useRef<NodeJS.Timeout | null>(null);
  // const keepAliveTimeoutRef = useRef<NodeJS.Timeout | null>(null); // Removed as logic is combined in interval

  const isCleaningUpRef = useRef<boolean>(false);
  const ttsAudioPlayerRef = useRef<AudioPlayer | null>(null);
  const lastServerMessageTimeRef = useRef<number>(Date.now());

  const [session, setSession] = useState<SessionState>({
    isConnected: false,
    isRecording: false,
    isMuted: false,
    isProcessing: false,
    isSpeaking: false, // This refers to AI speaking (TTS)
    sessionEnded: false,
    isReconnecting: false,
  });

  // Add reconnection state
  const reconnectAttemptsRef = useRef<number>(0);
  const maxReconnectAttempts = 5;
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const audioQueueRef = useRef<Blob[]>([]);

  const [transcript, setTranscript] = useState<TranscriptState>({
    partial: '',
    final: [],
    aiResponse: '',
    isAiResponding: false,
  });

  const [audioLevel, setAudioLevel] = useState(0);
  const [latency, setLatency] = useState<number>(0);
  const [isUserSpeaking, setIsUserSpeaking] = useState(false); // For client-side VAD indication

  // Get WebSocket URL from environment variable
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8080/ws';

  // Add a ref to track sessionEnded state without causing re-renders
  const sessionEndedRef = useRef(false);

  // Update the ref whenever session.sessionEnded changes
  useEffect(() => {
    sessionEndedRef.current = session.sessionEnded;
  }, [session.sessionEnded]);

  /**
   * Ends the current voice session and performs a full cleanup.
   * A regular (hoisted) function prevents "used before declaration" errors.
   */
  function handleEndSession(backendInitiated = false) {
    if (isCleaningUpRef.current) {
      console.log('handleEndSession: Cleanup already in progress, skipping.');
      return;
    }
    isCleaningUpRef.current = true;
    console.log(`Cleaning up session... Backend initiated: ${backendInitiated}`);

    if (keepAliveIntervalRef.current) {
      clearInterval(keepAliveIntervalRef.current);
      keepAliveIntervalRef.current = null;
    }

    const cleanupPromises = [];

    // Clean up VAD - make this awaitable
    if (vadRef.current && typeof (vadRef.current as any).destroy === 'function') {
      try {
        const vadCleanup = (vadRef.current as any).destroy();
        if (vadCleanup instanceof Promise) {
          cleanupPromises.push(vadCleanup.catch(e => {
            console.warn('Error destroying VAD:', e && (e as any).message ? (e as any).message : 'Unknown error');
          }));
        }
      } catch (e) {
        console.warn('Error initiating VAD destroy:', e && (e as any).message ? (e as any).message : 'Unknown error');
      }
    }
    vadRef.current = null;

    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== 'inactive') {
      try { mr.stop(); } catch (e) { 
        const errorMessage = e && (e as any).message ? (e as any).message : 'Unknown error';
        console.warn('Error stopping MediaRecorder during cleanup:', errorMessage); 
      }
    }
    mediaRecorderRef.current = null;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }

    if (pcmPlayerNodeRef.current) {
      pcmPlayerNodeRef.current.disconnect();
      pcmPlayerNodeRef.current = null;
    }

    const ctx = audioContextRef.current;
    if (ctx && ctx.state !== 'closed') {
      ctx.close().catch(e => console.warn('Error closing AudioContext:', e));
    }
    audioContextRef.current = null;

    if (ttsAudioPlayerRef.current) {
      ttsAudioPlayerRef.current.dispose();
      ttsAudioPlayerRef.current = null;
    }

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('Closing WebSocket connection from handleEndSession…');
      wsRef.current.close(1000, 'Session ended by client/cleanup');
    }
    wsRef.current = null;

    setSession(prev => ({
      ...prev,
      isConnected: false,
      isRecording: false,
      isMuted: false,
      isProcessing: false,
      isSpeaking: false,
      sessionEnded: true,
      isReconnecting: false,
      sessionId: undefined,
    }));
    setTranscript({ partial: '', final: [], aiResponse: '', isAiResponding: false });
    setAudioLevel(0);
    setLatency(0);
    setIsUserSpeaking(false);

    console.log('Session state reset.');

    // Wait for any cleanup promises to resolve before completing
    Promise.all(cleanupPromises).finally(() => {
      setTimeout(() => {
        isCleaningUpRef.current = false;
        console.log('Cleanup flag reset, ready for new session.');
      }, 500);
    });
  }

  const flushAudioQueue = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN && audioQueueRef.current.length > 0) {
      console.log(`Flushing ${audioQueueRef.current.length} queued audio chunks`);
      
      audioQueueRef.current.forEach(audioBlob => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(audioBlob);
        }
      });
      
      audioQueueRef.current = [];
    }
  }, []);

  const connectWebSocket = useCallback(() => {
    // Check if we should skip connection attempt
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('ConnectWebSocket: Already connected, skipping.');
      return;
    }
    if (isCleaningUpRef.current) {
      console.log('ConnectWebSocket: Cleanup in progress, skipping.');
      return;
    }
    if (wsRef.current?.readyState === WebSocket.CONNECTING) {
      console.log('ConnectWebSocket: Connection already in progress, skipping.');
      return;
    }

    console.log(`ConnectWebSocket: Attempting to connect to ${wsUrl}...`);

    try {
      // Close any existing connection first
      if (wsRef.current) {
        console.log('ConnectWebSocket: Closing existing connection before creating new one');
        wsRef.current.close();
        wsRef.current = null;
      }

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected successfully');
        setSession(prev => ({ 
          ...prev, 
          isConnected: true, 
          isReconnecting: false 
        }));
        reconnectAttemptsRef.current = 0;
        lastServerMessageTimeRef.current = Date.now();
        
        // Enable diagnostic mode
        sendDiagnosticMode();
        
        // Start keep-alive mechanism
        if (keepAliveIntervalRef.current) {
          clearInterval(keepAliveIntervalRef.current);
        }
        keepAliveIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
            
            // Check if we haven't received a message in too long
            const timeSinceLastMessage = Date.now() - lastServerMessageTimeRef.current;
            if (timeSinceLastMessage > 30000) { // 30 seconds
              console.warn('No server message received in 30 seconds, connection may be stale');
              // Optionally trigger reconnection here
            }
          }
        }, 10000); // Send ping every 10 seconds

        // Flush any queued audio
        flushAudioQueue();
      };

      ws.onmessage = async (event) => {
        lastServerMessageTimeRef.current = Date.now();
        
        if (typeof event.data === 'string') {
          try {
            const data = JSON.parse(event.data);
            await handleWebSocketMessage(data);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        }
      };

      ws.onerror = (error) => {
        // Add defensive error handling for potentially undefined error objects
        const errorMessage = error && (error as any).message ? (error as any).message : 'Unknown WebSocket error';
        console.error('WebSocket error:', errorMessage);
        setSession(prev => ({ ...prev, isConnected: false }));
        if (onError) {
          onError(`Connection failed: ${errorMessage}`);
        }
      };

      ws.onclose = (event) => {
        const code = event.code || 0;
        const reason = event.reason || 'No reason provided';
        console.log(`WebSocket closed: code=${code}, reason=${reason}`);
        setSession(prev => ({ 
          ...prev, 
          isConnected: false, 
          isRecording: false,
          isProcessing: false 
        }));
        
        // Clear keep-alive
        if (keepAliveIntervalRef.current) {
          clearInterval(keepAliveIntervalRef.current);
          keepAliveIntervalRef.current = null;
        }

        // Attempt reconnection if not a normal closure and not cleaning up
        if (code !== 1000 && !isCleaningUpRef.current && !sessionEndedRef.current) {
          attemptReconnection();
        }
      };

    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
      setSession(prev => ({ ...prev, isConnected: false }));
      if (onError) {
        onError(`Connection failed: ${error}`);
      }
    }
  }, [wsUrl, onError, flushAudioQueue]);

  const attemptReconnection = useCallback(() => {
    if (reconnectAttemptsRef.current >= maxReconnectAttempts || isCleaningUpRef.current || sessionEndedRef.current) {
      console.log('Max reconnection attempts reached or cleanup in progress');
      return;
    }

    reconnectAttemptsRef.current++;
    const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current - 1), 10000); // Exponential backoff, max 10s
    
    console.log(`Attempting reconnection ${reconnectAttemptsRef.current}/${maxReconnectAttempts} in ${delay}ms`);
    setSession(prev => ({ ...prev, isReconnecting: true }));

    reconnectTimeoutRef.current = setTimeout(() => {
      connectWebSocket();
    }, delay);
  }, [connectWebSocket]);

  async function handleWebSocketMessage(data: any) {
    switch (data.type) {
      case 'transcript':
        if (data.partial) {
          setTranscript(prev => ({ ...prev, partial: data.text || '' }));
        } else if (data.final) {
          setTranscript(prev => ({
            ...prev,
            final: [...prev.final, data.text || data.final],
            partial: ''
          }));
        }
        break;

      case 'ai_response':
        if (data.final) {
          setTranscript(prev => ({ 
            ...prev, 
            aiResponse: data.text || '', 
            isAiResponding: false 
          }));
          setSession(prev => ({ ...prev, isProcessing: false }));
        } else {
          // Streaming AI response
          setTranscript(prev => ({ 
            ...prev, 
            aiResponse: prev.aiResponse + (data.text || ''), 
            isAiResponding: true 
          }));
        }
        break;

      case 'audio_chunk':
        if (data.audio_data && ttsAudioPlayerRef.current) {
          try {
            // Validate base64 format before processing
            if (!isValidBase64(data.audio_data)) {
              console.warn('Received invalid base64 audio data, skipping playback');
              break;
            }
            
            // Decode base64 audio data
            const audioData = atob(data.audio_data);
            const audioArray = new Uint8Array(audioData.length);
            for (let i = 0; i < audioData.length; i++) {
              audioArray[i] = audioData.charCodeAt(i);
            }
            
            // Play the audio chunk
            await ttsAudioPlayerRef.current.playAudioData(audioArray.buffer);
          } catch (error) {
            console.error('Error playing audio chunk:', error);
          }
        }
        break;

      case 'tts_start':
        setSession(prev => ({ ...prev, isSpeaking: true }));
        break;

      case 'tts_complete':
        setSession(prev => ({ ...prev, isSpeaking: false }));
        break;

      case 'error':
        const errorMessage = data.message || 'Unknown server error';
        console.error('Server error:', errorMessage);
        if (onError) {
          onError(errorMessage);
        }
        setSession(prev => ({ ...prev, isProcessing: false }));
        break;

      case 'mute_status':
        setSession(prev => ({ ...prev, isMuted: data.muted }));
        break;

      case 'pong':
        // Keep-alive response received
        break;

      case 'status':
        // Session status update from server
        console.log('Session status:', data);
        if (data.ready) {
          console.log('Server confirmed session is ready');
        }
        break;

      case 'control':
        if (data.action === 'session_ended') {
          console.log('Server confirmed session end');
          handleEndSession(true);
        }
        break;

      case 'vad_status':
        // Handle VAD status updates
        if (data.status === 'speech_started') {
          setIsUserSpeaking(true);
        } else if (data.status === 'speech_ended') {
          setIsUserSpeaking(false);
        }
        break;

      case 'config_response':
        // Handle config response
        console.log('Server configuration received:', data);
        break;

      case 'eos_ack':
        // Handle end-of-stream acknowledgment
        console.log('End of stream acknowledged by server');
        break;

      default:
        console.log('Unknown message type:', data.type, data);
    }
  }

  const startRecording = useCallback(async () => {
    if (!session.isConnected || session.isRecording || isCleaningUpRef.current) {
      console.log('Cannot start recording: not connected, already recording, or cleaning up');
      return;
    }

    try {
      console.log('Starting audio recording...');
      
      // Log audio format support to help diagnose issues
      logAudioFormatSupport();
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });

      streamRef.current = stream;

      // Create AudioContext for raw PCM processing
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000
      });
      audioContextRef.current = audioContext;

      // Create audio source
      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;

      // Create ScriptProcessorNode for raw PCM extraction
      const bufferSize = 4096; // Larger buffer for efficiency
      const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
      
      processor.onaudioprocess = (event) => {
        if (!session.isConnected || session.isMuted || isCleaningUpRef.current) {
          return;
        }

        const inputBuffer = event.inputBuffer;
        const inputData = inputBuffer.getChannelData(0); // Get mono channel
        
        // Convert Float32Array to Int16Array (16-bit PCM)
        const pcmData = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          // Convert from [-1, 1] to [-32768, 32767]
          const sample = Math.max(-1, Math.min(1, inputData[i]));
          pcmData[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        }

        // Send raw PCM data directly
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          // Validate data before sending
          if (pcmData.length > 0 && !isAllZeros(pcmData)) {
            // Debug log on first few packets
            if (audioSentCounterRef.current < 5) {
              console.log(`Sending audio chunk #${audioSentCounterRef.current+1}: ${pcmData.length} samples, 16-bit PCM`);
              audioSentCounterRef.current++;
            } 
            wsRef.current.send(pcmData.buffer);
          } else {
            console.warn('Skipping empty audio buffer');
          }
        } else {
          if (audioQueueRef.current.length < 20) { // Limit queue size
            const blob = new Blob([pcmData.buffer], { type: 'audio/raw' });
            audioQueueRef.current.push(blob);
          }
        }

        // Update audio level for UI
        let sum = 0;
        for (let i = 0; i < inputData.length; i++) {
          sum += inputData[i] * inputData[i];
        }
        const rms = Math.sqrt(sum / inputData.length);
        setAudioLevel(rms * 100);
      };

      // Connect the audio processing chain
      source.connect(processor);
      processor.connect(audioContext.destination);

      // Initialize VAD for client-side speech detection
      try {
        const vad = await MicVAD.new({
          stream,
          positiveSpeechThreshold: 0.8, // Increase threshold to reduce false positives (default is 0.5)
          negativeSpeechThreshold: 0.3, // Set slightly lower negative threshold
          minSpeechFrames: 8, // Require more frames to consider valid speech (default is 3)
          redemptionFrames: 25, // Increase redemption frames to prevent quick cutoffs
          onSpeechStart: () => {
            console.log('VAD: Speech started');
            setIsUserSpeaking(true);
          },
          onSpeechEnd: () => {
            console.log('VAD: Speech ended');
            setIsUserSpeaking(false);
            // Send end-of-speech signal to backend
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({ type: 'eos' }));
            }
          },
          onVADMisfire: () => {
            console.log('VAD: Misfire detected');
            setIsUserSpeaking(false);
          },
        });
        vadRef.current = vad;
        vad.start();
      } catch (vadError) {
        console.warn('VAD initialization failed:', vadError);
        // Continue without VAD
      }

      // Initialize TTS audio player
      if (!ttsAudioPlayerRef.current) {
        ttsAudioPlayerRef.current = new AudioPlayer();
      }

      setSession(prev => ({ ...prev, isRecording: true }));
      console.log('Recording started successfully');

    } catch (error) {
      console.error('Error starting recording:', error);
      if (onError) {
        onError(`Failed to start recording: ${error}`);
      }
    }
  }, [session.isConnected, session.isRecording, session.isMuted, onError]);

  const initializeAudio = useCallback(async () => {
    // This function is no longer needed as audio initialization
    // is now handled directly in startRecording with raw PCM processing
    console.log('initializeAudio called - now handled in startRecording');
    return true;
  }, []);

  const sendWebSocketMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
        console.warn("sendWebSocketMessage: WebSocket not open. State:", wsRef.current?.readyState);
    }
  }, []);

  const toggleMute = useCallback(() => {
    const newMutedState = !session.isMuted;
    setSession(prev => ({...prev, isMuted: newMutedState})); // Optimistic update
    sendWebSocketMessage({ type: 'mute', muted: newMutedState });
  }, [session.isMuted, sendWebSocketMessage]);

  const endSession = useCallback(() => {
    console.log('User initiated endSession...');
    setSession(prev => ({ ...prev, sessionEnded: true })); // Mark session ended by user
    sendWebSocketMessage({ type: 'end_session' });
    // Delay cleanup to allow message to send
    setTimeout(() => handleEndSession(false), 100); // Pass false: user initiated
  }, [sendWebSocketMessage]);

  const startNewSession = useCallback(async () => {
    console.log('🚀 Starting new session attempt...');
    if (isCleaningUpRef.current) {
      console.log('StartNewSession: Cleanup in progress. Will retry in 550ms.');
      setTimeout(startNewSession, 550); // Retry after cleanup timeout
      return;
    }
    
    // Check if we need to clean up existing connections
    const hasActiveWebSocket = wsRef.current && wsRef.current.readyState === WebSocket.OPEN;
    const hasActiveStream = streamRef.current && streamRef.current.active;
    
    if (hasActiveWebSocket || hasActiveStream) {
        console.log("StartNewSession: Active connections detected, cleaning up first.");
        handleEndSession(false); // Force cleanup
        // Wait a moment for cleanup to complete if it was just triggered
        await new Promise(resolve => setTimeout(resolve, 600));
         if (isCleaningUpRef.current) {
            console.log('StartNewSession: Waiting for final cleanup flag reset.');
            await new Promise(resolve => setTimeout(resolve, 550));
        }
    } else {
        console.log("StartNewSession: No active connections, proceeding with new session.");
    }
    
    // Reset session state for new session
    setSession(prev => ({ 
      ...prev, 
      sessionEnded: false, 
      isConnected: false, 
      isReconnecting: false,
      isProcessing: false, 
      isSpeaking: false, 
      isRecording: false 
    }));
    setTranscript({ partial: '', final: [], aiResponse: '', isAiResponding: false });
    setIsUserSpeaking(false);
    
    console.log('StartNewSession: Initiating WebSocket connection...');
    connectWebSocket(); // This will attempt to establish a new WebSocket connection
    // initializeAudio will be called via useEffect when session.isConnected becomes true
  }, [connectWebSocket]); // Removed session.isConnected to prevent circular dependency

  useEffect(() => {
    // Initial connection attempt - only run once on mount
    // Use a timeout to allow the component to fully mount before connecting
    const connectionTimer = setTimeout(() => {
      if (!session.isConnected && !session.sessionEnded && !wsRef.current && !isCleaningUpRef.current) {
        console.log("🔗 Initial WebSocket connection attempt after component mount");
        connectWebSocket();
      }
    }, 100); // Small delay to ensure component is fully mounted

    // Cleanup on component unmount
    return () => {
      console.log("VoiceAgent component unmounting. Performing cleanup.");
      clearTimeout(connectionTimer); // Cancel pending connection attempt
      
      // Clear reconnection timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      
      // Clear audio queue
      audioQueueRef.current = [];
      
      handleEndSession(false); // User (component unmount) initiated
      if (keepAliveIntervalRef.current) {
        clearInterval(keepAliveIntervalRef.current);
        keepAliveIntervalRef.current = null;
      }
    };
  }, []); // Empty dependency array - only run on mount/unmount

  useEffect(() => {
    // Attempt to start recording once connected and if not already recording/cleaning up
    if (session.isConnected && !session.isRecording && !streamRef.current && !vadRef.current && !isCleaningUpRef.current) {
      console.log("🔊 useEffect: Connection active, attempting to start recording.");
      startRecording().then(() => {
        console.log(`🎯 Recording started successfully`);
      }).catch(error => {
        console.error('Failed to start recording:', error);
      });
    }
  }, [session.isConnected, session.isRecording, startRecording]); // Depend on isConnected and isRecording

  // Get activity indicator based on multiple states
  const getActivityState = () => {
    if (session.isSpeaking) return 'speaking'; // AI is speaking
    if (session.isProcessing) return 'processing'; // AI is thinking/STT finalizing
    if (isUserSpeaking && !session.isMuted) return 'listening'; // User is speaking (client VAD)
    if (session.isRecording && !session.isMuted) return 'recording'; // Mic is open, VAD might not be active yet
    return 'idle';
  };
  const activityState = getActivityState();

  // Helper function to validate base64 string
  const isValidBase64 = (str: string): boolean => {
    try {
      return btoa(atob(str)) === str;
    } catch (err) {
      return false;
    }
  };

  // Track number of audio packets sent for logging
  const audioSentCounterRef = useRef(0);
  
  // Helper function to log audio format support
  const logAudioFormatSupport = () => {
    const formats = [
      'audio/webm', 
      'audio/webm;codecs=opus', 
      'audio/ogg',
      'audio/ogg;codecs=opus',
      'audio/wav',
      'audio/mp4',
      'audio/mpeg'
    ];
    
    console.log('=== Audio Format Support ===');
    formats.forEach(format => {
      let supported = false;
      try {
        supported = MediaRecorder.isTypeSupported(format);
      } catch (e) {
        console.error(`Error checking support for ${format}:`, e);
      }
      console.log(`${format}: ${supported ? '✅' : '❌'}`);
    });
    
    // Log AudioContext information
    try {
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      console.log(`AudioContext: sampleRate=${ctx.sampleRate}, state=${ctx.state}`);
      ctx.close();
    } catch (e) {
      console.error('Error creating AudioContext:', e);
    }
  };
  
  // Helper function to check if array is all zeros
  const isAllZeros = (arr: Int16Array): boolean => {
    for (let i = 0; i < Math.min(arr.length, 100); i++) {
      if (arr[i] !== 0) return false;
    }
    return true;
  };

  const sendDiagnosticMode = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('Enabling diagnostic mode...');
      wsRef.current.send(JSON.stringify({ 
        type: 'config', 
        diagnostics: true,
        verbose_logging: true,
        client_info: {
          userAgent: navigator.userAgent,
          platform: navigator.platform,
          sampleRate: audioContextRef.current?.sampleRate || 'unknown'
        }
      }));
    }
  }, []);

  return (
    <div className="h-screen w-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex flex-col">
      
      <div className="absolute top-4 left-4 right-4 flex justify-between items-center z-10">
        <div className="flex items-center space-x-2 bg-black/30 backdrop-blur-sm rounded-full px-4 py-2">
          <div className={`w-2 h-2 rounded-full ${session.isConnected ? 'bg-green-400' : (session.isReconnecting ? 'bg-yellow-400 animate-pulse' : 'bg-red-400')}`} />
          <span className="text-white/80 text-sm">
            {session.isReconnecting ? 'Reconnecting...' : (session.isConnected ? 'Connected' : 'Disconnected')}
          </span>
        </div>
        {latency > 0 && (
          <div className="bg-black/30 backdrop-blur-sm rounded-full px-4 py-2">
            <span className="text-white/80 text-sm">{latency.toFixed(0)}ms</span>
          </div>
        )}
      </div>

      <div className="flex-1 flex items-center justify-center">
        <div className="relative">
          <div className={`absolute inset-0 rounded-full border-2 transition-all duration-300 ${
            activityState === 'listening' || activityState === 'recording' ? 'border-blue-400 animate-pulse scale-110' :
            activityState === 'processing' ? 'border-yellow-400 animate-pulse scale-105' :
            activityState === 'speaking' ? 'border-purple-400 animate-pulse scale-115' :
            'border-gray-600'
          }`} style={{ width: '200px', height: '200px' }} />
          
          <div 
            className={`absolute inset-0 rounded-full transition-all duration-100 ${
              activityState === 'listening' || activityState === 'recording' ? 'bg-blue-400/20' :
              activityState === 'processing' ? 'bg-yellow-400/20' :
              activityState === 'speaking' ? 'bg-purple-400/20' :
              'bg-gray-600/10'
            }`}
            style={{ width: '200px', height: '200px', transform: `scale(${1 + audioLevel * 0.3})` }}
          />
          
          <div className="relative w-48 h-48 bg-black/40 backdrop-blur-sm rounded-full flex items-center justify-center">
            {session.isMuted ? (
              <MicOff className="w-16 h-16 text-red-400" />
            ) : (
              <Mic className={`w-16 h-16 transition-colors ${
                activityState === 'listening' || activityState === 'recording' ? 'text-blue-400' :
                activityState === 'processing' ? 'text-yellow-400' :
                activityState === 'speaking' ? 'text-purple-400' :
                'text-white/60'
              }`} />
            )}
          </div>
        </div>
      </div>

      <div className="mx-8 mb-32 min-h-[120px] bg-black/30 backdrop-blur-sm rounded-lg p-6 border border-white/10">
        {transcript.partial && (
          <div className="text-blue-400 text-lg mb-2">
            <span className="text-white/60 text-sm">Listening: </span>
            {transcript.partial}
            <span className="animate-pulse">|</span>
          </div>
        )}
        {transcript.final.length > 0 && (
          <div className="text-white text-lg mb-2">
            <span className="text-white/60 text-sm">You: </span>
            {transcript.final[transcript.final.length - 1]}
          </div>
        )}
        {(transcript.aiResponse || transcript.isAiResponding) && (
          <div className="text-purple-400 text-lg">
            <span className="text-white/60 text-sm">Assistant: </span>
            {transcript.aiResponse}
            {transcript.isAiResponding && <span className="animate-pulse">|</span>}
          </div>
        )}
        {!transcript.partial && transcript.final.length === 0 && !transcript.aiResponse && !transcript.isAiResponding && (
          <div className="text-white/60 text-center py-8">
            {session.sessionEnded ? 'Session ended. Click "Start New Session" to begin again.' :
             session.isConnected ? (session.isMuted ? 'Microphone muted - Click to unmute' : 'Start speaking to begin conversation') :
             'Connecting to voice service...'}
          </div>
        )}
      </div>

      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 flex space-x-6">
        {session.sessionEnded ? (
          <button onClick={startNewSession}
            className="px-8 py-3 rounded-full bg-green-600 hover:bg-green-700 text-white flex items-center space-x-2 transition-all shadow-lg shadow-green-600/50">
            <Mic className="w-5 h-5" />
            <span>Start New Session</span>
          </button>
        ) : (
          <>
            <button onClick={toggleMute} disabled={!session.isConnected}
              className={`w-16 h-16 rounded-full flex items-center justify-center transition-all ${
                session.isMuted ? 'bg-red-500 hover:bg-red-600 text-white shadow-lg shadow-red-500/50' 
                                : 'bg-blue-500 hover:bg-blue-600 text-white shadow-lg shadow-blue-500/50'
              } disabled:opacity-50 disabled:cursor-not-allowed`}>
              {session.isMuted ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
            </button>
            <button onClick={endSession} disabled={!session.isConnected || session.isProcessing}
              className="w-16 h-16 rounded-full bg-red-600 hover:bg-red-700 text-white flex items-center justify-center transition-all shadow-lg shadow-red-600/50 disabled:opacity-50 disabled:cursor-not-allowed">
              <Square className="w-6 h-6" />
            </button>
          </>
        )}
      </div>

      <div className="absolute bottom-4 right-4 bg-black/30 backdrop-blur-sm rounded-full px-4 py-2">
        <span className="text-white/80 text-sm capitalize">{activityState}</span>
      </div>

      {session.isReconnecting && (
        <div className="absolute top-0 left-0 right-0 bg-yellow-500 text-black text-center p-2 z-50 flex items-center justify-center">
          <AlertTriangle className="h-5 w-5 mr-2" />
          Attempting to reconnect to server...
        </div>
      )}
      {/* Processing spinner overlay removed, as processing state is part of activityState visual */}
    </div>
  );
}