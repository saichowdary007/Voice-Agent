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
  const keepAliveIntervalRef = useRef<NodeJS.Timeout | null>(null);

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
  const [selectedMimeType, setSelectedMimeType] = useState<string | null>(null);

  // Get WebSocket URL from environment variable
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8080/ws';

  // Add a ref to track sessionEnded state without causing re-renders
  const sessionEndedRef = useRef(false);

  // Update the ref whenever session.sessionEnded changes
  useEffect(() => {
    sessionEndedRef.current = session.sessionEnded;
  }, [session.sessionEnded]);

  /**
   * Centralized resource cleanup function that handles all cleanup tasks
   * Returns a promise that resolves when all cleanup is complete
   */
  async function cleanupResources() {
    if (isCleaningUpRef.current) {
      console.log('Resource cleanup already in progress, skipping duplicate call.');
      // Return existing cleanup promise if one exists
      return;
    }
    
    isCleaningUpRef.current = true;
    console.log('Starting comprehensive resource cleanup...');
    
    const cleanupPromises = [];

    // Clear all timeouts and intervals first to prevent race conditions
    if (keepAliveIntervalRef.current) {
      clearInterval(keepAliveIntervalRef.current);
      keepAliveIntervalRef.current = null;
    }
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Clean up VAD with proper promise handling
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

    // MediaRecorder cleanup with error handling
    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== 'inactive') {
      try { 
        mr.stop(); 
      } catch (e) { 
        const errorMessage = e && (e as any).message ? (e as any).message : 'Unknown error';
        console.warn('Error stopping MediaRecorder during cleanup:', errorMessage); 
      }
    }
    mediaRecorderRef.current = null;

    // Media stream cleanup
    if (streamRef.current) {
      try {
        streamRef.current.getTracks().forEach(track => {
          try {
            track.stop();
          } catch (e) {
            console.warn('Error stopping media track:', e);
          }
        });
      } catch (e) {
        console.warn('Error cleaning up media stream:', e);
      }
      streamRef.current = null;
    }

    // Audio source cleanup
    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch (e) {
        console.warn('Error disconnecting audio source:', e);
      }
      sourceRef.current = null;
    }

    // AudioContext cleanup with proper promise handling
    const ctx = audioContextRef.current;
    if (ctx && ctx.state !== 'closed') {
      try {
        const ctxCleanup = ctx.close();
        cleanupPromises.push(ctxCleanup.catch(e => {
          console.warn('Error closing AudioContext:', e);
        }));
      } catch (e) {
        console.warn('Error initiating AudioContext close:', e);
      }
    }
    audioContextRef.current = null;

    // TTS audio player cleanup
    if (ttsAudioPlayerRef.current) {
      try {
        ttsAudioPlayerRef.current.dispose();
      } catch (e) {
        console.warn('Error disposing TTS audio player:', e);
      }
      ttsAudioPlayerRef.current = null;
    }

    // WebSocket cleanup last (after other resources)
    if (wsRef.current) {
      try {
        if (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING) {
          console.log('Closing WebSocket connection...');
          wsRef.current.close(1000, 'Session ended by client/cleanup');
        }
      } catch (e) {
        console.warn('Error closing WebSocket:', e);
      }
      wsRef.current = null;
    }

    // Reset all state
    try {
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
      audioQueueRef.current = [];
    } catch (e) {
      console.warn('Error resetting state during cleanup:', e);
    }

    // Wait for all cleanup promises to resolve
    try {
      await Promise.all(cleanupPromises);
      console.log('All async cleanup tasks completed successfully');
    } catch (e) {
      console.warn('Some async cleanup tasks failed:', e);
    }

    console.log('Resource cleanup completed');
    
    // Reset cleanup flag with a slight delay to prevent race conditions
    setTimeout(() => {
      isCleaningUpRef.current = false;
      console.log('Cleanup flag reset, ready for new session');
    }, 500);
  }

  /**
   * Ends the current voice session and performs a full cleanup.
   */
  function handleEndSession(backendInitiated = false) {
    console.log(`Ending session... Backend initiated: ${backendInitiated}`);
    cleanupResources().catch(e => {
      console.error('Error during session cleanup:', e);
      // Reset cleanup flag in case of error
      isCleaningUpRef.current = false;
    });
  }

  // Improved flush audio queue with error handling
  const flushAudioQueue = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN && audioQueueRef.current.length > 0) {
      console.log(`Flushing ${audioQueueRef.current.length} queued audio chunks`);
      
      // Take a snapshot of the current queue and then clear it
      const queueToSend = [...audioQueueRef.current];
      audioQueueRef.current = [];
      
      // Send all chunks
      queueToSend.forEach(audioBlob => {
        try {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(audioBlob);
          }
        } catch (e) {
          console.warn('Error sending queued audio chunk:', e);
          // Re-queue failed sends if WebSocket is still open
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            audioQueueRef.current.push(audioBlob);
          }
        }
      });
    }
  }, []);

  // Improved WebSocket connection with robust checks
  const connectWebSocket = useCallback(() => {
    // Enhanced connection checks
    if (isCleaningUpRef.current) {
      console.log('ConnectWebSocket: Cleanup in progress, skipping connection attempt');
      return Promise.resolve(false);
    }
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('ConnectWebSocket: Already connected, skipping');
      return Promise.resolve(true);
    }
    
    if (wsRef.current?.readyState === WebSocket.CONNECTING) {
      console.log('ConnectWebSocket: Connection already in progress, skipping');
      return Promise.resolve(false);
    }

    console.log(`ConnectWebSocket: Attempting to connect to ${wsUrl}...`);

    return new Promise((resolve) => {
      try {
        // Close any existing connection first
        if (wsRef.current) {
          try {
            console.log('ConnectWebSocket: Closing existing connection before creating new one');
            wsRef.current.close();
          } catch (e) {
            console.warn('Error closing existing WebSocket:', e);
          }
          wsRef.current = null;
        }

        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        // Set connection timeout
        const connectionTimeout = setTimeout(() => {
          if (ws.readyState !== WebSocket.OPEN) {
            console.warn('WebSocket connection timeout');
            try {
              ws.close();
            } catch (e) {
              console.warn('Error closing timed-out WebSocket:', e);
            }
            resolve(false);
          }
        }, 10000); // 10 second connection timeout

        ws.onopen = () => {
          clearTimeout(connectionTimeout);
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
          
          // Start keep-alive mechanism with watchdog capability
          if (keepAliveIntervalRef.current) {
            clearInterval(keepAliveIntervalRef.current);
          }
          
          keepAliveIntervalRef.current = setInterval(() => {
            if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
              console.warn('Keep-alive interval running but WebSocket is not open');
              clearInterval(keepAliveIntervalRef.current);
              keepAliveIntervalRef.current = null;
              return;
            }
            
            try {
              wsRef.current.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
              
              // Check if we haven't received a message in too long
              const timeSinceLastMessage = Date.now() - lastServerMessageTimeRef.current;
              const serverTimeout = parseInt(process.env.NEXT_PUBLIC_WS_SERVER_TIMEOUT || '60000', 10);
              
              if (timeSinceLastMessage > serverTimeout) {
                console.warn(`No server message received in ${serverTimeout/1000} seconds, connection may be stale`);
                // Close the stale connection which will trigger reconnection
                wsRef.current.close(4000, 'Server response timeout');
              }
            } catch (e) {
              console.warn('Error in keep-alive mechanism:', e);
            }
          }, parseInt(process.env.NEXT_PUBLIC_WS_PING_INTERVAL || '10000', 10)); // Configurable ping interval

          // Flush any queued audio
          flushAudioQueue();
          resolve(true);
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
          clearTimeout(connectionTimeout);
          // Add defensive error handling for potentially undefined error objects
          const errorMessage = error && (error as any).message ? (error as any).message : 'Unknown WebSocket error';
          console.error('WebSocket error:', errorMessage);
          setSession(prev => ({ ...prev, isConnected: false }));
          if (onError) {
            onError(`Connection failed: ${errorMessage}`);
          }
          resolve(false);
        };

        ws.onclose = (event) => {
          clearTimeout(connectionTimeout);
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
          resolve(false);
        };

      } catch (error) {
        console.error('Error creating WebSocket connection:', error);
        setSession(prev => ({ ...prev, isConnected: false }));
        if (onError) {
          onError(`Connection failed: ${error}`);
        }
        resolve(false);
      }
    });
  }, [wsUrl, onError, flushAudioQueue]);

  // Improved reconnection with exponential backoff
  const attemptReconnection = useCallback(() => {
    if (reconnectAttemptsRef.current >= maxReconnectAttempts || isCleaningUpRef.current || sessionEndedRef.current) {
      console.log('Max reconnection attempts reached or cleanup in progress, giving up');
      return;
    }

    // Clear any existing reconnection timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    reconnectAttemptsRef.current++;
    const baseDelay = parseInt(process.env.NEXT_PUBLIC_WS_RECONNECT_BASE_DELAY || '1000', 10);
    const delay = Math.min(baseDelay * Math.pow(2, reconnectAttemptsRef.current - 1), 30000); // Exponential backoff, max 30s
    
    console.log(`Attempting reconnection ${reconnectAttemptsRef.current}/${maxReconnectAttempts} in ${delay}ms`);
    setSession(prev => ({ ...prev, isReconnecting: true }));

    reconnectTimeoutRef.current = setTimeout(() => {
      connectWebSocket().then(success => {
        if (!success && !isCleaningUpRef.current && !sessionEndedRef.current) {
          // Schedule next reconnection attempt if this one failed
          attemptReconnection();
        }
      });
    }, delay);
  }, [connectWebSocket, maxReconnectAttempts]);

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
    if (!session.isConnected || session.isRecording || isCleaningUpRef.current || !selectedMimeType) {
      console.warn('Cannot start recording: not connected, already recording, cleaning up, or no supported MIME type');
      if (!selectedMimeType) {
          onError?.('Your browser does not support the required audio recording formats.');
      }
      return;
    }

    try {
      console.log(`Starting audio recording with format: ${selectedMimeType}...`);
      
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

      const mediaRecorder = new MediaRecorder(stream, { mimeType: selectedMimeType });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (wsRef.current?.readyState !== WebSocket.OPEN) {
            if(event.data.size > 0) audioQueueRef.current.push(event.data);
            return;
        }

        // Perform format-specific validation
        if (event.data.size > 0) {
            let minSize = 100; // General minimum size
            if (selectedMimeType.includes('webm') && audioQueueRef.current.length === 0) {
                minSize = 20; // WebM header can be small
            } else if (selectedMimeType.includes('wav')) {
                minSize = 44; // WAV header is at least 44 bytes
            }

            if (event.data.size < minSize) {
                console.warn(`Skipping very small audio chunk (size: ${event.data.size}, format: ${selectedMimeType}).`);
                return;
            }
            console.log(`Sending audio chunk: ${event.data.size} bytes, type: ${selectedMimeType}`);
            wsRef.current.send(event.data);
        }
      };
      
      mediaRecorder.onstart = () => {
          setSession(prev => ({ ...prev, isRecording: true }));
          console.log('Recording started successfully');
      };

      mediaRecorder.onerror = (event) => {
          console.error('MediaRecorder error:', event);
          const error = (event as any).error;
          onError?.(`Recording error: ${error.name} - ${error.message}`);
          handleEndSession(false);
      };
      
      mediaRecorder.start(1000); // Send data every 1000ms

      // Create AudioContext for VAD and audio level monitoring
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000
      });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;
      
      // VAD initialization
      try {
        const vad = await MicVAD.new({
          stream,
          positiveSpeechThreshold: 0.8,
          negativeSpeechThreshold: 0.3,
          minSpeechFrames: 8,
          redemptionFrames: 25,
          onSpeechStart: () => {
            console.log('VAD: Speech started');
            setIsUserSpeaking(true);
          },
          onSpeechEnd: () => {
            console.log('VAD: Speech ended');
            setIsUserSpeaking(false);
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
      }

      // Initialize TTS audio player
      if (!ttsAudioPlayerRef.current) {
        ttsAudioPlayerRef.current = new AudioPlayer();
      }

    } catch (error) {
      console.error('Error starting recording:', error);
      if (onError) {
        onError(`Failed to start recording: ${error}`);
      }
    }
  }, [session.isConnected, session.isRecording, onError, selectedMimeType]);

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
  }, [connectWebSocket]);

  const getBestSupportedMimeType = useCallback((): string | null => {
    const mimeTypes = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/ogg',
      'audio/wav',
    ];
    console.log('Checking for best supported MIME type...');
    for (const mimeType of mimeTypes) {
      if (MediaRecorder.isTypeSupported(mimeType)) {
        console.log(`Found supported MIME type: ${mimeType}`);
        return mimeType;
      }
    }
    console.warn('No supported MIME type found for MediaRecorder');
    onError?.('Your browser does not support the required audio recording formats.');
    return null;
  }, [onError]);

  useEffect(() => {
    const connectionTimer = setTimeout(() => {
      if (!session.isConnected && !session.sessionEnded && !wsRef.current && !isCleaningUpRef.current) {
        console.log("🔗 Initial WebSocket connection attempt after component mount");
        setSelectedMimeType(getBestSupportedMimeType());
        connectWebSocket();
      }
    }, 100);

    return () => {
      console.log("VoiceAgent component unmounting. Performing cleanup.");
      clearTimeout(connectionTimer);
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      
      audioQueueRef.current = [];
      
      handleEndSession(false);
      if (keepAliveIntervalRef.current) {
        clearInterval(keepAliveIntervalRef.current);
        keepAliveIntervalRef.current = null;
      }
    };
  }, [connectWebSocket, getBestSupportedMimeType]);

  useEffect(() => {
    if (session.isConnected && !session.isRecording && !streamRef.current && !vadRef.current && !isCleaningUpRef.current) {
      console.log("🔊 useEffect: Connection active, attempting to start recording.");
      startRecording().catch(error => {
        console.error('Failed to start recording:', error);
      });
    }
  }, [session.isConnected, session.isRecording, startRecording]);

  const getActivityState = () => {
    if (session.isSpeaking) return 'speaking';
    if (session.isProcessing) return 'processing';
    if (isUserSpeaking && !session.isMuted) return 'listening';
    if (session.isRecording && !session.isMuted) return 'recording';
    return 'idle';
  };
  const activityState = getActivityState();

  const isValidBase64 = (str: string): boolean => {
    try {
      return btoa(atob(str)) === str;
    } catch (err) {
      return false;
    }
  };
  
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
    
    console.log('=== Browser Audio Format Support ===');
    formats.forEach(format => {
      let supported = false;
      try {
        supported = MediaRecorder.isTypeSupported(format);
      } catch (e) {
        console.error(`Error checking support for ${format}:`, e);
      }
      console.log(`${format}: ${supported ? '✅ Supported' : '❌ Not Supported'}`);
    });
    
    try {
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      console.log(`AudioContext: sampleRate=${ctx.sampleRate}, state=${ctx.state}`);
      ctx.close();
    } catch (e) {
      console.error('Error creating AudioContext:', e);
    }
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
          sampleRate: audioContextRef.current?.sampleRate || 'unknown',
          mimeType: selectedMimeType || 'N/A'
        }
      }));
    }
  }, [selectedMimeType]);

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
            Press the button to start a new session
          </div>
        )}
      </div>

      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center space-y-4">
        {session.sessionEnded ? (
          <button
            onClick={startNewSession}
            className="px-8 py-4 bg-blue-600 text-white rounded-full shadow-lg hover:bg-blue-700 transition-all transform hover:scale-105"
          >
            Start New Session
          </button>
        ) : (
          <div className="flex items-center space-x-8">
            <button
              onClick={toggleMute}
              className={`p-4 rounded-full transition-colors ${session.isMuted ? 'bg-red-500/80' : 'bg-gray-700/60 hover:bg-gray-600'}`}
            >
              {session.isMuted ? <MicOff className="w-6 h-6 text-white" /> : <Mic className="w-6 h-6 text-white" />}
            </button>
            <div className="text-white/80 text-sm">
              {activityState.charAt(0).toUpperCase() + activityState.slice(1)}
            </div>
            <button
              onClick={endSession}
              className="p-4 rounded-full bg-red-700/80 hover:bg-red-600 transition-colors"
            >
              <Square className="w-6 h-6 text-white" />
            </button>
          </div>
        )}
        {session.isReconnecting && (
          <div className="flex items-center space-x-2 text-yellow-400">
            <AlertTriangle className="w-4 h-4" />
            <span>Connection lost, attempting to reconnect...</span>
          </div>
        )}
      </div>
    </div>
  );
}