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

  // Initialize audio recording
  const initializeAudio = useCallback(async () => {
    try {
      console.log('Requesting microphone access...');
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });

      console.log('Microphone access granted:', stream.getAudioTracks());
      streamRef.current = stream;

      // Initialize MediaRecorder
      const preferredMimeTypes = [
        'audio/ogg;codecs=opus', // TASK 2.a: Ensure this is prioritized
        'audio/webm;codecs=opus', 
        'audio/webm',
      ];

      let selectedMimeType = '';
      for (const mimeType of preferredMimeTypes) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          selectedMimeType = mimeType;
          break;
        }
      }

      if (!selectedMimeType) {
        console.error('No supported MIME type found for MediaRecorder. Supported types checked:', preferredMimeTypes);
        onError?.('MediaRecorder: No supported audio format found. Please try a different browser or check microphone settings.');
        // Optional: Clean up stream tracks if we can't proceed
        stream.getTracks().forEach(track => track.stop());
        streamRef.current = null; 
        return; // Stop further initialization
      }
      
      console.log(`Using MediaRecorder MIME type: ${selectedMimeType}`);
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: selectedMimeType,
        audioBitsPerSecond: 64000, // This might need adjustment based on codec
      });
      
      console.log('MediaRecorder initialized with MIME type:', selectedMimeType);
      mediaRecorderRef.current = mediaRecorder;

      // Audio chunk buffering to reduce FFmpeg overhead on small chunks
      let audioChunkBuffer: Blob[] = [];
      let bufferTimer: number | null = null;
      const BUFFER_TIME_MS = 200; // Buffer chunks for 200ms
      const MIN_CHUNK_SIZE = 1000; // Minimum chunk size before sending

      const flushBuffer = () => {
        if (audioChunkBuffer.length > 0) {
          const combinedBlob = new Blob(audioChunkBuffer, { type: selectedMimeType });
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(combinedBlob);
          }
          audioChunkBuffer = [];
        }
        bufferTimer = null;
      };

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
          // Add chunk to buffer
          audioChunkBuffer.push(event.data);
          
          // Calculate total buffered size
          const totalSize = audioChunkBuffer.reduce((sum, blob) => sum + blob.size, 0);
          
          // Send immediately if we have enough data or if this is the final chunk (when recording stops)
          if (totalSize >= MIN_CHUNK_SIZE || mediaRecorderRef.current?.state === 'inactive') {
            // Cancel pending timer
            if (bufferTimer) {
              window.clearTimeout(bufferTimer);
              bufferTimer = null;
            }
            flushBuffer();
          } else {
            // Set timer to flush buffer after delay if not already set
            if (!bufferTimer) {
              bufferTimer = window.setTimeout(flushBuffer, BUFFER_TIME_MS);
            }
          }
        }
      };

      mediaRecorderRef.current.onstart = () => {
        console.log('MediaRecorder started');
      };

      mediaRecorderRef.current.onstop = () => {
        console.log('MediaRecorder stopped. Sending EOS to backend.');
        
        // Flush any remaining buffered audio
        if (bufferTimer) {
          window.clearTimeout(bufferTimer);
          bufferTimer = null;
        }
        flushBuffer();
        
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: "eos" }));
        }
      };
      
      mediaRecorderRef.current.onerror = (event) => {
        console.error('MediaRecorder error:', event);
        onError?.('MediaRecorder error. Please check microphone.');
      };

      // Initialize VAD
      console.log('Initializing VAD...');
      if (vadRef.current) { // Destroy any existing VAD instance
        console.log('Destroying existing VAD instance...');
        vadRef.current.destroy();
        vadRef.current = null;
      }

      let myVad: MicVAD | null = null; // Explicitly define type
      try {
        myVad = await MicVAD.new({
          stream: streamRef.current,
          onSpeechStart: () => {
            console.log("VAD: Speech started");
            if (mediaRecorderRef.current?.state === 'inactive') {
              mediaRecorderRef.current.start(120); // Use consistent 120ms timeslice
            }
          },
          onSpeechEnd: (audio: Float32Array) => {
            // audio is a Float32Array of PCM data from the speech segment, can be used for local processing if needed
            console.log('VAD EOS detected by Silero VAD.'); // Clarified log
            if (mediaRecorderRef.current?.state === 'recording') {
              console.log('VAD EOS: Stopping MediaRecorder.');
              mediaRecorderRef.current.stop(); // Triggers ondataavailable (final) and then onstop
            }
            // EOS message is now sent in mediaRecorder.onstop
            // setSession(prev => ({ ...prev, isRecording: false })); // Moved to onstop for consistency
          },
          minSpeechFrames: 3,
          positiveSpeechThreshold: 0.8,
          negativeSpeechThreshold: 0.4,
          minSilenceFrames: 10, // ~100ms of silence - THIS IS A VALID OPTION
          redemptionFrames: 30,
          onVADMisfire: () => {
            console.warn("VAD Misfire detected");
          },
        } as any); // Type assertion to bypass strict type checking for VAD options
      } catch (vadCreationError) {
        console.error('Critical error during MicVAD.new():', vadCreationError);
        let errorMessage = 'Failed to initialize Voice Activity Detection.';
        if (vadCreationError instanceof Error) {
          errorMessage += ` Details: ${vadCreationError.message}`;
          if (vadCreationError.message.toLowerCase().includes('onnx') || vadCreationError.message.toLowerCase().includes('runtime')) {
            errorMessage += ' This might be due to ONNX runtime issues.';
          }
          if (vadCreationError.message.toLowerCase().includes('model') || vadCreationError.message.toLowerCase().includes('silero')) {
            errorMessage += ' This might be due to VAD model loading issues.';
          }
        }
        onError?.(errorMessage);
        return; // Exit initializeAudio
      }

      if (!myVad) {
        console.error('MicVAD.new() completed but returned a null instance. Cannot start VAD.');
        onError?.('Failed to initialize Voice Activity Detection: Null instance returned.');
        return; // Exit initializeAudio
      }

      vadRef.current = myVad;
      
      if (vadRef.current) {
        try {
          vadRef.current.start();
          console.log('VAD initialized and successfully started.'); // Moved success log here
        } catch (vadStartError) {
          console.error('Error when calling vadRef.current.start():', vadStartError);
          onError?.('Failed to start Voice Activity Detection after initialization.');
          if (vadRef.current) { // Attempt to clean up if start failed
            vadRef.current.destroy();
            vadRef.current = null;
          }
          return; // Exit initializeAudio if VAD start fails
        }
      } else {
        // This case should ideally not be hit if !myVad check above worked.
        console.error('VAD instance (vadRef.current) is null just before attempting to start. This indicates an unexpected issue.');
        onError?.('Internal error: VAD instance became null unexpectedly before start.');
        return; // Exit initializeAudio
      }

      // Set up audio level monitoring
      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      
      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;
      
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      
      const updateAudioLevel = () => {
        analyser.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
        setAudioLevel(average / 255);
        requestAnimationFrame(updateAudioLevel);
      };
      
      updateAudioLevel();

      // Start recording in 120ms chunks
      console.log('Starting MediaRecorder...');
      mediaRecorder.start(120);

    } catch (error) {
      console.error('Failed to initialize audio:', error);
      onError?.('Microphone access denied or not available');
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
      // Stop recording
      if (mediaRecorderRef.current?.state !== 'inactive') {
        mediaRecorderRef.current?.stop();
      }

      // Close audio resources with proper checks
      streamRef.current?.getTracks().forEach(track => track.stop());
      
      if (sourceRef.current) {
        try {
          sourceRef.current.disconnect();
        } catch (e) {
          // Already disconnected, ignore error
        }
        sourceRef.current = null;
      }
      
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(console.error);
        audioContextRef.current = null;
      }
      
      // Clean up any TTS audio player
      if (ttsAudioPlayerRef.current) {
        ttsAudioPlayerRef.current.cleanup();
        ttsAudioPlayerRef.current = null;
      }

      // Close WebSocket
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
      wsRef.current = null;
      
      // Clear stream reference
      streamRef.current = null;

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
      
    } catch (error) {
      console.error('Error during cleanup:', error);
    } finally {
      // Reset cleanup flag after a delay to allow for any pending operations
      setTimeout(() => {
        isCleaningUpRef.current = false;
      }, 100);
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