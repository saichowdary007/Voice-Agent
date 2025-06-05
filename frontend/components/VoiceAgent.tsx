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

  const [transcript, setTranscript] = useState<TranscriptState>({
    partial: '',
    final: [],
    aiResponse: '',
    isAiResponding: false,
  });

  const [audioLevel, setAudioLevel] = useState(0);
  const [latency, setLatency] = useState<number>(0);
  const [isUserSpeaking, setIsUserSpeaking] = useState(false); // For client-side VAD indication

  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';

  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN || isCleaningUpRef.current) {
      console.log('ConnectWebSocket: Already connected or cleaning up, skipping.');
      return;
    }
    console.log('ConnectWebSocket: Attempting to connect...');

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      lastServerMessageTimeRef.current = Date.now();

      ws.onopen = () => {
        console.log('WebSocket connected');
        setSession(prev => ({ ...prev, isConnected: true, sessionEnded: false, isReconnecting: false }));
        lastServerMessageTimeRef.current = Date.now();
        
        if (keepAliveIntervalRef.current) clearInterval(keepAliveIntervalRef.current);
        keepAliveIntervalRef.current = setInterval(() => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            const now = Date.now();
            if (now - lastServerMessageTimeRef.current > 30000) { 
              console.warn('No server message for over 30 seconds. Closing connection due to timeout.');
              wsRef.current.close(1000, "Keep-alive timeout"); 
            } else {
              wsRef.current.send(JSON.stringify({ type: "ping", timestamp: now }));
            }
          } else {
            console.log("Keep-alive: WebSocket not open, clearing interval.");
            if(keepAliveIntervalRef.current) clearInterval(keepAliveIntervalRef.current);
          }
        }, 15000);
      };

      ws.onmessage = async (event) => {
        lastServerMessageTimeRef.current = Date.now();
        setSession(prev => ({ ...prev, isReconnecting: false }));
        try {
          let data = JSON.parse(event.data as string);

          // ---- Back‑compat shim: older backend packets lack a `type` field ----
          if (!('type' in data) && 'status' in data) {
            // Treat legacy `{status:"processing"}`‑style payloads as modern `status` packets
            data = { type: 'status', ...data };
          }
          if (!data || typeof data !== 'object') {
            console.error('Invalid message format - not an object:', data);
            return;
          }
          if (!('type' in data)) {
            console.error('Message missing type field even after shim:', data);
            return;
          }
          await handleWebSocketMessage(data);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e, 'Raw data:', event.data);
        }
      };

      ws.onclose = (event: CloseEvent) => {
        console.log(`WebSocket disconnected. Code: ${event.code}, Reason: ${event.reason}`);
        setSession(prev => ({ 
          ...prev, 
          isConnected: false, 
          isRecording: false, 
          isProcessing: false, 
          isSpeaking: false, // AI speaking
          // sessionEnded: event.code === 1000 ? prev.sessionEnded : true // Keep sessionEnded if closed gracefully by user
        }));
        if (event.code !== 1000 && !session.sessionEnded) { // Avoid error if session was ended by user
          console.warn(`WebSocket closed unexpectedly. Code: ${event.code}. Reason: ${event.reason}`);
          onError?.(`Connection lost (Code: ${event.code}). Please try again.`);
        }
        if (keepAliveIntervalRef.current) {
          clearInterval(keepAliveIntervalRef.current);
          keepAliveIntervalRef.current = null;
        }
      };

      ws.onerror = (errorEvent) => {
        const error = errorEvent instanceof ErrorEvent ? errorEvent.message : 'Unknown WebSocket error';
        console.error('WebSocket error:', error, errorEvent);
        onError?.('WebSocket connection failed. Please check your connection and the server.');
        // Ensure state reflects disconnection
        setSession(prev => ({ ...prev, isConnected: false, isReconnecting: false }));
      };

    } catch (error) {
      console.error('Failed to initiate WebSocket connection:', error);
      onError?.('Failed to connect to the voice service.');
    }
  }, [wsUrl, onError, session.sessionEnded]);

async function handleWebSocketMessage(data: any) {
  switch (data.type) {
    case 'status':
      setSession(prev => ({ ...prev, sessionId: data.session_id, isProcessing: data.status === 'processing' }));
      if (data.config) console.log('Received backend config:', data.config);
      if (data.status === 'processing') setTranscript(prev => ({ ...prev, partial: '', aiResponse: '' }));
      break;
    case 'transcript':
      if (data.partial) setTranscript(prev => ({ ...prev, partial: data.partial }));
      if (data.final) {
        setTranscript(prev => ({ ...prev, final: [...prev.final, data.final], partial: '' }));
        setSession(prev => ({ ...prev, isProcessing: true }));
      }
      break;
    case 'ai_response':
      if (data.complete) {
        setTranscript(prev => ({ ...prev, isAiResponding: false }));
        setSession(prev => ({ ...prev, isProcessing: false }));
      } else if (data.token) {
        setTranscript(prev => ({ ...prev, aiResponse: prev.aiResponse + data.token, isAiResponding: true }));
      }
      break;
    case 'tts_start':
      setSession(prev => ({ ...prev, isSpeaking: true }));
      break;
    case 'audio_chunk':
      if (data.audio_data) await playTTSAudio(data.audio_data);
      break;
    case 'tts_complete':
      setSession(prev => ({ ...prev, isSpeaking: false }));
      break;
    case 'stop_audio':
      console.log('Received stop_audio from backend');
      ttsAudioPlayerRef.current?.stop();
      setSession(prev => ({ ...prev, isSpeaking: false }));
      break;
    case 'mute_status':
      setSession(prev => ({ ...prev, isMuted: data.muted }));
      break;
    case 'session_ended':
      console.log('Received session_ended from backend.');
      if (!isCleaningUpRef.current) handleEndSession(true);
      break;
    case 'pong':
      lastServerMessageTimeRef.current = Date.now();
      if (data.timestamp) {
        const rtt = Date.now() - data.timestamp;
        setLatency(rtt);
        console.log(`Pong received with ${rtt}ms latency`);
      } else {
        console.log('Pong received from server (no timestamp).');
      }
      break;
    case 'error':
      console.error('Server error message:', data.message);
      if (data.message?.includes('Audio processing failed')) {
        console.warn('Backend audio processing error. Potential client‑side audio format issue or server‑side decoding problem.');
      }
      onError?.(data.message);
      break;
    default:
      console.log('Unknown message type from server:', data.type, data);
  }
}

  const initializeAudio = useCallback(async () => {
    if (isCleaningUpRef.current) {
      console.log('InitializeAudio: Cleanup in progress, cannot initialize audio yet.');
      return false; // Indicate failure or inability to init
    }
    if (streamRef.current || vadRef.current) {
      console.log("InitializeAudio: Audio already initialized.");
      return true; // Already initialized
    }

    try {
      console.log('🎤 Starting microphone access request...');
      
      // Check if getUserMedia is available
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('getUserMedia not supported in this browser');
      }
      
      console.log('🔍 Checking available audio devices...');
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter(device => device.kind === 'audioinput');
        console.log(`Found ${audioInputs.length} audio input devices:`, audioInputs.map(d => d.label || 'Unnamed device'));
      } catch (e) {
        console.warn('Could not enumerate devices:', e);
      }

      console.log('📋 Requesting microphone access with constraints...');
      const constraints: MediaStreamConstraints = {
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        },
        video: false
      };
      console.log('Constraints:', constraints);
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      
      const track = stream.getAudioTracks()[0];
      if (!track) {
        throw new Error('No audio track found in the stream');
      }
      
      console.log('✅ Microphone access granted!');
      console.log('Track settings:', track.getSettings());
      console.log('Track capabilities:', track.getCapabilities());
      console.log('Track state:', track.readyState);
      console.log('Track enabled:', track.enabled);
      console.log('Track muted:', track.muted);

      const logFormatSupport = () => {
        const testFormats = ['audio/webm;codecs=opus', 'audio/webm', 'audio/wav', 'audio/ogg;codecs=opus', 'audio/mp4', 'audio/mpeg'];
        console.log('=== MediaRecorder Format Support ===');
        testFormats.forEach(format => console.log(`${format}: ${MediaRecorder.isTypeSupported(format) ? '✅' : '❌'}`));
        console.log('=====================================');
      };
      logFormatSupport();
      
      let mediaRecorder: MediaRecorder;
      let preferredMimeType = '';
      const formatTests = [
        { mimeType: 'audio/webm;codecs=opus', description: 'WebM/Opus' },
        { mimeType: 'audio/webm', description: 'WebM (generic)' },
        { mimeType: 'audio/wav', description: 'WAV (less preferred for streaming)' },
      ];
      
      let selectedFormatInfo = null;
      for (const format of formatTests) {
        if (MediaRecorder.isTypeSupported(format.mimeType)) {
          selectedFormatInfo = format;
          break;
        }
      }
      
      if (selectedFormatInfo) {
        mediaRecorder = new MediaRecorder(stream, { mimeType: selectedFormatInfo.mimeType, audioBitsPerSecond: selectedFormatInfo.mimeType.includes('wav') ? 128000 : 32000 });
        preferredMimeType = selectedFormatInfo.mimeType;
        console.log(`MediaRecorder initialized with ${selectedFormatInfo.description}`);
      } else {
        mediaRecorder = new MediaRecorder(stream); // Fallback to browser default
        preferredMimeType = mediaRecorder.mimeType || 'default/unknown';
        console.warn(`No preferred format supported. MediaRecorder initialized with browser default: ${preferredMimeType}`);
      }
      mediaRecorderRef.current = mediaRecorder;

      let audioChunks: Blob[] = [];
      let bufferTimer: NodeJS.Timeout | null = null;
      const BUFFER_TIMEOUT = 128; // 128ms (4 * 32ms backend frame)

      const flushBuffer = () => {
        if (isCleaningUpRef.current || wsRef.current?.readyState !== WebSocket.OPEN || audioChunks.length === 0) {
          if(audioChunks.length > 0) console.log('FlushBuffer: Skipping send due to cleanup or WebSocket not open.');
          audioChunks = []; // Clear chunks if not sending
          return;
        }
        
        const audioBlob = new Blob(audioChunks, { type: preferredMimeType });
        audioChunks = []; // Clear chunks after creating blob

        let minSize = 100; // Default minimum
        if (preferredMimeType.includes('wav')) minSize = 200; 
        if (preferredMimeType.includes('webm')) minSize = 150; 
        
        if (audioBlob.size > minSize) {
          audioBlob.arrayBuffer().then(buffer => {
            if (wsRef.current?.readyState === WebSocket.OPEN && !isCleaningUpRef.current) {
              const view = new Uint8Array(buffer);
              const hasValidData = view.some(byte => byte !== 0);
              if (hasValidData) {
                wsRef.current.send(buffer);
                console.log(`Sent audio chunk: ${buffer.byteLength} bytes, MIME: ${preferredMimeType}`);
              } else {
                console.warn(`Skipping empty audio data: ${buffer.byteLength} bytes, MIME: ${preferredMimeType}`);
              }
            }
          }).catch(error => console.error('Error converting audio blob to buffer:', error));
        } else {
          console.log(`Skipping small/empty audio chunk: ${audioBlob.size} bytes (min: ${minSize}), MIME: ${preferredMimeType}`);
        }
      };

      mediaRecorder.ondataavailable = (event) => {
        if (isCleaningUpRef.current || event.data.size === 0) return;
        console.log(`Audio chunk received: ${event.data.size} bytes`);
        audioChunks.push(event.data);
        if (bufferTimer) clearTimeout(bufferTimer);
        bufferTimer = setTimeout(flushBuffer, BUFFER_TIMEOUT);
      };

      mediaRecorder.onstart = () => {
        console.log('MediaRecorder started');
        setSession(prev => ({ ...prev, isRecording: true }));
      };

      mediaRecorder.onstop = () => {
        console.log('MediaRecorder stopped.');
        if (bufferTimer) clearTimeout(bufferTimer);
        flushBuffer(); // Send any remaining data
        setSession(prev => ({ ...prev, isRecording: false }));
        if (wsRef.current?.readyState === WebSocket.OPEN && !isCleaningUpRef.current) {
            console.log('Sending EOS to backend.');
            wsRef.current.send(JSON.stringify({ type: "eos" }));
        }
      };
      
      mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error:', event);
        onError?.(`MediaRecorder error: ${(event as any)?.error?.name || 'Unknown'}. Please check microphone and permissions.`);
      };

      if (vadRef.current) {
        console.log('Destroying existing VAD instance before creating new one...');
        if (typeof (vadRef.current as any).destroy === 'function') {
          try { (vadRef.current as any).destroy(); } catch (e) { console.warn('Error destroying old VAD:', e); }
        }
        vadRef.current = null;
      }
      console.log('Initializing VAD...');
      const newVad = await MicVAD.new({
        stream: streamRef.current,
        onSpeechStart: () => {
          if (isCleaningUpRef.current) return;
          console.log("VAD: User speech started");
          setIsUserSpeaking(true);
          const mr = mediaRecorderRef.current;
          if (mr && mr.state === 'inactive') {
            mr.start(); // Let ondataavailable handle chunking interval
          }
        },
        onSpeechEnd: (audio: Float32Array) => { // audio is Float32Array from VAD
          if (isCleaningUpRef.current) return;
          console.log('VAD: User speech ended, VAD audio length:', audio?.length || 0);
          setIsUserSpeaking(false);
          const mr = mediaRecorderRef.current;
          if (mr && mr.state === 'recording') {
            console.log('VAD EOS: Stopping MediaRecorder');
            mr.stop(); // This will trigger onstop and send EOS
          }
        },
        minSpeechFrames: 3,
        positiveSpeechThreshold: 0.7,
        negativeSpeechThreshold: 0.35,
        minSilenceFrames: 8,
        redemptionFrames: 25,
        preSpeechPadFrames: 1,
        onVADMisfire: () => console.warn('VAD: Misfire detected'),
      } as any);
      vadRef.current = newVad;
      vadRef.current.start();
      console.log('VAD initialized and started successfully');

      const existingCtx = audioContextRef.current;
      if (existingCtx && existingCtx.state !== 'closed') {
        await existingCtx.close();
      }
      const newAudioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = newAudioContext;
      const newSource = newAudioContext.createMediaStreamSource(stream);
      sourceRef.current = newSource;
      const analyser = newAudioContext.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.8;
      newSource.connect(analyser);
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const updateAudioLevel = () => {
        const ctx = audioContextRef.current;
        if (ctx && ctx.state !== 'closed' && analyser) {
          analyser.getByteFrequencyData(dataArray);
          const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
          setAudioLevel(Math.min(average / 255, 1.0));
          requestAnimationFrame(updateAudioLevel);
        }
      };
      updateAudioLevel();
      console.log('Audio initialization completed successfully');
      return true;
    } catch (error) {
      console.error('Failed to initialize audio:', error);
      const err = error as Error;
      if (err.name === 'NotAllowedError') onError?.('Microphone access denied. Please allow microphone access and refresh.');
      else if (err.name === 'NotFoundError') onError?.('No microphone found. Please connect a microphone.');
      else if (err.name ==='NotSupportedError') onError?.('Audio recording not supported by this browser.');
      else onError?.(`Audio initialization failed: ${err.message}`);
      return false;
    }
  }, [onError]); // Removed isCleaningUpRef from deps, it's a ref

  const sendWebSocketMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
        console.warn("sendWebSocketMessage: WebSocket not open. State:", wsRef.current?.readyState);
    }
  }, []);

  const playTTSAudio = useCallback(async (base64Audio: string) => {
    if (isCleaningUpRef.current || !session.isConnected || session.isMuted) {
      console.log("playTTSAudio: Skipped due to cleanup, disconnect, or mute.");
      return;
    }
    try {
      if (!ttsAudioPlayerRef.current) {
        ttsAudioPlayerRef.current = new AudioPlayer();
      }
      const binaryString = atob(base64Audio);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
      await ttsAudioPlayerRef.current.playAudioData(bytes.buffer);
      console.log('TTS audio chunk finished playing via AudioPlayer.');
    } catch (error) {
      console.error('Failed to play TTS audio via AudioPlayer:', error);
      try { // Fallback to HTML5 Audio for robustness
        const audio = new Audio(`data:audio/wav;base64,${base64Audio}`);
        await audio.play();
        console.log('TTS audio played via HTML5 Audio fallback.');
      } catch (fallbackError) {
        console.error('Fallback HTML5 audio play failed:', fallbackError);
        onError?.("Failed to play AI response audio.");
      }
    }
  }, [session.isConnected, session.isMuted, onError]); // isCleaningUpRef is a ref

  const toggleMute = useCallback(() => {
    const newMutedState = !session.isMuted;
    setSession(prev => ({...prev, isMuted: newMutedState})); // Optimistic update
    sendWebSocketMessage({ type: 'control', action: newMutedState ? 'mute' : 'unmute' });
    // VAD and MediaRecorder are managed by speech events, not directly by mute.
    // Mute mainly affects TTS playback and can be a signal to backend.
  }, [session.isMuted, sendWebSocketMessage]);

  const endSession = useCallback(() => {
    console.log('User initiated endSession...');
    setSession(prev => ({ ...prev, sessionEnded: true })); // Mark session ended by user
    sendWebSocketMessage({ type: 'end_session' });
    // Delay cleanup to allow message to send
    setTimeout(() => handleEndSession(false), 100); // Pass false: user initiated
  }, [sendWebSocketMessage]);


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

    if (vadRef.current && typeof (vadRef.current as any).destroy === 'function') {
      try { (vadRef.current as any).destroy(); } catch (e) { console.warn('Error destroying VAD:', e); }
    }
    vadRef.current = null;

    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== 'inactive') {
      try { mr.stop(); } catch (e) { console.warn('Error stopping MediaRecorder during cleanup:', e); }
    }
    mediaRecorderRef.current = null;

    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;

    sourceRef.current?.disconnect();
    sourceRef.current = null;

    pcmPlayerNodeRef.current?.disconnect();
    pcmPlayerNodeRef.current = null;

    const ctx = audioContextRef.current;
    if (ctx && ctx.state !== 'closed') {
      ctx.close().catch(e => console.warn('Error closing AudioContext:', e));
    }
    audioContextRef.current = null;

    ttsAudioPlayerRef.current?.dispose();
    ttsAudioPlayerRef.current = null;

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

    setTimeout(() => {
      isCleaningUpRef.current = false;
      console.log('Cleanup flag reset, ready for new session.');
    }, 500);
  }


  const startNewSession = useCallback(async () => {
    console.log('🚀 Starting new session attempt...');
    if (isCleaningUpRef.current) {
      console.log('StartNewSession: Cleanup in progress. Will retry in 550ms.');
      setTimeout(startNewSession, 550); // Retry after cleanup timeout
      return;
    }
    
    // Ensure previous session is fully ended if any refs still exist
    if(session.isConnected || wsRef.current || streamRef.current) {
        console.log("StartNewSession: Previous session seems active, ensuring full cleanup first.");
        await handleEndSession(false); // Force cleanup
        // Wait a moment for cleanup to complete if it was just triggered
        await new Promise(resolve => setTimeout(resolve, 600));
         if (isCleaningUpRef.current) {
            console.log('StartNewSession: Waiting for final cleanup flag reset.');
            await new Promise(resolve => setTimeout(resolve, 550));
        }
    }
    
    setSession(prev => ({ 
      ...prev, 
      sessionEnded: false, isConnected: false, isReconnecting: false,
      isProcessing: false, isSpeaking: false, isRecording: false
    }));
    setTranscript({ partial: '', final: [], aiResponse: '', isAiResponding: false });
    setIsUserSpeaking(false);
    
    connectWebSocket(); // This will attempt to establish a new WebSocket connection
    // initializeAudio will be called via useEffect when session.isConnected becomes true
  }, [connectWebSocket, session.isConnected]);

  useEffect(() => {
    // Initial connection attempt
    if(!session.isConnected && !session.sessionEnded && !wsRef.current) {
        connectWebSocket();
    }
    // Cleanup on component unmount
    return () => {
      console.log("VoiceAgent component unmounting. Performing cleanup.");
      handleEndSession(false); // User (component unmount) initiated
      if (keepAliveIntervalRef.current) {
        clearInterval(keepAliveIntervalRef.current);
        keepAliveIntervalRef.current = null;
      }
    };
  }, [connectWebSocket, session.isConnected, session.sessionEnded]);

  useEffect(() => {
    // Attempt to initialize audio once connected and if not already initialized/cleaning up
    if (session.isConnected && !streamRef.current && !vadRef.current && !isCleaningUpRef.current) {
      console.log("🔊 useEffect: Connection active, attempting to initialize audio.");
      initializeAudio().then(success => {
        console.log(`🎯 Audio initialization ${success ? 'succeeded' : 'failed'}`);
      });
    }
  }, [session.isConnected, initializeAudio]);


  // Get activity indicator based on multiple states
  const getActivityState = () => {
    if (session.isSpeaking) return 'speaking'; // AI is speaking
    if (session.isProcessing) return 'processing'; // AI is thinking/STT finalizing
    if (isUserSpeaking && !session.isMuted) return 'listening'; // User is speaking (client VAD)
    if (session.isRecording && !session.isMuted) return 'recording'; // Mic is open, VAD might not be active yet
    return 'idle';
  };
  const activityState = getActivityState();

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