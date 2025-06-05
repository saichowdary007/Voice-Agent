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

  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8003/ws';

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

    if (vadRef.current && typeof (vadRef.current as any).destroy === 'function') {
      try { (vadRef.current as any).destroy(); } catch (e) { console.warn('Error destroying VAD:', e); }
    }
    vadRef.current = null;

    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== 'inactive') {
      try { mr.stop(); } catch (e) { console.warn('Error stopping MediaRecorder during cleanup:', e); }
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

    setTimeout(() => {
      isCleaningUpRef.current = false;
      console.log('Cleanup flag reset, ready for new session.');
    }, 500);
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
        try {
          wsRef.current.close();
        } catch (e) {
          console.error('Error closing existing WebSocket:', e);
        }
        wsRef.current = null;
      }

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      lastServerMessageTimeRef.current = Date.now();

      // Set a connection timeout
      const connectionTimeoutId = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          console.error('WebSocket connection timeout');
          ws.close();
        }
      }, 10000); // 10 second timeout for initial connection

      ws.onopen = () => {
        clearTimeout(connectionTimeoutId);
        console.log('✅ WebSocket connected successfully');
        console.log('Setting session state: isConnected=true, sessionEnded=false');
        
        // Reset reconnection state on successful connection
        reconnectAttemptsRef.current = 0;
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
        
        setSession(prev => ({ ...prev, isConnected: true, sessionEnded: false, isReconnecting: false }));
        lastServerMessageTimeRef.current = Date.now();
        
        // Flush any queued audio after successful reconnection
        flushAudioQueue();
        
        if (keepAliveIntervalRef.current) clearInterval(keepAliveIntervalRef.current);
        keepAliveIntervalRef.current = setInterval(() => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            const now = Date.now();
            if (now - lastServerMessageTimeRef.current > 60000) { // Increased from 30s to 60s
              console.warn('No server message for over 60 seconds. Closing connection due to timeout.');
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
        clearTimeout(connectionTimeoutId);
        console.log(`❌ WebSocket disconnected. Code: ${event.code}, Reason: ${event.reason}`);
        setSession(prev => ({ 
          ...prev, 
          isConnected: false, 
          isRecording: false, 
          isProcessing: false, 
          isSpeaking: false, // AI speaking
          // Do not set sessionEnded here to allow reconnection
        }));
        
        if (keepAliveIntervalRef.current) {
          clearInterval(keepAliveIntervalRef.current);
          keepAliveIntervalRef.current = null;
        }

        // Handle reconnection for unexpected disconnections
        if (event.code !== 1000 && !sessionEndedRef.current) { // Use ref to avoid dependency issues
          console.warn(`WebSocket closed unexpectedly. Code: ${event.code}. Reason: ${event.reason}`);
          
          // Check if we should attempt reconnection
          if (reconnectAttemptsRef.current < maxReconnectAttempts) {
            const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 10000); // Exponential backoff, max 10s
            reconnectAttemptsRef.current += 1;
            
            console.log(`Attempting reconnection ${reconnectAttemptsRef.current}/${maxReconnectAttempts} in ${delay}ms`);
            setSession(prev => ({ ...prev, isReconnecting: true }));

            reconnectTimeoutRef.current = setTimeout(() => {
              console.log(`Reconnection attempt ${reconnectAttemptsRef.current}`);
              connectWebSocket();
            }, delay);
          } else {
            console.error('Max reconnection attempts reached');
            setSession(prev => ({ ...prev, isReconnecting: false, sessionEnded: true }));
            onError?.(`Connection lost (Code: ${event.code}). Max retries reached. Please start a new session.`);
          }
        } else if (event.code !== 1000) {
          // If normal closure, don't show error, just end session
          onError?.(`Connection lost (Code: ${event.code}). Please try again.`);
        }
      };

      ws.onerror = (errorEvent) => {
        clearTimeout(connectionTimeoutId);
        const error = errorEvent instanceof ErrorEvent ? errorEvent.message : 'Unknown WebSocket error';
        console.error(`❌ WebSocket error connecting to ${wsUrl}:`, error, errorEvent);
        
        // Only show error if not in reconnection mode
        if (!session.isReconnecting) {
          // Delay error reporting to prevent immediate re-renders that might cause unmounting
          setTimeout(() => {
            onError?.('WebSocket connection failed. Please check your connection and the server.');
          }, 100);
        }
      };

    } catch (error) {
      console.error(`❌ Failed to initiate WebSocket connection to ${wsUrl}:`, error);
      // Only show error if not in reconnection mode
      if (!session.isReconnecting) {
        onError?.('Failed to connect to the voice service.');
      }
    }
  }, [wsUrl, onError, flushAudioQueue, session.isReconnecting]);

async function handleWebSocketMessage(data: any) {
  console.log('📨 Received WebSocket message:', data.type, data);
  
  switch (data.type) {
    case 'status':
      setSession(prev => ({ 
        ...prev, 
        sessionId: data.session_id, 
        isProcessing: data.status === 'processing',
        // Set connected if we receive a status message with ready: true (initial connection)
        isConnected: data.ready === true ? true : prev.isConnected
      }));
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
      if (data.audio_data) {
        let audioBuffer;
        if (data.encoding === 'base64') {
          // Decode base64 audio data
          try {
            const binaryString = atob(data.audio_data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
              bytes[i] = binaryString.charCodeAt(i);
            }
            audioBuffer = bytes.buffer;
          } catch (error) {
            console.error('Error decoding base64 audio data:', error);
            break;
          }
        } else {
          // Assume it's already binary data (old format)
          audioBuffer = data.audio_data;
        }
        await playTTSAudio(audioBuffer);
      }
      break;
    case 'tts_complete':
      setSession(prev => ({ ...prev, isSpeaking: false }));
      break;
    case 'stop_audio':
      console.log('Received stop_audio from backend');
      if (ttsAudioPlayerRef.current) {
        ttsAudioPlayerRef.current.stop();
      }
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
            console.log(`Sent audio chunk: ${buffer.byteLength} bytes, MIME: ${preferredMimeType}`);
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(buffer);
            } else {
              // Queue audio if not connected
              console.log('WebSocket not open, queuing audio chunk');
              audioQueueRef.current.push(audioBlob);
              
              // Limit queue size to prevent memory issues
              if (audioQueueRef.current.length > 50) {
                console.warn('Audio queue size limit reached, dropping oldest chunk');
                audioQueueRef.current.shift();
              }
            }
          }).catch(e => console.error('Error sending audio chunk:', e));
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
        min_silence_frames: 8,
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
    sendWebSocketMessage({ type: 'mute', muted: newMutedState });
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
    // Attempt to initialize audio once connected and if not already initialized/cleaning up
    if (session.isConnected && !streamRef.current && !vadRef.current && !isCleaningUpRef.current) {
      console.log("🔊 useEffect: Connection active, attempting to initialize audio.");
      initializeAudio().then(success => {
        console.log(`🎯 Audio initialization ${success ? 'succeeded' : 'failed'}`);
      });
    }
  }, [session.isConnected]); // Only depend on isConnected, remove initializeAudio to prevent re-creation


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