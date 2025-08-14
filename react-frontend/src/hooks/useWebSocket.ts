import { useState, useEffect, useRef, useCallback } from 'react';
import { WebSocketMessage, ConnectionStatus } from '../types';
import authService from '../services/authService';

interface UseWebSocketReturn {
  socket: WebSocket | null;
  isConnected: boolean;
  isConnecting: boolean;
  lastMessage: WebSocketMessage | null;
  sendMessage: (message: any) => void;
  error: string | null;
  connectionStatus: ConnectionStatus;
}

interface UseWebSocketOptions {
  url?: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  onOpen?: () => void;
  onMessage?: (message: WebSocketMessage) => void;
  onError?: (error: Event) => void;
  onClose?: () => void;
}

export const useWebSocket = (options: UseWebSocketOptions = {}, authenticated: boolean = false): UseWebSocketReturn => {
  const {
    url = process.env.REACT_APP_WS_URL || 'ws://localhost:8000',
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
    onOpen,
    onMessage,
    onError,
    onClose,
  } = options;

  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const settingsAppliedRef = useRef(false);
  const settingsSentRef = useRef(false);
  // Enforce exactly-once settings per socket instance
  const socketGenerationRef = useRef(0);
  const settingsSendTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const stopPlaybackRef = useRef<(() => void) | null>(null);
  
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isConnectingRef = useRef(false);
  const isConnectedRef = useRef(false);
  const connectionHealthRef = useRef({ lastPong: 0, missedPongs: 0 });
  
  // Ultra-low latency audio playback with Web Audio API fallback
  const playAudioResponse = useCallback((audioData: string, mime: string) => {
    // Fallback to optimized HTML5 audio function
    const playWithHTMLAudio = () => {
      try {
        const audioBytes = atob(audioData);
        const audioArray = new Uint8Array(audioBytes.length);
        for (let i = 0; i < audioBytes.length; i++) {
          audioArray[i] = audioBytes.charCodeAt(i);
        }
        
        const audioBlob = new Blob([audioArray], { type: mime });
        const audioUrl = URL.createObjectURL(audioBlob);
        
        // Create audio element with low-latency optimizations
        const audio = new Audio(audioUrl);
        
        // Ultra-low latency settings
        audio.preload = 'auto';
        if ('mozAudioChannelType' in audio) {
          (audio as any).mozAudioChannelType = 'content';
        }
        
        // Start playback immediately
        const playPromise = audio.play();
        
        if (playPromise) {
          playPromise.then(() => {
            console.log('🔊 Playing AI response audio (HTML5 - optimized)');
          }).catch((error) => {
            console.error('❌ Failed to play audio:', error);
          });
        }
        
        // Cleanup URL after playback
        audio.onended = () => {
          URL.revokeObjectURL(audioUrl);
        };
        
        // Cleanup on error
        audio.onerror = () => {
          URL.revokeObjectURL(audioUrl);
        };

        // Expose stop for barge-in
        stopPlaybackRef.current = () => {
          try {
            audio.pause();
          } catch {}
          URL.revokeObjectURL(audioUrl);
          stopPlaybackRef.current = null;
        };
      } catch (error) {
        console.error('❌ Failed to process audio response:', error);
      }
    };

    try {
      // Convert base64 audio to ArrayBuffer for Web Audio API
      const audioBytes = atob(audioData);
      const audioArray = new Uint8Array(audioBytes.length);
      for (let i = 0; i < audioBytes.length; i++) {
        audioArray[i] = audioBytes.charCodeAt(i);
      }
      
      // Try Web Audio API first for lowest latency
      if (window.AudioContext || (window as any).webkitAudioContext) {
        try {
          const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
          const audioContext = new AudioContextClass();
          
          // Decode audio data with proper parameters
          audioContext.decodeAudioData(
            audioArray.buffer.slice(0),
            (audioBuffer) => {
              const source = audioContext.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(audioContext.destination);
              source.start(0);
              console.log('🔊 Playing AI response audio (Web Audio API - ultra-low latency)');

              // Expose stop for barge-in
              stopPlaybackRef.current = () => {
                try { source.stop(0); } catch {}
                try { audioContext.close(); } catch {}
                stopPlaybackRef.current = null;
              };
            },
            () => {
              // Fallback to HTML5 audio on decode error
              playWithHTMLAudio();
            }
          );
          
          return; // Exit early if Web Audio API works
        } catch (e) {
          // Fallback to HTML5 audio
          playWithHTMLAudio();
        }
      } else {
        // No Web Audio API support, use HTML5 audio
        playWithHTMLAudio();
      }
      
    } catch (error) {
      console.error('❌ Failed to process audio response:', error);
      // Final fallback
      playWithHTMLAudio();
    }
  }, []);

  // Track last time we played server-provided TTS audio to avoid double-speaking
  const lastTTSAudioAtRef = useRef<number>(0);

  const sendMessage = useCallback((message: any) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      try {
        socket.send(JSON.stringify(message));
      } catch (err) {
        console.error('Failed to send WebSocket message:', err);
        setError('Failed to send message');
      }
    } else {
      console.warn('WebSocket is not connected');
      setError('WebSocket is not connected');
    }
  }, [socket]);

  useEffect(() => {
    if (!authenticated) {
      // Reset state when not authenticated
      isConnectedRef.current = false;
      isConnectingRef.current = false;
      setIsConnected(false);
      setIsConnecting(false);
      setConnectionStatus('disconnected');
      return;
    }

    let currentSocket: WebSocket | null = null;

    const connectWebSocket = () => {
      if (isConnectingRef.current || isConnectedRef.current) {
        return;
      }

      const token = authService.getAccessToken();
      if (!token) {
        setError('No authentication token available');
        setConnectionStatus('error');
        return;
      }

      try {
        isConnectingRef.current = true;
        setIsConnecting(true);
        setConnectionStatus('connecting');
        setError(null);

          // Convert http(s) URL to ws(s) if needed
        const baseUrl = url.startsWith('http')
          ? url.replace(/^http/, 'ws')
          : url;

        // Use 127.0.0.1 instead of localhost to avoid DNS lookup latency
        const wsUrl = `${baseUrl.replace('localhost', '127.0.0.1')}/ws/${encodeURIComponent(token)}`;
        console.log('🔗 Attempting WebSocket connection to:', wsUrl);
        console.log('🎫 Using token:', token.substring(0, 20) + '...');
        
        // Establish WebSocket connection (no subprotocol; server does not negotiate any)
        const newSocket = new WebSocket(wsUrl);
        // Track a new socket generation and reset settings guard for this socket
        socketGenerationRef.current += 1;
        const myGeneration = socketGenerationRef.current;
        settingsSentRef.current = false;
        
          // Ultra-low latency optimizations
        if ('binaryType' in newSocket) {
          newSocket.binaryType = 'arraybuffer'; // Faster than blob
        }
        currentSocket = newSocket;

        // Heartbeat timer id
        let heartbeatId: NodeJS.Timeout | null = null;

          const startHeartbeat = () => {
          heartbeatId = setInterval(() => {
            if (newSocket.readyState === WebSocket.OPEN) {
              try {
                // Check connection health before sending heartbeat
                const now = Date.now();
                if (connectionHealthRef.current.lastPong > 0 && 
                    now - connectionHealthRef.current.lastPong > 90000) { // 1.5 minutes without pong
                  connectionHealthRef.current.missedPongs++;
                  console.warn(`Missed pong #${connectionHealthRef.current.missedPongs}`);
                  
                  if (connectionHealthRef.current.missedPongs >= 2) {
                    console.error('Connection appears dead, forcing reconnect');
                    newSocket.close(1000, 'Connection health check failed');
                    return;
                  }
                }
                
                // App-level heartbeat only
                newSocket.send(JSON.stringify({ type: 'heartbeat', timestamp: now }));
              } catch (error) {
                console.error('Failed to send heartbeat:', error);
                stopHeartbeat();
              }
            } else {
              stopHeartbeat();
            }
          }, 5000); // 5-second keepalive/heartbeat cadence to avoid 10s idle timeout
        };

        const stopHeartbeat = () => {
          if (heartbeatId) {
            clearInterval(heartbeatId);
            heartbeatId = null;
          }
        };

        newSocket.onopen = () => {
          console.log('WebSocket connected');
          console.log('🔗 Negotiated protocol:', newSocket.protocol || 'none');
          isConnectedRef.current = true;
          isConnectingRef.current = false;
          setIsConnected(true);
          setIsConnecting(false);
          setConnectionStatus('connected');
          setError(null);
          reconnectAttemptsRef.current = 0;
          
          // Reset connection health tracking
          connectionHealthRef.current = { lastPong: Date.now(), missedPongs: 0 };
          
          // Make WebSocket globally available for AudioVisualizer immediately
          (window as any).voiceAgentWebSocket = newSocket;
          console.log('🌐 WebSocket globally available for audio streaming');
          
          // Send initial connection message to establish the session
          try {
            newSocket.send(JSON.stringify({
              type: 'connection',
              message: 'Voice Agent client connected',
              timestamp: Date.now()
            }));
          } catch (error) {
            console.warn('Failed to send initial connection message:', error);
          }
          
          // Add a small delay before starting heartbeat to ensure connection is stable
          setTimeout(() => {
            if (newSocket.readyState === WebSocket.OPEN) {
              startHeartbeat();
            }
          }, 1000);
          
          onOpen?.();
        };

        // Handle automatic pong responses to server pings
        newSocket.addEventListener('ping', (event) => {
          console.log('📡 Received ping from server, sending pong');
          try {
            // Browser automatically sends pong, but we track it for health monitoring
            connectionHealthRef.current.lastPong = Date.now();
            connectionHealthRef.current.missedPongs = 0;
          } catch (error) {
            console.error('Failed to handle ping:', error);
          }
        });

        // Handle pong responses (for our custom heartbeat)
        newSocket.addEventListener('pong', (event) => {
          console.log('📡 Received pong from server');
          connectionHealthRef.current.lastPong = Date.now();
          connectionHealthRef.current.missedPongs = 0;
        });

        // Helper to build Settings payload consistently
          const buildSettingsPayload = () => ({
          type: 'Settings',
          tags: ['prod', 'voice_agent'],
          keepalive_interval: 10000,
          audio: {
            input: { encoding: 'linear16', sample_rate: 16000 },
            output: { encoding: 'linear16', sample_rate: 16000, container: 'none' }
          },
          agent: {
            language: 'en-US',
            listen: { provider: { type: 'deepgram', model: 'nova-3', smart_format: false } },
            // Use Google/Gemini directly to match backend configuration
            think: {
              provider: { type: 'google', model: 'gemini-2.0-flash', temperature: 0.7 }
            },
            speak: { provider: { type: 'deepgram', model: 'aura-2-thalia-en' } },
            greeting: 'Hi! How can I help today?'
          }
        });

        newSocket.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            
            // Send settings exactly once per socket, only on connection_ack
            if (message.type === 'connection_ack' || message.type === 'connection') {
              if (!settingsAppliedRef.current && !settingsSentRef.current && myGeneration === socketGenerationRef.current) {
                if (settingsSendTimeoutRef.current) clearTimeout(settingsSendTimeoutRef.current);
                settingsSendTimeoutRef.current = setTimeout(() => {
                  if (settingsAppliedRef.current || settingsSentRef.current || myGeneration !== socketGenerationRef.current) return;
                  try {
                    const settings = buildSettingsPayload();
                    newSocket.send(JSON.stringify(settings));
                    settingsSentRef.current = true;
                    console.log('⚙️ Sent Settings to server (once)');
                  } catch (e) {
                    console.error('Failed to send Settings:', e);
                  }
                }, 200); // small debounce to allow any immediate server setup
              }
            }

            // Settings applied -> enable mic streaming
            if ((message.type === 'settings_applied' || (message as any).type === 'SettingsApplied') && !settingsAppliedRef.current) {
              settingsAppliedRef.current = true;
              (window as any).voiceAgentReady = true;
              if (settingsSendTimeoutRef.current) {
                clearTimeout(settingsSendTimeoutRef.current);
                settingsSendTimeoutRef.current = null;
              }
              console.log('✅ SettingsApplied received; mic streaming enabled');
              setLastMessage(message);
              onMessage?.(message);
              return;
            }

            // If backend complains about duplicate settings, treat as success and ignore error
            if (message.type === 'error' && typeof (message as any).message === 'string' && (
              (message as any).message.includes('SETTINGS_ALREADY_APPLIED') ||
              (message as any).message.includes('settings were already established') ||
              (message as any).message.includes('SETTINGS_ALREADY_ESTABLISHED')
            )) {
              settingsAppliedRef.current = true;
              (window as any).voiceAgentReady = true;
              if (settingsSendTimeoutRef.current) {
                clearTimeout(settingsSendTimeoutRef.current);
                settingsSendTimeoutRef.current = null;
              }
              console.warn('✅ Duplicate settings reported by server; treating as applied and continuing');
              setLastMessage({ ...message, type: 'settings_applied' } as any);
              onMessage?.({ ...message, type: 'settings_applied' } as any);
              return;
            }

            // If backend complains we sent audio before settings, immediately resend settings for this connection
            if (message.type === 'error' && typeof (message as any).message === 'string' && /before\s+Settings/i.test((message as any).message)) {
              console.warn('⚠️ Server reported audio before Settings; re-sending Settings now');
              try {
                const settings = buildSettingsPayload();
                settingsAppliedRef.current = false;
                settingsSentRef.current = false;
                newSocket.send(JSON.stringify(settings));
                settingsSentRef.current = true;
              } catch (e) {
                console.error('Failed to re-send Settings after error:', e);
              }
            }

            // Barge-in: user started speaking -> stop TTS playback immediately
            if (message.type === 'speech_started' || message.type === 'UserStartedSpeaking' || message.type === 'user_started_speaking') {
              if (stopPlaybackRef.current) {
                try { stopPlaybackRef.current(); } catch {}
                stopPlaybackRef.current = null;
                console.log('🛑 Barge-in: Stopped TTS playback');
              }
            }

            // ONLY play the final WAV blob to avoid duplicate audio
            // Skip chunked audio (tts_audio) and only play complete utterance (tts_wav)
            if (message.type === 'tts_audio') {
              // Just log but don't play - wait for tts_wav
              console.log(`🔊 Received tts_audio (${message.data?.length || 0} bytes) - buffering for final playback`);
              setLastMessage(message);
              onMessage?.(message);
              return;
            }

            // Play only the final complete utterance WAV
            if (message.type === 'tts_wav' && message.data) {
              console.log(`🔊 Received tts_wav (${message.data.length} bytes) - playing complete utterance`);
              playAudioResponse(message.data, 'audio/wav');
              lastTTSAudioAtRef.current = Date.now();
              setLastMessage(message);
              onMessage?.(message);
              return;
            }
            
            // Handle heartbeat responses for connection health monitoring
            if (message.type === 'heartbeat_ack') {
              connectionHealthRef.current.lastPong = Date.now();
              connectionHealthRef.current.missedPongs = 0;
              return; // Don't process heartbeat acks further
            }
            
            // Handle ping messages from server (respond with pong)
            if (message.type === 'ping') {
              try {
                newSocket.send(JSON.stringify({ 
                  type: 'pong', 
                  timestamp: message.timestamp || Date.now() 
                }));
                connectionHealthRef.current.lastPong = Date.now();
                connectionHealthRef.current.missedPongs = 0;
              } catch (error) {
                console.error('Failed to send pong response:', error);
              }
              return; // Don't process ping messages further
            }
            
            // Surface STT feedback to console for diagnostics
            if (message.type === 'stt_result') {
              if (message.transcript && message.transcript.trim().length > 0) {
                console.log(`🗣️ Final transcript: "${message.transcript}"`);
              } else if ((message as any).message) {
                // Backend may include helpful feedback + debug_info
                const dbg = (message as any).debug_info;
                console.warn('🛈 STT feedback:', (message as any).message, dbg ? { debug_info: dbg } : undefined);
              }
            }

            // Fallback: if we receive text but no server audio, synthesize speech client-side
            if ((message.type === 'text_response' && (message as any).text) || (message.type === 'agent_text' && (message as any).content)) {
              const text = (message as any).text || (message as any).content;
              const now = Date.now();
              // Only speak if we haven't played server audio very recently (avoid double)
              if (typeof window !== 'undefined' && 'speechSynthesis' in window && now - lastTTSAudioAtRef.current > 1500) {
                try {
                  const utterance = new SpeechSynthesisUtterance(text);
                  utterance.rate = 1.0;
                  utterance.pitch = 1.0;
                  window.speechSynthesis.cancel();
                  window.speechSynthesis.speak(utterance);
                  console.log('🗣️ Spoke response via SpeechSynthesis fallback');
                } catch (e) {
                  console.warn('SpeechSynthesis fallback failed:', e);
                }
              }
            }
            
            setLastMessage(message);
            onMessage?.(message);
          } catch (err) {
            console.error('Failed to parse WebSocket message:', err);
            setError('Failed to parse message from server');
          }
        };

        newSocket.onerror = (event) => {
          console.error('WebSocket error:', event);
          console.error('WebSocket readyState:', newSocket.readyState);
          console.error('WebSocket URL:', wsUrl);
          
          // More specific error handling
          let errorMessage = 'Network error';
          if (newSocket.readyState === WebSocket.CLOSED) {
            errorMessage = 'Connection rejected by server';
          } else if (newSocket.readyState === WebSocket.CLOSING) {
            errorMessage = 'Connection closing';
          } else if (newSocket.readyState === WebSocket.CONNECTING) {
            errorMessage = 'Connection timeout';
          }
          
          setError(`WebSocket connection error: ${errorMessage}`);
          setConnectionStatus('error');
          onError?.(event);
        };

        newSocket.onclose = (event) => {
          console.log('WebSocket disconnected', event.code, event.reason);
          isConnectedRef.current = false;
          isConnectingRef.current = false;
          setIsConnected(false);
          setIsConnecting(false);
          setSocket(null);
          setConnectionStatus('disconnected');
          
          // Cleanup global WebSocket reference
          if ((window as any).voiceAgentWebSocket === newSocket) {
            (window as any).voiceAgentWebSocket = null;
          }
          
          onClose?.();

          // Don't reconnect immediately if the connection was rejected
          if (event.code === 1008) {
            setError('Authentication failed');
            setConnectionStatus('error');
            return;
          }

          // Attempt to reconnect if we haven't exceeded max attempts
          if (reconnectAttemptsRef.current < maxReconnectAttempts) {
            reconnectAttemptsRef.current += 1;
            console.log(`Attempting to reconnect (${reconnectAttemptsRef.current}/${maxReconnectAttempts})...`);
            
            // Exponential backoff: 3s, 5s, 8s, 12s, 20s as per forensic analysis
            const backoffDelays = [3000, 5000, 8000, 12000, 20000];
            const backoffDelay = backoffDelays[Math.min(reconnectAttemptsRef.current - 1, backoffDelays.length - 1)];
            
            reconnectTimeoutRef.current = setTimeout(() => {
              connectWebSocket();
            }, backoffDelay);
          } else {
            setError('Max reconnection attempts reached');
            setConnectionStatus('error');
          }

          stopHeartbeat();
          settingsAppliedRef.current = false;
          settingsSentRef.current = false;
          (window as any).voiceAgentReady = false;
        };

        setSocket(newSocket);
      } catch (err) {
        console.error('Failed to create WebSocket connection:', err);
        setError('Failed to establish WebSocket connection');
        isConnectingRef.current = false;
        setIsConnecting(false);
        setConnectionStatus('error');
      }
    };

    connectWebSocket();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (currentSocket) {
        // Avoid closing a connecting socket during React StrictMode double-invocation
        // Only close if the socket is OPEN or CLOSING; leave CONNECTING sockets alone
        if (
          currentSocket.readyState === WebSocket.OPEN ||
          currentSocket.readyState === WebSocket.CLOSING
        ) {
          currentSocket.close();
        }
      }
    };
  }, [authenticated]); // Minimal dependencies to avoid constant reconnections

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (socket) {
        // Only close established or closing sockets; avoid tearing down CONNECTING sockets in StrictMode
        if (
          socket.readyState === WebSocket.OPEN ||
          socket.readyState === WebSocket.CLOSING
        ) {
          socket.close();
        }
      }
    };
  }, [socket]);

  return {
    socket,
    isConnected,
    isConnecting,
    lastMessage,
    sendMessage,
    error,
    connectionStatus,
  };
}; 