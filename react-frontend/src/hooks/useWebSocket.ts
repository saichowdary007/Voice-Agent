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
  
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isConnectingRef = useRef(false);
  const isConnectedRef = useRef(false);
  const connectionHealthRef = useRef({ lastPong: 0, missedPongs: 0 });
  
  // Audio playback function
  const playAudioResponse = useCallback((audioData: string, mime: string) => {
    try {
      // Convert base64 audio to blob
      const audioBytes = atob(audioData);
      const audioArray = new Uint8Array(audioBytes.length);
      for (let i = 0; i < audioBytes.length; i++) {
        audioArray[i] = audioBytes.charCodeAt(i);
      }
      
      const audioBlob = new Blob([audioArray], { type: mime });
      const audioUrl = URL.createObjectURL(audioBlob);
      
      // Play the audio
      const audio = new Audio(audioUrl);
      audio.play().then(() => {
        console.log('🔊 Playing AI response audio');
      }).catch((error) => {
        console.error('❌ Failed to play audio:', error);
      });
      
      // Cleanup URL after playback
      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
      };
      
    } catch (error) {
      console.error('❌ Failed to process audio response:', error);
    }
  }, []);

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

        // Use 127.0.0.1 instead of localhost to avoid potential DNS issues
        const wsUrl = `${baseUrl.replace('localhost', '127.0.0.1')}/ws/${encodeURIComponent(token)}`;
        console.log('🔗 Attempting WebSocket connection to:', wsUrl);
        console.log('🎫 Using token:', token.substring(0, 20) + '...');
        const newSocket = new WebSocket(wsUrl);
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
                    now - connectionHealthRef.current.lastPong > 60000) { // 1 minute without pong
                  connectionHealthRef.current.missedPongs++;
                  console.warn(`Missed pong #${connectionHealthRef.current.missedPongs}`);
                  
                  if (connectionHealthRef.current.missedPongs >= 3) {
                    console.error('Connection appears dead, forcing reconnect');
                    newSocket.close(1000, 'Connection health check failed');
                    return;
                  }
                }
                
                newSocket.send(JSON.stringify({ type: 'heartbeat', timestamp: now }));
              } catch (error) {
                console.error('Failed to send heartbeat:', error);
                stopHeartbeat();
              }
            } else {
              stopHeartbeat();
            }
          }, 30000); // 30-second heartbeat (less aggressive)
        };

        const stopHeartbeat = () => {
          if (heartbeatId) {
            clearInterval(heartbeatId);
            heartbeatId = null;
          }
        };

        newSocket.onopen = () => {
          console.log('WebSocket connected');
          isConnectedRef.current = true;
          isConnectingRef.current = false;
          setIsConnected(true);
          setIsConnecting(false);
          setConnectionStatus('connected');
          setError(null);
          reconnectAttemptsRef.current = 0;
          
          // Reset connection health tracking
          connectionHealthRef.current = { lastPong: Date.now(), missedPongs: 0 };
          
          // Make WebSocket globally available for AudioVisualizer with delay to ensure stability
          setTimeout(() => {
            (window as any).voiceAgentWebSocket = newSocket;
          }, 100);
          
          startHeartbeat();
          onOpen?.();
        };

        newSocket.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            
            // Handle heartbeat responses for connection health monitoring
            if (message.type === 'heartbeat_ack') {
              connectionHealthRef.current.lastPong = Date.now();
              connectionHealthRef.current.missedPongs = 0;
              return; // Don't process heartbeat acks further
            }
            
            // Handle audio response from server
            if ((message.type === 'audio_response' || message.type === 'audio_stream' || message.type === 'tts_audio') && message.data) {
              const mime = message.mime || (message.type === 'audio_stream' || message.type === 'tts_audio' ? 'audio/mp3' : 'audio/wav');
              playAudioResponse(message.data, mime);
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
            
            // More conservative backoff: 3s, 5s, 8s, 12s, 20s
            const backoffDelay = Math.min(reconnectInterval + (reconnectAttemptsRef.current * 2000), 20000);
            
            reconnectTimeoutRef.current = setTimeout(() => {
              connectWebSocket();
            }, backoffDelay);
          } else {
            setError('Max reconnection attempts reached');
            setConnectionStatus('error');
          }

          stopHeartbeat();
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
        currentSocket.close();
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
        socket.close();
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