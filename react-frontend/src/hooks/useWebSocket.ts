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
    url = process.env.REACT_APP_WS_URL || 'ws://localhost:8080',
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
  
  // Audio playback function
  const playAudioResponse = useCallback((audioData: string) => {
    try {
      // Convert base64 audio to blob
      const audioBytes = atob(audioData);
      const audioArray = new Uint8Array(audioBytes.length);
      for (let i = 0; i < audioBytes.length; i++) {
        audioArray[i] = audioBytes.charCodeAt(i);
      }
      
      const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
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

        const wsUrl = `${url}/ws/${encodeURIComponent(token)}`;
        console.log('🔗 Attempting WebSocket connection to:', wsUrl);
        console.log('🎫 Using token:', token.substring(0, 20) + '...');
        const newSocket = new WebSocket(wsUrl);
        currentSocket = newSocket;

        newSocket.onopen = () => {
          console.log('WebSocket connected');
          isConnectedRef.current = true;
          isConnectingRef.current = false;
          setIsConnected(true);
          setIsConnecting(false);
          setConnectionStatus('connected');
          setError(null);
          reconnectAttemptsRef.current = 0;
          
          // Make WebSocket globally available for AudioVisualizer
          (window as any).voiceAgentWebSocket = newSocket;
          
          onOpen?.();
        };

        newSocket.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            
            // Handle audio response from server
            if (message.type === 'audio_response' && message.data) {
              playAudioResponse(message.data);
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
          setError(`WebSocket connection error: ${newSocket.readyState === WebSocket.CLOSED ? 'Connection rejected by server' : 'Network error'}`);
          setConnectionStatus('error');
          onError?.(event);
        };

        newSocket.onclose = () => {
          console.log('WebSocket disconnected');
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

          // Attempt to reconnect if we haven't exceeded max attempts
          if (reconnectAttemptsRef.current < maxReconnectAttempts) {
            reconnectAttemptsRef.current += 1;
            console.log(`Attempting to reconnect (${reconnectAttemptsRef.current}/${maxReconnectAttempts})...`);
            
            reconnectTimeoutRef.current = setTimeout(() => {
              connectWebSocket();
            }, reconnectInterval);
          } else {
            setError('Max reconnection attempts reached');
            setConnectionStatus('error');
          }
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