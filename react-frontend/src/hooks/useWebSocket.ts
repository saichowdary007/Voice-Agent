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
          onOpen?.();
        };

        newSocket.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            setLastMessage(message);
            onMessage?.(message);
          } catch (err) {
            console.error('Failed to parse WebSocket message:', err);
            setError('Failed to parse message from server');
          }
        };

        newSocket.onerror = (event) => {
          console.error('WebSocket error:', event);
          setError('WebSocket connection error');
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