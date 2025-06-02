import {
  VoiceAgentConfig,
  WebSocketMessage,
  WebSocketResponse,
  VoiceAgentState,
  ConnectionInfo,
  SessionMetrics,
} from './types';

export interface WebSocketState {
  isConnected: boolean;
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
  error?: string;
}

/**
 * WebSocket manager for voice agent communication
 */
export class WebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private config: any;
  private onMessage?: (data: any) => void;
  private onAudioData?: (audioData: ArrayBuffer) => void;
  private onOpen?: () => void;
  private onClose?: () => void;
  private onError?: (error: string) => void;
  private onStateChange?: (state: WebSocketState) => void;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  constructor(config: any) {
    this.config = config;
    this.url = config.wsUrl || 'ws://localhost:8001/ws';
    this.maxReconnectAttempts = config.reconnectAttempts || 5;
    this.reconnectDelay = config.reconnectDelay || 1000;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.updateState({ connectionStatus: 'connecting', isConnected: false });
        
        this.ws = new WebSocket(this.url);
        
        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.reconnectAttempts = 0;
          this.updateState({ connectionStatus: 'connected', isConnected: true });
          this.onOpen?.();
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            if (event.data instanceof ArrayBuffer) {
              // Handle binary audio data (TTS)
              console.log('Received TTS audio data:', event.data.byteLength, 'bytes');
              this.onAudioData?.(event.data);
            } else if (typeof event.data === 'string') {
              // Handle JSON messages
              const data = JSON.parse(event.data);
              this.onMessage?.(data);
            } else if (event.data instanceof Blob) {
              // Handle blob data
              event.data.arrayBuffer().then((buffer) => {
                console.log('Received TTS audio blob:', buffer.byteLength, 'bytes');
                this.onAudioData?.(buffer);
              });
            }
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        this.ws.onclose = () => {
          console.log('WebSocket disconnected');
          this.updateState({ connectionStatus: 'disconnected', isConnected: false });
          this.onClose?.();
          
          // Attempt reconnection if not manually disconnected
          if (this.reconnectAttempts < this.maxReconnectAttempts) {
            setTimeout(() => {
              this.reconnectAttempts++;
              this.connect().catch(console.error);
            }, this.reconnectDelay);
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.updateState({ connectionStatus: 'error', isConnected: false, error: 'Connection failed' });
          this.onError?.('WebSocket connection error');
          reject(new Error('WebSocket connection failed'));
        };
      } catch (error) {
        this.updateState({ connectionStatus: 'error', isConnected: false, error: String(error) });
        reject(error);
      }
    });
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.updateState({ connectionStatus: 'disconnected', isConnected: false });
  }

  send(data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket is not connected');
    }
  }

  async sendAudio(audioData: ArrayBuffer): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      // Convert ArrayBuffer to base64 for transmission
      const base64Audio = this.arrayBufferToBase64(audioData);
      this.send({
        type: 'audio_chunk',
        data: base64Audio,
        timestamp: Date.now()
      });
    } else {
      throw new Error('WebSocket is not connected');
    }
  }

  async sendControl(action: string): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.send({
        type: 'control',
        action: action,
        timestamp: Date.now()
      });
    } else {
      throw new Error('WebSocket is not connected');
    }
  }

  setMessageHandler(handler: (data: any) => void): void {
    this.onMessage = handler;
  }

  setAudioDataHandler(handler: (audioData: ArrayBuffer) => void): void {
    this.onAudioData = handler;
  }

  setOpenHandler(handler: () => void): void {
    this.onOpen = handler;
  }

  setCloseHandler(handler: () => void): void {
    this.onClose = handler;
  }

  setErrorHandler(handler: (error: string) => void): void {
    this.onError = handler;
  }

  setStateChangeHandler(handler: (state: WebSocketState) => void): void {
    this.onStateChange = handler;
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  cleanup(): void {
    this.disconnect();
    this.onMessage = undefined;
    this.onAudioData = undefined;
    this.onOpen = undefined;
    this.onClose = undefined;
    this.onError = undefined;
    this.onStateChange = undefined;
    console.log('WebSocket manager cleaned up');
  }

  private updateState(state: Partial<WebSocketState>): void {
    if (this.onStateChange) {
      const currentState: WebSocketState = {
        isConnected: this.isConnected(),
        connectionStatus: 'disconnected',
        ...state
      };
      this.onStateChange(currentState);
    }
  }

  private arrayBufferToBase64(buffer: ArrayBuffer): string {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }
} 