export interface User {
  id: string;
  email: string;
  created_at?: string;
  updated_at?: string;
}

export interface AuthResponse {
  access_token: string;
  refresh_token?: string;
  expires_in?: number;
  user: User;
}

export interface ChatMessage {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  language?: string;
  audioUrl?: string;
}

export interface ChatRequest {
  text: string;
  language: string;
}

export interface ChatResponse {
  response: string;
  audio_url?: string;
  language?: string;
}

export interface WebSocketMessage {
  type: 'message' | 'audio' | 'status' | 'error' | 'audio_response' | 'vad_status' | 'text_response' | 'audio_processed' | 'listening_started' | 'listening_stopped' | 'heartbeat';
  data?: any;
  text?: string;
  isActive?: boolean;
  timestamp?: string;
}

export interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string) => Promise<void>;
  loginAsGuest: () => Promise<void>;
  logout: () => void;
  error: string | null;
}

export interface WebSocketContextType {
  socket: WebSocket | null;
  isConnected: boolean;
  isConnecting: boolean;
  lastMessage: WebSocketMessage | null;
  sendMessage: (message: any) => void;
  error: string | null;
}

export interface VoiceSettings {
  language: 'en' | 'es' | 'fr' | 'de' | 'it' | 'pt' | 'ru' | 'ja' | 'ko' | 'zh';
  voice: string;
  speed: number;
  pitch: number;
}

export interface ApiError {
  detail: string;
  status_code: number;
}

export type ConnectionStatus = 'connected' | 'connecting' | 'disconnected' | 'error';

export type Theme = 'light' | 'dark' | 'system'; 