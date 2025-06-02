// WebSocket Message Types
export interface AudioChunkMessage {
  type: 'audio_chunk';
  data: Uint8Array;
}

export interface ControlMessage {
  type: 'control';
  action: 'mute' | 'unmute' | 'end_session';
}

export interface TranscriptMessage {
  type: 'transcript';
  partial?: string;
  final?: string;
}

export interface AIResponseMessage {
  type: 'ai_response';
  token: string;
  complete?: boolean;
}

export interface AudioResponseMessage {
  type: 'audio_chunk';
  data: Uint8Array;
}

export interface ControlResponseMessage {
  type: 'control';
  action: 'stop_audio' | 'session_ended';
}

export interface ErrorMessage {
  type: 'error';
  message: string;
}

export type WebSocketMessage = 
  | AudioChunkMessage 
  | ControlMessage;

export type WebSocketResponse = 
  | TranscriptMessage 
  | AIResponseMessage 
  | AudioResponseMessage 
  | ControlResponseMessage 
  | ErrorMessage;

// Voice Agent State Types
export interface VoiceAgentState {
  isConnected: boolean;
  isRecording: boolean;
  isMuted: boolean;
  isProcessing: boolean;
  isSpeaking: boolean;
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
  error?: string;
}

export interface TranscriptState {
  partial: string;
  final: string[];
  aiResponse: string;
  isAiResponding: boolean;
}

export interface AudioState {
  isRecording: boolean;
  isPlayingTTS: boolean;
  audioLevel: number;
  deviceId?: string;
  sampleRate: number;
}

// Configuration Types
export interface AudioConfig {
  sampleRate: number;
  channelCount: number;
  frameDuration: number;
  enableEchoCancellation: boolean;
  enableNoiseSuppression: boolean;
  enableAutoGainControl: boolean;
}

export interface AudioConstraints {
  deviceId?: string;
  sampleRate?: number;
  channelCount?: number;
  echoCancellation?: boolean;
  noiseSuppression?: boolean;
  autoGainControl?: boolean;
}

export interface MediaDeviceInfo {
  deviceId: string;
  label: string;
  kind: 'audioinput';
}

export interface VoiceAgentConfig {
  wsUrl: string;
  apiUrl: string;
  apiKey: string;
  autoConnect: boolean;
  reconnectAttempts: number;
  reconnectDelay: number;
  audio: AudioConfig;
}

// Performance Metrics Types
export interface LatencyMetrics {
  endToEndLatency: number;
  speechToTranscript: number;
  transcriptToResponse: number;
  responseToAudio: number;
  websocketRoundTrip: number;
}

export interface SessionMetrics {
  sessionDuration: number;
  audioPacketsSent: number;
  audioPacketsReceived: number;
  transcriptWords: number;
  aiResponseWords: number;
  errors: number;
}

// Avatar Types
export interface AvatarState {
  isVisible: boolean;
  isSpeaking: boolean;
  emotion: 'neutral' | 'happy' | 'thinking' | 'speaking';
  lipSyncData?: number[];
}

export interface Viseme {
  time: number;
  value: string;
  weight: number;
}

// UI Component Props
export interface VoiceAgentProps {
  config?: Partial<VoiceAgentConfig>;
  onStateChange?: (state: VoiceAgentState) => void;
  onTranscript?: (text: string, isFinal: boolean) => void;
  onAIResponse?: (text: string, isComplete: boolean) => void;
  onError?: (error: string) => void;
  onMetrics?: (metrics: SessionMetrics) => void;
  className?: string;
}

export interface ControlButtonsProps {
  isMuted: boolean;
  isConnected: boolean;
  isProcessing: boolean;
  onMuteToggle: () => void;
  onEndSession: () => void;
  className?: string;
}

export interface TranscriptDisplayProps {
  partialText: string;
  finalTexts: string[];
  aiResponse: string;
  isAiResponding: boolean;
  maxLines?: number;
  className?: string;
  transparent?: boolean;
}

export interface AvatarProps {
  isVisible: boolean;
  isSpeaking: boolean;
  lipSyncData?: number[];
  emotion?: 'neutral' | 'happy' | 'thinking' | 'speaking';
  className?: string;
}

// Event Handler Types
export type StateChangeHandler = (state: VoiceAgentState) => void;
export type TranscriptHandler = (transcript: string, isFinal: boolean) => void;
export type AIResponseHandler = (response: string, isComplete: boolean) => void;
export type ErrorHandler = (error: string) => void;
export type MetricsHandler = (metrics: SessionMetrics) => void;

// Utility Types
export interface ConnectionInfo {
  isConnected: boolean;
  connectionStatus: string;
  error?: string;
}

// Default Configurations
export const DEFAULT_AUDIO_CONFIG: AudioConfig = {
  sampleRate: 16000,
  channelCount: 1,
  frameDuration: 120, // 120ms frames
  enableNoiseSuppression: true,
  enableEchoCancellation: true,
  enableAutoGainControl: true,
};

export const DEFAULT_VOICE_AGENT_CONFIG: VoiceAgentConfig = {
  wsUrl: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001/ws',
  apiUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001',
  apiKey: process.env.NEXT_PUBLIC_API_KEY || '',
  autoConnect: false,
  reconnectAttempts: 5,
  reconnectDelay: 1000,
  audio: {
    sampleRate: 16000,
    channelCount: 1,
    frameDuration: 100,
    enableEchoCancellation: true,
    enableNoiseSuppression: true,
    enableAutoGainControl: true,
  },
}; 