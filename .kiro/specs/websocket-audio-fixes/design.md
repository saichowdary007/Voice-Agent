# Design Document

## Overview

This design addresses critical WebSocket connection failures (1011 errors), audio processing inefficiencies, and VAD re-initialization loops in the Voice-Agent system. Based on forensic analysis comparing the working "loco" frontend with the current implementation, we need to fix three core mismatches: WebSocket protocol alignment, audio format standardization, and resource lifecycle management.

The solution implements a systematic approach to achieve stable WebSocket connections, optimized audio processing, and reliable voice activity detection while maintaining the target sub-3-second response times.

## Architecture

### Current Issues Identified

1. **WebSocket Protocol Mismatch**: Frontend expects `wss://<HOST>/ws/audio?model=whisper` with `"stream-audio"` protocol, but backend serves `ws://<HOST>:8000/api/v1/stt` expecting `"binary"` protocol
2. **Audio Format Incompatibility**: Frontend sends stereo 48kHz 1000ms chunks, backend expects mono 16kHz 250ms chunks
3. **Resource Lifecycle Problems**: VAD instances are reset on every reconnection, causing initialization storms
4. **Audio Processing Bottlenecks**: ffmpeg spawns new processes for every blob, causing CPU spikes and ping timeouts

### Target Architecture

```mermaid
graph TB
    subgraph "Frontend (React)"
        A[MediaRecorder] --> B[Audio Chunks 250ms]
        B --> C[WebSocket Client]
        C --> D[Reconnection Logic]
    end
    
    subgraph "Backend (FastAPI)"
        E[WebSocket Handler] --> F[Protocol Negotiation]
        F --> G[Audio Stream Processor]
        G --> H[Persistent VAD Instance]
        H --> I[STT Pipeline]
        I --> J[LLM Processing]
        J --> K[TTS Response]
    end
    
    C -.->|"binary" or "stream-audio"| E
    B -.->|"mono 16kHz 250ms opus"| G
    G -.->|"streaming ffmpeg"| H
    
    subgraph "Connection Health"
        L[Heartbeat Manager]
        M[Connection Monitor]
        N[Backoff Strategy]
    end
    
    D --> L
    E --> M
    M --> N
```

## Components and Interfaces

### 1. WebSocket Protocol Alignment

#### Frontend Changes (`useWebSocket.ts`)
```typescript
interface WebSocketConfig {
  endpoint: string;           // "/api/v1/stt" 
  protocols: string[];        // ["binary", "stream-audio"]
  heartbeatInterval: number;  // 30000ms
  reconnectBackoff: number[]; // [3000, 5000, 8000, 12000, 20000]
}

class WebSocketManager {
  connect(config: WebSocketConfig): Promise<WebSocket>
  enableHeartbeat(): void
  handleProtocolNegotiation(): void
  reconnectWithBackoff(): Promise<void>
}
```

#### Backend Changes (`websocket_handlers.py`)
```python
class WebSocketProtocolHandler:
    SUPPORTED_PROTOCOLS = ["binary", "stream-audio"]
    
    async def negotiate_protocol(self, websocket: WebSocket) -> str
    async def handle_heartbeat(self, websocket: WebSocket, message: dict) -> None
    async def validate_connection(self, websocket: WebSocket) -> bool
```

### 2. Audio Format Standardization

#### Frontend Audio Configuration (`VoiceInterface.tsx`)
```typescript
interface AudioConfig {
  mimeType: "audio/webm;codecs=opus";
  channelCount: 1;           // Mono
  sampleRate: 16000;         // 16kHz
  chunkDuration: 250;        // 250ms chunks
  audioBitsPerSecond: 128000; // 128kbps
}

class AudioProcessor {
  configureMediaRecorder(config: AudioConfig): MediaRecorder
  processAudioChunk(chunk: Blob): Promise<ArrayBuffer>
  validateAudioFormat(chunk: ArrayBuffer): boolean
}
```

#### Backend Audio Processing (`audio_preprocessor.py`)
```python
class StreamingAudioProcessor:
    def __init__(self):
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.audio_buffer = bytearray()
        
    async def initialize_streaming_ffmpeg(self) -> None
    async def process_audio_stream(self, audio_chunk: bytes) -> bytes
    async def cleanup_ffmpeg_process(self) -> None
```

### 3. VAD Lifecycle Management

#### Persistent VAD Instance (`vad.py`)
```python
class PersistentVAD:
    def __init__(self):
        self.vad_instance: webrtcvad.Vad = None
        self.session_state: Dict[str, Any] = {}
        self.is_initialized = False
        
    async def initialize_session(self, websocket_id: str) -> None
    async def process_audio_chunk(self, audio: bytes, websocket_id: str) -> VADResult
    async def cleanup_session(self, websocket_id: str) -> None
    def should_reset_vad(self, error_count: int) -> bool
```

#### WebSocket Session Management (`websocket_handlers.py`)
```python
class WebSocketSessionManager:
    def __init__(self):
        self.active_sessions: Dict[str, SessionState] = {}
        self.vad_manager = PersistentVAD()
        
    async def create_session(self, websocket: WebSocket) -> str
    async def cleanup_session(self, session_id: str) -> None
    async def get_session_vad(self, session_id: str) -> VAD
```

### 4. Connection Health Monitoring

#### Heartbeat System
```python
class HeartbeatManager:
    def __init__(self, interval: int = 30):
        self.interval = interval
        self.active_connections: Dict[str, ConnectionHealth] = {}
        
    async def start_heartbeat(self, websocket: WebSocket) -> None
    async def handle_pong(self, websocket: WebSocket) -> None
    async def check_connection_health(self) -> List[str]  # Returns unhealthy connections
```

#### Connection Recovery
```typescript
class ConnectionRecovery {
  private backoffDelays = [3000, 5000, 8000, 12000, 20000];
  private currentAttempt = 0;
  
  async reconnectWithBackoff(): Promise<WebSocket>
  resetBackoff(): void
  shouldGiveUp(): boolean
}
```

## Data Models

### WebSocket Message Protocol
```typescript
interface WebSocketMessage {
  type: 'audio_chunk' | 'heartbeat' | 'speech_started' | 'speech_final' | 'stt_result' | 'error';
  data?: any;
  timestamp: string;
  session_id?: string;
}

interface AudioChunkMessage extends WebSocketMessage {
  type: 'audio_chunk';
  data: string;           // base64 encoded audio
  is_final: boolean;
  chunk_index: number;
  format: AudioFormat;
}

interface AudioFormat {
  sample_rate: 16000;
  channels: 1;
  bit_depth: 16;
  codec: 'opus';
  duration_ms: 250;
}
```

### Session State Management
```python
@dataclass
class SessionState:
    session_id: str
    websocket: WebSocket
    vad_instance: VAD
    audio_processor: StreamingAudioProcessor
    last_heartbeat: float
    connection_health: ConnectionHealth
    audio_buffer: bytearray
    speech_state: SpeechState
    
@dataclass
class SpeechState:
    is_speaking: bool = False
    speech_start_time: Optional[float] = None
    silence_start_time: Optional[float] = None
    total_audio_duration: float = 0.0
    chunk_count: int = 0
```

## Error Handling

### WebSocket Error Recovery
```python
class WebSocketErrorHandler:
    ERROR_CODES = {
        1011: "Unexpected server condition",
        1008: "Authentication failed", 
        1006: "Connection lost",
        1000: "Normal closure"
    }
    
    async def handle_connection_error(self, error_code: int, websocket: WebSocket) -> None
    async def should_attempt_reconnect(self, error_code: int) -> bool
    async def cleanup_failed_connection(self, websocket: WebSocket) -> None
```

### Audio Processing Error Handling
```python
class AudioProcessingErrorHandler:
    async def handle_ffmpeg_error(self, error: Exception) -> None
    async def handle_vad_error(self, error: Exception, session_id: str) -> None
    async def handle_buffer_overflow(self, buffer_size: int) -> None
    
    def is_recoverable_error(self, error: Exception) -> bool
    async def attempt_recovery(self, error: Exception, context: Dict) -> bool
```

### Frontend Error Handling
```typescript
class AudioErrorHandler {
  handleMediaRecorderError(error: MediaRecorderErrorEvent): void
  handleWebSocketError(error: Event): void
  handlePermissionError(error: DOMException): void
  
  shouldRetryOperation(error: Error): boolean
  getErrorRecoveryStrategy(error: Error): RecoveryStrategy
}
```

## Testing Strategy

### Unit Tests

#### WebSocket Protocol Tests
```python
class TestWebSocketProtocol:
    async def test_protocol_negotiation_binary(self)
    async def test_protocol_negotiation_stream_audio(self)
    async def test_heartbeat_handling(self)
    async def test_connection_timeout_recovery(self)
```

#### Audio Processing Tests
```python
class TestAudioProcessing:
    async def test_streaming_ffmpeg_initialization(self)
    async def test_audio_chunk_processing_250ms(self)
    async def test_mono_16khz_conversion(self)
    async def test_buffer_overflow_handling(self)
```

#### VAD Lifecycle Tests
```python
class TestVADLifecycle:
    async def test_persistent_vad_creation(self)
    async def test_session_cleanup(self)
    async def test_vad_state_preservation(self)
    async def test_speech_boundary_detection(self)
```

### Integration Tests

#### End-to-End Voice Pipeline
```python
class TestVoicePipeline:
    async def test_complete_voice_interaction(self)
    async def test_connection_recovery_during_speech(self)
    async def test_concurrent_sessions(self)
    async def test_performance_under_load(self)
```

#### Frontend Integration Tests
```typescript
describe('WebSocket Audio Integration', () => {
  test('establishes stable connection with correct protocol')
  test('sends properly formatted audio chunks')
  test('handles connection recovery gracefully')
  test('maintains audio quality during reconnection')
})
```

### Performance Tests

#### Latency Benchmarks
```python
class TestLatencyBenchmarks:
    async def test_websocket_connection_time(self)
    async def test_audio_processing_latency(self)
    async def test_vad_response_time(self)
    async def test_end_to_end_latency(self)
    
    def assert_latency_under_threshold(self, latency_ms: float, threshold: float)
```

#### Load Testing
```python
class TestLoadHandling:
    async def test_concurrent_websocket_connections(self)
    async def test_audio_processing_under_load(self)
    async def test_memory_usage_stability(self)
    async def test_connection_recovery_at_scale(self)
```

### Diagnostic Tools

#### Connection Health Monitor
```python
class ConnectionHealthMonitor:
    def monitor_websocket_metrics(self) -> Dict[str, Any]
    def track_audio_processing_performance(self) -> Dict[str, Any]
    def analyze_vad_behavior(self) -> Dict[str, Any]
    def generate_health_report(self) -> HealthReport
```

#### Audio Pipeline Diagnostics
```python
class AudioPipelineDiagnostics:
    def analyze_audio_format_compliance(self, audio_data: bytes) -> FormatAnalysis
    def measure_processing_latency(self) -> LatencyMetrics
    def detect_audio_quality_issues(self) -> List[QualityIssue]
    def validate_vad_accuracy(self) -> VADAccuracyReport
```

The design ensures that all identified issues from the forensic analysis are systematically addressed while maintaining the existing architecture's strengths and adding robust error handling and monitoring capabilities.