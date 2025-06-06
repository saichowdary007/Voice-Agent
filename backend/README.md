# Voice Agent Backend - Rewritten Architecture

This is a completely rewritten backend for the Voice Agent application with improved architecture, better error handling, and enhanced performance.

## Key Improvements

### 🏗️ **Unified Architecture**
- **Single Voice Service**: Combines VAD, STT, LLM, and TTS into one cohesive service
- **Simplified WebSocket Management**: Clean session handling with proper cleanup
- **Better Error Recovery**: Robust error handling and automatic recovery mechanisms

### 🚀 **Performance Enhancements**
- **Async-First Design**: All operations are properly async with no blocking calls
- **Efficient Resource Management**: Better memory usage and cleanup
- **Optimized Audio Pipeline**: Streamlined audio processing with reduced latency

### 🔧 **Technical Improvements**
- **Proper Async/Await**: Fixed coroutine warnings and blocking operations
- **Clean Session Management**: Proper WebSocket lifecycle management
- **Structured Logging**: Better logging with session context
- **Health Monitoring**: Comprehensive health checks and metrics

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │────│ WebSocket Manager │────│ Session Handler │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Voice Service  │
                       └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              ┌─────────┐ ┌─────────┐ ┌─────────┐
              │   VAD   │ │   STT   │ │   LLM   │
              └─────────┘ └─────────┘ └─────────┘
                                │
                                ▼
                          ┌─────────┐
                          │   TTS   │
                          └─────────┘
```

## Key Components

### 1. **Voice Service** (`services/voice_service.py`)
- Unified interface for all AI services
- Handles audio processing pipeline
- Manages service state and health
- Provides performance metrics

### 2. **WebSocket Manager** (`app/websocket_manager.py`)
- Manages WebSocket connections and sessions
- Handles message routing and error recovery
- Implements proper cleanup and resource management
- Provides session timeouts and health checks

### 3. **Individual Services**
- **VAD Service**: Voice Activity Detection
- **STT Service**: Speech-to-Text (Azure Speech)
- **LLM Service**: Language Model (Gemini 2.0 Flash)
- **TTS Service**: Text-to-Speech (Piper TTS)

## Configuration

The backend uses environment variables for configuration:

```bash
# Core Settings
DEBUG=false
LOG_LEVEL=INFO
ALLOWED_ORIGINS=*

# Audio Settings
SAMPLE_RATE=16000
CHANNELS=1
FRAME_DURATION_MS=32
VAD_THRESHOLD=0.6

# Session Management
MAX_CONCURRENT_SESSIONS=3
SESSION_TIMEOUT=300
SPEECH_TIMEOUT=3.0

# API Keys
GOOGLE_API_KEY=your_gemini_api_key
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=eastus

# Feature Flags
ENABLE_METRICS=true
PRELOAD_MODELS=true
```

## API Endpoints

### WebSocket
- `ws://localhost:8000/ws` - Main voice communication endpoint

### HTTP
- `GET /` - Service information
- `GET /health` - Health check with service status
- `GET /metrics` - Performance metrics (if enabled)

## WebSocket Protocol

### Client → Server Messages

**Audio Data (Binary)**
```
Raw PCM audio data (16-bit, 16kHz, mono)
```

**Control Messages (JSON)**
```json
{
  "type": "ping"
}

{
  "type": "mute",
  "muted": true
}

{
  "type": "end_speech"
}
```

### Server → Client Messages

**Status Messages**
```json
{
  "type": "status",
  "session_id": "session_123",
  "ready": true,
  "config": {
    "sample_rate": 16000,
    "channels": 1,
    "frame_duration_ms": 32
  }
}
```

**Speech Recognition**
```json
{
  "type": "partial_transcript",
  "text": "Hello wor..."
}

{
  "type": "final_transcript",
  "text": "Hello world"
}
```

**AI Response**
```json
{
  "type": "ai_response",
  "text": "Hi there! How can I help you?"
}
```

**TTS Audio**
```json
{
  "type": "tts_audio",
  "audio": "hex_encoded_audio_data"
}

{
  "type": "tts_complete"
}
```

## Running the Backend

### Development
```bash
cd backend
pip install -r requirements.txt
python -m app.main
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t voice-agent-backend .
docker run -p 8000:8000 voice-agent-backend
```

## Error Handling

The new backend includes comprehensive error handling:

- **Service Failures**: Graceful degradation when services are unavailable
- **WebSocket Errors**: Automatic reconnection and cleanup
- **Audio Processing**: Robust error recovery in the audio pipeline
- **Resource Management**: Proper cleanup of resources and sessions

## Monitoring

### Health Checks
The `/health` endpoint provides detailed service status:

```json
{
  "status": "healthy",
  "timestamp": 1704067200,
  "services": {
    "voice_service": {
      "available": true,
      "processing_stats": {
        "frames_processed": 1234,
        "speech_sessions": 56,
        "errors": 0
      }
    },
    "vad": {"available": true},
    "stt": {"available": true},
    "llm": {"available": true},
    "tts": {"available": true}
  }
}
```

### Metrics
When enabled, the `/metrics` endpoint provides performance data:

```json
{
  "timestamp": 1704067200,
  "uptime_seconds": 3600,
  "counters": {
    "websocket_connections": 10,
    "audio_frames_processed": 5000,
    "ai_requests": 25
  },
  "derived": {
    "requests_per_second": 0.5,
    "error_rate": 0.1,
    "average_latencies": {
      "audio_processing": 15.2,
      "ai_response": 850.5
    }
  }
}
```

## Migration from Old Backend

The new backend is a complete rewrite with breaking changes:

1. **WebSocket Protocol**: Updated message format
2. **Service Architecture**: Unified voice service instead of separate services
3. **Configuration**: New environment variable names
4. **Error Handling**: Different error response format

Please update your frontend client to work with the new protocol.

## Troubleshooting

### Common Issues

1. **Service Unavailable**: Check API keys and network connectivity
2. **Audio Issues**: Verify audio format (16-bit PCM, 16kHz, mono)
3. **High Latency**: Check network and service response times
4. **Memory Usage**: Monitor metrics and adjust session limits

### Logs

The backend uses structured logging. Set `LOG_LEVEL=DEBUG` for detailed logs:

```bash
export LOG_LEVEL=DEBUG
python -m app.main
```

## Contributing

When contributing to the backend:

1. Follow async/await patterns
2. Add proper error handling
3. Include logging with context
4. Update tests and documentation
5. Monitor performance impact 