# Ultra-Fast Voice Agent 🚀

A production-ready, real-time voice conversation agent powered by Google's Gemini 2.0 Flash, optimized for ultra-low latency interactions.

## ✨ Recent Improvements

This repository has undergone systematic optimization addressing 12+ critical issues across infrastructure, audio processing, state management, and code quality:

### 🔧 Infrastructure Fixes
- **Standardized API Configuration**: Unified `GOOGLE_API_KEY` usage across all components
- **Consistent Model Paths**: All models now use `/app/models` path consistently
- **Optimized Docker Build**: Streamlined model downloading and consistent file structure

### 🎵 Audio Pipeline Enhancements  
- **Improved TTS Playback**: Eliminated per-chunk AudioContext creation with persistent AudioPlayer
- **Robust FFmpeg Processing**: Enhanced format handling with WebM prioritization and better error recovery
- **Simplified EOS Handling**: Streamlined end-of-speech detection to prevent race conditions

### ⚡ State Management & Reliability
- **Unified Keep-alive**: Simplified to client-side ping/server pong pattern
- **Enhanced Error Propagation**: Comprehensive error handling with user-friendly messages
- **Pipeline State Tracking**: Robust watchdog timer and processing state management

### 🏗️ Code Quality Improvements
- **Consistent Logging**: Elevated critical errors from debug to warning/error levels
- **Better Error Messages**: More descriptive error propagation to frontend
- **Cleaner Architecture**: Removed redundant components and simplified control flow

## 🏗️ Architecture

### Backend Components
- **Speech-to-Text**: sherpa-ncnn with Zipformer-20M model
- **LLM**: Google Gemini 2.0 Flash for conversational AI
- **Text-to-Speech**: Piper TTS with en_US-libritts-high model
- **VAD**: Silero Voice Activity Detection
- **WebSocket Handler**: Real-time bidirectional communication

### Frontend Components  
- **React/Next.js**: Modern web interface
- **WebSocket Client**: Real-time communication with audio streaming
- **Audio Processing**: MediaRecorder API with Opus encoding
- **Voice Activity Detection**: Client-side VAD for responsive interaction

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Google AI API key

### Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Configure your Google AI API key
echo "GOOGLE_API_KEY=your_gemini_api_key_here" >> .env
```

### Launch with Docker
```bash
# Build and start all services
docker-compose up --build

# Access the application
open http://localhost:3000
```

### Local Development
```bash
# Backend
cd backend
pip install -r requirements.txt
python -m app.models.download_models  # Download models
uvicorn app.main:app --host 0.0.0.0 --port 8001

# Frontend  
cd frontend
npm install
npm run dev
```

## 🔧 Configuration

### Environment Variables
```env
# Required
GOOGLE_API_KEY=your_gemini_api_key

# Optional - Audio Processing
SAMPLE_RATE=16000
AUDIO_FRAME_MS=120
MAX_CONCURRENT_SESSIONS=3

# Optional - WebSocket
WS_URL=ws://localhost:8001/ws
RECONNECT_ATTEMPTS=5
RECONNECT_DELAY=1000
```

### Model Configuration
Models are automatically downloaded to `/app/models`:
- **STT**: sherpa-ncnn Zipformer-20M (English)
- **TTS**: Piper en_US-libritts-high (English)
- **VAD**: Silero VAD (Universal)

## 📊 Performance Optimizations

### Audio Processing
- **WebM/Opus Priority**: Optimized format selection for minimal latency
- **Chunk Buffering**: Smart buffering to reduce FFmpeg overhead
- **Failed Chunk Recovery**: Accumulation and retry logic for partial audio

### Real-time Communication
- **Persistent Audio Context**: Eliminates per-chunk initialization overhead
- **Pipeline State Management**: Prevents race conditions and ensures proper sequencing  
- **Barge-in Support**: Instant TTS interruption for natural conversation flow

### Error Handling
- **Graceful Degradation**: Service failures don't crash the system
- **User-friendly Messages**: Clear error communication to frontend
- **Automatic Recovery**: Retry mechanisms for transient failures

## 🛠️ API Reference

### WebSocket Messages

#### Client → Server
```json
// Audio data (binary)
Binary data: WebM/Opus encoded audio chunks

// Control messages  
{"type": "ping", "t": timestamp}
{"type": "eos"}  // End of speech
{"type": "control", "action": "mute|unmute|end_session"}
```

#### Server → Client
```json
// Status updates
{"type": "status", "session_id": "...", "status": "processing"}

// Transcription
{"type": "transcript", "partial": "...", "final": "..."}

// AI responses
{"type": "ai_response", "token": "...", "complete": true}

// Audio playback
{"type": "audio_chunk", "audio_data": "base64..."}

// Keep-alive
{"type": "pong", "t": timestamp}

// Errors
{"type": "error", "message": "..."}
```

## 🧪 Testing

### Health Checks
```bash
# Backend health
curl http://localhost:8001/health

# WebSocket connection
wscat -c ws://localhost:8001/ws
```

### Load Testing
```bash
# Install artillery
npm install -g artillery

# Run WebSocket load test
artillery run tests/websocket-load-test.yml
```

## 📈 Monitoring

### Metrics Available
- WebSocket connection count
- Processing latency (end-to-end)
- Barge-in frequency  
- Error rates by component
- Audio chunk processing stats

### Logging Levels
- **ERROR**: Critical failures requiring attention
- **WARNING**: Important issues that don't break functionality
- **INFO**: Operational information and key events
- **DEBUG**: Detailed tracing for development

## 🔒 Security

### API Key Protection
- Environment variable configuration
- No hardcoded credentials
- Docker secrets support

### WebSocket Security
- Connection limits (configurable)
- Session isolation
- Graceful error handling

## 🐳 Docker Configuration

### Production Deployment
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  backend:
    build: ./backend
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - MAX_CONCURRENT_SESSIONS=10
    volumes:
      - models_cache:/app/models
    restart: unless-stopped

  frontend:
    build: ./frontend
    environment:
      - NEXT_PUBLIC_WS_URL=wss://your-domain.com/ws
    restart: unless-stopped
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Guidelines
- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation for API changes
- Ensure Docker builds work correctly

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini 2.0 Flash for conversational AI
- Piper TTS for high-quality speech synthesis  
- sherpa-ncnn for efficient speech recognition
- Silero team for robust voice activity detection
- FFmpeg community for audio processing tools

---

**Built with ❤️ for real-time voice interaction** 