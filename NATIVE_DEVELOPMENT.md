# Native Development Setup

This guide explains how to run the Voice Agent natively (without Docker) using the new Kokoro TTS engine.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with virtual environment
2. **Docker** (for Kokoro TTS service)
3. **Node.js 16+** (for frontend)

### 1. Start All Services

```bash
# Start backend + Kokoro TTS automatically
make start-native

# OR manually:
./start_voice_agent.sh
```

This will:
- ✅ Activate virtual environment
- ✅ Install missing dependencies
- ✅ Start Kokoro TTS service (Docker)
- ✅ Start backend on port 8080
- ✅ Verify all services are healthy

### 2. Start Frontend (separate terminal)

```bash
cd react-frontend
npm start
```

### 3. Test the System

```bash
# Test Kokoro TTS connection
make test-kokoro

# OR manually:
python test_kokoro_connection.py
```

## 🛠️ Services Overview

| Service | Port | Purpose |
|---------|------|---------|
| **Backend API** | 8080 | Main Voice Agent server |
| **Kokoro TTS** | 8880 | Text-to-Speech synthesis |
| **Frontend** | 3000 | React web interface |

## 🔧 Configuration

### Backend Configuration (`src/config.py`)

```python
# Kokoro TTS Settings
KOKORO_TTS_URL = "http://localhost:8880"
KOKORO_TTS_VOICE = "af_bella"  # Available: af_bella, af_nicole, etc.
KOKORO_TTS_MODEL = "kokoro"
KOKORO_TTS_SPEED = 1.0
```

### Environment Variables

```bash
# Backend
export PORT=8080
export HOST=0.0.0.0

# Kokoro TTS
export KOKORO_TTS_URL=http://localhost:8880
export KOKORO_TTS_VOICE=af_bella
```

## 🎤 Kokoro TTS Engine

### Features
- **Ultra-fast synthesis** (50-200ms)
- **High-quality voices** (af_bella, af_nicole, etc.)
- **Streaming support** for real-time audio
- **OpenAI-compatible API**

### Manual Docker Setup

```bash
# Pull and run Kokoro TTS
docker pull ghcr.io/remsky/kokoro-fastapi-cpu:latest
docker run -d --name kokoro-tts -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest

# Check status
curl http://localhost:8880/health
```

### Available Voices

- `af_bella` - Clear, professional female voice
- `af_nicole` - Warm, friendly female voice  
- `am_adam` - Professional male voice
- `am_michael` - Casual male voice

## 🛠️ Troubleshooting

### Port Conflicts

If you get port conflicts:

```bash
# Check what's using port 8080
lsof -i :8080

# Kill existing processes
./stop_voice_agent.sh

# Restart
./start_voice_agent.sh
```

### TTS Not Working

1. **Check Kokoro TTS status:**
   ```bash
   curl http://localhost:8880/health
   ```

2. **Test TTS synthesis:**
   ```bash
   python test_kokoro_connection.py
   ```

3. **Restart Kokoro TTS:**
   ```bash
   docker stop kokoro-tts && docker rm kokoro-tts
   docker run -d --name kokoro-tts -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest
   ```

### WebSocket Connection Issues

1. **Verify backend is on port 8080:**
   ```bash
   curl http://localhost:8080/health
   ```

2. **Check frontend WebSocket URL:**
   - Should connect to `ws://localhost:8080/ws/`
   - Check browser console for connection errors

### Audio Issues

1. **Check browser permissions:**
   - Allow microphone access
   - Enable autoplay for audio

2. **Test audio pipeline:**
   ```bash
   # Check if ffplay is available (for audio output)
   which ffplay
   
   # Install if missing (macOS)
   brew install ffmpeg
   ```

## 📊 Performance Expectations

| Component | Expected Latency |
|-----------|------------------|
| **Speech-to-Text** | 500-800ms |
| **LLM Processing** | 1-3 seconds |
| **Kokoro TTS** | 50-200ms |
| **Total Pipeline** | **2-5 seconds** |

## 🔄 Development Workflow

### Start Development Session

```bash
# Terminal 1: Backend services
make start-native

# Terminal 2: Frontend
cd react-frontend && npm start

# Terminal 3: Testing
make test-kokoro
```

### Stop All Services

```bash
# Stop everything
make stop-native

# OR manually
./stop_voice_agent.sh
```

### Code Changes

- **Backend changes**: Server auto-reloads with file changes
- **Frontend changes**: React hot reload active
- **TTS changes**: Restart backend to pick up config changes

## 🚨 Common Issues

### 1. "WebSocket connection failed"
- **Cause**: Backend not running on port 8080
- **Fix**: Run `./start_voice_agent.sh`

### 2. "TTS synthesis failed"  
- **Cause**: Kokoro TTS service not running
- **Fix**: Check Docker container with `docker ps`

### 3. "Audio not playing"
- **Cause**: Browser autoplay policy or missing ffplay
- **Fix**: Allow autoplay in browser, install ffmpeg

### 4. "99+ second latency"
- **Cause**: Services not communicating properly
- **Fix**: Restart all services with `./stop_voice_agent.sh && ./start_voice_agent.sh`

## 📝 Logs and Debugging

### View Backend Logs
```bash
# Logs are shown when running start_voice_agent.sh
# Or check specific service:
curl http://localhost:8080/health
```

### View Kokoro TTS Logs
```bash
docker logs kokoro-tts -f
```

### Enable Debug Mode
```bash
export DEBUG_MODE=true
./start_voice_agent.sh
```

## 🎯 Next Steps

1. **Customize TTS voice**: Edit `KOKORO_TTS_VOICE` in config
2. **Optimize latency**: Tune buffer sizes and streaming parameters  
3. **Add new features**: Extend the WebSocket API
4. **Deploy to production**: Use Docker Compose for full deployment

---

**Happy coding! 🎉**

For more help, check the main [README.md](README.md) or open an issue. 