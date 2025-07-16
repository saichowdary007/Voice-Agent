# Voice Agent Technology Stack

## Backend Stack

- **Framework**: FastAPI with async/await for high-performance WebSocket handling
- **Language**: Python 3.8+ with asyncio for concurrent processing
- **WebSocket**: Native FastAPI WebSocket support for real-time audio streaming
- **Authentication**: Supabase Auth with JWT tokens, fallback to demo mode
- **Database**: PostgreSQL via Supabase or local Docker container
- **Caching**: Redis for session management and performance optimization

## Frontend Stack

- **Framework**: React 18 with TypeScript
- **UI Library**: Radix UI components with Tailwind CSS
- **Build Tool**: Create React App (CRA) with TypeScript template
- **WebSocket Client**: Native WebSocket API with custom hooks
- **Audio Processing**: Web Audio API for real-time visualization
- **State Management**: React Context API and custom hooks

## AI/ML Components

- **Speech-to-Text**: 
  - Primary: Faster-Whisper (tiny/base models for speed)
  - Alternative: Deepgram API
  - Fallback: Browser Web Speech API
- **Large Language Model**: Google Gemini via official SDK
- **Text-to-Speech**:
  - Primary: Microsoft Edge TTS
  - Alternative: Deepgram TTS, Gemini native audio
- **Voice Activity Detection**: WebRTC VAD with custom preprocessing

## Infrastructure & Deployment

- **Containerization**: Docker with multi-stage builds
- **Orchestration**: Docker Compose for local development
- **Web Server**: Uvicorn ASGI server with production settings
- **Reverse Proxy**: Nginx (in frontend container)
- **Environment Management**: python-dotenv for configuration

## Key Dependencies

### Backend (requirements.txt)
```
fastapi>=0.104.0          # Web framework
uvicorn[standard]>=0.24.0 # ASGI server
websockets>=11.0.0        # WebSocket support
faster-whisper==1.1.1     # STT engine
supabase>=2.0.0           # Database & auth
deepgram-sdk>=3.0.0       # Alternative STT/TTS
google-genai>=0.3.0       # Gemini LLM
torch>=2.0.0              # ML framework
```

### Frontend (package.json)
```
react: ^18.3.1            # UI framework
typescript: ^4.9.5        # Type safety
@radix-ui/*               # UI components
tailwindcss: ^3.4.17      # Styling
framer-motion: ^12.18.1   # Animations
axios: ^1.7.2             # HTTP client
```

## Common Commands

### Development Setup
```bash
# Backend setup
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Frontend setup
cd react-frontend
npm install

# Environment configuration
cp env-template.txt .env
# Edit .env with your API keys
```

### Running the Application
```bash
# Full stack with Docker
docker-compose up --build

# Backend only (development)
python server.py

# Frontend only (development)
cd react-frontend && npm start

# CLI mode (local testing)
python main.py
```

### Testing & Diagnostics
```bash
# Run comprehensive diagnostics
python run_diagnostics.py --test comprehensive

# Test voice pipeline
python test_voice_pipeline.py

# Monitor WebSocket connections
python websocket_monitor.py

# Test specific components
python test_deepgram_integration.py
python test_audio_pipeline.py
```

### Production Deployment
```bash
# Build and deploy
docker-compose -f docker-compose.yml up -d

# Health check
curl http://localhost:8080/health

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

## Configuration Management

- **Environment Variables**: All configuration via `.env` file
- **Voice Settings**: `voice_config.json` for audio processing parameters
- **Docker Settings**: `docker-compose.yml` for service orchestration
- **Build Configuration**: Separate Dockerfiles for backend/frontend

## Performance Optimization

- **Ultra-fast Mode**: `ULTRA_FAST_MODE=true` for ~500ms latency
- **Model Selection**: Configurable Whisper model sizes (tiny/base/small)
- **Concurrent Processing**: Async/await throughout the pipeline
- **Connection Pooling**: WebSocket connection management
- **Resource Monitoring**: Built-in performance tracking and diagnostics