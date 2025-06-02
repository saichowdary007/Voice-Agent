# 🎙️ Ultra-Fast Voice Agent

A real-time voice conversation agent with sub-500ms latency, powered by cutting-edge AI models and optimized audio processing pipeline.

## ✨ Features

- **Real-time Speech Recognition** - Sherpa-NCNN 2.1.11 for ultra-fast STT
- **Voice Activity Detection** - Silero VAD for automatic speech detection
- **AI-Powered Responses** - Google Gemini 2.0 Flash for intelligent conversations
- **Natural Text-to-Speech** - Piper TTS for high-quality voice synthesis
- **WebSocket Communication** - Real-time bidirectional audio streaming
- **Docker Deployment** - Easy setup with Docker Compose

## 🏗️ Architecture

```
🎤 Audio Input → 🔍 VAD → 📝 STT → 🤖 LLM → 🔊 TTS → 🎧 Audio Output
```

- **Frontend**: Next.js with TypeScript, Tailwind CSS
- **Backend**: FastAPI with async WebSocket support
- **Audio Pipeline**: WebM/Opus → PCM conversion with FFmpeg
- **AI Models**: Gemini 2.0 Flash, Sherpa-NCNN, Piper TTS, Silero VAD

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Google AI API Key

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd voice-agent
```

2. **Configure environment**
Create `backend/.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
VAD_THRESHOLD=0.6
SAMPLE_RATE=16000
MAX_CONCURRENT_SESSIONS=3
```

3. **Start the application**
```bash
docker-compose up --build
```

4. **Access the application**
- Frontend: http://localhost:3001
- Backend API: http://localhost:8001
- Health Check: http://localhost:8001/health

## 🎯 Usage

1. Open the frontend in your browser
2. Click "Start New Session"
3. Grant microphone permissions
4. Start speaking - the agent automatically detects speech
5. Listen to the AI's response
6. Continue the conversation naturally

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | Required | Google AI API key for Gemini |
| `VAD_THRESHOLD` | 0.6 | Voice activity detection sensitivity (0.1-0.9) |
| `SAMPLE_RATE` | 16000 | Audio sample rate in Hz |
| `MAX_CONCURRENT_SESSIONS` | 3 | Maximum simultaneous users |
| `TTS_SPEED` | 1.0 | Text-to-speech speed (0.5-2.0) |

## 🛠️ Development

### Project Structure
```
voice-agent/
├── backend/           # FastAPI backend
│   ├── main.py       # Main application
│   ├── services/     # AI and audio services
│   └── Dockerfile    # Backend container
├── frontend/         # Next.js frontend
│   ├── app/          # Next.js app directory
│   ├── components/   # React components
│   └── Dockerfile    # Frontend container
└── docker-compose.yml
```

### Local Development

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## 📊 Performance

- **Latency**: < 500ms end-to-end
- **STT**: Real-time factor < 0.1 (10x faster than real-time)
- **LLM**: ~200ms average response time
- **TTS**: ~100ms synthesis time
- **VAD**: ~1ms processing per frame

## 🔧 Technical Details

### Audio Pipeline
1. Browser captures audio via MediaRecorder (WebM/Opus)
2. Real-time WebSocket streaming to backend
3. FFmpeg converts WebM to PCM if needed
4. Silero VAD detects speech boundaries
5. Sherpa-NCNN transcribes speech to text
6. Gemini generates intelligent response
7. Piper synthesizes response to audio
8. Browser plays TTS audio

### WebSocket Protocol
- **Audio**: Binary messages (WebM/Opus chunks)
- **Control**: JSON text messages (`eos`, `ping/pong`)
- **Keep-alive**: Automatic connection monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Sherpa-NCNN](https://github.com/k2-fsa/sherpa-ncnn) for speech recognition
- [Silero VAD](https://github.com/snakers4/silero-vad) for voice activity detection
- [Piper TTS](https://github.com/rhasspy/piper) for text-to-speech
- [Google Gemini](https://ai.google.dev/) for language model 