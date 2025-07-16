# Voice Agent Project Structure

## Root Directory Layout

```
Voice-Agent/
├── src/                          # Core backend modules
├── react-frontend/               # React TypeScript frontend
├── debug_audio/                  # Audio debugging samples
├── voice_metrics/                # Performance monitoring data
├── logs/                         # Application logs
├── uploads/                      # File upload storage
├── server.py                     # Main FastAPI server entry point
├── main.py                       # CLI interface entry point
├── requirements.txt              # Python dependencies
├── docker-compose.yml            # Multi-service orchestration
├── Dockerfile.backend            # Backend container build
└── voice_config.json             # Audio processing configuration
```

## Backend Architecture (`src/`)

```
src/
├── __init__.py                   # Package initialization
├── config.py                     # Environment configuration
├── server.py                     # Alternative server entry
├── auth.py                       # Authentication management
├── conversation.py               # Conversation history & user profiles
├── llm.py                        # Large Language Model interface
├── stt.py                        # Speech-to-Text engines
├── stt_deepgram.py              # Deepgram STT implementation

├── tts.py                        # Text-to-Speech engines
├── tts_deepgram.py              # Deepgram TTS implementation
├── vad.py                        # Voice Activity Detection
├── audio_preprocessor.py         # Audio signal processing
└── websocket_handlers.py         # WebSocket message handlers
```

## Frontend Architecture (`react-frontend/`)

```
react-frontend/
├── src/
│   ├── components/               # React components
│   │   ├── VoiceInterface.tsx    # Main voice interaction UI
│   │   ├── AudioVisualizer.tsx   # Real-time audio visualization
│   │   ├── Login.tsx             # Authentication forms
│   │   └── magicui/              # Custom UI components
│   ├── hooks/
│   │   └── useWebSocket.ts       # WebSocket connection management
│   ├── contexts/
│   │   └── AuthContext.tsx       # Authentication state
│   ├── services/                 # API communication
│   │   ├── authService.ts        # Authentication API calls
│   │   ├── chatService.ts        # Chat/voice API calls
│   │   └── axiosInstance.ts      # HTTP client configuration
│   ├── types/
│   │   └── index.ts              # TypeScript type definitions
│   └── lib/
│       └── utils.ts              # Utility functions
├── public/
│   └── index.html                # HTML template
├── package.json                  # Node.js dependencies
├── tailwind.config.js            # Tailwind CSS configuration
└── Dockerfile                    # Frontend container build
```

## Key File Responsibilities

### Entry Points
- **`server.py`**: Main production server (FastAPI + WebSocket)
- **`main.py`**: CLI interface for local testing and development
- **`react-frontend/src/App.tsx`**: React application root component

### Core Backend Modules
- **`src/config.py`**: Centralized configuration management via environment variables
- **`src/auth.py`**: Supabase authentication with JWT token management
- **`src/websocket_handlers.py`**: WebSocket message routing and processing
- **`src/conversation.py`**: Persistent conversation history and user profile learning
- **`src/llm.py`**: LLM interface with context management and fact extraction
- **`src/stt.py`**: Deepgram STT integration for high-accuracy transcription
- **`src/tts.py`**: Multi-engine TTS with Edge/Deepgram/Gemini support
- **`src/vad.py`**: Voice Activity Detection for speech segmentation

### Frontend Components
- **`VoiceInterface.tsx`**: Main voice interaction component with recording controls
- **`AudioVisualizer.tsx`**: Real-time audio waveform and frequency visualization
- **`useWebSocket.ts`**: WebSocket connection management with reconnection logic
- **`AuthContext.tsx`**: Global authentication state management

## Configuration Files

### Environment & Deployment
- **`.env`**: Environment variables (API keys, database URLs, feature flags)
- **`voice_config.json`**: Audio processing parameters and tuning settings
- **`docker-compose.yml`**: Multi-service container orchestration
- **`Dockerfile.backend`** & **`react-frontend/Dockerfile`**: Container build definitions

### Package Management
- **`requirements.txt`**: Python dependencies with version constraints
- **`react-frontend/package.json`**: Node.js dependencies and build scripts

## Testing & Diagnostics

```
├── run_diagnostics.py            # Comprehensive system diagnostics
├── test_voice_pipeline.py        # End-to-end voice pipeline testing
├── test_deepgram_integration.py  # Deepgram API integration tests
├── test_audio_pipeline.py        # Audio processing pipeline tests
├── websocket_monitor.py          # Real-time WebSocket monitoring
└── monitor_voice_metrics.py      # Performance metrics collection
```

## Data & Logs

```
├── debug_audio/                  # Audio debugging samples and analysis
├── voice_metrics/                # Performance monitoring data and charts
├── logs/                         # Application logs (structured JSON)
└── uploads/                      # User file uploads (if applicable)
```

## Naming Conventions

### Python Files
- **Snake case**: `websocket_handlers.py`, `audio_preprocessor.py`
- **Descriptive names**: Files clearly indicate their primary responsibility
- **Module prefixes**: `stt_*.py`, `tts_*.py` for engine-specific implementations

### React/TypeScript Files
- **PascalCase components**: `VoiceInterface.tsx`, `AudioVisualizer.tsx`
- **camelCase hooks**: `useWebSocket.ts`, `useAuth.ts`
- **Descriptive suffixes**: `.tsx` for components, `.ts` for utilities

### Configuration
- **Lowercase with underscores**: `voice_config.json`, `docker-compose.yml`
- **Environment variables**: `UPPER_SNAKE_CASE` in `.env`

## Import Patterns

### Backend Imports
```python
# Relative imports within src/
from src.config import USE_SUPABASE, DEEPGRAM_STT_MODEL
from src.stt import STT
from src.llm import LLM

# External dependencies
import asyncio
from fastapi import FastAPI, WebSocket
```

### Frontend Imports
```typescript
// React and hooks
import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

// Components and utilities
import { AudioVisualizer } from './AudioVisualizer';
import { cn } from '../lib/utils';
```

## Development Workflow

1. **Backend changes**: Modify `src/` modules, test with `python server.py`
2. **Frontend changes**: Edit `react-frontend/src/`, test with `npm start`
3. **Full stack testing**: Use `docker-compose up --build`
4. **Diagnostics**: Run `python run_diagnostics.py --test comprehensive`
5. **Performance monitoring**: Check `voice_metrics/` for performance data