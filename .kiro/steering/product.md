# Voice Agent Product Overview

## What is Voice Agent?

Voice Agent is a real-time conversational AI system that enables natural voice interactions through a web interface. Users can speak to the AI and receive spoken responses, creating a seamless voice-first experience.

## Core Features

- **Real-time Voice Interaction**: WebSocket-based streaming audio processing with sub-3-second response times
- **Multi-modal STT/TTS**: Uses Deepgram for speech-to-text and supports multiple text-to-speech services (Deepgram TTS, Gemini TTS)
- **User Authentication**: Supabase-powered user management with JWT tokens
- **Conversation Memory**: Persistent conversation history and user profile learning
- **3D Audio Visualization**: Interactive React frontend with real-time audio visualization
- **Ultra-fast Mode**: Optimized for ~500ms end-to-end latency

## Target Use Cases

- Voice assistants and chatbots
- Accessibility applications
- Interactive voice response systems
- Real-time customer support
- Educational and training applications

## Architecture Philosophy

The system prioritizes speed and reliability with a microservices-like approach:
- Modular components (STT, LLM, TTS, VAD) that can be swapped independently
- WebSocket-first design for real-time performance
- Comprehensive error handling and graceful degradation
- Production-ready with Docker containerization