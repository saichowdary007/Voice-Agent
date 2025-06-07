# Voice Agent

A real-time voice agent application with speech recognition, natural language processing, and speech synthesis capabilities.

## Features

- **Real-time Voice Activity Detection (VAD)**: Silero VAD model for accurate speech detection
- **Speech-to-Text (STT)**: Azure Speech Services for accurate transcription
- **Natural Language Processing**: Google Gemini for intelligent responses
- **Text-to-Speech (TTS)**: Fast and natural-sounding voice synthesis
- **Real-time WebSocket Communication**: Bidirectional real-time communication
- **Magic UI Frontend**: Stylish interface built with [Magic UI](https://magicui.design)

## Setup

### Prerequisites

- Python 3.8+
- Node.js and npm
- API keys for:
  - Azure Speech Services
  - Google Gemini API

### Environment Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Voice\ Agent
   ```

2. Set up Python virtual environment:
   ```
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install frontend dependencies:
   ```
   cd frontend
   npm install
   cd ..
   ```

5. Copy `.env.example` to `.env` and add your API keys:
   ```
   # Core settings
   DEBUG=false
   LOG_LEVEL=INFO

   # Azure Speech-to-Text
   AZURE_SPEECH_KEY=your_azure_speech_key
   AZURE_SPEECH_REGION=eastus
   AZURE_SPEECH_ENDPOINT=https://eastus.api.cognitive.microsoft.com/
   AZURE_SPEECH_LANGUAGE=en-US

   # Google Gemini LLM
   GOOGLE_API_KEY=your_google_api_key
   LLM_MODEL=gemini-1.0-pro

   # Optional: Use mock services for testing
   ENABLE_MOCK_SERVICES=false
   ```

## Running the Application

### All-in-One Launcher

The easiest way to run the application is using the provided launcher scripts:

#### Local Development

```bash
# Use our local development script (recommended)
./start_local.sh

```

#### Docker Deployment

```bash
# Start with Docker Compose
./start_docker.sh
```

### Running Components Separately

If you prefer to run the components separately:

1. Start the backend server:
   ```bash
   # Using our helper script (recommended)
   ./start_backend.sh
   
   # Or manually
   PYTHONPATH=. python3 -m backend.app.main
   ```

2. In a separate terminal, start the frontend:
   ```bash
   # Using our helper script (recommended)
   ./start_frontend.sh
   
   # Or manually
   cd frontend
   npm run dev
   ```

3. Access the application at http://localhost:3000

## Verification

To verify that all services are working correctly, you can run the verification script:

```
cd backend
PYTHONPATH=.. python verify_services.py
```

This will test each component (VAD, STT, LLM, TTS) individually and as an integrated system.

## Architecture

The Voice Agent consists of several key components:

- **Voice Activity Detection (VAD)**: Detects when speech is present in an audio stream
- **Speech-to-Text (STT)**: Converts spoken words to text
- **Language Model (LLM)**: Processes text input and generates intelligent responses
- **Text-to-Speech (TTS)**: Converts text responses to spoken words
- **WebSocket Manager**: Handles real-time communication between frontend and backend

## Troubleshooting

- **Missing API Keys**: If you don't provide valid API keys, the system will use mock services by setting `ENABLE_MOCK_SERVICES=true` in your `.env` file.
- **Port Conflicts**: If port 8000 is already in use, modify the port in `backend/app/main.py`.
- **Audio Issues**: Make sure your microphone is properly connected and permissions are granted in your browser.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 