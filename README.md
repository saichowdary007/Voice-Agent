# Voice Agent

A real-time, high-performance voice agent application featuring advanced speech recognition, natural language processing, and text-to-speech capabilities. This project has been refactored for a unified and robust architecture.

## 🚀 Key Features & Improvements

-   **Unified Architecture**: A single, cohesive voice service for VAD, STT, LLM, and TTS.
-   **High Performance**: Async-first design ensures non-blocking operations and reduced latency.
-   **Real-time WebSocket Communication**: Bidirectional communication for a seamless user experience.
-   **Advanced Error Recovery**: Robust error handling and automatic recovery mechanisms.
-   **Magic UI Frontend**: A stylish and responsive interface built with [Magic UI](https://magicui.design).

---

## 🛠️ Setup and Installation

### Prerequisites

-   Python 3.8+
-   Node.js and npm
-   FFmpeg (required for audio processing)
-   API keys for:
    -   Azure Speech Services
    -   Google Gemini API

### Environment Setup

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd Voice-Agent
    ```

2.  **Backend Setup**:
    -   **Create a Virtual Environment**:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```
    -   **Install Python Dependencies**:
        ```bash
        pip install -r requirements.txt
        ```

3.  **Frontend Setup**:
    ```bash
    cd frontend
    npm install
    cd ..
    ```

4.  **Configure Environment Variables**:
    -   Copy `.env.example` to `.env` and add your API keys. This file configures both the backend and frontend.
    -   The backend uses environment variables for services, audio settings, and session management.
    -   The frontend uses `NEXT_PUBLIC_WS_URL` to connect to the backend WebSocket.

---

## 🏃‍♀️ Running the Application

### All-in-One (Recommended)

Use the provided shell scripts to launch the entire application stack.

-   **Local Development**:
    ```bash
    ./start_local.sh
    ```
-   **Docker Deployment**:
    ```bash
    ./start_docker.sh
    ```

### Running Components Separately

1.  **Start the Backend Server**:
    ```bash
    # Run the FastAPI backend (defaults to port 8000)
    PYTHONPATH=. python backend/app/main.py
    ```

2.  **Start the Frontend Server**:
    ```bash
    # In a separate terminal, run the Next.js frontend
    cd frontend
    npm run dev
    ```

3.  Access the application at `http://localhost:3000`.

---

##  architecture-overview

The application is composed of a FastAPI backend and a Next.js frontend, communicating via WebSockets.

```
┌──────────────────┐      ┌──────────────────┐
│ Next.js Frontend │ ◀───▶│  FastAPI Backend │
│ (Magic UI)       │      │ (WebSocket)      │
└──────────────────┘      └──────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  Voice Service  │
                     └─────────────────┘
                          │
              ┌───────────┼───────────┬───────────┐
              ▼           ▼           ▼           ▼
        ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
        │   VAD   │ │   STT   │ │   LLM   │ │   TTS   │
        └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

### Backend Components

-   **WebSocket Manager**: Manages all WebSocket connections, sessions, and message routing.
-   **Voice Service**: A unified service that orchestrates the entire voice processing pipeline, including VAD, STT, LLM, and TTS.
-   **AI Services**:
    -   **VAD**: Silero VAD for voice activity detection.
    -   **STT**: Azure Speech for speech-to-text.
    -   **LLM**: Gemini 2.0 Flash for natural language understanding.
    -   **TTS**: Piper TTS for text-to-speech synthesis.

### Frontend Components

-   **VoiceAgent Component**: The main React component that manages the user interface, audio recording, and WebSocket communication.
-   **AudioPlayer**: A utility for playing back TTS audio streams.

---

## ⚙️ API and WebSocket Protocol

### API Endpoints

-   `GET /`: Service information.
-   `GET /health`: System health check with detailed service status.
-   `GET /metrics`: Performance metrics (if enabled).
-   `ws://localhost:8000/ws`: Main WebSocket endpoint for voice communication.

### WebSocket Messages

The frontend and backend communicate using a JSON-based protocol over WebSockets. Key message types include:

-   **Client → Server**: `ping`, `mute`, `end_speech`, and binary audio chunks.
-   **Server → Client**: `status`, `partial_transcript`, `final_transcript`, `ai_response`, `tts_audio`, and `tts_complete`.

---

## ✅ Verification and Troubleshooting

### Verification Script

To ensure all backend services are configured and running correctly, use the verification script:

```bash
cd backend
PYTHONPATH=.. python verify_services.py
```

### Troubleshooting

-   **API Keys**: If you encounter authentication errors, ensure your API keys are correctly set in the `.env` file. If keys are missing, the system can be configured to use mock services by setting `ENABLE_MOCK_SERVICES=true`.
-   **Audio Issues**: Verify that your microphone is properly connected and that you have granted microphone permissions in your browser.
-   **Port Conflicts**: If you encounter port conflicts, you can modify the default ports in the startup scripts or application configuration.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details. 