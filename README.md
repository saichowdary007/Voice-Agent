# Lightning-Fast Bilingual Voice Assistant

This project implements a real-time, bilingual (English/Telugu) voice assistant as specified in the Product Requirements Document. It features VAD, STT, LLM, and TTS in a continuous loop with barge-in support and long-term memory.

## Features
- **Bilingual:** Supports English and Telugu.
- **Real-Time:** Low-latency interaction with barge-in capability.
- **Intelligent:** Uses a Large Language Model (Gemini 1.5 Flash) for responses.
- **Persistent Memory:** Leverages a PostgreSQL database with `pgvector` for long-term, semantically searchable conversation history.

## Project Structure
- `main.py`: Main application entry point.
- `Dockerfile`: Container configuration for deployment.
- `requirements.txt`: Python dependencies.
- `config.py`: Configuration management.
- `src/`: Source code directory.
  - `vad.py`: Voice Activity Detection.
  - `stt.py`: Speech-to-Text processing.
  - `language_detection.py`: Identifies the language from text.
  - `llm.py`: Large Language Model interaction.
  - `tts.py`: Text-to-Speech synthesis (EN only for now).
  - `audio_utils.py`: Audio processing and non-blocking playback.
  - `conversation.py`: Conversation state management with PostgreSQL.

## Setup

### 1. Prerequisites
- **Python 3.8+**
- **PostgreSQL Database:** A running instance with the `pgvector` extension enabled.
  ```sql
  -- Run this in your PostgreSQL instance
  CREATE EXTENSION IF NOT EXISTS vector;
  ```
- **fastText Model:** Download the `lid.176.bin` model from the [fastText website](https://fasttext.cc/docs/en/language-identification.html) and place it in the root of the project directory.

### 2. Installation
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Voice-Agent
    ```
2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configuration
1.  **Create a `.env` file** in the project root. You can copy the `.env.example` if it exists, or create a new one.
2.  **Add your Gemini API Key:**
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```
3.  **(Optional) Configure the database connection** in the `.env` file if it's not running on the default `localhost:5432` with user `postgres`:
    ```
    DB_USER="your_db_user"
    DB_PASSWORD="your_db_password"
    DB_HOST="your_db_host"
    DB_PORT="your_db_port"
    DB_NAME="voice_agent"
    ```

### 4. Running the Application
Once all prerequisites are met and the configuration is set, start the voice agent:
```bash
python main.py
```
The agent will initialize all components. Say `"Hey Gemini"` to start a conversation.

### 5. Docker (Advanced)
A `Dockerfile` is provided for containerized deployment.
1.  **Build the image:**
    ```bash
    docker build -t voice-agent .
    ```
2.  **Run the container:** You must provide the Gemini API key and database credentials as environment variables.
    ```bash
    docker run --rm -it \
      -e GEMINI_API_KEY="your_api_key" \
      -e DB_HOST="host.docker.internal" \
      --device /dev/snd \
      voice-agent
    ```
    *Note: Audio device mapping (`--device /dev/snd`) is platform-specific and may require adjustments.* 