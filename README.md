# Ultra-Fast AI Voice Agent

This project is a streamlined, high-performance voice agent that uses:
- **Speech-to-Text**: `speech_recognition` library for fast, accurate transcription.
- **AI/LLM**: Direct API calls to Google's Gemini 1.5 Flash for low-latency responses.
- **Text-to-Speech**: `edge-tts` for high-quality, natural-sounding voice output.
- **Conversation Memory**: `supabase` with `pgvector` for intelligent, context-aware conversations.
- **Audio Playback**: `ffmpeg` for robust, crash-free audio playback.

## 1. Installation

### Prerequisites
You must have **FFmpeg** installed on your system. It's used for stable audio playback.

- **On macOS (using Homebrew):**
  ```bash
  brew install ffmpeg
  ```
- **On Debian/Ubuntu:**
  ```bash
  sudo apt update && sudo apt install ffmpeg
  ```
- **On Windows:**
  Download from the [official FFmpeg site](https://ffmpeg.org/download.html) and add the `bin` directory to your system's PATH.

### Project Setup
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Voice-Agent
    ```
2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## 2. Configuration

1.  **Create a `.env` file** in the project root.
2.  **Add your API keys** to the file. You can get these from the Google AI and Supabase dashboards.

    ```env
    # .env
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    SUPABASE_URL="YOUR_SUPABASE_URL"
    SUPABASE_KEY="YOUR_SUPABASE_ANON_KEY"
    ```

## 3. Database Setup (for Conversation Memory)

To enable the agent to remember past conversations, you need to set up a `pgvector` table in your Supabase project.

1.  Go to the **SQL Editor** in your Supabase project dashboard.
2.  Click **"New query"**.
3.  Open the `supabase_setup.sql` file from this repository.
4.  Copy its entire content, paste it into the Supabase SQL editor, and click **RUN**.

This only needs to be done once. If you skip this step, the agent will still work but will not have any long-term memory.

## 4. Running the Agent

With your virtual environment activated and your `.env` file configured, start the agent with:

```bash
python main.py
```

The agent will calibrate your microphone for 1-2 seconds, and then it will be ready to listen for your commands. 