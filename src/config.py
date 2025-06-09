import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Audio Settings ---
# Input
INPUT_SAMPLE_RATE = 16000  # 16kHz
INPUT_CHANNELS = 1
INPUT_FORMAT = "int16"  # 16-bit PCM

# VAD
VAD_AGGRESSIVENESS = 3  # 0 to 3, 3 is most aggressive
VAD_FRAME_MS = 30  # ms
VAD_SILENCE_TIMEOUT_MS = 3000 # ms of silence to mark end of speech

# Output
OUTPUT_SAMPLE_RATE = 22050  # 22.05kHz for Piper
OUTPUT_CHANNELS = 1
OUTPUT_FORMAT = "int16"

# --- Models ---
# STT
WHISPER_MODEL = "base" # Using multilingual base model

# TTS
PIPER_VOICE = "en_US-libritts-high" # As per PRD
VAKYANSH_VOICE_TE = "te_IN-cmu-male" # Placeholder for Vakyansh Telugu voice

# --- Real-time settings ---
MIN_INTERRUPTION_DELAY_MS = 100 # To prevent accidental barge-in

# --- Conversation ---
MAX_CONTEXT_TOKENS = 2000

# --- Database ---
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "voice_agent") 