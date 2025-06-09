import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
# It's recommended to set your API key in the .env file for security
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Supabase Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# A boolean flag to easily check if Supabase is configured
USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_KEY)

# --- TTS Configuration ---
# Voice for Microsoft Edge TTS, find more at `edge-tts --list-voices`
EDGE_TTS_VOICE = "en-US-AriaNeural"

# --- STT Configuration ---
# Energy threshold for silence detection with speech_recognition
# Higher values mean you have to speak louder
ENERGY_THRESHOLD = 300
# Seconds of non-speaking audio before a phrase is considered complete
PAUSE_THRESHOLD = 0.4

# --- Audio Settings ---
# Input
INPUT_SAMPLE_RATE = 16000  # 16kHz
INPUT_CHANNELS = 1
INPUT_FORMAT = "int16"  # 16-bit PCM

# VAD - More sensitive settings
VAD_AGGRESSIVENESS = 1  # Reduced from 3 to 1 for more sensitivity
VAD_FRAME_MS = 30  # ms
VAD_SILENCE_TIMEOUT_MS = 2000  # Reduced from 3000 to 2000ms for faster response

# Audio processing
AUDIO_GAIN = 5.0  # Amplify audio by this factor
MIN_AUDIO_THRESHOLD = 50  # Minimum audio level to consider

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
# PostgreSQL (legacy support)
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "voice_agent") 