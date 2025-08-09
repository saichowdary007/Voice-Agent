import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
# It's recommended to set your API key in the .env file for security
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

# --- Supabase Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
# Check if Supabase is explicitly enabled and has valid credentials
USE_SUPABASE_ENV = os.getenv("USE_SUPABASE", "false").lower() == "true"
USE_SUPABASE = USE_SUPABASE_ENV and bool(SUPABASE_URL and SUPABASE_KEY)

# --- TTS Configuration ---
# TTS handled by Deepgram Agent; legacy provider settings removed

# --- STT Configuration ---
# Legacy thresholds removed; Deepgram Agent handles VAD/STT

# --- Deepgram STT Configuration ---
# Voice activity detection sensitivity (0.0 to 1.0)
DEEPGRAM_STT_VAD_SENSITIVITY = float(os.getenv("DEEPGRAM_STT_VAD_SENSITIVITY", "0.6"))
# Enable real-time transcription updates
DEEPGRAM_STT_ENABLE_REALTIME = os.getenv("DEEPGRAM_STT_ENABLE_REALTIME", "true").lower() == "true"

ULTRA_FAST_MODE = False

# --- Audio Settings ---
# Input
INPUT_SAMPLE_RATE = 16000  # 16kHz
INPUT_CHANNELS = 1
INPUT_FORMAT = "int16"  # 16-bit PCM

VAD_AGGRESSIVENESS = 1

# Audio processing - Optimized for speed
AUDIO_GAIN = 3.0  # Moderate amplification 
MIN_AUDIO_THRESHOLD = 30  # Lower threshold for faster detection
AUDIO_BUFFER_SIZE = 512  # Smaller buffer for lower latency

OUTPUT_SAMPLE_RATE = 24000
OUTPUT_CHANNELS = 1
OUTPUT_FORMAT = "linear16"

DEEPGRAM_STT_MODEL_FAST = "nova-3"

MIN_INTERRUPTION_DELAY_MS = 100

# --- Conversation ---
MAX_CONTEXT_TOKENS = 500 if ULTRA_FAST_MODE else 2000  # Smaller context for speed

# --- Database ---
# PostgreSQL (legacy support)
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "voice_agent")

# --- Security Configuration ---
# CORS settings for production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

# Production environment detection
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT.lower() == "production"

# Debug mode settings (default to true in non-production for local dev convenience)
_debug_default = "false" if IS_PRODUCTION else "true"
DEBUG_MODE = os.getenv("DEBUG_MODE", _debug_default).lower() == "true"

# Rate limiting
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "1000"))
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))

# Session timeouts
SESSION_TIMEOUT_HOURS = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))
REFRESH_TOKEN_TIMEOUT_DAYS = int(os.getenv("REFRESH_TOKEN_TIMEOUT_DAYS", "30"))

USE_REALTIME_STT = False

KOKORO_TTS_URL = None
KOKORO_TTS_VOICE = None
KOKORO_TTS_MODEL = None
KOKORO_TTS_SPEED = 1.0
GEMINI_TTS_MODEL = None
GEMINI_TTS_VOICE = None
GEMINI_TTS_SPEAKING_RATE = 1.0

# --- Deepgram Configuration ---
# STT Configuration - Optimized for better speech recognition
DEEPGRAM_STT_MODEL = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
# Force nova-3 for Nova-3 optimization
if DEEPGRAM_STT_MODEL != "nova-3":
    DEEPGRAM_STT_MODEL = "nova-3"
DEEPGRAM_STT_LANGUAGE = os.getenv("DEEPGRAM_STT_LANGUAGE", "en-US")
DEEPGRAM_STT_SMART_FORMAT = os.getenv("DEEPGRAM_STT_SMART_FORMAT", "true").lower() == "true"
DEEPGRAM_STT_PUNCTUATE = os.getenv("DEEPGRAM_STT_PUNCTUATE", "true").lower() == "true"
DEEPGRAM_STT_DIARIZE = os.getenv("DEEPGRAM_STT_DIARIZE", "false").lower() == "true"
# Additional STT parameters for better recognition
DEEPGRAM_STT_FILLER_WORDS = os.getenv("DEEPGRAM_STT_FILLER_WORDS", "true").lower() == "true"
DEEPGRAM_STT_NUMERALS = os.getenv("DEEPGRAM_STT_NUMERALS", "true").lower() == "true"
DEEPGRAM_STT_ENDPOINTING = int(os.getenv("DEEPGRAM_STT_ENDPOINTING", "300"))  # 300ms silence detection

# Voice Agent toggle
USE_DEEPGRAM_AGENT = os.getenv("USE_DEEPGRAM_AGENT", "true").lower() == "true"

# TTS Configuration  
DEEPGRAM_TTS_MODEL = os.getenv("DEEPGRAM_TTS_MODEL", "aura-asteria-en")
DEEPGRAM_TTS_ENCODING = os.getenv("DEEPGRAM_TTS_ENCODING", "linear16")
DEEPGRAM_TTS_SAMPLE_RATE = int(os.getenv("DEEPGRAM_TTS_SAMPLE_RATE", "24000")) 