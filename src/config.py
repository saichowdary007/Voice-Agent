import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
# It's recommended to set your API key in the .env file for security
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "84a7d0c60019baa83e8fd7b3097ef0a23e104c98")

# --- Supabase Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
# Check if Supabase is explicitly enabled and has valid credentials
USE_SUPABASE_ENV = os.getenv("USE_SUPABASE", "false").lower() == "true"
USE_SUPABASE = USE_SUPABASE_ENV and bool(SUPABASE_URL and SUPABASE_KEY)

# --- TTS Configuration ---
# Voice for Microsoft Edge TTS, find more at `edge-tts --list-voices`
EDGE_TTS_VOICE = "en-US-AriaNeural"

# --- STT Configuration ---
# Energy threshold for silence detection with speech_recognition
# Lower values make it more sensitive to quieter speech
# Higher values mean you have to speak louder
ENERGY_THRESHOLD = 150  # Reduced from 300 to 150 for better sensitivity
# Seconds of non-speaking audio before a phrase is considered complete
PAUSE_THRESHOLD = 0.8  # Increased from 0.4 to 0.8 for better phrase detection

# --- Deepgram STT Configuration ---
# Voice activity detection sensitivity (0.0 to 1.0)
DEEPGRAM_STT_VAD_SENSITIVITY = float(os.getenv("DEEPGRAM_STT_VAD_SENSITIVITY", "0.6"))
# Enable real-time transcription updates
DEEPGRAM_STT_ENABLE_REALTIME = os.getenv("DEEPGRAM_STT_ENABLE_REALTIME", "true").lower() == "true"

# --- Ultra-Fast Performance Configuration ---
# Ultra-fast mode settings for ~500ms latency target
ULTRA_FAST_MODE = os.getenv("ULTRA_FAST_MODE", "false").lower() == "true"

# STT Ultra-Fast Settings
ULTRA_FAST_STT_MODEL = "nova-3"  # Fastest Deepgram model
ULTRA_FAST_VAD_SENSITIVITY = 0.3  # Aggressive VAD for instant detection
ULTRA_FAST_PAUSE_THRESHOLD = 0.4  # Shorter pause for faster cutoff
ULTRA_FAST_MIN_PHRASE_LENGTH = 0.3  # Minimum phrase length in seconds
ULTRA_FAST_SILENCE_TIMEOUT = 1000  # 1 second max silence before cutoff

# LLM Ultra-Fast Settings  
ULTRA_FAST_LLM_MAX_TOKENS = 100  # Limit response length for speed
ULTRA_FAST_LLM_TEMPERATURE = 0.7  # Balanced creativity vs speed
ULTRA_FAST_SKIP_CONTEXT = True  # Skip conversation history for speed
ULTRA_FAST_SKIP_PROFILE = True  # Skip user profile for speed
ULTRA_FAST_PARALLEL_PROCESSING = True  # Start LLM while speaking

# TTS Ultra-Fast Settings
ULTRA_FAST_TTS_VOICE = "en-US-JennyNeural"  # Fast neural voice
ULTRA_FAST_TTS_RATE = "+20%"  # Slightly faster speech
ULTRA_FAST_TTS_STREAMING = True  # Stream sentence by sentence
ULTRA_FAST_TTS_BUFFER_SIZE = 1024  # Smaller buffer for lower latency

# Performance Monitoring
ULTRA_FAST_PERFORMANCE_TRACKING = True  # Enable detailed timing
ULTRA_FAST_TARGET_LATENCY_MS = 500  # Target total response time

# --- Audio Settings ---
# Input
INPUT_SAMPLE_RATE = 16000  # 16kHz
INPUT_CHANNELS = 1
INPUT_FORMAT = "int16"  # 16-bit PCM

# VAD - Ultra-fast settings for immediate response
VAD_AGGRESSIVENESS = 0  # Most sensitive for ultra-fast mode
VAD_FRAME_MS = 10  # Smaller frames for faster detection
VAD_SILENCE_TIMEOUT_MS = 800  # Very short silence timeout
VAD_SPEECH_THRESHOLD = 0.3  # Lower threshold for faster detection

# Audio processing - Optimized for speed
AUDIO_GAIN = 3.0  # Moderate amplification 
MIN_AUDIO_THRESHOLD = 30  # Lower threshold for faster detection
AUDIO_BUFFER_SIZE = 512  # Smaller buffer for lower latency

# Output
OUTPUT_SAMPLE_RATE = 22050  # 22.05kHz for Piper
OUTPUT_CHANNELS = 1
OUTPUT_FORMAT = "int16"

# --- Models ---
# STT - Using Deepgram models only
DEEPGRAM_STT_MODEL_FAST = "nova-3" if ULTRA_FAST_MODE else "nova-3"  # Always use nova-3 for best performance

# TTS
PIPER_VOICE = "en_US-libritts-high" # As per PRD
VAKYANSH_VOICE_TE = "te_IN-cmu-male" # Placeholder for Vakyansh Telugu voice

# --- Real-time settings ---
MIN_INTERRUPTION_DELAY_MS = 50 if ULTRA_FAST_MODE else 100  # Faster interruption in ultra-fast mode

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

# Debug mode settings
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Production environment detection
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT.lower() == "production"

# Rate limiting
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "1000"))
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))

# Session timeouts
SESSION_TIMEOUT_HOURS = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))
REFRESH_TOKEN_TIMEOUT_DAYS = int(os.getenv("REFRESH_TOKEN_TIMEOUT_DAYS", "30"))

# --- STT Global Toggle ---
# Enable server-side STT by default. Set USE_REALTIME_STT=false in the
# environment when you want the browser-only Web-Speech fallback instead.
USE_REALTIME_STT = os.getenv("USE_REALTIME_STT", "true").lower() == "true"

# --- Kokoro TTS Configuration ---
KOKORO_TTS_URL = os.getenv("KOKORO_TTS_URL", "http://localhost:8880")
KOKORO_TTS_VOICE = os.getenv("KOKORO_TTS_VOICE", "af_bella")
KOKORO_TTS_MODEL = os.getenv("KOKORO_TTS_MODEL", "kokoro")
KOKORO_TTS_SPEED = float(os.getenv("KOKORO_TTS_SPEED", "1.0"))

# --- Gemini TTS Configuration ---
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-native-audio-dialog")
GEMINI_TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Aoede")
GEMINI_TTS_SPEAKING_RATE = float(os.getenv("GEMINI_TTS_SPEAKING_RATE", "1.0"))

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

# TTS Configuration  
DEEPGRAM_TTS_MODEL = os.getenv("DEEPGRAM_TTS_MODEL", "aura-asteria-en")
DEEPGRAM_TTS_ENCODING = os.getenv("DEEPGRAM_TTS_ENCODING", "linear16")
DEEPGRAM_TTS_SAMPLE_RATE = int(os.getenv("DEEPGRAM_TTS_SAMPLE_RATE", "24000")) 