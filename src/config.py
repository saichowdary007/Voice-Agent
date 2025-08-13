import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Supabase Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
USE_SUPABASE_ENV = os.getenv("USE_SUPABASE", "false").lower() == "true"
USE_SUPABASE = USE_SUPABASE_ENV and bool(SUPABASE_URL and SUPABASE_KEY)

# --- Deepgram Voice Agent Configuration ---
# Use Deepgram Voice Agent API for end-to-end voice processing
USE_DEEPGRAM_AGENT = True  # Always enabled for this implementation
DEEPGRAM_AGENT_ENDPOINT = "wss://agent.deepgram.com/v1/agent/converse"

# Audio settings tuned for latency
AUDIO_INPUT_ENCODING = "linear16"
AUDIO_INPUT_SAMPLE_RATE = 16000  # 16kHz aligns with microphone downsampling and DG low-latency STT
AUDIO_OUTPUT_ENCODING = "linear16"
AUDIO_OUTPUT_SAMPLE_RATE = 24000  # 24kHz for high-quality TTS playback

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

# --- Deepgram Configuration ---
# STT and TTS settings for Deepgram Agent
DEEPGRAM_STT_LANGUAGE = os.getenv("DEEPGRAM_STT_LANGUAGE", "en-US")
DEEPGRAM_TTS_MODEL = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")
DEEPGRAM_TTS_ENCODING = os.getenv("DEEPGRAM_TTS_ENCODING", "linear16")
DEEPGRAM_TTS_SAMPLE_RATE = int(os.getenv("DEEPGRAM_TTS_SAMPLE_RATE", "24000"))

# --- LLM Provider Configuration ---
# LLM provider type: open_ai, anthropic, google, groq
# Use DG_THINK_PROVIDER from .env if set, otherwise choose based on available keys
_dg_provider = os.getenv("DG_THINK_PROVIDER")
if _dg_provider:
    LLM_PROVIDER_TYPE = _dg_provider
else:
    _default_provider = (
        "open_ai" if OPENAI_API_KEY else
        ("anthropic" if ANTHROPIC_API_KEY else "google")
    )
    LLM_PROVIDER_TYPE = os.getenv("LLM_PROVIDER_TYPE", _default_provider)

# Use DG_THINK_MODEL from .env if set, otherwise choose based on provider
_dg_model = os.getenv("DG_THINK_MODEL")
if _dg_model:
    LLM_MODEL = _dg_model
else:
    # Choose model based on provider
    if LLM_PROVIDER_TYPE == "open_ai":
        LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    elif LLM_PROVIDER_TYPE == "google":
        LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    else:
        LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "150"))

# Custom LLM endpoint configuration (optional for open_ai/anthropic, required for others)
LLM_ENDPOINT_URL = os.getenv("LLM_ENDPOINT_URL")
LLM_ENDPOINT_HEADERS = {}

# --- Feature Flags ---
# Enable backend function calling bridge between agent and server
USE_FUNCTION_CALLS = os.getenv("USE_FUNCTION_CALLS", "false").lower() == "true"

# Set up Google/Gemini configuration for Deepgram Agent
if LLM_PROVIDER_TYPE == "google" and GEMINI_API_KEY:
    # Use v1beta endpoint which supports system_instruction and more features
    LLM_ENDPOINT_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent?key={GEMINI_API_KEY}"
    LLM_ENDPOINT_HEADERS = {
        "Content-Type": "application/json"
    }
    print(f"Configuring Google provider for Deepgram Agent: {LLM_MODEL}")
elif os.getenv("LLM_ENDPOINT_AUTH_TOKEN"):
    LLM_ENDPOINT_HEADERS["authorization"] = f"Bearer {os.getenv('LLM_ENDPOINT_AUTH_TOKEN')}"

 