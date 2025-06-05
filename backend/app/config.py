from pydantic_settings import BaseSettings
from pydantic import Field
import os
from typing import Optional

class AppSettings(BaseSettings):
    # Core App Settings
    app_name: str = Field("Ultra-Fast Voice Agent Backend", description="Application name")
    app_version: str = Field("2.0.0", description="Application version")
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper(), description="Logging level")
    log_format: str = Field(default_factory=lambda: os.getenv("LOG_FORMAT", "console").lower(), description="Log format (json or console)") # console or json

    # Audio Processing - Standardized PCM format for all services
    sample_rate: int = Field(16000, description="Audio sample rate in Hz - standardized across all services")
    channels: int = Field(1, description="Number of audio channels - mono for voice processing")
    frame_duration_ms: int = Field(32, description="Duration of audio frames in milliseconds - used for Opus codec and VAD (512 samples at 16kHz)")
    vad_threshold: float = Field(default_factory=lambda: float(os.getenv("VAD_THRESHOLD", "0.6")), description="Voice activity detection sensitivity (0.1-0.9)")
    tts_speed: float = Field(default_factory=lambda: float(os.getenv("TTS_SPEED", "1.0")), description="Text-to-speech speed (0.5-2.0)")

    # WebSocket & Session Management
    max_concurrent_sessions: int = Field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_SESSIONS", "3")),
                                       description="Maximum simultaneous WebSocket sessions")
    ws_server_host: str = Field(default_factory=lambda: os.getenv("UVICORN_HOST", "0.0.0.0"), description="WebSocket server host")
    ws_server_port: int = Field(default_factory=lambda: int(os.getenv("UVICORN_PORT", "8000")), description="WebSocket server port")
    # Note: Frontend WS_URL should be constructed from these in a real deployment scenario if not directly provided.

    # Model Paths - Consistent across all services
    model_path: str = Field(default_factory=lambda: os.getenv("MODEL_PATH", "/app/models"), description="Path to AI models directory")

    # API Keys - Standardized naming
    google_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"), description="Google AI API Key for Gemini")
    azure_speech_key: Optional[str] = Field(default_factory=lambda: os.getenv("AZURE_SPEECH_KEY", "3AbXttS4rrG8QqKcH9dEEJePfgzR6Rkssg6rCX3FAioVRBzdKonlJQQJ99BFACYeBjFXJ3w3AAAYACOGaRsS"), description="Azure Speech API Key")
    azure_speech_region: Optional[str] = Field(default_factory=lambda: os.getenv("AZURE_SPEECH_REGION", "eastus"), description="Azure Speech Service Region")
    
    # Azure STT specific settings
    azure_speech_language: str = Field(default_factory=lambda: os.getenv("AZURE_SPEECH_LANGUAGE", "en-US"), description="Speech recognition language")
    azure_speech_initial_silence_timeout_ms: int = Field(default_factory=lambda: int(os.getenv("AZURE_SPEECH_INITIAL_SILENCE_TIMEOUT", "5000")), 
                                                description="Initial silence timeout in milliseconds")
    azure_speech_end_silence_timeout_ms: int = Field(default_factory=lambda: int(os.getenv("AZURE_SPEECH_END_SILENCE_TIMEOUT", "1000")), 
                                               description="End of speech silence timeout in milliseconds")

    # TTS Settings
    tts_model: str = Field(default_factory=lambda: os.getenv("TTS_MODEL", "en_US-libritts-high"), description="TTS model name")
    tts_voice: str = Field(default_factory=lambda: os.getenv("TTS_VOICE", "en-US-Neural"), description="TTS voice name")
    
    # LLM Settings
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gemini-2.0-flash-exp"), description="LLM model name")
    llm_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("LLM_API_KEY"), description="LLM API key")
    llm_temperature: float = Field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7")), description="LLM temperature (0.0-1.0)")
    llm_max_tokens: int = Field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "1024")), description="LLM maximum tokens")

    # Feature Flags
    enable_metrics: bool = Field(default_factory=lambda: os.getenv("ENABLE_METRICS", "true").lower() == 'true', description="Enable Prometheus metrics collection")
    preload_models: bool = Field(default_factory=lambda: os.getenv("PRELOAD_MODELS", "true").lower() == 'true', description="Preload AI models on startup")
    
    # Audio Processing Pipeline Settings
    max_failed_chunk_retries: int = Field(5, description="Maximum retry attempts for processing failed audio chunks")
    failed_chunk_buffer_max_size: int = Field(100000, description="Maximum size of failed chunk buffer in bytes")
    small_chunk_buffer_threshold: int = Field(100, description="Minimum chunk size in bytes to process immediately")
    
    # Watchdog Timer Settings
    watchdog_inactivity_timeout: int = Field(120, description="Seconds of inactivity before closing WebSocket connection")
    watchdog_check_interval: int = Field(5, description="Watchdog timer check interval in seconds")

    # Timeout Settings
    audio_conversion_timeout: int = Field(20, description="Timeout in seconds for audio conversion")
    audio_processing_timeout: int = Field(15, description="Timeout in seconds for audio processing")
    ai_response_timeout: int = Field(60, description="Timeout in seconds for AI response generation")
    message_send_timeout: int = Field(10, description="Timeout in seconds for sending WebSocket messages")

    class Config:
        env_prefix = 'APP_' # Optional: prefix for environment variables e.g. APP_LOG_LEVEL
        env_file = ".env"  # If you have a .env file for local overrides
        env_file_encoding = 'utf-8'
        extra = 'ignore' # Ignore extra fields from environment

# Instantiate the settings
settings = AppSettings() 

# Helper function to get standardized sample rate
def get_sample_rate() -> int:
    """Get the standardized sample rate for all audio processing"""
    return settings.sample_rate

# Helper function to get standardized channels
def get_channels() -> int:
    """Get the standardized number of audio channels"""
    return settings.channels

# Helper function to validate Google API key
def validate_google_api_key() -> bool:
    """Validate that Google API key is configured"""
    return bool(settings.google_api_key and settings.google_api_key.strip()) 