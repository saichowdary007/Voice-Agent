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

    # Audio Processing
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")
    audio_frame_ms: int = Field(120, description="Duration of audio frames in milliseconds")
    vad_threshold: float = Field(default_factory=lambda: float(os.getenv("VAD_THRESHOLD", "0.6")), description="Voice activity detection sensitivity (0.1-0.9)")
    tts_speed: float = Field(default_factory=lambda: float(os.getenv("TTS_SPEED", "1.0")), description="Text-to-speech speed (0.5-2.0)")

    # WebSocket & Session Management
    max_concurrent_sessions: int = Field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_SESSIONS", "3")),
                                       description="Maximum simultaneous WebSocket sessions")
    ws_server_host: str = Field(default_factory=lambda: os.getenv("UVICORN_HOST", "0.0.0.0"), description="WebSocket server host")
    ws_server_port: int = Field(default_factory=lambda: int(os.getenv("UVICORN_PORT", "8000")), description="WebSocket server port")
    # Note: Frontend WS_URL should be constructed from these in a real deployment scenario if not directly provided.

    # Model Paths
    model_path: str = Field(default_factory=lambda: os.getenv("MODEL_PATH", "/app/models"), description="Path to AI models directory")

    # API Keys
    google_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"), description="Google AI API Key for Gemini")

    # Feature Flags
    enable_metrics: bool = Field(default_factory=lambda: os.getenv("ENABLE_METRICS", "true").lower() == 'true', description="Enable Prometheus metrics collection")
    preload_models: bool = Field(default_factory=lambda: os.getenv("PRELOAD_MODELS", "true").lower() == 'true', description="Preload AI models on startup")

    class Config:
        env_prefix = 'APP_' # Optional: prefix for environment variables e.g. APP_LOG_LEVEL
        env_file = ".env"  # If you have a .env file for local overrides
        env_file_encoding = 'utf-8'
        extra = 'ignore' # Ignore extra fields from environment

# Instantiate the settings
settings = AppSettings() 