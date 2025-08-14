"""
Deepgram Voice Agent - Simplified Implementation
This module provides a simple interface to Deepgram's Voice Agent API
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DeepgramVoiceAgent:
    """
    Simplified Deepgram Voice Agent class
    
    In the new architecture, the actual Voice Agent functionality is handled
    by the WebSocket proxy in websocket_handlers.py. This class is kept for
    compatibility but is no longer the primary implementation.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the agent (compatibility only)"""
        logger.info("DeepgramVoiceAgent initialized (proxy mode)")
        self._active = False
    
    async def start(self) -> bool:
        """Start the agent (compatibility only)"""
        logger.info("DeepgramVoiceAgent start called (proxy mode)")
        self._active = True
        return True
    
    def send_audio(self, audio_data: bytes) -> None:
        """Send audio data (compatibility only)"""
        logger.debug(f"DeepgramVoiceAgent send_audio called with {len(audio_data)} bytes (proxy mode)")
    
    def send_silence(self, duration_ms: int = 200) -> None:
        """Send silence (compatibility only)"""
        logger.debug(f"DeepgramVoiceAgent send_silence called with {duration_ms}ms (proxy mode)")
    
    async def stop(self) -> None:
        """Stop the agent (compatibility only)"""
        logger.info("DeepgramVoiceAgent stop called (proxy mode)")
        self._active = False
    
    def register_function(self, name: str, func, description: str = "") -> None:
        """Register a function (compatibility only)"""
        logger.debug(f"DeepgramVoiceAgent register_function called: {name} (proxy mode)")


# Legacy compatibility
class VoiceAgent(DeepgramVoiceAgent):
    """Legacy compatibility class"""
    pass