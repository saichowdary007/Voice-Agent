import asyncio
import logging
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
import structlog

# Import from app.audio.vad
from app.audio.vad import VADEngine, VADResult

# Re-export VADResult for convenience
__all__ = ["VADService", "VADResult"]

logger = structlog.get_logger(__name__)

class VADService:
    """Voice Activity Detection service wrapper"""
    
    def __init__(self, threshold: float = 0.6):
        """Initialize VAD service with given sensitivity threshold (0.1-0.9)"""
        self.threshold = threshold
        self.vad_engine = VADEngine()
        self.vad_engine.threshold = threshold  # Set threshold property after creation
        self.is_available = False
        
        # State tracking
        self._last_activity_time = time.time()
        self._speech_frames = 0
        self._silence_frames = 0
        self._total_frames = 0
        
        logger.info(f"VAD Service initialized with threshold: {threshold}")
        
    async def initialize(self):
        """Initialize VAD engine"""
        try:
            await self.vad_engine.initialize()
            self.is_available = True
            logger.info("VAD service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VAD service: {e}", exc_info=True)
            self.is_available = False
            
    async def process_frame(self, audio_frame: bytes) -> VADResult:
        """Process a single audio frame for voice activity detection"""
        if not self.is_available:
            logger.warning("VAD service not available or not initialized")
            return VADResult(is_speech=False, is_end_of_speech=False, confidence=0.0, timestamp=0.0)
            
        try:
            # Process frame with VAD engine
            vad_result = await self.vad_engine.process_frame(audio_frame)
            
            # Update internal state for metrics
            self._total_frames += 1
            if vad_result.is_speech:
                self._speech_frames += 1
                self._last_activity_time = time.time()
            else:
                self._silence_frames += 1
                
            return vad_result
        except Exception as e:
            logger.error(f"Error processing audio frame with VAD: {e}", exc_info=True)
            # Return a default negative result on error
            return VADResult(is_speech=False, is_end_of_speech=False, confidence=0.0, timestamp=0.0)
            
    def reset_state(self):
        """Reset VAD state for a new session"""
        self._last_activity_time = time.time()
        self._speech_frames = 0
        self._silence_frames = 0
        self._total_frames = 0
        if hasattr(self.vad_engine, 'reset_state') and callable(self.vad_engine.reset_state):
            self.vad_engine.reset_state()
        logger.info("VAD state reset")
        
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the VAD service"""
        status = {
            "available": self.is_available,
            "threshold": self.threshold,
            "speech_ratio": self._speech_frames / max(1, self._total_frames),
            "frames_processed": self._total_frames
        }
        
        # Add engine-specific details if available
        if hasattr(self.vad_engine, 'get_model_info') and callable(self.vad_engine.get_model_info):
            status.update(self.vad_engine.get_model_info())
            
        return status
        
    async def cleanup(self):
        """Clean up resources"""
        if hasattr(self.vad_engine, 'cleanup') and callable(self.vad_engine.cleanup):
            await self.vad_engine.cleanup()
        logger.info("VAD service cleaned up") 