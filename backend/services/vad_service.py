import asyncio
import logging
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, AsyncGenerator
import structlog
import torch

# Import from app.audio.vad
from backend.app.audio.vad import VADEngine, VADResult

# Re-export VADResult for convenience
__all__ = ["VADService", "VADResult"]

logger = structlog.get_logger(__name__)

class VADService:
    """Optimized Voice Activity Detection service using Silero VAD with real-time performance"""
    
    def __init__(self, 
                 speech_threshold: float = 0.5,  # Lowered from 0.75 for higher sensitivity
                 silence_threshold: float = 0.2,  # Lowered from 0.3 for more reliable silence detection
                 min_speech_duration: float = 0.25,  # Reduced to detect shorter speech segments
                 min_silence_duration: float = 0.5,  # Unchanged
                 sample_rate: int = 16000):
        
        self.speech_threshold = speech_threshold
        self.silence_threshold = silence_threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.sample_rate = sample_rate
        
        self.model = None
        self.utils = None
        self.is_available = False
        self.is_speaking = False
        
        # State tracking for hysteresis
        self.speech_start_time = None
        self.silence_start_time = None
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.frame_duration = 0.032  # 32ms frames (512 samples at 16kHz)
        
        # Performance tracking
        self.processing_times = []
        
    async def initialize(self):
        """Initialize the Silero VAD model"""
        try:
            logger.info("Initializing Silero VAD model for real-time processing...")
            
            # Load Silero VAD model
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            # Enable evaluation mode for inference
            self.model.eval()
            
            # Warm up the model with dummy data
            dummy_audio = torch.zeros(1, 512)  # 32ms at 16kHz
            with torch.no_grad():
                _ = self.model(dummy_audio, self.sample_rate)
            
            self.is_available = True
            logger.info("✅ Silero VAD model initialized and warmed up for real-time processing")
            
        except Exception as e:
            logger.error(f"Failed to initialize VAD service: {e}", exc_info=True)
            self.is_available = False

    async def process_audio_chunk(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """
        Process audio chunk for voice activity detection with optimized performance
        Expected input: 16kHz, 32ms chunks (512 samples)
        """
        if not self.is_available:
            return {"speech_detected": False, "confidence": 0.0, "processing_time": 0.0}
        
        start_time = time.time()
        
        try:
            # Ensure audio is the right format
            if len(audio_chunk) != 512:  # 32ms at 16kHz
                # Pad or truncate to 512 samples
                if len(audio_chunk) < 512:
                    audio_chunk = np.pad(audio_chunk, (0, 512 - len(audio_chunk)))
                else:
                    audio_chunk = audio_chunk[:512]
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
            
            # Run VAD inference
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
            # Apply hysteresis and temporal filtering
            result = self._apply_temporal_logic(speech_prob)
            
            # Track processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            result["processing_time"] = processing_time
            result["confidence"] = speech_prob
            
            return result
            
        except Exception as e:
            logger.error(f"Error during VAD processing: {e}")
            return {"speech_detected": False, "confidence": 0.0, "processing_time": 0.0, "error": str(e)}

    def _apply_temporal_logic(self, speech_prob: float) -> Dict[str, Any]:
        """Apply temporal logic to reduce false positives and improve stability"""
        current_time = time.time()
        speech_detected = False
        speech_started = False
        speech_ended = False
        
        # Determine current frame state
        is_speech_frame = speech_prob > self.speech_threshold
        is_silence_frame = speech_prob < self.silence_threshold
        
        if is_speech_frame:
            self.consecutive_speech_frames += 1
            self.consecutive_silence_frames = 0
            
            if not self.is_speaking:
                if self.speech_start_time is None:
                    self.speech_start_time = current_time
                elif (current_time - self.speech_start_time) >= self.min_speech_duration:
                    # Check if we have enough consecutive speech frames to confirm it's real speech
                    if self.consecutive_speech_frames >= max(3, int(self.min_speech_duration / self.frame_duration)):
                        # Confirmed speech start
                        self.is_speaking = True
                        speech_started = True
                        speech_detected = True
                        self.silence_start_time = None
                        logger.debug(f"Speech started (confidence: {speech_prob:.3f}, frames: {self.consecutive_speech_frames})")
                    else:
                        # Not enough consecutive frames yet
                        speech_detected = False
            else:
                speech_detected = True
                self.silence_start_time = None
                
        elif is_silence_frame:
            self.consecutive_silence_frames += 1
            self.consecutive_speech_frames = 0
            
            if self.is_speaking:
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                elif (current_time - self.silence_start_time) >= self.min_silence_duration:
                    # Only end speech if we have enough consecutive silence frames
                    if self.consecutive_silence_frames >= max(5, int(self.min_silence_duration / self.frame_duration)):
                        # Confirmed speech end
                        self.is_speaking = False
                        speech_ended = True
                        speech_detected = False
                        self.speech_start_time = None
                        logger.debug(f"Speech ended (confidence: {speech_prob:.3f}, silence frames: {self.consecutive_silence_frames})")
                    else:
                        # Not enough consecutive silence frames yet
                        speech_detected = True
            else:
                # Reset speech start if we haven't confirmed speech yet
                if self.speech_start_time is not None:
                    self.speech_start_time = None
                    
        else:
            # Ambiguous region - maintain current state
            if self.is_speaking:
                speech_detected = True
        
        return {
            "speech_detected": speech_detected,
            "speech_started": speech_started,
            "speech_ended": speech_ended,
            "is_speaking": self.is_speaking,
            "consecutive_speech_frames": self.consecutive_speech_frames,
            "consecutive_silence_frames": self.consecutive_silence_frames
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current VAD service status and performance metrics"""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        
        return {
            "available": self.is_available,
            "service_type": "Silero_VAD",
            "speech_threshold": self.speech_threshold,
            "silence_threshold": self.silence_threshold,
            "min_speech_duration": self.min_speech_duration,
            "min_silence_duration": self.min_silence_duration,
            "is_currently_speaking": self.is_speaking,
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "consecutive_speech_frames": self.consecutive_speech_frames,
            "consecutive_silence_frames": self.consecutive_silence_frames
        }

    async def cleanup(self):
        """Cleanup VAD resources"""
        logger.info("Cleaning up VAD service")
        self.model = None
        self.utils = None
        self.is_available = False 