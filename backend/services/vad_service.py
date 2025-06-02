import logging
import numpy as np
import torch
import asyncio
from typing import Optional, Tuple, Dict, Any
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

logger = logging.getLogger(__name__)

class VADService:
    """Voice Activity Detection service using Silero VAD"""
    
    def __init__(self, threshold: float = 0.6, min_speech_duration_ms: int = 250):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.sample_rate = 16000
        self.model = None
        self.is_available = False
        
        # Speech state tracking
        self.is_speech_active = False
        self.speech_start_time = None
        self.silence_duration = 0
        self.max_silence_ms = 1000  # 1 second of silence ends speech
        
    async def initialize(self):
        """Initialize the VAD model"""
        try:
            logger.info("Loading Silero VAD model...")
            self.model = load_silero_vad()
            self.is_available = True
            logger.info("✅ Silero VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            self.is_available = False
            
    def process_audio_frame(self, audio_data: np.ndarray) -> Tuple[bool, bool]:
        """
        Process audio frame and detect speech activity
        
        Args:
            audio_data: Audio data as numpy array (16kHz, mono)
            
        Returns:
            Tuple of (is_speech, speech_ended)
        """
        if not self.is_available or self.model is None:
            return False, False
            
        try:
            # Ensure audio is the right format
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # Convert to mono
                
            # Normalize audio
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
                
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_data)
            
            # Get VAD prediction
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
            is_speech = speech_prob > self.threshold
            
            # Track speech state
            current_time = len(audio_data) / self.sample_rate * 1000  # ms
            speech_ended = False
            
            if is_speech:
                if not self.is_speech_active:
                    self.is_speech_active = True
                    self.speech_start_time = current_time
                    logger.debug("Speech started")
                self.silence_duration = 0
            else:
                if self.is_speech_active:
                    self.silence_duration += len(audio_data) / self.sample_rate * 1000
                    if self.silence_duration >= self.max_silence_ms:
                        self.is_speech_active = False
                        speech_ended = True
                        logger.debug("Speech ended")
                        
            return is_speech, speech_ended
            
        except Exception as e:
            logger.error(f"VAD processing error: {e}")
            return False, False
    
    def process_audio_bytes(self, audio_bytes: bytes) -> Tuple[bool, bool]:
        """
        Process raw audio bytes
        
        Args:
            audio_bytes: Raw audio bytes (PCM 16-bit)
            
        Returns:
            Tuple of (is_speech, speech_ended)
        """
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return self.process_audio_frame(audio_data)
        except Exception as e:
            logger.error(f"Error processing audio bytes: {e}")
            return False, False
    
    def detect_end_of_speech(self, audio_data: np.ndarray) -> bool:
        """
        Detect if speech has ended using Silero VAD
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if speech has ended
        """
        if not self.is_available:
            return False
            
        try:
            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(
                audio_data, 
                self.model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_duration_ms
            )
            
            # If no speech detected in recent audio, consider speech ended
            return len(speech_timestamps) == 0 and self.is_speech_active
            
        except Exception as e:
            logger.error(f"End of speech detection error: {e}")
            return False
    
    def reset_state(self):
        """Reset speech detection state"""
        self.is_speech_active = False
        self.speech_start_time = None
        self.silence_duration = 0
        
    def get_status(self) -> Dict[str, Any]:
        """Get VAD service status"""
        return {
            "available": self.is_available,
            "threshold": self.threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "is_speech_active": self.is_speech_active,
            "sample_rate": self.sample_rate
        } 