import logging
import numpy as np
import torch
import asyncio
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

logger = logging.getLogger(__name__)

@dataclass
class VADResult:
    """Voice activity detection result"""
    is_speech: bool
    is_end_of_speech: bool
    confidence: float
    timestamp: float

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
        self.frame_count = 0
        
    async def initialize(self):
        """Initialize the VAD model"""
        try:
            logger.info("Loading Silero VAD model...")
            self.model = await asyncio.to_thread(load_silero_vad)
            self.is_available = True
            logger.info("✅ Silero VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            self.is_available = False

    async def process_frame(self, audio_frame: bytes) -> VADResult:
        """Process audio frame and detect voice activity"""
        try:
            self.frame_count += 1
            current_time = self.frame_count * len(audio_frame) / (self.sample_rate * 2)  # 2 bytes per sample
            
            if not self.is_available or self.model is None:
                return VADResult(
                    is_speech=False,
                    is_end_of_speech=False,
                    confidence=0.0,
                    timestamp=current_time
                )
            
            # Convert bytes to numpy array
            if isinstance(audio_frame, bytes):
                audio_data = np.frombuffer(audio_frame, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_data = audio_frame
                
            # DEBUG: Log the actual audio data size
            logger.debug(f"VAD processing: received {len(audio_frame)} bytes = {len(audio_data)} samples")
                
            # Process audio frame
            def process_vad():
                # Ensure audio is the right format
                if len(audio_data.shape) > 1:
                    processed_audio = audio_data.mean(axis=1)  # Convert to mono
                else:
                    processed_audio = audio_data
                    
                # Normalize audio
                if processed_audio.dtype != np.float32:
                    processed_audio = processed_audio.astype(np.float32)
                if np.max(np.abs(processed_audio)) > 1.0:
                    processed_audio = processed_audio / np.max(np.abs(processed_audio))
                
                # DEBUG: Log the processed audio size before sending to VAD model
                logger.debug(f"VAD model input: {len(processed_audio)} samples")
                    
                # Convert to tensor
                audio_tensor = torch.from_numpy(processed_audio)
                
                # Get VAD prediction
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
                return speech_prob
            
            speech_prob = await asyncio.to_thread(process_vad)
            is_speech = speech_prob > self.threshold
            
            # Track speech state
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
                        
            return VADResult(
                is_speech=is_speech and self.is_speech_active,
                is_end_of_speech=speech_ended,
                confidence=speech_prob,
                timestamp=current_time
            )
            
        except Exception as e:
            logger.error(f"VAD processing error: {e}")
            current_time = self.frame_count * len(audio_frame) / (self.sample_rate * 2) if isinstance(audio_frame, bytes) else 0
            return VADResult(
                is_speech=False,
                is_end_of_speech=False,
                confidence=0.0,
                timestamp=current_time
            )
            
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
        self.frame_count = 0
        
    def get_status(self) -> Dict[str, Any]:
        """Get VAD service status"""
        return {
            "available": self.is_available,
            "threshold": self.threshold,
            "sample_rate": self.sample_rate,
            "is_speech_active": self.is_speech_active,
            "frame_count": self.frame_count
        }
    
    async def cleanup(self):
        """Clean up VAD resources"""
        try:
            logger.info("Cleaning up VAD service...")
            
            # Mark service as unavailable first to prevent new operations
            self.is_available = False
            
            # Reset state first
            self.reset_state()
            
            # Clean up model with proper error handling
            if self.model is not None:
                try:
                    # Clear CUDA cache before deleting model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as cuda_error:
                    logger.warning(f"Error clearing CUDA cache: {cuda_error}")
                
                try:
                    # Don't explicitly delete - let Python's GC handle it
                    # This prevents double free errors
                    pass
                except Exception as del_error:
                    logger.warning(f"Error during VAD model cleanup: {del_error}")
                finally:
                    # Always set to None to prevent double cleanup
                    self.model = None
                    
            logger.info("VAD service cleaned up")
            
        except Exception as e:
            logger.error(f"VAD cleanup error: {e}")
            # Ensure model is None even if cleanup fails
            self.model = None
            self.is_available = False 