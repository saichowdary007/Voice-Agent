import os
import asyncio
from typing import Optional, NamedTuple
import numpy as np
import torch
import torchaudio
import structlog
from dataclasses import dataclass

logger = structlog.get_logger()


@dataclass
class VADResult:
    """Voice activity detection result"""
    is_speech: bool
    is_end_of_speech: bool
    confidence: float
    timestamp: float


class VADEngine:
    """Silero VAD engine for real-time voice activity detection"""
    
    def __init__(self):
        self.sample_rate = int(os.getenv('SAMPLE_RATE', '16000'))
        self.threshold = float(os.getenv('VAD_THRESHOLD', '0.6'))
        self.min_speech_duration = 0.1  # 100ms minimum speech
        self.min_silence_duration = 0.5  # 500ms minimum silence for end-of-speech
        
        # Model and state
        self.model: Optional[torch.nn.Module] = None
        self.utils: Optional[tuple] = None
        
        # Detection state
        self.speech_start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None
        self.is_speaking = False
        self.frame_count = 0
        
        # Audio buffer for processing
        self.audio_buffer = []
        self.buffer_size = 1024  # Process in 1024 sample chunks
        
    async def initialize(self):
        """Initialize Silero VAD model"""
        try:
            logger.info("Loading Silero VAD model...")
            
            # Load model
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            self.model = model
            self.utils = utils
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("Silero VAD model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize VAD: {e}")
            raise
            
    async def is_ready(self) -> bool:
        """Check if VAD engine is ready"""
        return self.model is not None
        
    async def preload(self):
        """Preload VAD model"""
        if self.model is None:
            await self.initialize()
            
        # Warm up with dummy audio (512 samples for 16kHz)
        dummy_audio = torch.zeros(512, dtype=torch.float32)
        with torch.no_grad():
            _ = self.model(dummy_audio, self.sample_rate)
            
        logger.info("VAD model preloaded")
        
    async def process_frame(self, audio_frame: bytes) -> VADResult:
        """Process audio frame and detect voice activity"""
        try:
            # Convert bytes to numpy array
            if isinstance(audio_frame, bytes):
                audio_data = np.frombuffer(audio_frame, dtype=np.int16)
            else:
                audio_data = audio_frame
                
            # Normalize to float32 [-1, 1]
            audio_float = audio_data.astype(np.float32) / 32767.0
            
            # Add to buffer
            self.audio_buffer.extend(audio_float)
            
            # Process when we have enough samples
            if len(self.audio_buffer) >= self.buffer_size:
                # Extract chunk
                chunk = np.array(self.audio_buffer[:self.buffer_size])
                self.audio_buffer = self.audio_buffer[self.buffer_size:]
                
                # Run VAD
                result = await self._detect_speech(chunk)
                return result
            else:
                # Not enough data yet
                current_time = self.frame_count * self.buffer_size / self.sample_rate
                return VADResult(
                    is_speech=self.is_speaking,
                    is_end_of_speech=False,
                    confidence=0.0,
                    timestamp=current_time
                )
                
        except Exception as e:
            logger.error(f"VAD processing error: {e}")
            # Return safe default
            current_time = self.frame_count * self.buffer_size / self.sample_rate
            return VADResult(
                is_speech=False,
                is_end_of_speech=False,
                confidence=0.0,
                timestamp=current_time
            )
            
    async def _detect_speech(self, audio_chunk: np.ndarray) -> VADResult:
        """Run speech detection on audio chunk"""
        try:
            self.frame_count += 1
            current_time = self.frame_count * self.buffer_size / self.sample_rate
            
            # Convert to torch tensor
            audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)
            
            # Run VAD model
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
                
            # Determine if speech is detected
            is_speech = speech_prob > self.threshold
            is_end_of_speech = False
            
            # State management
            if is_speech:
                if not self.is_speaking:
                    # Speech start
                    if self.speech_start_time is None:
                        self.speech_start_time = current_time
                    elif current_time - self.speech_start_time >= self.min_speech_duration:
                        # Confirmed speech start
                        self.is_speaking = True
                        logger.debug(f"Speech detected at {current_time:.2f}s")
                        
                self.last_speech_time = current_time
                
            else:
                # No speech detected
                if self.is_speaking and self.last_speech_time:
                    silence_duration = current_time - self.last_speech_time
                    if silence_duration >= self.min_silence_duration:
                        # End of speech
                        self.is_speaking = False
                        is_end_of_speech = True
                        self.speech_start_time = None
                        logger.debug(f"End of speech detected at {current_time:.2f}s")
                        
                elif not self.is_speaking:
                    # Reset speech start if false alarm
                    self.speech_start_time = None
                    
            return VADResult(
                is_speech=is_speech and self.is_speaking,
                is_end_of_speech=is_end_of_speech,
                confidence=speech_prob,
                timestamp=current_time
            )
            
        except Exception as e:
            logger.error(f"Speech detection error: {e}")
            return VADResult(
                is_speech=False,
                is_end_of_speech=False,
                confidence=0.0,
                timestamp=current_time
            )
            
    def reset_state(self):
        """Reset VAD state"""
        self.speech_start_time = None
        self.last_speech_time = None
        self.is_speaking = False
        self.frame_count = 0
        self.audio_buffer.clear()
        logger.debug("VAD state reset")
        
    async def cleanup(self):
        """Clean up VAD resources"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
                
            if self.utils is not None:
                del self.utils
                self.utils = None
                
            # Clear audio buffer
            self.audio_buffer.clear()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("VAD engine cleaned up")
            
        except Exception as e:
            logger.error(f"VAD cleanup error: {e}")


class AdaptiveVAD:
    """Adaptive VAD with dynamic threshold adjustment"""
    
    def __init__(self, base_vad: VADEngine):
        self.base_vad = base_vad
        self.noise_level = 0.0
        self.adaptation_rate = 0.01
        self.min_threshold = 0.3
        self.max_threshold = 0.8
        
    async def process_frame(self, audio_frame: bytes) -> VADResult:
        """Process frame with adaptive threshold"""
        # Get base result
        result = await self.base_vad.process_frame(audio_frame)
        
        # Update noise level estimate
        if not result.is_speech:
            self.noise_level = (
                (1 - self.adaptation_rate) * self.noise_level +
                self.adaptation_rate * result.confidence
            )
            
        # Adjust threshold based on noise level
        adaptive_threshold = np.clip(
            self.base_vad.threshold + self.noise_level * 2,
            self.min_threshold,
            self.max_threshold
        )
        
        # Re-evaluate with adaptive threshold
        result.is_speech = result.confidence > adaptive_threshold
        
        return result 