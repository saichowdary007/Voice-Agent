import os
import asyncio
import logging
import numpy as np
import soundfile as sf
import tempfile
from typing import Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass
import sherpa_ncnn

logger = logging.getLogger(__name__)

@dataclass
class STTResult:
    """STT processing result"""
    partial_text: str
    final_text: str
    is_final: bool
    confidence: float
    timestamp: float = 0.0

class STTService:
    """Speech-to-Text service using sherpa-ncnn 2.1.11 for ultra-fast streaming ASR"""
    
    def __init__(self):
        self.recognizer = None
        self.is_available = False
        self.sample_rate = 16000
        
        # Model paths - will be downloaded at build time
        self.model_dir = os.getenv("MODEL_PATH", "/app/models")
        
        # Audio buffer for streaming
        self.audio_buffer = []
        self.total_audio_duration = 0.0
        
    async def initialize(self):
        """Initialize the sherpa-ncnn ASR model"""
        try:
            logger.info("Initializing sherpa-ncnn 2.1.11 model...")
            
            # Use the correct API for sherpa-ncnn 2.1.11
            # Based on documentation: direct Recognizer initialization
            def create_recognizer():
                return sherpa_ncnn.Recognizer(
                    tokens=f"{self.model_dir}/tokens.txt",
                    encoder_param=f"{self.model_dir}/encoder_jit_trace-pnnx.ncnn.param",
                    encoder_bin=f"{self.model_dir}/encoder_jit_trace-pnnx.ncnn.bin",
                    decoder_param=f"{self.model_dir}/decoder_jit_trace-pnnx.ncnn.param",
                    decoder_bin=f"{self.model_dir}/decoder_jit_trace-pnnx.ncnn.bin",
                    joiner_param=f"{self.model_dir}/joiner_jit_trace-pnnx.ncnn.param",
                    joiner_bin=f"{self.model_dir}/joiner_jit_trace-pnnx.ncnn.bin",
                    num_threads=2,
                )
            
            self.recognizer = await asyncio.to_thread(create_recognizer)
            
            if not self.recognizer:
                raise Exception("Failed to create recognizer")
                
            self.is_available = True
            logger.info("✅ sherpa-ncnn 2.1.11 initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize sherpa-ncnn: {e}")
            self.is_available = False

    async def process_frame(self, audio_frame: bytes) -> STTResult:
        """Process audio frame and return streaming STT result"""
        if not self.is_available:
            return STTResult(
                partial_text="",
                final_text="",
                is_final=False,
                confidence=0.0,
                timestamp=0.0
            )
        
        try:
            # Convert bytes to numpy array (mono 16-bit PCM)
            if isinstance(audio_frame, bytes):
                audio_samples = np.frombuffer(audio_frame, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_samples = audio_frame
                
            # Add to audio buffer
            self.audio_buffer.extend(audio_samples.tolist())
            
            # Wait for enough audio before processing
            min_samples = int(self.sample_rate * 0.03)  # 30ms
            
            if len(self.audio_buffer) >= min_samples:
                # Extract audio chunk
                audio_chunk = np.array(self.audio_buffer)
                self.audio_buffer = []
                    
                # Feed audio to recognizer (sherpa-ncnn 2.1.11 API)
                def process_audio():
                    logger.debug(f"STT:process_audio: About to call accept_waveform. Audio chunk shape: {audio_chunk.shape}, dtype: {audio_chunk.dtype}")
                    
                    # Ensure audio data is 1D before sending to recognizer
                    if len(audio_chunk.shape) > 1:
                        logger.warning(f"Audio chunk has {len(audio_chunk.shape)} dimensions, flattening to 1D")
                        audio_chunk_1d = audio_chunk.flatten()
                    else:
                        audio_chunk_1d = audio_chunk
                        
                    self.recognizer.accept_waveform(self.sample_rate, audio_chunk_1d)
                    logger.debug("STT:process_audio: accept_waveform completed. About to get text.")
                    text_result = self.recognizer.text.strip()
                    logger.debug(f"STT:process_audio: Got text: '{text_result}'")
                    return text_result
                
                partial_text = await asyncio.to_thread(process_audio)
                
                # Update audio duration
                audio_duration = len(audio_chunk) / self.sample_rate
                self.total_audio_duration += audio_duration
                
                return STTResult(
                    partial_text=partial_text,
                    final_text="",
                    is_final=False,
                    confidence=0.95,
                    timestamp=self.total_audio_duration
                )
            else:
                # Not enough audio yet
                return STTResult(
                    partial_text="",
                    final_text="",
                    is_final=False,
                    confidence=0.0,
                    timestamp=self.total_audio_duration
                )
                
        except Exception as e:
            logger.error(f"STT processing error: {e}", exc_info=True)
            return STTResult(
                partial_text="",
                final_text="",
                is_final=False,
                confidence=0.0,
                timestamp=self.total_audio_duration
            )

    async def finalize(self) -> Optional[STTResult]:
        """Finalize current utterance and get final result"""
        try:
            if not self.is_available:
                return None
                
            def finalize_recognition():
                # Process any remaining audio
                if self.audio_buffer:
                    audio_chunk = np.array(self.audio_buffer)
                    self.audio_buffer = []
                    self.recognizer.accept_waveform(self.sample_rate, audio_chunk)
                    
                # Add tail padding for complete recognition
                tail_paddings = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
                self.recognizer.accept_waveform(self.sample_rate, tail_paddings)
                
                # Signal input finished for final result
                self.recognizer.input_finished()
                
                # Get final result
                final_text = self.recognizer.text.strip()
                
                # Reset for next utterance
                self.recognizer.reset()
                
                return final_text
            
            final_text = await asyncio.to_thread(finalize_recognition)
            
            if final_text:
                logger.debug(f"STT finalized with text: {final_text}")
                
                return STTResult(
                    partial_text="",
                    final_text=final_text,
                    is_final=True,
                    confidence=0.95,
                    timestamp=self.total_audio_duration
                )
            else:
                logger.debug("STT finalization returned no text")
                return None
                
        except Exception as e:
            logger.error(f"STT finalization error: {e}")
            return None
    
    async def transcribe_streaming(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Transcribe audio data in streaming mode
        
        Args:
            audio_data: Raw audio bytes (PCM 16-bit, 16kHz, mono)
            
        Returns:
            Dict with partial transcript and completion status
        """
        if not self.is_available:
            return {
                "partial": "",
                "final": "",
                "is_complete": False,
                "confidence": 0.0,
                "error": "STT service not available"
            }
        
        try:
            # Convert bytes to numpy array
            audio_samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Use sherpa-ncnn 2.1.11 streaming API
            def process_streaming():
                self.recognizer.accept_waveform(self.sample_rate, audio_samples)
                return self.recognizer.text
            
            result_text = await asyncio.to_thread(process_streaming)
            
            # For streaming, we return partial results
            return {
                "partial": result_text,
                "final": "",
                "is_complete": False,
                "confidence": 0.95,  # sherpa-ncnn doesn't provide confidence
                "tokens": []
            }
                
        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            return {
                "partial": "",
                "final": "",
                "is_complete": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def transcribe_audio(self, audio_data: bytes, language: str = "en") -> Dict[str, Any]:
        """
        Transcribe complete audio data (offline mode)
        
        Args:
            audio_data: Raw audio bytes
            language: Language code (ignored for now)
            
        Returns:
            Dict with transcript, confidence, and metadata
        """
        if not self.is_available:
            return {
                "transcript": "",
                "confidence": 0.0,
                "language": "en",
                "error": "STT service not available"
            }
        
        try:
            # Convert bytes to numpy array
            audio_samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            def transcribe_complete():
                # Use sherpa-ncnn 2.1.11 API for complete transcription
                self.recognizer.accept_waveform(self.sample_rate, audio_samples)
                
                # Add tail padding for complete recognition
                tail_paddings = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
                self.recognizer.accept_waveform(self.sample_rate, tail_paddings)
                
                # Signal input finished for final result
                self.recognizer.input_finished()
                
                return self.recognizer.text.strip()
            
            transcript = await asyncio.to_thread(transcribe_complete)
                
            logger.info(f"STT transcribed: '{transcript}'")
                
            return {
                "transcript": transcript,
                "confidence": 0.95,
                "language": "en",
                "duration": len(audio_samples) / self.sample_rate,
                "tokens": []
            }
                    
        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            return {
                "transcript": "",
                "confidence": 0.0,
                "language": "en",
                "error": str(e)
            }
    
    def reset_stream(self):
        """Reset the streaming session"""
        try:
            if self.recognizer:
                # Reset recognizer state for new stream
                self.recognizer.reset()
                self.audio_buffer = []
                self.total_audio_duration = 0.0
                logger.debug("STT stream reset successfully")
        except Exception as e:
            logger.warning(f"Failed to reset stream: {e}")
    
    async def test_connection(self) -> bool:
        """Test STT service"""
        if not self.is_available:
            return False
        
        try:
            # Test with small audio buffer
            test_audio = np.zeros(1600, dtype=np.float32)  # 100ms of silence
            test_bytes = (test_audio * 32768).astype(np.int16).tobytes()
            result = await self.transcribe_audio(test_bytes)
            return not result.get("error")
        except Exception as e:
            logger.error(f"STT connection test failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get STT service status"""
        return {
            "available": self.is_available,
            "service": "Sherpa-NCNN" if self.is_available else "Not Available",
            "sample_rate": self.sample_rate,
            "model_path": self.model_dir,
            "streaming": True
        }
    
    async def cleanup(self):
        """Clean up STT resources"""
        try:
            logger.info("Cleaning up STT service...")
            
            # Mark service as unavailable first to prevent new operations
            self.is_available = False
            
            # Clear audio buffer first
            self.audio_buffer = []
            
            # Clean up recognizer with proper error handling
            if self.recognizer:
                try:
                    # Try to reset before deletion to ensure clean state
                    self.recognizer.reset()
                except Exception as reset_error:
                    logger.warning(f"Error resetting recognizer during cleanup: {reset_error}")
                
                try:
                    # Don't explicitly delete - let Python's GC handle it
                    # This prevents double free errors
                    pass
                except Exception as del_error:
                    logger.warning(f"Error during recognizer cleanup: {del_error}")
                finally:
                    # Always set to None to prevent double cleanup
                    self.recognizer = None
                    
            logger.info("STT service cleaned up")
            
        except Exception as e:
            logger.error(f"STT cleanup error: {e}")
            # Ensure recognizer is None even if cleanup fails
            self.recognizer = None 