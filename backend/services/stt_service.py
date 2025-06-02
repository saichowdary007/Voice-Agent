import os
import asyncio
import logging
import numpy as np
import soundfile as sf
import tempfile
from typing import Optional, Dict, Any, AsyncGenerator
import sherpa_ncnn

logger = logging.getLogger(__name__)

class STTService:
    """Speech-to-Text service using sherpa-ncnn 2.1.11 for ultra-fast streaming ASR"""
    
    def __init__(self):
        self.recognizer = None
        self.is_available = False
        self.sample_rate = 16000
        
        # Model paths - will be downloaded at build time
        self.model_dir = os.getenv("MODEL_PATH", "/app/models")
        
    async def initialize(self):
        """Initialize the sherpa-ncnn ASR model"""
        try:
            logger.info("Initializing sherpa-ncnn 2.1.11 model...")
            
            # Use the correct API for sherpa-ncnn 2.1.11
            # Based on documentation: direct Recognizer initialization
            self.recognizer = sherpa_ncnn.Recognizer(
                tokens=f"{self.model_dir}/tokens.txt",
                encoder_param=f"{self.model_dir}/encoder_jit_trace-pnnx.ncnn.param",
                encoder_bin=f"{self.model_dir}/encoder_jit_trace-pnnx.ncnn.bin",
                decoder_param=f"{self.model_dir}/decoder_jit_trace-pnnx.ncnn.param",
                decoder_bin=f"{self.model_dir}/decoder_jit_trace-pnnx.ncnn.bin",
                joiner_param=f"{self.model_dir}/joiner_jit_trace-pnnx.ncnn.param",
                joiner_bin=f"{self.model_dir}/joiner_jit_trace-pnnx.ncnn.bin",
                num_threads=2,
            )
            
            if not self.recognizer:
                raise Exception("Failed to create recognizer")
                
            self.is_available = True
            logger.info("✅ sherpa-ncnn 2.1.11 initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize sherpa-ncnn: {e}")
            self.is_available = False
    
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
            self.recognizer.accept_waveform(self.sample_rate, audio_samples)
            result_text = self.recognizer.text
            
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
            
            # Use sherpa-ncnn 2.1.11 API for complete transcription
            self.recognizer.accept_waveform(self.sample_rate, audio_samples)
            
            # Add tail padding for complete recognition
            tail_paddings = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
            self.recognizer.accept_waveform(self.sample_rate, tail_paddings)
            
            # Signal input finished for final result
            self.recognizer.input_finished()
            
            transcript = self.recognizer.text.strip()
                
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
            "service": "sherpa-ncnn 2.1.11" if self.is_available else "Not Available",
            "sample_rate": self.sample_rate,
            "streaming": True,
            "model_dir": self.model_dir
        } 