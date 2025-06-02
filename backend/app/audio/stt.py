import os
import asyncio
from typing import Optional, AsyncGenerator
import numpy as np
import sherpa_ncnn
import structlog
from dataclasses import dataclass

logger = structlog.get_logger()


@dataclass
class STTResult:
    """Speech-to-text result"""
    partial_text: str
    final_text: str
    is_final: bool
    confidence: float
    timestamp: float


class STTEngine:
    """Sherpa-NCNN streaming speech-to-text engine"""
    
    def __init__(self):
        self.sample_rate = int(os.getenv('SAMPLE_RATE', '16000'))
        self.model_path = os.getenv('MODEL_PATH', '/opt/render/project/src/models')
        
        # Model components
        self.recognizer: Optional[sherpa_ncnn.Recognizer] = None
        
        # Processing state
        self.audio_buffer = []
        self.last_result = ""
        self.accumulated_text = ""
        
        # Performance tracking
        self.total_audio_duration = 0.0
        self.total_processing_time = 0.0
        
    async def initialize(self):
        """Initialize Sherpa-NCNN recognizer"""
        try:
            logger.info("Initializing sherpa-ncnn 2.1.11 model...")
            
            # Use the correct API for sherpa-ncnn 2.1.11
            self.recognizer = sherpa_ncnn.Recognizer(
                tokens=f"{self.model_path}/tokens.txt",
                encoder_param=f"{self.model_path}/encoder_jit_trace-pnnx.ncnn.param",
                encoder_bin=f"{self.model_path}/encoder_jit_trace-pnnx.ncnn.bin",
                decoder_param=f"{self.model_path}/decoder_jit_trace-pnnx.ncnn.param",
                decoder_bin=f"{self.model_path}/decoder_jit_trace-pnnx.ncnn.bin",
                joiner_param=f"{self.model_path}/joiner_jit_trace-pnnx.ncnn.param",
                joiner_bin=f"{self.model_path}/joiner_jit_trace-pnnx.ncnn.bin",
                num_threads=2,
            )
            
            if not self.recognizer:
                raise RuntimeError("Failed to create recognizer")
                
            logger.info("✅ sherpa-ncnn 2.1.11 initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize STT: {e}")
            raise
            
    async def is_ready(self) -> bool:
        """Check if STT engine is ready"""
        return self.recognizer is not None
        
    async def preload(self):
        """Preload STT model"""
        if self.recognizer is None:
            await self.initialize()
            
        # Warm up with silence
        dummy_audio = np.zeros(self.sample_rate, dtype=np.float32)
        self.recognizer.accept_waveform(self.sample_rate, dummy_audio)
        
        logger.info("STT model preloaded")
        
    async def process_frame(self, audio_frame: bytes) -> STTResult:
        """Process audio frame and return streaming STT result"""
        try:
            import time
            start_time = time.time()
            
            # Convert bytes to float32 audio
            if isinstance(audio_frame, bytes):
                audio_data = np.frombuffer(audio_frame, dtype=np.int16)
            else:
                audio_data = audio_frame
                
            # Normalize to float32 [-1, 1]
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Add to buffer
            self.audio_buffer.extend(audio_float)
            
            # Process if we have enough samples (30ms minimum)
            min_samples = self.sample_rate * 30 // 1000  # 30ms
            if len(self.audio_buffer) >= min_samples:
                # Extract audio chunk
                audio_chunk = np.array(self.audio_buffer)
                self.audio_buffer = []
                    
                # Feed audio to recognizer (sherpa-ncnn 2.1.11 API)
                self.recognizer.accept_waveform(self.sample_rate, audio_chunk)
                
                # Get partial result
                partial_text = self.recognizer.text.strip()
                
                # Update performance metrics
                processing_time = time.time() - start_time
                audio_duration = len(audio_chunk) / self.sample_rate
                self.total_processing_time += processing_time
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
            logger.error(f"STT processing error: {e}")
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
            
            if final_text:
                logger.debug(f"STT finalized with text: {final_text}")
                
                # Reset for next utterance
                self.recognizer.reset()
                
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
            
    def get_performance_metrics(self) -> dict:
        """Get STT performance metrics"""
        if self.total_audio_duration > 0:
            rtf = self.total_processing_time / self.total_audio_duration
        else:
            rtf = 0.0
            
        return {
            "real_time_factor": rtf,
            "total_audio_duration": self.total_audio_duration,
            "total_processing_time": self.total_processing_time,
            "average_latency": self.total_processing_time / max(1, self.total_audio_duration)
        }
        
    def reset_session(self):
        """Reset STT session state"""
        try:
            if self.recognizer:
                self.recognizer.reset()
                
            self.audio_buffer = []
            self.last_result = ""
            self.accumulated_text = ""
            
            logger.debug("STT session reset")
            
        except Exception as e:
            logger.error(f"STT reset error: {e}")
            
    async def cleanup(self):
        """Clean up STT resources"""
        try:
            if self.recognizer:
                del self.recognizer
                self.recognizer = None
                
            self.audio_buffer = []
            
            logger.info("STT engine cleaned up")
            
        except Exception as e:
            logger.error(f"STT cleanup error: {e}")


class StreamingSTT:
    """Streaming STT wrapper with enhanced features"""
    
    def __init__(self, base_stt: STTEngine):
        self.base_stt = base_stt
        self.sentence_buffer = []
        self.word_timestamps = []
        
    async def process_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[STTResult, None]:
        """Process audio stream and yield STT results"""
        try:
            async for audio_chunk in audio_stream:
                result = await self.base_stt.process_frame(audio_chunk)
                
                if result.partial_text or result.is_final:
                    yield result
                    
                # Handle sentence boundaries
                if result.is_final and result.final_text:
                    self.sentence_buffer.append(result.final_text)
                    
        except Exception as e:
            logger.error(f"Streaming STT error: {e}")
            
    def get_full_transcript(self) -> str:
        """Get full transcript from sentence buffer"""
        return " ".join(self.sentence_buffer)
        
    def clear_transcript(self):
        """Clear transcript buffer"""
        self.sentence_buffer = []
        self.word_timestamps = [] 