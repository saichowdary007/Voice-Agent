import os
import asyncio
import tempfile
from typing import Optional, AsyncGenerator, List
import numpy as np
import subprocess
import structlog
from dataclasses import dataclass
import json
import io

from .codec import OpusCodec, AudioConverter

logger = structlog.get_logger()


@dataclass
class TTSResult:
    """Text-to-speech result"""
    audio_data: bytes
    duration: float
    sample_rate: int
    text: str


class TTSEngine:
    """Piper TTS engine for high-quality speech synthesis"""
    
    def __init__(self):
        self.sample_rate = int(os.getenv('SAMPLE_RATE', '16000'))
        self.model_path = os.getenv('MODEL_PATH', '/app/models')
        self.speed = float(os.getenv('TTS_SPEED', '1.0'))
        
        # Piper configuration
        self.piper_model = f"{self.model_path}/en_US-libritts-high.onnx"
        self.piper_config = f"{self.model_path}/en_US-libritts-high.onnx.json"
        
        # Audio processing
        self.opus_codec = OpusCodec()
        
        # Streaming state
        self.is_speaking = False
        self.current_process: Optional[subprocess.Popen] = None
        
        # Performance tracking
        self.total_chars_processed = 0
        self.total_processing_time = 0.0
        
    async def initialize(self):
        """Initialize TTS engine"""
        try:
            logger.info("Initializing Piper TTS engine...")
            
            # Initialize Opus codec
            await self.opus_codec.initialize()
            
            # Verify Piper model files exist
            if not os.path.exists(self.piper_model):
                raise FileNotFoundError(f"Piper model not found: {self.piper_model}")
                
            if not os.path.exists(self.piper_config):
                raise FileNotFoundError(f"Piper config not found: {self.piper_config}")
                
            logger.info("Piper TTS engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            raise
            
    async def is_ready(self) -> bool:
        """Check if TTS engine is ready"""
        return (
            os.path.exists(self.piper_model) and 
            os.path.exists(self.piper_config) and
            self.opus_codec.encoder is not None
        )
        
    async def preload(self):
        """Preload TTS model"""
        # Test synthesis with short text
        test_text = "Hello"
        async for _ in self.generate_speech_stream(test_text):
            break  # Just test the first chunk
            
        logger.info("TTS model preloaded")
        
    async def generate_speech(self, text: str) -> TTSResult:
        """Generate complete speech for text"""
        try:
            import time
            start_time = time.time()
            
            if not text.strip():
                return TTSResult(
                    audio_data=b"",
                    duration=0.0,
                    sample_rate=self.sample_rate,
                    text=text
                )
                
            logger.debug(f"Generating TTS for: '{text[:50]}...'")
            
            # Generate audio using Piper
            audio_data = await self._synthesize_with_piper(text)
            
            # Convert to Opus
            opus_data = await self._convert_to_opus(audio_data)
            
            # Calculate duration
            duration = len(audio_data) / (self.sample_rate * 2)  # 16-bit samples
            
            # Update metrics
            processing_time = time.time() - start_time
            self.total_chars_processed += len(text)
            self.total_processing_time += processing_time
            
            logger.debug(f"TTS generated {duration:.2f}s audio in {processing_time:.3f}s")
            
            return TTSResult(
                audio_data=opus_data,
                duration=duration,
                sample_rate=self.sample_rate,
                text=text
            )
            
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            raise
            
    async def generate_speech_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate streaming speech for text"""
        try:
            if not text.strip():
                return
                
            # Split text into sentences for streaming
            sentences = self._split_sentences(text)
            
            for sentence in sentences:
                if sentence.strip():
                    result = await self.generate_speech(sentence)
                    if result.audio_data:
                        # Split audio into chunks for streaming
                        chunk_size = 4096  # 4KB chunks
                        for i in range(0, len(result.audio_data), chunk_size):
                            chunk = result.audio_data[i:i + chunk_size]
                            yield chunk
                            
                        # Small delay between sentences
                        await asyncio.sleep(0.01)
                        
        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
            
    async def _synthesize_with_piper(self, text: str) -> bytes:
        """Synthesize speech using Piper"""
        try:
            # Prepare command
            cmd = [
                "piper",
                "--model", self.piper_model,
                "--config", self.piper_config,
                "--output-raw",
                "--output_file", "-"
            ]
            
            # Add speed control if supported
            if self.speed != 1.0:
                cmd.extend(["--length_scale", str(1.0 / self.speed)])
                
            # Run Piper process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Send text and get audio
            stdout, stderr = await process.communicate(input=text.encode('utf-8'))
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8') if stderr else "Unknown Piper error"
                raise RuntimeError(f"Piper synthesis failed: {error_msg}")
                
            return stdout
            
        except Exception as e:
            logger.error(f"Piper synthesis error: {e}")
            raise
            
    async def _convert_to_opus(self, pcm_data: bytes) -> bytes:
        """Convert PCM audio to Opus format"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            
            # Resample if needed (Piper outputs at 22050Hz by default)
            if len(audio_array) > 0:
                # Simple resampling by taking every nth sample
                # For production, use proper resampling
                target_length = len(audio_array) * self.sample_rate // 22050
                indices = np.linspace(0, len(audio_array) - 1, target_length).astype(int)
                resampled = audio_array[indices]
            else:
                resampled = audio_array
                
            # Encode with Opus
            frame_size = self.opus_codec.frame_size
            opus_frames = []
            
            # Process in frames
            for i in range(0, len(resampled), frame_size):
                frame = resampled[i:i + frame_size]
                
                # Pad frame if needed
                if len(frame) < frame_size:
                    frame = np.pad(frame, (0, frame_size - len(frame)), 'constant')
                    
                opus_frame = self.opus_codec.encode_frame(frame)
                opus_frames.append(opus_frame)
                
            # Combine all frames
            return b''.join(opus_frames)
            
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            raise
            
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences for streaming"""
        # Simple sentence splitting
        import re
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter
        cleaned = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Add punctuation back
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                cleaned.append(sentence)
                
        return cleaned if cleaned else [text]
        
    async def stop(self):
        """Stop current TTS generation"""
        try:
            self.is_speaking = False
            
            if self.current_process and self.current_process.poll() is None:
                self.current_process.terminate()
                try:
                    await asyncio.wait_for(self.current_process.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    self.current_process.kill()
                    
            logger.debug("TTS stopped")
            
        except Exception as e:
            logger.error(f"TTS stop error: {e}")
            
    def get_performance_metrics(self) -> dict:
        """Get TTS performance metrics"""
        if self.total_chars_processed > 0:
            chars_per_second = self.total_chars_processed / self.total_processing_time
        else:
            chars_per_second = 0.0
            
        return {
            "chars_per_second": chars_per_second,
            "total_chars_processed": self.total_chars_processed,
            "total_processing_time": self.total_processing_time,
            "average_latency": self.total_processing_time / max(1, self.total_chars_processed) * 100
        }
        
    async def cleanup(self):
        """Clean up TTS resources"""
        try:
            await self.stop()
            
            if self.opus_codec:
                # Cleanup Opus codec if needed
                pass
                
            logger.info("TTS engine cleaned up")
            
        except Exception as e:
            logger.error(f"TTS cleanup error: {e}")


class StreamingTTS:
    """Enhanced streaming TTS with buffering and optimization"""
    
    def __init__(self, base_tts: TTSEngine):
        self.base_tts = base_tts
        self.audio_buffer = asyncio.Queue()
        self.is_buffering = False
        
    async def generate_buffered_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate buffered audio stream"""
        try:
            # Start buffering in background
            buffer_task = asyncio.create_task(self._buffer_audio(text))
            
            # Yield buffered audio
            while True:
                try:
                    # Get audio chunk with timeout
                    chunk = await asyncio.wait_for(
                        self.audio_buffer.get(), 
                        timeout=1.0
                    )
                    
                    if chunk is None:  # End marker
                        break
                        
                    yield chunk
                    
                except asyncio.TimeoutError:
                    # Check if buffering is still active
                    if buffer_task.done():
                        break
                        
            # Wait for buffering to complete
            await buffer_task
            
        except Exception as e:
            logger.error(f"Buffered streaming error: {e}")
            
    async def _buffer_audio(self, text: str):
        """Buffer audio in background"""
        try:
            self.is_buffering = True
            
            async for chunk in self.base_tts.generate_speech_stream(text):
                await self.audio_buffer.put(chunk)
                
            # Send end marker
            await self.audio_buffer.put(None)
            
        except Exception as e:
            logger.error(f"Audio buffering error: {e}")
        finally:
            self.is_buffering = False 