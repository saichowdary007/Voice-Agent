import os
import asyncio
import logging
import tempfile
import subprocess
import json
from typing import Optional, Dict, Any, AsyncGenerator
import numpy as np
import soundfile as sf
import io
import wave

logger = logging.getLogger(__name__)

class TTSService:
    """Text-to-Speech service using Piper TTS with automatic voice downloads"""
    
    def __init__(self):
        self.is_available = False
        self.sample_rate = 22050  # Piper model's native sample rate
        self.voice_name = "en_US-lessac-medium"  # Default voice
        
        self.is_speaking = False
        self.should_stop = False
        
    async def initialize(self):
        """Initialize TTS engine with selected voice"""
        logger.info(f"Initializing Piper TTS with {self.voice_name} voice...")
        
        try:
            # Check if piper command exists
            result = await asyncio.create_subprocess_exec(
                'piper', '--help',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.communicate()
            
            if result.returncode != 0:
                logger.warning("Piper command not found, using mock TTS service")
                self.use_mock = True
                self.is_available = True
                return
                
            # Continue with normal initialization...
            self.is_available = True
            logger.info(f"✅ Piper TTS initialized successfully with voice: {self.voice_name}")
            
        except FileNotFoundError:
            logger.warning("Piper command not found, using mock TTS service")
            self.use_mock = True
            self.is_available = True
        except Exception as e:
            logger.error(f"Failed to initialize Piper TTS: {e}")
            self.use_mock = True
            self.is_available = True

    async def generate_speech_stream(self, text: str, speed: float = 1.0) -> AsyncGenerator[bytes, None]:
        """
        Generate TTS audio for the given text in streaming chunks
        """
        if not self.is_available:
            logger.warning("TTS service not available, cannot generate speech.")
            return
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS, skipping generation.")
            return
            
        self.is_speaking = True
        self.should_stop = False

        logger.info(f"TTS: Generating streaming audio for text: '{text[:70]}...' (Speed: {speed}x)")
        
        try:
            # Use mock implementation if piper is not available
            if self.use_mock:
                logger.info("Using mock TTS implementation")
                # Generate silence as audio data (simple 1-second 16kHz silence)
                sample_rate = 16000
                duration_sec = 1.0
                silence = np.zeros(int(sample_rate * duration_sec), dtype=np.int16)
                
                # Convert to WAV format
                with io.BytesIO() as wav_buffer:
                    with wave.open(wav_buffer, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(silence.tobytes())
                        
                    wav_data = wav_buffer.getvalue()
                    
                # Split into chunks to simulate streaming
                chunk_size = 4000  # bytes
                for i in range(0, len(wav_data), chunk_size):
                    if self.should_stop:
                        logger.info("TTS synthesis was interrupted.")
                        break
                    yield wav_data[i:i+chunk_size]
                    await asyncio.sleep(0.1)  # Simulate delay between chunks
                
                self.is_speaking = False
                return
                
            # Break text into smaller chunks for streaming
            import re
            
            sentences = re.split(r'[.!?]+', text)
            chunks = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(sentence) > 100:
                    sub_chunks = re.split(r'[,;:]+', sentence)
                    for sub_chunk in sub_chunks:
                        sub_chunk = sub_chunk.strip()
                        if sub_chunk and len(sub_chunk) > 5:
                            chunks.append(sub_chunk)
                else:
                    if len(sentence) > 5:
                        chunks.append(sentence)
            
            if not chunks:
                chunks = [text.strip()]
            
            logger.info(f"TTS: Split text into {len(chunks)} chunks for streaming")
            
            # Generate audio for each chunk
            for i, chunk in enumerate(chunks):
                if self.should_stop:
                    logger.info("TTS synthesis was interrupted.")
                    break
                
                logger.debug(f"TTS: Processing chunk {i+1}/{len(chunks)}: '{chunk[:50]}...'")
                
                try:
                    # Create temporary file for audio output
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # Run piper command for this chunk
                    process = await asyncio.create_subprocess_exec(
                        'piper',
                        '--model', self.voice_name,
                        '--output_file', temp_path,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    # Send text to piper
                    stdout, stderr = await process.communicate(input=chunk.encode())
                    
                    if process.returncode == 0:
                        # Read the generated audio file
                        if os.path.exists(temp_path):
                            with open(temp_path, 'rb') as f:
                                audio_data = f.read()
                            
                            logger.debug(f"TTS: Generated {len(audio_data)} bytes of audio for chunk {i+1}")
                            yield audio_data
                            
                            # Clean up temp file
                            os.unlink(temp_path)
                        else:
                            logger.error(f"TTS: Output file not created for chunk {i+1}")
                    else:
                        logger.error(f"TTS: Piper failed for chunk {i+1}: {stderr.decode()}")
                
                except Exception as e:
                    logger.error(f"TTS: Error processing chunk {i+1}: {e}")
                
                if self.should_stop:
                    break
                
                # Small delay between chunks
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error during TTS streaming speech generation: {e}", exc_info=True)
        finally:
            self.is_speaking = False

    async def stop(self):
        """Signal to stop any ongoing TTS generation."""
        logger.debug("TTS stop signaled.")
        self.should_stop = True
        self.is_speaking = False

    async def synthesize_speech(self, text: str, speed: float = 1.0) -> Dict[str, Any]:
        """
        Convert text to speech (complete, non-streaming) using Piper TTS.
        Returns WAV audio data.
        """
        if not self.is_available:
            logger.warning("TTS service not available for synthesize_speech.")
            return {"audio_data": None, "error": "TTS service not available"}
        
        logger.info(f"TTS (blocking): Synthesizing '{text[:50]}...'")
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Run piper command
            process = await asyncio.create_subprocess_exec(
                'piper',
                '--model', self.voice_name,
                '--output_file', temp_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate(input=text.encode())
            
            if process.returncode == 0 and os.path.exists(temp_path):
                with open(temp_path, 'rb') as f:
                    audio_data = f.read()
                
                os.unlink(temp_path)
                
                return {
                    "audio_data": audio_data,
                    "sample_rate": self.sample_rate,
                    "format": "wav"
                }
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"TTS synthesis failed: {error_msg}")
                return {"audio_data": None, "error": error_msg}
                
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}", exc_info=True)
            return {"audio_data": None, "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get current TTS service status"""
        return {
            "available": self.is_available,
            "service_type": "Piper_TTS",
            "voice_model": self.voice_name,
            "model_native_sample_rate": self.sample_rate,
            "is_currently_speaking": self.is_speaking
        }

    async def cleanup(self):
        """Cleanup TTS resources"""
        logger.info("Cleaning up TTS service")
        self.should_stop = True
        self.is_speaking = False