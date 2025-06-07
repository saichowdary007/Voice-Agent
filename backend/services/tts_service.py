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

from backend.app.config import settings

logger = logging.getLogger(__name__)

class TTSService:
    """Text-to-Speech service using Piper TTS with automatic voice downloads"""
    
    def __init__(self):
        self.is_available = False
        self.sample_rate = 22050  # Piper model's native sample rate
        self.voice_name = settings.tts_model
        self.piper_process = None
        
        self.is_speaking = False
        self.should_stop = False
        
    async def initialize(self):
        """Initialize TTS engine and start a long-running Piper process."""
        logger.info(f"Initializing Piper TTS with {self.voice_name} voice...")
        
        try:
            # Check if piper command exists
            piper_path = "piper"  # Assuming it's in the PATH
            
            self.piper_process = await asyncio.create_subprocess_exec(
                piper_path,
                '--model', self.voice_name,
                '--output-raw',  # Output raw PCM data
                '--json-input',  # Expect JSON on stdin
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Check if the process started successfully
            await asyncio.sleep(1) # Give it a moment to start
            if self.piper_process.returncode is not None:
                stderr = await self.piper_process.stderr.read()
                logger.error(f"Failed to start Piper process. Return code: {self.piper_process.returncode}. Stderr: {stderr.decode()}")
                self.piper_process = None
                self.is_available = False
                return

            self.is_available = True
            logger.info(f"✅ Piper TTS process started successfully with voice: {self.voice_name}")

        except FileNotFoundError:
            logger.warning("Piper command not found, TTS service is unavailable.")
            self.is_available = False
        except Exception as e:
            logger.error(f"Failed to initialize Piper TTS: {e}", exc_info=True)
            self.is_available = False

    async def generate_speech_stream(self, text: str, speed: float = 1.0) -> AsyncGenerator[bytes, None]:
        """Generate TTS audio by feeding text to the long-running Piper process."""
        if not self.is_available or not self.piper_process:
            logger.warning("TTS service not available, cannot generate speech.")
            return
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS, skipping generation.")
            return

        self.is_speaking = True
        self.should_stop = False

        logger.info(f"TTS: Generating streaming audio for text: '{text[:70]}...'")
        
        try:
            request = {
                "text": text,
                "speaker_id": 0,  # Default speaker
                "length_scale": 1.0 / speed, # Piper uses length_scale
            }
            
            # Write JSON request to piper's stdin
            self.piper_process.stdin.write(json.dumps(request).encode('utf-8'))
            self.piper_process.stdin.write(b'\n') # Newline to signal end of request
            await self.piper_process.stdin.drain()

            # Read raw PCM from stdout
            while not self.should_stop:
                try:
                    # Read a chunk of audio data
                    audio_chunk = await asyncio.wait_for(self.piper_process.stdout.read(4096), timeout=5.0)
                    if not audio_chunk:
                        # End of stream for this text
                        break
                    yield audio_chunk
                except asyncio.TimeoutError:
                    logger.warning("TTS stdout read timed out. Assuming end of audio for this request.")
                    break
        
        except Exception as e:
            logger.error(f"Error during TTS streaming: {e}", exc_info=True)
            # Attempt to restart the process if it has crashed
            await self.cleanup()
            await self.initialize()
        finally:
            self.is_speaking = False

    async def stop(self):
        """Signal to stop any ongoing TTS generation."""
        logger.debug("TTS stop signaled.")
        self.should_stop = True
        self.is_speaking = False

    async def synthesize_speech(self, text: str, speed: float = 1.0) -> Dict[str, Any]:
        """
        Convert text to speech (complete, non-streaming) using the long-running process.
        Returns WAV audio data.
        """
        if not self.is_available:
            logger.warning("TTS service not available for synthesize_speech.")
            return {"audio_data": None, "error": "TTS service not available"}
        
        logger.info(f"TTS (blocking): Synthesizing '{text[:50]}...'")
        audio_chunks = []
        try:
            async for chunk in self.generate_speech_stream(text, speed):
                audio_chunks.append(chunk)

            if not audio_chunks:
                return {"audio_data": None, "error": "No audio generated"}

            # Combine chunks and create a WAV header
            pcm_data = b"".join(audio_chunks)
            
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(pcm_data)
                wav_data = wav_buffer.getvalue()

            return {
                "audio_data": wav_data,
                "sample_rate": self.sample_rate,
                "format": "wav"
            }
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
        """Clean up TTS resources by terminating the Piper process."""
        logger.info("Cleaning up TTS service and terminating Piper process...")
        if self.piper_process:
            if self.piper_process.returncode is None: # Process is still running
                try:
                    self.piper_process.terminate()
                    await self.piper_process.wait()
                    logger.info("Piper process terminated.")
                except Exception as e:
                    logger.error(f"Error terminating Piper process: {e}")
            self.piper_process = None
        self.is_available = False