"""
Deepgram Text-to-Speech (TTS) module using Deepgram's Aura models.
Provides high-quality, low-latency speech synthesis with streaming capabilities.
"""

import asyncio
import logging
import tempfile
import os
import subprocess
import platform
from typing import AsyncGenerator, Optional

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    SpeakOptions,
)

from src.config import (
    DEEPGRAM_API_KEY,
    DEEPGRAM_TTS_MODEL,
    DEEPGRAM_TTS_ENCODING,
    DEEPGRAM_TTS_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


class DeepgramTTS:
    """Deepgram TTS client for high-quality speech synthesis."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or DEEPGRAM_API_KEY
        self.model = model or DEEPGRAM_TTS_MODEL
        
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY is required")

        # Initialize Deepgram client
        config = DeepgramClientOptions(options={"keepalive": "true"})
        self.client = DeepgramClient(self.api_key, config)
        
        logger.info(f"✅ Deepgram TTS initialized with model: {self.model}")

    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize speech from text using Deepgram's TTS API.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Raw audio bytes
        """
        if not text.strip():
            return b""

        try:
            logger.info(f"🎤 Synthesizing text: '{text[:50]}...' with model: {self.model}")
            
            # Configure synthesis options
            options = SpeakOptions(
                model=self.model,
                encoding=DEEPGRAM_TTS_ENCODING,
                sample_rate=DEEPGRAM_TTS_SAMPLE_RATE,
            )

            # Use the save method and read file (most reliable approach)
            def _synthesize():
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    filename = tmp_file.name
                
                try:
                    # Save to file using the working API
                    logger.debug(f"Saving TTS audio to: {filename}")
                    response = self.client.speak.v("1").save(filename, {"text": text}, options)
                    logger.debug(f"Deepgram TTS response: {response}")
                    
                    # Read file content
                    with open(filename, 'rb') as f:
                        audio_data = f.read()
                    
                    logger.info(f"✅ Generated {len(audio_data)} bytes of audio")
                    return audio_data
                    
                except Exception as e:
                    logger.error(f"Deepgram TTS API error: {e}")
                    raise
                finally:
                    # Clean up
                    try:
                        os.unlink(filename)
                    except Exception:
                        pass
            
            # Run in thread to avoid blocking
            return await asyncio.to_thread(_synthesize)

        except Exception as e:
            logger.error(f"Deepgram TTS synthesis failed: {e}")
            return b""

    async def stream_synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesized audio chunks for the given text.
        
        Args:
            text: Text to synthesize
            
        Yields:
            Audio chunks as bytes
        """
        if not text.strip():
            return

        try:
            # For now, we'll synthesize the full text and yield it in chunks
            # Deepgram's streaming TTS might be available in future SDK versions
            audio_bytes = await self.synthesize(text)
            
            if audio_bytes:
                # Yield audio in chunks for streaming playback
                chunk_size = 4096  # 4KB chunks
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    if chunk:
                        yield chunk
                        # Small delay to simulate streaming
                        await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Deepgram TTS streaming failed: {e}")

    async def speak(self, text: str) -> None:
        """
        Synthesize and play audio for the given text.
        
        Args:
            text: Text to speak
        """
        if not text.strip():
            return

        try:
            # Get audio bytes
            audio_bytes = await self.synthesize(text)
            if not audio_bytes:
                logger.warning("No audio data generated")
                return

            # Save to temporary file and play
            await self._play_audio_bytes(audio_bytes)

        except Exception as e:
            logger.error(f"Failed to speak text: {e}")

    async def _play_audio_bytes(self, audio_bytes: bytes) -> None:
        """
        Play audio bytes using system audio player.
        
        Args:
            audio_bytes: Raw audio bytes to play
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name

            # Play the audio file based on platform
            system = platform.system()
            
            if system == "Darwin":  # macOS
                await asyncio.to_thread(
                    subprocess.run, ["afplay", tmp_path], check=True
                )
            elif system == "Linux":
                # Try different players
                players = ["aplay", "paplay", "play"]
                for player in players:
                    try:
                        await asyncio.to_thread(
                            subprocess.run, [player, tmp_path], 
                            check=True, capture_output=True
                        )
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                else:
                    logger.warning("No suitable audio player found on Linux")
            elif system == "Windows":
                await asyncio.to_thread(
                    subprocess.run, ["start", "", tmp_path], 
                    shell=True, check=True
                )
            else:
                logger.warning(f"Audio playback not supported on {system}")

        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    async def get_available_voices(self) -> list:
        """
        Get list of available Deepgram TTS voices.
        
        Returns:
            List of available voice models
        """
        # Deepgram Aura voice models as of current API
        return [
            "aura-asteria-en",    # Female, American English
            "aura-luna-en",       # Female, American English  
            "aura-stella-en",     # Female, American English
            "aura-athena-en",     # Female, British English
            "aura-hera-en",       # Female, American English
            "aura-orion-en",      # Male, American English
            "aura-arcas-en",      # Male, American English
            "aura-perseus-en",    # Male, American English
            "aura-angus-en",      # Male, Irish English
            "aura-orpheus-en",    # Male, American English
            "aura-helios-en",     # Male, British English
            "aura-zeus-en",       # Male, American English
        ]


# Compatibility wrapper to match existing TTS interface
class TTS(DeepgramTTS):
    """Compatibility wrapper for existing TTS interface."""
    
    def __init__(self, *, api_key: str = None, voice: str = None, model: str = None, **kwargs):
        # Map voice parameter to Deepgram model if provided
        if voice and not model:
            model = voice
        elif not model:
            model = DEEPGRAM_TTS_MODEL
            
        super().__init__(api_key=api_key, model=model)

    async def synthesize_streaming(self, text: str) -> AsyncGenerator[bytes, None]:
        """Alternative method name for streaming synthesis."""
        async for chunk in self.stream_synthesize(text):
            yield chunk


# Create default TTS instance
tts = TTS()


async def main():
    """Test the Deepgram TTS functionality."""
    try:
        test_tts = TTS()
        print("Testing Deepgram TTS...")
        
        # List available voices
        voices = await test_tts.get_available_voices()
        print(f"Available voices: {voices}")
        
        # Test synthesis
        test_text = "Hello! This is a test of Deepgram's Aura text-to-speech."
        print(f"Synthesizing: {test_text}")
        
        audio_data = await test_tts.synthesize(test_text)
        
        if audio_data:
            print(f"✓ Successfully generated {len(audio_data)} bytes of audio")
            print("Playing audio...")
            await test_tts.speak(test_text)
        else:
            print("✗ No audio data generated")
            
        # Test streaming
        print("\nTesting streaming synthesis...")
        chunk_count = 0
        async for chunk in test_tts.stream_synthesize("This is a streaming test."):
            chunk_count += 1
            
        print(f"✓ Received {chunk_count} audio chunks")
            
    except Exception as e:
        print(f"✗ Deepgram TTS test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())