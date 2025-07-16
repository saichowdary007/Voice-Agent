"""
Text-to-Speech (TTS) module with Deepgram integration.
Uses Deepgram's Aura models for high-quality, low-latency speech synthesis.
Falls back to Gemini TTS if Deepgram is not available.
"""

import asyncio
import logging
from typing import AsyncGenerator

# Try to import Deepgram TTS first
try:
    from src.tts_deepgram import TTS as DeepgramTTS
    _DEEPGRAM_AVAILABLE = True
except ImportError:
    _DEEPGRAM_AVAILABLE = False

# Fallback imports for Gemini TTS
if not _DEEPGRAM_AVAILABLE:
    import base64
    import json
    import re
    import tempfile
    import os
    
    # Google Gen AI SDK for Live API
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
        _GENAI_AVAILABLE = True
    except ImportError:
        genai = None  # type: ignore
        types = None  # type: ignore
        _GENAI_AVAILABLE = False

    from src.config import (
        GEMINI_API_KEY,
        GEMINI_TTS_MODEL,
        GEMINI_TTS_VOICE,
        GEMINI_TTS_SPEAKING_RATE,
    )

logger = logging.getLogger(__name__)


# Use Deepgram TTS if available, otherwise fallback to Gemini
if _DEEPGRAM_AVAILABLE:
    logger.info("🚀 Using Deepgram TTS for high-quality speech synthesis")
    TTS = DeepgramTTS
else:
    logger.info("🔄 Deepgram not available, using Gemini TTS fallback")
    
    class TTS:
        """Gemini TTS using Google Gen AI SDK with Live API for native audio generation."""

        def __init__(
            self,
            *,
            api_key: str | None = None,
            voice: str | None = None,
            model: str | None = None,
            speaking_rate: float | None = None,
        ):
            if not _GENAI_AVAILABLE:
                raise ImportError(
                    "google-genai package is required for TTS. "
                    "Install it with: pip install 'google-genai>=0.3.0'"
                )

            self.api_key = api_key or GEMINI_API_KEY
            self.voice = voice or GEMINI_TTS_VOICE
            self.model = model or "gemini-2.0-flash-live-001"  # Use stable live model
            self.speaking_rate = speaking_rate or GEMINI_TTS_SPEAKING_RATE

            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is required for TTS")

            # Initialize the client
            self._client = genai.Client(api_key=self.api_key)  # type: ignore

        async def _synthesize_with_live_api(self, text: str) -> bytes:
            """Synthesize speech using Gemini Live API for native audio generation."""
            if not text.strip():
                return b""

            try:
                # Configure Live API for audio generation
                config = types.LiveConnectConfig(  # type: ignore
                    response_modalities=[types.Modality.AUDIO],  # type: ignore
                    temperature=0.1,  # Low temperature for consistent speech
                )

                # Collect audio chunks
                audio_chunks = []

                async with self._client.aio.live.connect(  # type: ignore
                    model=self.model, config=config
                ) as session:
                    # Send the text to synthesize
                    await session.send_client_content(  # type: ignore
                        turns=types.Content(  # type: ignore
                            role="user", parts=[types.Part(text=text)]  # type: ignore
                        )
                    )

                    # Collect audio response
                    async for message in session.receive():  # type: ignore
                        if message.data:  # Audio data
                            audio_chunks.append(message.data)

                # Concatenate all audio chunks
                return b"".join(audio_chunks)

            except Exception as e:
                logger.error(f"Gemini Live API synthesis failed: {e}")
                raise

        async def _synthesize_chunk_live(self, text: str) -> AsyncGenerator[bytes, None]:
            """Stream audio chunks from Live API."""
            try:
                audio_data = await self._synthesize_with_live_api(text)
                if audio_data:
                    # Yield the complete audio in one chunk for simplicity
                    yield audio_data
            except Exception as e:
                logger.error(f"Live API chunk synthesis failed: {e}")
                raise

        async def synthesize(self, text: str) -> bytes:
            """Return raw audio bytes for the given text using Gemini Live API."""
            if not text:
                return b""

            try:
                return await self._synthesize_with_live_api(text)
            except Exception as e:
                logger.error(f"TTS synthesis failed: {e}")
                return b""

        async def stream_synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
            """Stream synthesized audio chunks for the given text."""
            if not text:
                return

            async for chunk in self._synthesize_chunk_live(text):  # type: ignore[misc]
                yield chunk

        async def speak(self, text: str) -> None:
            """Synthesize and play audio for the given text."""
            if not text:
                return

            # Get raw audio data from Gemini Live API
            audio_bytes = await self.synthesize(text)
            if not audio_bytes:
                logger.warning("No audio data generated")
                return

            try:
                # Convert raw PCM to WAV format for playback
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    import wave
                    with wave.open(tmp_file.name, 'wb') as wav_file:
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(24000)  # 24kHz sample rate
                        wav_file.writeframes(audio_bytes)
                    tmp_path = tmp_file.name

                # Play the WAV file
                import subprocess
                import platform
                
                system = platform.system()
                if system == "Darwin":  # macOS
                    subprocess.run(["afplay", tmp_path], check=True)
                elif system == "Linux":
                    subprocess.run(["aplay", tmp_path], check=True)
                elif system == "Windows":
                    subprocess.run(["start", "", tmp_path], shell=True, check=True)
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


# Create a default TTS instance for easy importing
tts = TTS()


async def main():
    """Test the TTS functionality."""
    if not _GENAI_AVAILABLE:
        print("Error: google-genai package not available")
        return

    try:
        test_tts = TTS()
        print("Testing Gemini Live API TTS...")
        
        test_text = "Hello! This is a test of Gemini's native audio generation."
        print(f"Synthesizing: {test_text}")
        
        audio_data = await test_tts.synthesize(test_text)
        
        if audio_data:
            print(f"✓ Successfully generated {len(audio_data)} bytes of audio")
            print("Playing audio...")
            await test_tts.speak(test_text)
        else:
            print("✗ No audio data generated")
            
    except Exception as e:
        print(f"✗ TTS test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
