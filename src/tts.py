"""
Text-to-Speech (TTS) module backed by the Kokoro FastAPI service.

The implementation keeps the same public interface that the rest of the codebase
relies on (`stream_synthesize`, `speak`, `synthesize`, etc.) but swaps out the
underlying engine from Microsoft Edge-TTS to the Kokoro engine that is exposed
over an OpenAI-compatible HTTP API (see https://github.com/remsky/Kokoro-FastAPI).

Latency-optimised streaming is preserved by issuing the request with
`stream=true` and yielding the base64-encoded audio chunks as they arrive.
"""

import asyncio
import base64
import json
import logging
import re
from typing import AsyncGenerator, AsyncIterator

import aiohttp

from src.config import (
    KOKORO_TTS_URL,
    KOKORO_TTS_VOICE,
    KOKORO_TTS_MODEL,
    KOKORO_TTS_SPEED,
)

logger = logging.getLogger(__name__)

class TTS:
    """Kokoro-backed Text-to-Speech helper.

    The rest of the application instantiates this class with no arguments, so
    we take our configuration from environment variables that are surfaced in
    `src.config`. All parameters are overridable at runtime without code
    changes.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        voice: str | None = None,
        model: str | None = None,
        speed: float | None = None,
    ):
        self.base_url = base_url or KOKORO_TTS_URL.rstrip("/")
        self.voice = voice or KOKORO_TTS_VOICE
        self.model = model or KOKORO_TTS_MODEL
        self.speed = speed or KOKORO_TTS_SPEED

        # Buffer used by stream_synthesize to accumulate partial sentences.
        self._sentence_buffer = ""
        
    def _split_into_speech_chunks(self, text: str) -> list:
        """Split text into speakable chunks (sentences or clauses)."""
        # Split on sentence endings, keeping the punctuation
        chunks = re.split(r'([.!?]+)', text)
        
        # Recombine with punctuation and filter out empty chunks
        speech_chunks = []
        for i in range(0, len(chunks) - 1, 2):
            chunk = chunks[i].strip()
            punct = chunks[i + 1] if i + 1 < len(chunks) else ""
            if chunk:
                speech_chunks.append(chunk + punct)
        
        # If no sentence endings found, split on long clauses
        if not speech_chunks and text.strip():
            # Split on commas for long phrases
            clause_chunks = [chunk.strip() for chunk in text.split(',') if chunk.strip()]
            if len(clause_chunks) > 1:
                speech_chunks = [chunk + ',' for chunk in clause_chunks[:-1]] + [clause_chunks[-1]]
            else:
                speech_chunks = [text.strip()]
                
        return speech_chunks

    async def stream_synthesize(self, text_stream: AsyncIterator[str]) -> AsyncGenerator[bytes, None]:
        """
        Stream TTS synthesis from a stream of text chunks.
        Starts generating audio as soon as complete sentences are available.
        
        Args:
            text_stream: Async iterator yielding text chunks from LLM
            
        Yields:
            Audio bytes as they become available
        """
        try:
            async for text_chunk in text_stream:
                if not text_chunk:
                    continue
                    
                # Add to buffer
                self._sentence_buffer += text_chunk
                
                # Extract complete sentences from buffer
                speech_chunks = self._split_into_speech_chunks(self._sentence_buffer)
                
                # If we have complete sentences, synthesize them
                if len(speech_chunks) > 1:
                    # Keep the last incomplete chunk in buffer
                    complete_chunks = speech_chunks[:-1]
                    self._sentence_buffer = speech_chunks[-1]
                    
                    # Synthesize complete chunks
                    for chunk in complete_chunks:
                        if chunk.strip():
                            async for audio_bytes in self._synthesize_chunk(chunk):
                                yield audio_bytes
                                
                # If buffer is getting long, force synthesis
                elif len(self._sentence_buffer) > 100:
                    chunk_to_synthesize = self._sentence_buffer
                    self._sentence_buffer = ""
                    
                    if chunk_to_synthesize.strip():
                        async for audio_bytes in self._synthesize_chunk(chunk_to_synthesize):
                            yield audio_bytes
            
            # Synthesize any remaining text in buffer
            if self._sentence_buffer.strip():
                async for audio_bytes in self._synthesize_chunk(self._sentence_buffer):
                    yield audio_bytes
                self._sentence_buffer = ""
                
        except Exception as e:
            logger.error(f"Stream synthesis error: {e}")
            # Fallback: synthesize any buffered text
            if self._sentence_buffer.strip():
                async for audio_bytes in self._synthesize_chunk(self._sentence_buffer):
                    yield audio_bytes

    async def _synthesize_chunk(self, text: str) -> AsyncGenerator[bytes, None]:
        """Synthesize a single text chunk and yield audio bytes."""
        if not text.strip():
            return
            
        url = f"{self.base_url}/v1/audio/speech"
        payload = {
            "input": text.strip(),
            "model": self.model,
            "voice": self.voice,
            "speed": self.speed,
            "response_format": "mp3",
            "stream": True,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        logger.error(
                            "Kokoro TTS error (%s): %s", resp.status, err_text[:200]
                        )
                        return

                    # The Kokoro API streams JSON lines with base64 encoded audio.
                    buffer = b""
                    async for chunk_bytes in resp.content.iter_chunked(4096):
                        if not chunk_bytes:
                            continue
                        buffer += chunk_bytes
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data_json = json.loads(line.decode("utf-8"))
                                audio_b64 = data_json.get("audio")
                                if audio_b64:
                                    yield base64.b64decode(audio_b64)
                            except json.JSONDecodeError:
                                # Partial line or keep-alive heartbeat; continue accumulating
                                continue
        except Exception as e:
            logger.error("Chunk synthesis error for '%s...': %s", text[:50], e)

    async def speak(self, text: str):
        """
        Synthesizes text and streams it directly to ffplay's stdin for immediate playback.

        Args:
            text: The text to be spoken.
        """
        if not text:
            return

        try:
            audio_bytes = await self.synthesize(text)
            if not audio_bytes:
                logger.warning("No audio returned for speak() call")
                return

            process = await asyncio.create_subprocess_exec(
                "ffplay",
                "-i",
                "-",
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

            if process.stdin:
                process.stdin.write(audio_bytes)
                await process.stdin.drain()
                process.stdin.close()

            await process.wait()

        except FileNotFoundError:
            print("❌ Error: `ffplay` not found. Please install ffmpeg.")
            print("   On macOS, run: brew install ffmpeg")
            print("   On Debian/Ubuntu, run: sudo apt-get install ffmpeg")
        except Exception as e:
            logger.error("Error in TTS speak(): %s", e)

    async def synthesize(self, text: str) -> bytes:
        """Return raw audio bytes for the given *text*.

        This is useful when running inside the FastAPI backend so we can ship
        the audio to the front-end over WebSocket instead of playing it on the
        server.
        """
        if not text:
            return b""

        url = f"{self.base_url}/v1/audio/speech"
        payload = {
            "input": text,
            "model": self.model,
            "voice": self.voice,
            "speed": self.speed,
            "response_format": "mp3",
            "stream": False,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        logger.error("Kokoro TTS synthesis error (%s)", resp.status)
                        return b""
                    data_json = await resp.json()
                    audio_b64 = data_json.get("audio")
                    if not audio_b64:
                        logger.warning("No 'audio' key in Kokoro response")
                        return b""
                    return base64.b64decode(audio_b64)
        except Exception as e:
            logger.error("Error during TTS synthesis: %s", e)
            return b""

    async def synthesize_stream_from_text(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text and yield audio chunks immediately.
        Useful for converting regular text to streaming audio.
        """
        if not text:
            return
            
        # Split text into chunks and synthesize each
        chunks = self._split_into_speech_chunks(text)
        for chunk in chunks:
            if chunk.strip():
                async for audio_bytes in self._synthesize_chunk(chunk):
                    yield audio_bytes

# --- Example Usage ---
async def main():
    """Example of how to use the TTS module."""
    tts = TTS()
    print("--- TTS Module Example ---")
    text_to_speak = "Hello, this is a test of the refactored text-to-speech engine using Microsoft Edge."
    print(f"Speaking: '{text_to_speak}'")
    await tts.speak(text_to_speak)
    print("--- TTS Example Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
