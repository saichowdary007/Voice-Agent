"""
Refactored Text-to-Speech (TTS) module using edge-tts for a fast and high-quality voice.
Enhanced with streaming capabilities for ultra-low latency voice responses.
"""
import asyncio
import edge_tts
import re
import logging
from typing import AsyncGenerator, AsyncIterator
from src.config import EDGE_TTS_VOICE

logger = logging.getLogger(__name__)

class TTS:
    """
    Handles Text-to-Speech synthesis using Microsoft Edge's TTS engine.
    Enhanced with streaming capabilities for immediate audio playback as text becomes available.
    """
    def __init__(self, voice: str = EDGE_TTS_VOICE):
        self.voice = voice
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
            
        try:
            communicate = edge_tts.Communicate(text.strip(), self.voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        except Exception as e:
            logger.error(f"Chunk synthesis error for '{text[:50]}...': {e}")

    async def speak(self, text: str):
        """
        Synthesizes text and streams it directly to ffplay's stdin for immediate playback.

        Args:
            text: The text to be spoken.
        """
        if not text:
            return

        process = None
        try:
            # Start ffplay process, configured to read from stdin ('-')
            process = await asyncio.create_subprocess_exec(
                'ffplay',
                '-i', '-',          # Input from stdin
                '-nodisp',
                '-autoexit',
                '-loglevel', 'quiet',
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

            # Stream audio from edge-tts directly to the ffplay process
            communicate = edge_tts.Communicate(text, self.voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio" and process.stdin:
                    try:
                        process.stdin.write(chunk["data"])
                        await process.stdin.drain()
                    except (BrokenPipeError, ConnectionResetError):
                        # This can happen if ffplay closes unexpectedly
                        print("⚠️  TTS stream pipe broke. Playback may have been interrupted.")
                        break

            # Close stdin to signal that we're done sending audio
            if process.stdin:
                process.stdin.close()
            
            # Wait for the ffplay process to finish
            await process.wait()
                
        except FileNotFoundError:
            print("❌ Error: `ffplay` not found. Please install ffmpeg.")
            print("   On macOS, run: brew install ffmpeg")
            print("   On Debian/Ubuntu, run: sudo apt-get install ffmpeg")
        except Exception as e:
            print(f"❌ Error in TTS: {e}")
            if process:
                process.kill()
                await process.wait()

    async def synthesize(self, text: str) -> bytes:
        """Return raw audio bytes for the given *text*.

        This is useful when running inside the FastAPI backend so we can ship
        the audio to the front-end over WebSocket instead of playing it on the
        server.
        """
        if not text:
            return b""

        audio_data = bytearray()
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])
        except Exception as e:
            print(f"❌ Error during TTS synthesis: {e}")
            return b""
        return bytes(audio_data)

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
