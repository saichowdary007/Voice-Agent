"""
Refactored Text-to-Speech (TTS) module using edge-tts for a fast and high-quality voice.
This version uses an external `ffplay` process and streams audio for ultra-low latency.
"""
import asyncio
import edge_tts
from src.config import EDGE_TTS_VOICE

class TTS:
    """
    Handles Text-to-Speech synthesis using Microsoft Edge's TTS engine.
    Audio is streamed directly to an external `ffplay` process to ensure stability
    and begin playback almost instantaneously.
    """
    def __init__(self, voice: str = EDGE_TTS_VOICE):
        self.voice = voice

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
