"""
Deepgram Speech-to-Text (STT) module for real-time and batch transcription.
Provides both streaming and batch transcription capabilities using Deepgram's Nova-2 model.
"""

import asyncio
import logging
import json
from typing import AsyncGenerator, Optional, Dict, Any
import io
import wave

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    PrerecordedOptions,
    FileSource,
)

from src.config import (
    DEEPGRAM_API_KEY,
    DEEPGRAM_STT_MODEL,
    DEEPGRAM_STT_LANGUAGE,
    DEEPGRAM_STT_SMART_FORMAT,
    DEEPGRAM_STT_PUNCTUATE,
    DEEPGRAM_STT_DIARIZE,
)

logger = logging.getLogger(__name__)


class DeepgramSTT:
    """Deepgram STT client for real-time and batch transcription."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or DEEPGRAM_API_KEY
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY is required")

        # Initialize Deepgram client
        config = DeepgramClientOptions(options={"keepalive": "true"})
        self.client = DeepgramClient(self.api_key, config)
        
        # Streaming connection state
        self._connection = None
        self._is_connected = False
        self._transcript_buffer = []

    async def transcribe_bytes(self, audio_bytes: bytes, sr: int = 16000) -> str:
        """
        Transcribe audio bytes using Deepgram's prerecorded API.
        
        Args:
            audio_bytes: Raw audio bytes (PCM or WAV format)
            sr: Sample rate of the audio
            
        Returns:
            Transcribed text string
        """
        if not audio_bytes:
            logger.warning("Empty audio bytes provided")
            return ""

        try:
            logger.info(f"🎯 Transcribing {len(audio_bytes)} bytes at {sr}Hz")
            
            # Convert raw PCM to WAV if needed
            if not self._is_wav_format(audio_bytes):
                logger.debug("Converting PCM to WAV format")
                audio_bytes = self._pcm_to_wav(audio_bytes, sr)
                logger.debug(f"WAV conversion complete: {len(audio_bytes)} bytes")

            # Prepare the audio source
            payload = {"buffer": audio_bytes}

            # Configure transcription options
            options = PrerecordedOptions(
                model=DEEPGRAM_STT_MODEL,
                language=DEEPGRAM_STT_LANGUAGE,
                smart_format=DEEPGRAM_STT_SMART_FORMAT,
                punctuate=DEEPGRAM_STT_PUNCTUATE,
                diarize=DEEPGRAM_STT_DIARIZE,
            )

            logger.debug(f"Deepgram options: model={DEEPGRAM_STT_MODEL}, language={DEEPGRAM_STT_LANGUAGE}")

            # Transcribe the audio
            response = await asyncio.to_thread(
                self.client.listen.prerecorded.v("1").transcribe_file,
                payload,
                options
            )

            logger.debug(f"Deepgram response received: {response}")

            # Extract transcript from response
            if response.results and response.results.channels:
                alternatives = response.results.channels[0].alternatives
                if alternatives and len(alternatives) > 0:
                    transcript = alternatives[0].transcript.strip()
                    confidence = getattr(alternatives[0], 'confidence', 0.0)
                    logger.info(f"✅ Transcript: '{transcript}' (confidence: {confidence})")
                    return transcript
                else:
                    logger.warning("No alternatives found in Deepgram response")
            else:
                logger.warning("No results or channels found in Deepgram response")

            return ""

        except Exception as e:
            logger.error(f"Deepgram transcription failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ""

    async def start_streaming(self, sample_rate: int = 16000) -> bool:
        """
        Start a streaming transcription connection.
        
        Args:
            sample_rate: Audio sample rate
            
        Returns:
            True if connection started successfully
        """
        try:
            logger.info(f"🔄 Starting Deepgram streaming with model: {DEEPGRAM_STT_MODEL}")
            
            # Configure live transcription options
            options = LiveOptions(
                model=DEEPGRAM_STT_MODEL,
                language=DEEPGRAM_STT_LANGUAGE,
                smart_format=DEEPGRAM_STT_SMART_FORMAT,
                punctuate=DEEPGRAM_STT_PUNCTUATE,
                diarize=DEEPGRAM_STT_DIARIZE,
                sample_rate=sample_rate,
                channels=1,
                encoding="linear16",
                interim_results=True,
                utterance_end_ms="1000",
                vad_events=True,
            )

            # Create live transcription connection
            self._connection = self.client.listen.websocket.v("1")
            
            # Set up event handlers
            self._connection.on(LiveTranscriptionEvents.Open, self._on_open)
            self._connection.on(LiveTranscriptionEvents.Transcript, self._on_message)
            self._connection.on(LiveTranscriptionEvents.Error, self._on_error)
            self._connection.on(LiveTranscriptionEvents.Close, self._on_close)

            # Start the connection synchronously (don't use asyncio.to_thread)
            result = self._connection.start(options)
            if result:
                self._is_connected = True
                logger.info("✅ Deepgram streaming connection started successfully")
                # Give it a moment to establish
                await asyncio.sleep(0.1)
                return True
            else:
                logger.error("❌ Failed to start Deepgram streaming connection")
                return False

        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    async def stream_audio_chunk(self, audio_chunk: bytes) -> None:
        """
        Send audio chunk to streaming connection.
        
        Args:
            audio_chunk: Raw audio bytes to stream
        """
        if not self._is_connected or not self._connection:
            logger.warning("Streaming connection not active")
            return

        try:
            # Send audio chunk synchronously (Deepgram SDK handles threading internally)
            self._connection.send(audio_chunk)
            logger.debug(f"📤 Sent audio chunk: {len(audio_chunk)} bytes")
        except Exception as e:
            logger.error(f"Failed to send audio chunk: {e}")
            # Mark connection as disconnected on error
            self._is_connected = False

    async def get_streaming_transcript(self) -> Optional[str]:
        """
        Get the latest transcript from streaming buffer.
        
        Returns:
            Latest transcript or None if no new transcript available
        """
        if self._transcript_buffer:
            return self._transcript_buffer.pop(0)
        return None

    async def stop_streaming(self) -> None:
        """Stop the streaming transcription connection."""
        if self._connection and self._is_connected:
            try:
                await asyncio.to_thread(self._connection.finish)
                self._is_connected = False
                logger.info("🔌 Deepgram streaming connection closed")
            except Exception as e:
                logger.error(f"Error closing streaming connection: {e}")

    def _on_open(self, *args, **kwargs):
        """Handle connection open event."""
        logger.info("🔗 Deepgram streaming connection opened successfully")

    def _on_message(self, *args, **kwargs):
        """Handle transcript message event."""
        try:
            # Handle different argument patterns from Deepgram SDK
            result = None
            if len(args) >= 2:
                result = args[1]
            elif 'result' in kwargs:
                result = kwargs['result']
            
            if result and hasattr(result, 'channel'):
                alternatives = result.channel.alternatives
                if alternatives and len(alternatives) > 0:
                    transcript = alternatives[0].transcript
                    if transcript.strip():
                        # Add both interim and final transcripts to buffer
                        self._transcript_buffer.append(transcript.strip())
                        if result.is_final:
                            logger.info(f"✅ Final transcript: {transcript}")
                        else:
                            logger.debug(f"📝 Interim transcript: {transcript}")
        except Exception as e:
            logger.error(f"Error processing transcript: {e}")
            logger.error(f"Args: {args}, Kwargs: {kwargs}")

    def _on_error(self, *args, **kwargs):
        """Handle connection error event."""
        error = kwargs.get('error') or (args[1] if len(args) > 1 else 'Unknown error')
        logger.error(f"Deepgram streaming error: {error}")

    def _on_close(self, *args, **kwargs):
        """Handle connection close event."""
        self._is_connected = False
        close_code = kwargs.get('close_code') or (args[1] if len(args) > 1 else 'Unknown')
        logger.warning(f"🔌 Deepgram streaming connection closed (code: {close_code})")

    def _is_wav_format(self, audio_bytes: bytes) -> bool:
        """Check if audio bytes are in WAV format."""
        return audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[:12]

    def _pcm_to_wav(self, pcm_bytes: bytes, sample_rate: int) -> bytes:
        """Convert raw PCM bytes to WAV format."""
        try:
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_bytes)
            
            wav_buffer.seek(0)
            return wav_buffer.read()
        except Exception as e:
            logger.error(f"Failed to convert PCM to WAV: {e}")
            return pcm_bytes


# Compatibility wrapper to match existing STT interface
class STT(DeepgramSTT):
    """Compatibility wrapper for existing STT interface."""
    
    def __init__(self, model_size="nova-2", device="auto"):
        # Map model_size to Deepgram models
        model_map = {
            "tiny": "nova-2",
            "base": "nova-2", 
            "small": "nova-2",
            "medium": "nova-2",
            "large": "nova-2",
            "nova-2": "nova-2",
            "nova": "nova",
            "enhanced": "enhanced",
        }
        
        # Override config with mapped model
        import src.config as config
        config.DEEPGRAM_STT_MODEL = model_map.get(model_size, "nova-2")
        
        super().__init__()
        logger.info(f"✅ Deepgram STT initialized with model: {config.DEEPGRAM_STT_MODEL}")

    async def _reset_state(self):
        """Reset STT state to prevent 'stuck in silence' issues."""
        # Close and restart Deepgram connection to reset internal endpointing
        if self._connection and self._is_connected:
            try:
                logger.debug("🔄 Resetting Deepgram connection to prevent silence issues")
                await self.stop_streaming()
                # Small delay to ensure clean shutdown
                await asyncio.sleep(0.1)
                # Connection will be restarted on next audio chunk if needed
            except Exception as e:
                logger.warning(f"Error during Deepgram reset: {e}")
        logger.debug("🔄 STT state reset (Deepgram)")
        pass

    async def stream_transcribe_chunk(self, audio_chunk: bytes, is_final: bool = False) -> Optional[str]:
        """
        Stream transcription compatibility method.
        
        Args:
            audio_chunk: Audio bytes to transcribe
            is_final: Whether this is the final chunk
            
        Returns:
            Transcript text or None
        """
        # For now, use batch transcription for reliability
        # TODO: Fix streaming implementation later
        if is_final and audio_chunk and len(audio_chunk) > 320:
            logger.info(f"🎯 Processing final audio chunk: {len(audio_chunk)} bytes")
            
            # Check if audio contains actual speech (basic volume check)
            import numpy as np
            try:
                # Convert bytes to numpy array for analysis
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                rms_level = np.sqrt(np.mean(audio_np**2))
                max_level = np.max(np.abs(audio_np))
                
                logger.info(f"🔊 Audio analysis - RMS: {rms_level:.4f}, Max: {max_level:.4f}")
                
                # Skip transcription if audio is too quiet (likely silence)
                # Very relaxed thresholds to allow quiet speech
                if rms_level < 0.001 and max_level < 0.01:
                    logger.warning("⚠️ Audio appears to be silence, skipping transcription")
                    return None
                    
            except Exception as analysis_e:
                logger.warning(f"Audio analysis failed: {analysis_e}")
            
            try:
                transcript = await self.transcribe_bytes(audio_chunk)
                if transcript and transcript.strip():
                    logger.info(f"✅ Deepgram transcript: '{transcript}'")
                    return transcript.strip()
                else:
                    logger.warning("⚠️ No transcript returned from Deepgram - trying fallback")
                    
                    # Fallback to local Whisper if available
                    try:
                        # Import the fallback Whisper STT class directly
                        import speech_recognition as sr
                        from faster_whisper import WhisperModel
                        import numpy as np
                        import asyncio
                        
                        logger.info("🔄 Trying Whisper fallback...")
                        
                        # Initialize Whisper model
                        model = WhisperModel("tiny", device="cpu", compute_type="int8")
                        
                        # Convert bytes to numpy array (assumes 16-bit PCM)
                        audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        # Run transcription in thread
                        def _transcribe():
                            segments, _ = model.transcribe(
                                audio_np,
                                language=None,
                                beam_size=1,
                                vad_filter=False,  # Disable Whisper VAD - rely on WebRTC/RNNoise instead
                                condition_on_previous_text=False
                            )
                            return " ".join(s.text for s in segments).strip()
                        
                        fallback_transcript = await asyncio.to_thread(_transcribe)
                        
                        if fallback_transcript and fallback_transcript.strip():
                            logger.info(f"✅ Whisper fallback transcript: '{fallback_transcript}'")
                            return fallback_transcript.strip()
                        else:
                            logger.warning("⚠️ Whisper fallback also returned empty transcript")
                            
                    except Exception as fallback_e:
                        logger.warning(f"Whisper fallback failed: {fallback_e}")
                    
            except Exception as e:
                logger.error(f"❌ Deepgram transcription failed: {e}")
                return None
        
        # For non-final chunks, just return None (no partial transcripts for now)
        return None

    def listen_and_transcribe(self, timeout: int = None, phrase_time_limit: int = None) -> str:
        """
        CLI compatibility method for microphone input using Deepgram.
        
        Args:
            timeout: Maximum time to wait for speech to start (seconds)
            phrase_time_limit: Maximum time for the phrase (seconds)
            
        Returns:
            Transcribed text string
        """
        try:
            import speech_recognition as sr
            import asyncio
            
            # Initialize microphone and recognizer
            recognizer = sr.Recognizer()
            microphone = sr.Microphone()
            
            logger.info("🎤 Listening for speech...")
            
            # Adjust for ambient noise
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            # Listen for audio
            with microphone as source:
                audio = recognizer.listen(
                    source, 
                    timeout=timeout or 5, 
                    phrase_time_limit=phrase_time_limit or 10
                )
            
            # Convert audio to bytes
            audio_bytes = audio.get_wav_data()
            
            logger.info(f"🎯 Audio captured: {len(audio_bytes)} bytes")
            
            # Use async transcription in sync context
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Transcribe using Deepgram
            transcript = loop.run_until_complete(self.transcribe_bytes(audio_bytes))
            
            if transcript and transcript.strip():
                logger.info(f"✅ Transcribed: '{transcript}'")
                return transcript.strip()
            else:
                logger.warning("⚠️ No speech detected or transcription failed")
                return ""
                
        except sr.WaitTimeoutError:
            logger.warning("⏰ Listening timeout - no speech detected")
            return ""
        except Exception as e:
            logger.error(f"❌ listen_and_transcribe failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ""


# Create default instance
stt = STT()


async def main():
    """Test the Deepgram STT functionality."""
    try:
        test_stt = STT()
        print("Testing Deepgram STT...")
        
        # Test with sample audio (you would need actual audio bytes here)
        print("✓ Deepgram STT initialized successfully")
        print("Note: Add actual audio bytes to test transcription")
        
    except Exception as e:
        print(f"✗ Deepgram STT test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())