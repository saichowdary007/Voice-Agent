"""
Speech-to-Text (STT) module with Deepgram integration.
Uses Deepgram's Nova-2 model for high-accuracy, low-latency transcription.
Falls back to faster-whisper for offline scenarios.
"""
import logging
from typing import AsyncGenerator, Optional

# Try to import Deepgram STT first
try:
    from src.stt_deepgram import STT as DeepgramSTT
    _DEEPGRAM_AVAILABLE = True
except ImportError:
    _DEEPGRAM_AVAILABLE = False

# Fallback imports for Whisper
if not _DEEPGRAM_AVAILABLE:
    import speech_recognition as sr
    from faster_whisper import WhisperModel
    from src.config import ENERGY_THRESHOLD, PAUSE_THRESHOLD
    import asyncio
    import io
    import numpy as np
    import threading
    from collections import deque

logger = logging.getLogger(__name__)

# Use Deepgram STT if available, otherwise fallback to Whisper
if _DEEPGRAM_AVAILABLE:
    logger.info("🚀 Using Deepgram STT for high-accuracy transcription")
    STT = DeepgramSTT
else:
    logger.info("🔄 Deepgram not available, using Whisper STT fallback")
    
    class STT:
        def __init__(self, model_size="small", device="auto"):
            # Auto-detect best device for maximum performance
            if device == "auto":
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda"
                        compute_type = "float16"
                        logger.info("🚀 Using GPU acceleration for STT")
                    else:
                        device = "cpu" 
                        compute_type = "int8"
                        logger.info("💻 Using CPU for STT")
                except ImportError:
                    device = "cpu"
                    compute_type = "int8"
                    logger.info("💻 Using CPU for STT (PyTorch not available)")
            else:
                compute_type = "float16" if device == "cuda" else "int8"
            
            # faster-whisper model for server-side transcription
            try:
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
                logger.info(f"✅ Whisper model '{model_size}' loaded on {device} with {compute_type}")
            except Exception as e:
                logger.error(f"❌ Failed to load Whisper model: {e}")
                # Fallback to CPU with int8
                self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
                logger.info("🔄 Fallback: Using CPU with int8")
            
            # speech_recognition for CLI microphone input (fallback compatibility)
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = ENERGY_THRESHOLD
            self.recognizer.pause_threshold = PAUSE_THRESHOLD
            self.microphone = None
            try:
                self.microphone = sr.Microphone()
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
            except Exception:
                pass  # Microphone not available (headless server)
            
            # Streaming transcription state
            self._audio_buffer = deque(maxlen=160000)  # ~10s at 16kHz
            self._last_transcript = ""
            self._stream_lock = threading.Lock()
        
        async def transcribe_bytes(self, audio_bytes: bytes, sr: int = 16_000) -> str:
            """Fast transcription of complete audio bytes."""
            if not audio_bytes:
                return ""
            
            # Convert bytes to numpy array (assumes 16-bit PCM)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Run in executor to avoid blocking event loop
            def _transcribe():
                segments, _ = self.model.transcribe(
                    audio_np,
                    language=None,
                    beam_size=1,  # Reduced for speed
                    vad_filter=False,  # Disable Whisper VAD - rely on WebRTC/RNNoise instead
                    condition_on_previous_text=False  # Disable for speed
                )
                return " ".join(s.text for s in segments).strip()
            
            return await asyncio.to_thread(_transcribe)
        
        async def stream_transcribe_chunk(self, audio_chunk: bytes, is_final: bool = False) -> Optional[str]:
            """
            Stream transcription with partial results.
            Returns partial transcript on each call, full transcript when is_final=True.
            """
            if not audio_chunk:
                return None
                
            # Convert chunk to numpy array
            chunk_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add to rolling buffer
            with self._stream_lock:
                self._audio_buffer.extend(chunk_np)
                
                # For partial results, use last 3 seconds of audio
                if not is_final:
                    buffer_size = min(9600, len(self._audio_buffer))  # 0.6s at 16kHz
                    if buffer_size < 3200:  # Need at least 0.2s of audio
                        return None
                        
                    audio_segment = np.array(list(self._audio_buffer)[-buffer_size:])
                else:
                    # For final, use entire buffer
                    if len(self._audio_buffer) < 1600:  # Need at least 0.1s
                        return None
                    audio_segment = np.array(list(self._audio_buffer))
            
            # Transcribe in background thread
            def _stream_transcribe():
                try:
                    segments, _ = self.model.transcribe(
                        audio_segment,
                        language=None,
                        beam_size=1,
                        vad_filter=False,  # Disable Whisper VAD - rely on WebRTC/RNNoise instead
                        condition_on_previous_text=False,
                        word_timestamps=False  # Disable for speed
                    )
                    return " ".join(s.text for s in segments).strip()
                except Exception as e:
                    logger.error(f"Stream transcription error: {e}")
                    return None
            
            transcript = await asyncio.to_thread(_stream_transcribe)
            
            if is_final:
                # Clear buffer after final transcription
                with self._stream_lock:
                    self._audio_buffer.clear()
                    self._last_transcript = ""
            else:
                # Track partial results
                self._last_transcript = transcript or self._last_transcript
                
            return transcript
        
        def listen_and_transcribe(self, timeout: int = None, phrase_time_limit: int = None) -> str:
            """CLI compatibility method for microphone input"""
            if self.microphone is None:
                return None
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                return self.recognizer.recognize_google(audio)
            except Exception:
                return None
