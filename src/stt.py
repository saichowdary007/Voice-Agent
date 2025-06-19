"""
Drop-in replacement STT module using faster-whisper for multilingual, offline speech recognition.
4x faster than vanilla Whisper, supports 50+ languages, quantized to <1GB RAM.
"""
import speech_recognition as sr
from faster_whisper import WhisperModel
from src.config import ENERGY_THRESHOLD, PAUSE_THRESHOLD
import asyncio
import io
import numpy as np

class STT:
    def __init__(self, model_size="small", device="cpu"):
        # faster-whisper model for server-side transcription
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        
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
    
    async def transcribe_bytes(self, audio_bytes: bytes, sr: int = 16_000) -> str:
        if not audio_bytes:
            return ""
        
        # Convert bytes to numpy array (assumes 16-bit PCM)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Run in executor to avoid blocking event loop
        def _transcribe():
            segments, _ = self.model.transcribe(audio_np,
                                                language=None,
                                                beam_size=5,
                                                vad_filter=True)
            return " ".join(s.text for s in segments)
        
        return await asyncio.to_thread(_transcribe)
    
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
