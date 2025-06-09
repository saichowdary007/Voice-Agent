import numpy as np
import os
from src.config import OUTPUT_SAMPLE_RATE

# Try to import Piper TTS, fall back to a simple TTS if not available
try:
    from piper.voice import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("Piper TTS not available. Using fallback TTS implementation.")

class TTS:
    """
    Handles Text-to-Speech synthesis using multiple TTS engines.
    """
    def __init__(self):
        self.engines = {}
        if PIPER_AVAILABLE:
            self._load_piper_voice()
        else:
            self._load_fallback_voice()

    def _load_piper_voice(self):
        """Loads the Piper TTS engine for English."""
        try:
            from src.config import PIPER_VOICE
            print("Loading Piper TTS voice for English (en)...")
            # Use the model name directly - Piper will handle model downloading/caching
            voice = PiperVoice.load(PIPER_VOICE)
            self.engines['en'] = {'engine': voice, 'type': 'piper'}
            print("Piper TTS voice for English loaded.")
        except Exception as e:
            print(f"Could not load Piper voice: {e}")
            print("Falling back to simple TTS implementation.")
            self._load_fallback_voice()

    def _load_fallback_voice(self):
        """Loads a simple fallback TTS that generates silence for testing."""
        print("Loading fallback TTS for testing...")
        self.engines['en'] = {'engine': None, 'type': 'fallback'}
        print("Fallback TTS loaded (generates silence for testing).")

    def _load_telugu_voice(self):
        """
        Placeholder for loading the Telugu TTS engine (e.g., Vakyansh).
        This will require downloading models and setting up the Vakyansh library.
        """
        print("Telugu TTS engine is not yet implemented.")
        # Example of what this might look like:
        # from vakyansh_tts_wrapper import VakyanshTTS
        # self.engines['te'] = {'engine': VakyanshTTS(), 'type': 'vakyansh'}
        pass

    def synthesize(self, text: str, lang: str = 'en') -> bytes:
        """
        Synthesizes speech from text using the appropriate engine for the language.

        Args:
            text: The text to be converted to speech.
            lang: The language of the text (e.g., 'en', 'te').

        Returns:
            Raw audio data in bytes (16-bit PCM).
        """
        if lang not in self.engines:
            print(f"No TTS engine loaded for language: '{lang}'. Falling back to 'en'.")
            lang = 'en'
            if 'en' not in self.engines:
                 print("English TTS engine not available. Cannot synthesize audio.")
                 return b''

        engine_info = self.engines[lang]
        
        if engine_info['type'] == 'piper':
            tts_engine = engine_info['engine']
            # Use the new Piper API - synthesize returns raw audio bytes
            audio_bytes = b''
            for chunk in tts_engine.synthesize_stream_raw(text):
                audio_bytes += chunk
            return audio_bytes
        
        elif engine_info['type'] == 'fallback':
            # Generate silence for testing - approximate 1 second per 10 characters
            duration_seconds = max(1.0, len(text) / 10.0)
            num_samples = int(OUTPUT_SAMPLE_RATE * duration_seconds)
            silence = np.zeros(num_samples, dtype=np.int16)
            print(f"Fallback TTS: Generated {duration_seconds:.1f}s of silence for text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            return silence.tobytes()
        
        # Add logic for other TTS engines here
        # elif engine_info['type'] == 'vakyansh':
        #     audio_samples, sr = tts_engine.synthesize(text)
        #     return audio_samples.tobytes()

        return b''

    def synthesize_stream(self, text: str, lang: str = 'en'):
        """
        Synthesizes speech from text and yields audio chunks as they become available.

        Args:
            text: The text to be converted to speech.
            lang: The language of the text.
        
        Yields:
            bytes: Chunks of raw audio data (16-bit PCM).
        """
        if lang not in self.engines:
            print(f"No TTS engine for '{lang}', falling back to 'en'.")
            lang = 'en'
            if 'en' not in self.engines:
                return

        engine_info = self.engines[lang]

        if engine_info['type'] == 'piper':
            tts_engine = engine_info['engine']
            # Use the new Piper streaming API
            for audio_chunk in tts_engine.synthesize_stream_raw(text):
                yield audio_chunk
        
        elif engine_info['type'] == 'fallback':
            # Stream the fallback audio in chunks
            audio_data = self.synthesize(text, lang)
            chunk_size = 1024  # bytes
            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i:i+chunk_size]
        
        # When a true streaming TTS is added (like Vakyansh might offer),
        # the logic would change to yield directly from the engine's stream.
        # for chunk in tts_engine.synthesize_stream(text):
        #     yield chunk
