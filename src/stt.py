import whisper
import numpy as np
from src.config import WHISPER_MODEL, INPUT_SAMPLE_RATE

class STT:
    """
    Handles Speech-to-Text using the OpenAI Whisper model.
    """
    def __init__(self):
        print("Loading Whisper model...")
        self.model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribes a given audio segment.

        Args:
            audio_data: A numpy array containing the audio data.
                        The audio must be in the sample rate the model was trained on (16kHz).

        Returns:
            The transcribed text.
        """
        if audio_data.dtype != np.float32:
            # Whisper expects float32, so we convert from int16 and normalize
            audio_data = audio_data.astype(np.float32) / 32768.0

        result = self.model.transcribe(audio_data, fp16=False) # fp16=False for CPU
        return result['text']

    @staticmethod
    def audio_bytes_to_numpy(audio_bytes: bytes) -> np.ndarray:
        """
        Converts raw audio bytes (16-bit PCM) to a numpy array.
        """
        return np.frombuffer(audio_bytes, dtype=np.int16)

if __name__ == '__main__':
    # Example usage:
    # This part will only run when the script is executed directly
    # It demonstrates how to use the STT class with a dummy audio file.
    import soundfile as sf

    # Create a dummy audio file for testing
    samplerate = INPUT_SAMPLE_RATE
    duration = 5  # seconds
    frequency = 440  # Hz
    t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)
    
    # Save dummy audio
    dummy_file = "dummy_audio.wav"
    sf.write(dummy_file, data, samplerate)
    print(f"Created dummy audio file: {dummy_file}")

    # Initialize STT and transcribe
    stt = STT()
    audio_np = stt.audio_bytes_to_numpy(data.tobytes())
    transcribed_text = stt.transcribe(audio_np)
    
    print("\n--- Transcription ---")
    print(f"Audio length: {duration} seconds")
    # The dummy audio is a pure sine wave, so Whisper might not "hear" words.
    # The output will likely be empty or repetitive noise. This is expected.
    print(f"Transcription result: '{transcribed_text}'")
    print("-----------------------\n")

    # Clean up the dummy file
    import os
    os.remove(dummy_file)
    print(f"Removed dummy audio file: {dummy_file}")
