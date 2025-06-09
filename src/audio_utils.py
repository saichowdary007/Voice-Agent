import pyaudio
from src.config import OUTPUT_SAMPLE_RATE, OUTPUT_CHANNELS, OUTPUT_FORMAT
import numpy as np

class AudioPlayer:
    """
    A non-blocking audio player that allows for interruption.
    """
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None

    def start_playback(self):
        """Opens a new non-blocking audio stream for playback."""
        if self.stream:
            self.stop_playback()

        pyaudio_format = pyaudio.paInt16 if OUTPUT_FORMAT == 'int16' else None
        if not pyaudio_format:
            raise ValueError(f"Unsupported output format: {OUTPUT_FORMAT}")

        self.stream = self.p.open(
            format=pyaudio_format,
            channels=OUTPUT_CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True
        )

    def write_chunk(self, audio_chunk: bytes):
        """Writes a chunk of audio data to the stream if it's active."""
        if self.stream and self.stream.is_active():
            self.stream.write(audio_chunk)

    def is_playing(self) -> bool:
        """Checks if the audio stream is currently active."""
        return self.stream and self.stream.is_active()

    def stop_playback(self):
        """Stops the playback and closes the stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
    
    def __del__(self):
        """Ensure PyAudio is terminated when the object is destroyed."""
        self.stop_playback()
        self.p.terminate()

# The old blocking function, kept for simplicity in other scripts if needed.
def play_audio_blocking(audio_data: bytes):
    p = pyaudio.PyAudio()
    pyaudio_format = pyaudio.paInt16 if OUTPUT_FORMAT == 'int16' else None
    stream = p.open(format=pyaudio_format,
                    channels=OUTPUT_CHANNELS,
                    rate=OUTPUT_SAMPLE_RATE,
                    output=True)
    stream.write(audio_data)
    stream.stop_stream()
    stream.close()
    p.terminate()

def get_input_device_index(p: pyaudio.PyAudio, device_name: str) -> int:
    """
    Gets the index of an input device by name.
    """
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['name'].lower() == device_name.lower() and dev_info['maxInputChannels'] > 0:
            return i
    return -1

def get_output_device_index(p: pyaudio.PyAudio, device_name: str) -> int:
    """
    Gets the index of an output device by name.
    """
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['name'].lower() == device_name.lower() and dev_info['maxOutputChannels'] > 0:
            return i
    return -1
