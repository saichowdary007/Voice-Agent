import webrtcvad
from src.config import INPUT_SAMPLE_RATE, VAD_AGGRESSIVENESS, VAD_FRAME_MS

class VAD:
    """
    Handles Voice Activity Detection using the webrtcvad library.
    """
    def __init__(self):
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(VAD_AGGRESSIVENESS)
        self.sample_rate = INPUT_SAMPLE_RATE
        self.frame_duration_ms = VAD_FRAME_MS
        self.frame_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0))

        # Check for valid frame duration
        if self.frame_duration_ms not in [10, 20, 30]:
            raise ValueError("VAD frame duration must be 10, 20, or 30 ms.")

    def is_speech(self, frame_data: bytes) -> bool:
        """
        Determines if a given audio frame contains speech.

        Args:
            frame_data: A bytes object containing the raw audio data for one frame.
                        The frame must be 16-bit PCM mono.

        Returns:
            True if the frame contains speech, False otherwise.
        """
        if len(frame_data) != self.frame_size * 2: # 2 bytes per sample for 16-bit
            raise ValueError(f"Frame data size must be {self.frame_size * 2} bytes for the configured sample rate and frame duration.")
        
        return self.vad.is_speech(frame_data, self.sample_rate)

    def process_audio_stream(self, audio_stream):
        """
        A generator that yields speech segments from a continuous audio stream.
        This is a placeholder for a more complete implementation that would handle
        buffering and silence detection to yield meaningful chunks of speech.
        """
        # Note: A full implementation would buffer audio and yield larger segments
        # of speech, not just individual frames. This is a simplified version.
        for frame in audio_stream:
            if self.is_speech(frame):
                yield frame
