import webrtcvad

class VAD:
    """Lightweight wrapper around WebRTC VAD for 16-kHz mono PCM audio.

    The `is_speech` method scans the provided byte string in 20-ms frames and
    returns True as soon as any frame is classified as speech.  Designed to be
    extremely fast (micro-seconds) so it can be run for every incoming audio
    chunk before STT to skip leading / trailing silence.
    """

    def __init__(self, sample_rate: int = 16000, mode: int = 1):
        """Create a new VAD instance.

        Args:
            sample_rate: PCM sample-rate of incoming audio. 16 kHz is the sweet-spot for Whisper.
            mode: Aggressiveness (0-3). 0 = least aggressive (more speech detected),
                  1 = less aggressive (better for quiet speech), 3 = most aggressive.
        """
        self.sample_rate = sample_rate
        self._vad = webrtcvad.Vad(mode)

        # 20 ms frame == (sample_rate * 0.02) samples * 2 bytes per sample (16-bit).
        self._frame_bytes = int(sample_rate * 0.02) * 2

    def is_speech(self, pcm_bytes: bytes) -> bool:
        """Return True if any 20-ms frame in *pcm_bytes* contains speech."""
        if not pcm_bytes or len(pcm_bytes) < self._frame_bytes:
            return False

        # Iterate over contiguous 20-ms frames.
        for i in range(0, len(pcm_bytes) - self._frame_bytes + 1, self._frame_bytes):
            frame = pcm_bytes[i : i + self._frame_bytes]
            if self._vad.is_speech(frame, self.sample_rate):
                return True
        return False 