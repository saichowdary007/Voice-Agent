import webrtcvad
import time
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class VAD:
    """Enhanced Voice Activity Detection with improved speech boundary detection.
    
    Features:
    - Adaptive speech/silence detection with configurable thresholds
    - Speech boundary tracking to prevent premature cutoffs
    - Energy-based validation alongside WebRTC VAD
    - Configurable silence padding for natural speech patterns
    """

    def __init__(self, sample_rate: int = 16000, mode: int = 1):
        """Create a new VAD instance with enhanced speech detection.

        Args:
            sample_rate: PCM sample-rate of incoming audio. 16 kHz is optimal for speech processing.
            mode: Aggressiveness (0-3). 0 = least aggressive (more speech detected),
                  1 = less aggressive (better for quiet speech), 3 = most aggressive.
        """
        self.sample_rate = sample_rate
        self._vad = webrtcvad.Vad(mode)

        # 20 ms frame == (sample_rate * 0.02) samples * 2 bytes per sample (16-bit).
        self._frame_bytes = int(sample_rate * 0.02) * 2
        
        # Enhanced speech detection parameters
        self._speech_frames = 0
        self._silence_frames = 0
        self._last_speech_time = 0
        self._speech_started = False
        
        # Configurable thresholds for better speech detection
        self.min_speech_frames = 3  # Minimum consecutive speech frames to start
        self.min_silence_frames = 25  # Minimum silence frames to end (500ms at 20ms frames)
        self.energy_threshold = 500  # Minimum energy level for speech
        self.speech_padding_ms = 200  # Extra padding after speech ends
        
        logger.info(f"Enhanced VAD initialized: mode={mode}, min_silence={self.min_silence_frames * 20}ms")

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

    def analyze_speech_boundaries(self, pcm_bytes: bytes) -> Tuple[bool, bool, float]:
        """Analyze speech boundaries with enhanced detection.
        
        Args:
            pcm_bytes: Raw PCM audio bytes
            
        Returns:
            Tuple of (has_speech, speech_ended, confidence)
        """
        if not pcm_bytes or len(pcm_bytes) < self._frame_bytes:
            return False, False, 0.0
        
        current_time = time.time()
        speech_frames_in_chunk = 0
        total_frames = 0
        total_energy = 0.0
        
        # Analyze each 20ms frame
        for i in range(0, len(pcm_bytes) - self._frame_bytes + 1, self._frame_bytes):
            frame = pcm_bytes[i : i + self._frame_bytes]
            total_frames += 1
            
            # Calculate frame energy
            try:
                frame_array = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
                frame_energy = np.sum(frame_array ** 2) / len(frame_array)
                total_energy += frame_energy
            except Exception:
                frame_energy = 0
            
            # Check if frame contains speech (WebRTC VAD + energy threshold)
            is_speech_frame = (
                self._vad.is_speech(frame, self.sample_rate) and 
                frame_energy > self.energy_threshold
            )
            
            if is_speech_frame:
                speech_frames_in_chunk += 1
                self._speech_frames += 1
                self._silence_frames = 0
                self._last_speech_time = current_time
                
                # Mark speech as started if we have enough consecutive frames
                if self._speech_frames >= self.min_speech_frames:
                    self._speech_started = True
            else:
                self._silence_frames += 1
                # Reset speech frame counter if we hit silence
                if self._silence_frames > 2:  # Allow brief pauses
                    self._speech_frames = 0
        
        # Calculate confidence based on speech ratio and energy
        avg_energy = total_energy / max(total_frames, 1)
        speech_ratio = speech_frames_in_chunk / max(total_frames, 1)
        confidence = min(1.0, (speech_ratio * 0.7) + (min(avg_energy / 10000, 1.0) * 0.3))
        
        has_speech = speech_frames_in_chunk > 0
        
        # Determine if speech has ended (with padding)
        time_since_speech = current_time - self._last_speech_time
        speech_ended = (
            self._speech_started and 
            self._silence_frames >= self.min_silence_frames and
            time_since_speech > (self.speech_padding_ms / 1000.0)
        )
        
        if speech_ended:
            logger.info(f"Speech boundary detected - ending after {time_since_speech:.2f}s silence")
            # Reset state for next speech segment
            self._speech_started = False
            self._speech_frames = 0
            self._silence_frames = 0
        
        logger.debug(f"VAD analysis: speech_frames={speech_frames_in_chunk}/{total_frames}, "
                    f"energy={avg_energy:.1f}, confidence={confidence:.2f}, "
                    f"speech_started={self._speech_started}, ended={speech_ended}")
        
        return has_speech, speech_ended, confidence

    def should_process_audio(self, pcm_bytes: bytes, force_final: bool = False) -> bool:
        """Determine if audio should be processed for transcription.
        
        Args:
            pcm_bytes: Raw PCM audio bytes
            force_final: Force processing regardless of speech detection
            
        Returns:
            True if audio should be sent for transcription
        """
        if force_final:
            return True
            
        has_speech, speech_ended, confidence = self.analyze_speech_boundaries(pcm_bytes)
        
        # Process if we have confident speech or if speech has clearly ended
        should_process = (
            (has_speech and confidence > 0.3) or 
            speech_ended or
            (self._speech_started and confidence > 0.1)  # Lower threshold once speech started
        )
        
        logger.debug(f"Should process audio: {should_process} (speech={has_speech}, "
                    f"ended={speech_ended}, confidence={confidence:.2f})")
        
        return should_process

    def reset_state(self):
        """Reset VAD state for new audio session."""
        self._speech_frames = 0
        self._silence_frames = 0
        self._last_speech_time = 0
        self._speech_started = False
        logger.debug("VAD state reset")

    def configure_sensitivity(self, sensitivity: str = "medium"):
        """Configure VAD sensitivity for different use cases.
        
        Args:
            sensitivity: "low", "medium", "high", or "ultra"
        """
        if sensitivity == "low":
            self.min_speech_frames = 5
            self.min_silence_frames = 35  # 700ms
            self.energy_threshold = 1000
            self.speech_padding_ms = 300
        elif sensitivity == "medium":
            self.min_speech_frames = 3
            self.min_silence_frames = 25  # 500ms
            self.energy_threshold = 500
            self.speech_padding_ms = 200
        elif sensitivity == "high":
            self.min_speech_frames = 2
            self.min_silence_frames = 20  # 400ms
            self.energy_threshold = 300
            self.speech_padding_ms = 150
        elif sensitivity == "ultra":
            self.min_speech_frames = 1
            self.min_silence_frames = 15  # 300ms
            self.energy_threshold = 200
            self.speech_padding_ms = 100
        
        logger.info(f"VAD sensitivity set to {sensitivity}: "
                   f"min_silence={self.min_silence_frames * 20}ms, "
                   f"energy_threshold={self.energy_threshold}")

    def get_speech_stats(self) -> dict:
        """Get current speech detection statistics."""
        return {
            "speech_frames": self._speech_frames,
            "silence_frames": self._silence_frames,
            "speech_started": self._speech_started,
            "time_since_speech": time.time() - self._last_speech_time if self._last_speech_time > 0 else 0
        } 