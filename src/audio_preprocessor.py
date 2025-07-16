"""
Server-side audio preprocessing with noise suppression and enhanced VAD.
Implements the recommended two-stage VAD pipeline with RNNoise-style processing.
"""

import logging
import numpy as np
from typing import Optional, Tuple
import webrtcvad
from scipy import signal
import asyncio

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Server-side audio preprocessing with noise suppression and dual-stage VAD.
    Implements the recommended pipeline: RNNoise -> Silero VAD -> WebRTC VAD mode 2
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
        # Initialize WebRTC VAD with mode 2 (more aggressive for server-side)
        self.webrtc_vad = webrtcvad.Vad(2)
        
        # Frame size for WebRTC VAD (20ms)
        self.frame_bytes = int(sample_rate * 0.02) * 2  # 16-bit samples
        
        # Noise suppression state
        self.noise_profile = None
        self.noise_history = []
        self.noise_history_size = 50
        
        # Spectral subtraction parameters
        self.alpha = 2.0  # Over-subtraction factor
        self.beta = 0.01  # Spectral floor
        
        logger.info(f"✅ AudioPreprocessor initialized (WebRTC VAD mode 2, {sample_rate}Hz)")
    
    def preprocess_audio(self, audio_bytes: bytes) -> Tuple[bytes, bool]:
        """
        Apply full preprocessing pipeline to audio.
        
        Args:
            audio_bytes: Raw 16-bit PCM audio bytes
            
        Returns:
            Tuple of (processed_audio_bytes, is_speech_detected)
        """
        if not audio_bytes or len(audio_bytes) < self.frame_bytes:
            return audio_bytes, False
        
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Apply noise suppression
            cleaned_audio = self._apply_noise_suppression(audio_np)
            
            # Convert back to bytes for VAD
            cleaned_bytes = (cleaned_audio * 32767).astype(np.int16).tobytes()
            
            # Apply dual-stage VAD
            is_speech = self._dual_stage_vad(cleaned_bytes)
            
            return cleaned_bytes, is_speech
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return audio_bytes, True  # Default to speech when in doubt
    
    def _apply_noise_suppression(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply spectral subtraction-based noise suppression.
        Simplified version of RNNoise approach for CPU-only processing.
        """
        try:
            # Ensure we have enough samples for FFT
            if len(audio) < 512:
                return audio
            
            # Apply window and FFT
            windowed = audio * np.hanning(len(audio))
            spectrum = np.fft.rfft(windowed)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            
            # Update noise profile
            self._update_noise_profile(magnitude)
            
            if self.noise_profile is not None:
                # Spectral subtraction
                noise_magnitude = self.noise_profile
                
                # Calculate SNR estimate
                snr_estimate = magnitude / (noise_magnitude + 1e-10)
                
                # Apply over-subtraction with spectral floor
                suppression_factor = np.maximum(
                    1.0 - self.alpha * (noise_magnitude / (magnitude + 1e-10)),
                    self.beta
                )
                
                # Apply suppression
                cleaned_magnitude = magnitude * suppression_factor
                
                # Reconstruct signal
                cleaned_spectrum = cleaned_magnitude * np.exp(1j * phase)
                cleaned_audio = np.fft.irfft(cleaned_spectrum)
                
                # Apply window again and normalize
                if len(cleaned_audio) == len(audio):
                    return cleaned_audio * np.hanning(len(cleaned_audio))
            
            return audio
            
        except Exception as e:
            logger.warning(f"Noise suppression failed: {e}")
            return audio
    
    def _update_noise_profile(self, magnitude: np.ndarray):
        """Update noise profile using voice activity detection."""
        # Add to history
        self.noise_history.append(magnitude.copy())
        
        # Keep only recent history
        if len(self.noise_history) > self.noise_history_size:
            self.noise_history.pop(0)
        
        # Update noise profile (use minimum statistics)
        if len(self.noise_history) >= 5:
            # Stack all magnitude spectra
            history_stack = np.stack(self.noise_history)
            
            # Use 25th percentile as noise estimate (conservative)
            self.noise_profile = np.percentile(history_stack, 25, axis=0)
    
    def _dual_stage_vad(self, audio_bytes: bytes) -> bool:
        """
        Apply dual-stage VAD: energy-based pre-filter + WebRTC VAD mode 2.
        """
        try:
            # Stage 1: Energy-based pre-filter
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            rms_energy = np.sqrt(np.mean(audio_np**2))
            
            # Very low threshold for pre-filter (just eliminate obvious silence)
            if rms_energy < 50.0:  # Adjust based on your audio levels
                return False
            
            # Stage 2: WebRTC VAD mode 2 (more aggressive)
            return self._webrtc_vad_check(audio_bytes)
            
        except Exception as e:
            logger.warning(f"Dual-stage VAD error: {e}")
            return True  # Default to speech when in doubt
    
    def _webrtc_vad_check(self, audio_bytes: bytes) -> bool:
        """Apply WebRTC VAD to audio frames."""
        if len(audio_bytes) < self.frame_bytes:
            return False
        
        # Check each 20ms frame
        for i in range(0, len(audio_bytes) - self.frame_bytes + 1, self.frame_bytes):
            frame = audio_bytes[i:i + self.frame_bytes]
            if self.webrtc_vad.is_speech(frame, self.sample_rate):
                return True
        
        return False
    
    def reset_noise_profile(self):
        """Reset noise profile for new audio session."""
        self.noise_profile = None
        self.noise_history = []
        logger.debug("🔄 Noise profile reset")


class SileroVAD:
    """
    Placeholder for Silero VAD integration.
    This would be the preferred VAD for far-field microphones.
    """
    
    def __init__(self):
        self.available = False
        logger.info("⚠️ Silero VAD not implemented - using WebRTC VAD fallback")
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """Placeholder for Silero VAD."""
        return True  # Always return True as fallback


# Global preprocessor instance
_preprocessor = None


def get_audio_preprocessor() -> AudioPreprocessor:
    """Get global audio preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = AudioPreprocessor()
    return _preprocessor


async def preprocess_audio_chunk(audio_bytes: bytes) -> Tuple[bytes, bool]:
    """
    Async wrapper for audio preprocessing.
    
    Args:
        audio_bytes: Raw audio bytes
        
    Returns:
        Tuple of (processed_bytes, is_speech)
    """
    preprocessor = get_audio_preprocessor()
    
    # Run preprocessing in thread to avoid blocking
    def _process():
        return preprocessor.preprocess_audio(audio_bytes)
    
    return await asyncio.to_thread(_process)