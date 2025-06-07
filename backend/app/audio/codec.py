import os
import json
import tempfile
import subprocess
import asyncio
from typing import Optional, AsyncGenerator
from enum import Enum
import numpy as np
import opuslib
import ffmpeg
import structlog

# Import standardized configuration
from backend.app.config import settings

logger = structlog.get_logger()


class AudioFormat(Enum):
    """Enumeration for supported audio formats."""
    WAV = "wav"
    WEBM = "webm"
    OGG = "ogg"
    OPUS = "opus"
    UNKNOWN = "unknown"

def identify_format(data: bytes) -> AudioFormat:
    """
    Identifies the audio format based on magic bytes.
    
    Args:
        data: The audio data in bytes.
        
    Returns:
        The identified AudioFormat.
    """
    if len(data) < 12:
        return AudioFormat.UNKNOWN
        
    # Check for WAV (RIFF header)
    if data.startswith(b'RIFF') and data[8:12] == b'WAVE':
        logger.debug("Identified audio format: WAV")
        return AudioFormat.WAV
        
    # Check for WebM (EBML header)
    if data.startswith(b'\x1aE\xdf\xa3'):
        logger.debug("Identified audio format: WebM")
        return AudioFormat.WEBM
        
    # Check for Ogg
    if data.startswith(b'OggS'):
        logger.debug("Identified audio format: Ogg")
        return AudioFormat.OGG
        
    # Default to unknown
    logger.debug("Could not identify audio format from magic bytes.")
    return AudioFormat.UNKNOWN


class OpusCodec:
    """High-performance Opus codec for real-time audio streaming"""
    
    def __init__(self):
        # Use standardized configuration instead of hardcoded values
        self.sample_rate = settings.sample_rate
        self.channels = settings.channels
        self.frame_duration = settings.frame_duration_ms  # Use config instead of hardcoded 120ms
        self.bitrate = 32000  # 32 kbps for voice
        
        # Opus encoder/decoder
        self.encoder: Optional[opuslib.Encoder] = None
        self.decoder: Optional[opuslib.Decoder] = None
        
        # Frame size calculation
        self.frame_size = self.sample_rate * self.frame_duration // 1000
        
    async def initialize(self):
        """Initialize Opus encoder and decoder"""
        try:
            # Create encoder
            self.encoder = opuslib.Encoder(
                fs=self.sample_rate,
                channels=self.channels,
                application=opuslib.APPLICATION_VOIP  # Optimized for voice
            )
            
            # Configure encoder for low latency
            self.encoder.bitrate = self.bitrate
            self.encoder.signal = opuslib.SIGNAL_VOICE
            self.encoder.complexity = 5  # Balance quality vs latency
            self.encoder.packet_loss_perc = 1  # Expect some packet loss
            self.encoder.dtx = True  # Discontinuous transmission
            
            # Create decoder
            self.decoder = opuslib.Decoder(
                fs=self.sample_rate,
                channels=self.channels
            )
            
            logger.info(f"Opus codec initialized: {self.sample_rate}Hz, {self.frame_duration}ms frames")
            
        except Exception as e:
            logger.error(f"Failed to initialize Opus codec: {e}")
            raise
            
    def encode_frame(self, pcm_data: np.ndarray) -> bytes:
        """Encode PCM frame to Opus"""
        try:
            if self.encoder is None:
                raise RuntimeError("Encoder not initialized")
                
            # Ensure correct data type and shape
            if pcm_data.dtype != np.int16:
                pcm_data = (pcm_data * 32767).astype(np.int16)
                
            # Encode frame
            opus_data = self.encoder.encode(pcm_data.tobytes(), self.frame_size)
            return opus_data
            
        except Exception as e:
            logger.error(f"Opus encoding error: {e}")
            raise
            
    def decode_frame(self, opus_data: bytes) -> np.ndarray:
        """Decode Opus frame to PCM"""
        try:
            if self.decoder is None:
                raise RuntimeError("Decoder not initialized")
                
            # Decode frame
            pcm_bytes = self.decoder.decode(opus_data, self.frame_size)
            
            # Convert to numpy array
            pcm_data = np.frombuffer(pcm_bytes, dtype=np.int16)
            return pcm_data
            
        except Exception as e:
            logger.error(f"Opus decoding error: {e}")
            raise
            
    async def encode_stream(self, pcm_stream: AsyncGenerator[np.ndarray, None]) -> AsyncGenerator[bytes, None]:
        """Encode PCM stream to Opus stream"""
        try:
            async for pcm_frame in pcm_stream:
                opus_frame = self.encode_frame(pcm_frame)
                yield opus_frame
                
        except Exception as e:
            logger.error(f"Opus stream encoding error: {e}")
            raise
            
    async def decode_stream(self, opus_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[np.ndarray, None]:
        """Decode Opus stream to PCM stream"""
        try:
            async for opus_frame in opus_stream:
                pcm_frame = self.decode_frame(opus_frame)
                yield pcm_frame
                
        except Exception as e:
            logger.error(f"Opus stream decoding error: {e}")
            raise


class AudioConverter:
    """Audio format conversion utilities using FFmpeg"""
    
    @staticmethod
    async def convert_to_wav(input_data: bytes, input_format: Optional[str] = None) -> bytes:
        """
        Convert audio data to WAV format with automatic format detection.
        If input_format is not provided, it will be auto-detected.
        """
        detected_format = identify_format(input_data)
        
        # Use provided format if valid, otherwise use detected format
        final_format = input_format if input_format and input_format != "auto" else detected_format.value

        if final_format == AudioFormat.UNKNOWN.value:
            raise ValueError("Could not determine input audio format for conversion.")
            
        logger.info(f"Converting audio from '{final_format}' to WAV.")

        try:
            # For WebM/Ogg, ffmpeg needs a format hint.
            # Opus is a raw codec, not a container, so we handle it separately.
            if final_format in [AudioFormat.WEBM.value, AudioFormat.OGG.value]:
                # These are container formats that ffmpeg can handle directly
                ffmpeg_input_format = final_format
            elif final_format == AudioFormat.OPUS.value:
                # opus is a raw codec, not a container, 'libopus' is a decoder name
                ffmpeg_input_format = 'opus'
            else:
                 # For WAV, no specific format hint is needed, but being explicit is good
                ffmpeg_input_format = 'wav'

            process = (
                ffmpeg
                .input('pipe:', format=ffmpeg_input_format)
                .output('pipe:', format='wav', acodec='pcm_s16le', ar=settings.sample_rate, ac=settings.channels)
                .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
            )
            
            stdout, stderr = await asyncio.to_thread(process.communicate, input=input_data)
            
            if process.returncode != 0:
                error_message = stderr.decode()
                logger.error(f"FFmpeg error during {final_format} to WAV conversion: {error_message}")
                # Provide more specific error messages
                if "EBML header parsing failed" in error_message:
                    raise RuntimeError("FFmpeg error: Invalid WebM header. The file may be corrupt or not a valid WebM file.")
                if "Error opening input: End of file" in error_message:
                    raise RuntimeError("FFmpeg error: End of file reached unexpectedly. The input data may be incomplete.")
                raise RuntimeError(f"FFmpeg conversion failed with exit code {process.returncode}")
                
            logger.info(f"Successfully converted {len(input_data)} bytes of {final_format} to {len(stdout)} bytes of WAV.")
            return stdout
            
        except Exception as e:
            logger.error(f"Audio conversion failed for format '{final_format}': {e}")
            raise
            
    @staticmethod
    async def convert_from_wav(wav_data: bytes, output_format: str = "opus") -> bytes:
        """Convert WAV data to other format"""
        try:
            process = (
                ffmpeg
                .input('pipe:', format='wav')
                .output('pipe:', format=output_format, acodec='libopus', ar=16000, ac=1, b='32k')
                .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
            )
            
            stdout, stderr = await asyncio.to_thread(process.communicate, wav_data)
            
            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {stderr.decode()}")
                
            return stdout
            
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            raise
            
    @staticmethod
    def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude"""
        if len(audio_data) == 0:
            return audio_data
            
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        
        if rms > 0:
            # Normalize to -12dB
            target_rms = 0.25
            gain = target_rms / rms
            # Prevent clipping
            gain = min(gain, 1.0)
            audio_data = (audio_data.astype(np.float32) * gain).astype(np.int16)
            
        return audio_data
        
    @staticmethod
    def apply_noise_gate(audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Apply noise gate to reduce background noise"""
        # Calculate signal level
        signal_level = np.abs(audio_data.astype(np.float32)) / 32767.0
        
        # Apply gate
        gate = signal_level > threshold
        audio_data = audio_data * gate
        
        return audio_data 