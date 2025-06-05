import logging
import numpy as np
import asyncio
import io
from typing import Optional, Tuple, Dict, Any, List
import opuslib # Ensure this is installed: pip install opuslib
import soundfile as sf # Ensure this is installed: pip install soundfile
import subprocess
import shutil # For shutil.which

# Relative import for app config, assuming services is a package sibling to app
from ..app.config import settings, get_sample_rate, get_channels 

# Use standard Python logging for services, or pass structlog logger if standardized
logger = logging.getLogger(__name__)
# If using structlog throughout:
# import structlog
# logger = structlog.get_logger(__name__)


class AudioService:
    """Audio processing service for Opus decoding, PCM conversion, and format management"""
    
    def __init__(self, sample_rate: Optional[int] = None, channels: Optional[int] = None):
        self.sample_rate: int = sample_rate or get_sample_rate()
        self.channels: int = channels or get_channels()
        self.opus_decoder: Optional[opuslib.Decoder] = None
        self.opus_encoder: Optional[opuslib.Encoder] = None
        self.is_available: bool = False # Overall service availability
        self.ffmpeg_available: bool = False
        
        # Chunk buffering for handling incomplete WebM/audio fragments
        self.chunk_buffer: bytearray = bytearray()
        self.buffer_size_threshold: int = 25000  # 25KB - buffer until we have substantial data
        self.stream_mode_active: bool = False  # True after first successful decode
        
        # Frame size in samples per channel for Opus codec
        self.opus_frame_duration_ms: int = settings.audio_frame_ms # Use configured frame duration for Opus
        self.opus_frame_size_samples: int = int(self.sample_rate * self.opus_frame_duration_ms / 1000)
        
        # Frame size in bytes for general PCM processing (16-bit)
        self.pcm_frame_size_bytes: int = self.opus_frame_size_samples * self.channels * 2 # 2 bytes per sample for 16-bit
        
        logger.info(f"AudioService configured: SR={self.sample_rate}Hz, Chan={self.channels}, FrameDur={self.opus_frame_duration_ms}ms, OpusFrameSamples={self.opus_frame_size_samples}")

    async def initialize(self):
        """Initialize audio processing components (Opus codec, FFmpeg check)."""
        logger.info("Initializing AudioService...")
        await self._check_ffmpeg_availability()
        
        try:
            self.opus_decoder = opuslib.Decoder(fs=self.sample_rate, channels=self.channels)
            logger.info(f"Opus decoder initialized: {self.sample_rate}Hz, {self.channels}ch.")
        except Exception as e:
            logger.error(f"Failed to initialize Opus decoder: {e}", exc_info=True)
            self.opus_decoder = None
        
        try:
            self.opus_encoder = opuslib.Encoder(fs=self.sample_rate, channels=self.channels, application=opuslib.APPLICATION_AUDIO)
            try: # Set preferred Opus encoder options, fail gracefully if any are not supported
                self.opus_encoder.bitrate = 32000 # 32 kbps for voice
                self.opus_encoder.signal = opuslib.SIGNAL_VOICE
                self.opus_encoder.complexity = 5 # Balance between quality and CPU
                self.opus_encoder.packet_loss_perc = 5 # Expect some packet loss
                self.opus_encoder.dtx = True # Discontinuous Transmission for silence periods
            except Exception as enc_opt_e:
                logger.warning(f"Could not set all preferred Opus encoder parameters: {enc_opt_e}")
            logger.info(f"Opus encoder initialized: {self.sample_rate}Hz, {self.channels}ch.")
        except Exception as e:
            logger.error(f"Failed to initialize Opus encoder: {e}", exc_info=True)
            self.opus_encoder = None
        
        # Service is considered available if FFmpeg or Opus components are working
        self.is_available = self.ffmpeg_available or (self.opus_decoder is not None and self.opus_encoder is not None)
        if self.is_available:
            logger.info("✅ AudioService initialized successfully.")
        else:
            logger.error("❌ AudioService failed to initialize critical components (FFmpeg and/or Opus). It will operate in a degraded state or fail.")

    async def _check_ffmpeg_availability(self):
        """Check if FFmpeg command-line tool is available."""
        try:
            ffmpeg_path = shutil.which('ffmpeg')
            if not ffmpeg_path:
                logger.warning("FFmpeg command not found in system PATH. FFmpeg-based conversions will be unavailable.")
                self.ffmpeg_available = False
                return

            process = await asyncio.create_subprocess_exec(
                ffmpeg_path, '-version',
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                version_info = stdout.decode('utf-8', errors='ignore').split('\n')[0].strip()
                logger.info(f"FFmpeg is available: {version_info}")
                self.ffmpeg_available = True
            else:
                err_msg = stderr.decode('utf-8', errors='ignore').strip()
                logger.error(f"FFmpeg test execution failed (Return Code: {process.returncode}): {err_msg}")
                self.ffmpeg_available = False
        except FileNotFoundError:
            logger.error("FFmpeg command not found. Ensure FFmpeg is installed and in PATH.")
            self.ffmpeg_available = False
        except Exception as e:
            logger.error(f"Error during FFmpeg availability check: {e}", exc_info=True)
            self.ffmpeg_available = False

    def decode_opus_frame(self, opus_frame: bytes) -> Optional[np.ndarray]:
        """Decodes a single Opus frame to PCM (float32 numpy array)."""
        if not self.opus_decoder:
            logger.warning("Opus decoder not initialized. Cannot decode Opus frame.")
            return None
        try:
            # opuslib.Decoder.decode expects frame_size in samples per channel
            pcm_s16le_bytes = self.opus_decoder.decode(opus_frame, frame_size=self.opus_frame_size_samples)
            # Convert s16le bytes to float32 numpy array, normalized to [-1.0, 1.0]
            pcm_float32_array = np.frombuffer(pcm_s16le_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return pcm_float32_array
        except opuslib.OpusError as e: # Catch specific Opus errors
            logger.error(f"Opus decoding error: {e}. Frame size: {len(opus_frame)} bytes.", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error during Opus decoding: {e}", exc_info=True)
            return None

    def encode_pcm_to_opus(self, pcm_float32_array: np.ndarray) -> Optional[bytes]:
        """Encodes a PCM float32 numpy array to an Opus frame."""
        if not self.opus_encoder:
            logger.warning("Opus encoder not initialized. Cannot encode PCM to Opus.")
            return None
        try:
            # Ensure PCM data is 16-bit signed integers for opuslib
            pcm_s16le_array = (pcm_float32_array.clip(-1.0, 1.0) * 32767.0).astype(np.int16)
            pcm_s16le_bytes = pcm_s16le_array.tobytes()
            
            # Encoder expects frame_size in samples per channel
            num_samples_per_channel = len(pcm_s16le_array) // self.channels
            if num_samples_per_channel != self.opus_frame_size_samples:
                logger.warning(f"PCM data for Opus encoding has {num_samples_per_channel} samples, but encoder expects {self.opus_frame_size_samples}. This might lead to issues.")
                # Consider padding or truncating here if strict frame sizes are required by Opus encoder for certain modes
            
            opus_frame = self.opus_encoder.encode(pcm_s16le_bytes, frame_size=num_samples_per_channel)
            return opus_frame
        except opuslib.OpusError as e:
            logger.error(f"Opus encoding error: {e}. PCM array shape: {pcm_float32_array.shape}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error during PCM to Opus encoding: {e}", exc_info=True)
            return None
            
    def extract_pcm_from_raw(self, audio_bytes: bytes, expected_dtype=np.int16) -> Optional[np.ndarray]:
        """Attempts to interpret raw audio bytes as PCM (defaulting to 16-bit signed int)."""
        if not audio_bytes: return None
        bytes_per_sample = np.dtype(expected_dtype).itemsize
        if len(audio_bytes) % bytes_per_sample != 0:
            logger.warning(f"Raw audio data length ({len(audio_bytes)}) is not a multiple of sample size ({bytes_per_sample}). Cannot interpret as raw {expected_dtype}.")
            return None
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=expected_dtype)
            # Normalize if int type, assuming it's PCM that needs to be in [-1,1] float range
            if np.issubdtype(audio_array.dtype, np.integer):
                max_val = np.iinfo(audio_array.dtype).max
                audio_array = audio_array.astype(np.float32) / max_val
            elif np.issubdtype(audio_array.dtype, np.floating) and (np.max(np.abs(audio_array)) > 1.0):
                 logger.warning("Raw float audio data seems to be out of [-1,1] range. Clipping.")
                 audio_array = np.clip(audio_array, -1.0, 1.0)

            logger.debug(f"Successfully interpreted raw bytes as {expected_dtype} PCM: {len(audio_array)} samples.")
            return audio_array
        except Exception as e:
            logger.error(f"Failed to interpret raw bytes as PCM: {e}", exc_info=True)
            return None

    async def _run_ffmpeg_conversion_internal_async(self, audio_data: bytes, input_args: List[str]) -> Optional[np.ndarray]:
        """Asynchronous helper to run FFmpeg conversion using asyncio.create_subprocess_exec."""
        if not self.ffmpeg_available:
            logger.error("FFmpeg not available for internal async conversion.")
            return None
        if not audio_data: return None

        ffmpeg_cmd = [
            'ffmpeg',
            *input_args,
            '-i', 'pipe:0',
            '-f', 's16le',
            '-ar', str(self.sample_rate),
            '-ac', str(self.channels),
            '-hide_banner',
            '-loglevel', 'error', # Changed to error to reduce noise, but FFmpeg warnings can be useful
            'pipe:1'
        ]
        logger.debug(f"Executing FFmpeg (async): {' '.join(ffmpeg_cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            # Increased timeout for potentially larger/complex conversions
            pcm_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(input=audio_data), timeout=15.0)
            
            if process.returncode != 0:
                stderr_str = stderr_bytes.decode('utf-8', errors='ignore').strip()
                logger.warning(f"FFmpeg async conversion failed (Code: {process.returncode}) for args {input_args}. Stderr: {stderr_str}")
                return None
            if not pcm_bytes:
                logger.warning(f"FFmpeg async conversion with args {input_args} produced no PCM data.")
                return None
            
            audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            logger.debug(f"FFmpeg async conversion success with args {input_args}: {len(pcm_bytes)}B PCM -> {len(audio_array)} samples.")
            return audio_array
        except asyncio.TimeoutError:
            logger.error(f"FFmpeg async conversion timed out after 15s for args {input_args}. Terminating process.")
            try: process.terminate() # Try graceful termination
            except ProcessLookupError: pass # Process might have already exited
            await asyncio.sleep(0.1) # Give it a moment
            if process.returncode is None: # Still running
                try: process.kill() # Force kill
                except ProcessLookupError: pass 
            return None
        except Exception as e:
            logger.error(f"FFmpeg async conversion with args {input_args} failed with exception: {e}", exc_info=True)
            if process.returncode is None:
                try: process.kill()
                except ProcessLookupError: pass
            return None

    def _has_valid_header(self, audio_bytes: bytes) -> bool:
        """Check if audio data has a valid container header."""
        if len(audio_bytes) < 20:
            return False
            
        # Check for WebM/Matroska (EBML header)
        if audio_bytes.startswith(b'\x1a\x45\xdf\xa3'):
            # Check for enough data after EBML header
            return len(audio_bytes) > 100
            
        # Check for WAV header
        if audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[:20]:
            return len(audio_bytes) > 44  # WAV header is 44 bytes minimum
            
        # Check for Ogg header
        if audio_bytes.startswith(b'OggS'):
            return len(audio_bytes) > 28  # Ogg page header minimum
            
        return False

    async def extract_pcm_smart_async(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """Asynchronous smart PCM extraction with chunk buffering for WebM fragments."""
        if not audio_bytes: 
            return None
            
        # Add incoming data to buffer
        self.chunk_buffer.extend(audio_bytes)
        logger.debug(f"Chunk buffer size: {len(self.chunk_buffer)} bytes")
        
        # If we're in stream mode (after first successful decode), try immediate processing
        if self.stream_mode_active:
            # In stream mode, try to process chunks more aggressively
            if len(self.chunk_buffer) >= 1000:  # Minimum 1KB for stream processing
                buffer_copy = bytes(self.chunk_buffer)
                self.chunk_buffer.clear()
                
                # Try stream processing with raw PCM format
                pcm_array = await self._run_ffmpeg_conversion_internal_async(buffer_copy, ['-f', 's16le'])
                if pcm_array is not None:
                    logger.debug(f"Stream mode processing successful: {len(buffer_copy)} bytes -> {len(pcm_array)} samples")
                    return pcm_array
                    
                # If stream processing fails, fall back to container format processing
                self.chunk_buffer = bytearray(buffer_copy)  # Restore buffer
        
        # Check if we have a valid header or enough data
        buffer_data = bytes(self.chunk_buffer)
        has_header = self._has_valid_header(buffer_data)
        has_enough_data = len(self.chunk_buffer) >= self.buffer_size_threshold
        
        if not has_header and not has_enough_data:
            logger.debug(f"Buffering: no header and insufficient data ({len(self.chunk_buffer)}/{self.buffer_size_threshold} bytes)")
            return None
            
        # Process the accumulated buffer
        logger.info(f"Processing accumulated buffer: {len(self.chunk_buffer)} bytes (header: {has_header})")
        
        # Make a copy and clear the buffer
        process_data = bytes(self.chunk_buffer)
        self.chunk_buffer.clear()
        
        # Try different format approaches
        pcm_array = None
        
        # 1. Try based on detected format
        if has_header:
            is_webm = process_data.startswith(b'\x1a\x45\xdf\xa3')
            is_wav = process_data.startswith(b'RIFF') and b'WAVE' in process_data[:20]
            is_ogg = process_data.startswith(b'OggS')
            
            if is_webm:
                logger.debug("Processing WebM/Matroska container")
                pcm_array = await self._run_ffmpeg_conversion_internal_async(process_data, ['-f', 'matroska'])
            elif is_wav:
                logger.debug("Processing WAV container")
                pcm_array = await self._run_ffmpeg_conversion_internal_async(process_data, ['-f', 'wav'])
            elif is_ogg:
                logger.debug("Processing Ogg container")
                pcm_array = await self._run_ffmpeg_conversion_internal_async(process_data, ['-f', 'ogg'])
                
        # 2. If no header or header processing failed, try WebM/Matroska (most common from browsers)
        if pcm_array is None:
            logger.debug("Trying WebM/Matroska format fallback")
            pcm_array = await self._run_ffmpeg_conversion_internal_async(process_data, ['-f', 'matroska'])
            
        # 3. Try auto-detection as last resort
        if pcm_array is None and self.ffmpeg_available:
            logger.debug("Trying FFmpeg auto-detection")
            pcm_array = await self._run_ffmpeg_conversion_internal_async(process_data, [])
            
        # 4. Final fallback: raw PCM interpretation
        if pcm_array is None:
            logger.debug("Trying raw PCM interpretation")
            pcm_array = self.extract_pcm_from_raw(process_data)
            
        if pcm_array is not None:
            logger.info(f"Successfully processed {len(process_data)} bytes -> {len(pcm_array)} samples")
            if not self.stream_mode_active:
                logger.info("Activating stream mode for subsequent chunks")
                self.stream_mode_active = True
            return pcm_array
        else:
            logger.warning(f"All processing methods failed for {len(process_data)} bytes")
            return None

    def extract_pcm_smart(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """
        Synchronous wrapper that just processes immediately without buffering.
        This method should be avoided in favor of extract_pcm_smart_async.
        """
        logger.warning("Using synchronous extract_pcm_smart - consider using async version for better buffering")
        
        # Basic format detection and processing without buffering
        if not audio_bytes: 
            return None
            
        is_webm = audio_bytes.startswith(b'\x1a\x45\xdf\xa3')
        is_wav = audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[:20]
        is_ogg = audio_bytes.startswith(b'OggS')
        
        # Try raw PCM first (fastest)
        pcm_array = self.extract_pcm_from_raw(audio_bytes)
        if pcm_array is not None:
            return pcm_array
            
        # Note: Cannot use async FFmpeg methods in sync context
        logger.warning("Synchronous processing limited - FFmpeg conversion requires async context")
        return None

    def reset_session(self):
        """Reset the audio service state for a new session."""
        self.chunk_buffer.clear()
        self.stream_mode_active = False
        logger.info("AudioService session state reset")

    def get_status(self) -> Dict[str, Any]:
        """Get audio service status."""
        return {
            "overall_available": self.is_available,
            "ffmpeg_available": self.ffmpeg_available,
            "opus_decoder_ready": self.opus_decoder is not None,
            "opus_encoder_ready": self.opus_encoder is not None,
            "configured_sample_rate": self.sample_rate,
            "configured_channels": self.channels,
            "opus_frame_duration_ms": self.opus_frame_duration_ms,
            "opus_frame_size_samples": self.opus_frame_size_samples,
            "chunk_buffer_size": len(self.chunk_buffer),
            "stream_mode_active": self.stream_mode_active
        }

    async def cleanup(self):
        """Clean up audio service resources."""
        logger.info("Cleaning up AudioService...")
        # Opuslib objects are C extensions, Python's GC should handle them when no longer referenced.
        self.opus_decoder = None
        self.opus_encoder = None
        self.chunk_buffer.clear()
        self.stream_mode_active = False
        self.is_available = False # Mark service as unavailable after cleanup
        logger.info("AudioService cleaned up.")