import logging
import numpy as np
import asyncio
import io
from typing import Optional, Tuple, Dict, Any, List
import opuslib # Ensure this is installed: pip install opuslib
import soundfile as sf # Ensure this is installed: pip install soundfile
import subprocess
import shutil # For shutil.which
import os
import time
import tempfile
from dataclasses import dataclass

# Import for app config, using absolute path for Docker compatibility
from backend.app.config import settings, get_sample_rate, get_channels

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
        self.last_processed_chunk_size: int = 0  # Track size of last successfully processed chunk
        
        # Improved error handling for failed chunks
        self.failed_chunks: List[bytes] = []
        self.max_failed_chunks: int = settings.max_failed_chunk_retries
        self.failed_chunks_buffer_size: int = 0
        self.max_failed_chunks_buffer_size: int = settings.failed_chunk_buffer_max_size
        
        # Frame size in samples per channel for Opus codec
        self.opus_frame_duration_ms: int = settings.frame_duration_ms # Use configured frame duration for Opus
        self.opus_frame_size_samples: int = int(self.sample_rate * self.opus_frame_duration_ms / 1000)
        
        # Frame size in bytes for general PCM processing (16-bit)
        self.pcm_frame_size_bytes: int = self.opus_frame_size_samples * self.channels * 2 # 2 bytes per sample for 16-bit
        
        # Format detection stats
        self.format_stats = {
            "wav_success": 0,
            "webm_success": 0,
            "ogg_success": 0,
            "auto_success": 0,
            "raw_success": 0,
            "total_attempts": 0,
            "failures": 0
        }
        
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
            return len(audio_bytes) > 28  # Ogg page header is 28 bytes minimum
            
        return False

    def _detect_audio_format(self, audio_bytes: bytes) -> str:
        """Enhanced audio format detection using magic bytes."""
        if not audio_bytes or len(audio_bytes) < 12:
            return "unknown"
            
        # WAV detection (RIFF header)
        if audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[:20]:
            return "wav"
            
        # WebM detection (EBML header)
        if audio_bytes.startswith(b'\x1a\x45\xdf\xa3'):
            return "webm"
            
        # Ogg detection
        if audio_bytes.startswith(b'OggS'):
            return "ogg"
            
        # MP3 detection (ID3 or MP3 frame sync)
        if audio_bytes.startswith(b'ID3') or (audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
            return "mp3"
            
        # Raw PCM detection (heuristic - check if data looks like 16-bit PCM)
        if len(audio_bytes) % 2 == 0:  # Must be even number of bytes for 16-bit
            # Check if data has reasonable variance for audio
            try:
                pcm_array = np.frombuffer(audio_bytes, dtype=np.int16)
                if len(pcm_array) > 10:
                    std_dev = np.std(pcm_array)
                    if 100 < std_dev < 20000:  # Typical range for voice audio
                        return "raw_pcm"
            except:
                pass
                
        return "unknown"

    async def extract_pcm_smart_async(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        Enhanced smart PCM extraction with stream mode and format prioritization.
        
        Args:
            audio_data: Raw audio bytes in any supported format
            
        Returns:
            Normalized float32 PCM data as numpy array or None if extraction failed
        """
        self.format_stats["total_attempts"] += 1
        
        if not audio_data:
            logger.warning("Received empty audio data")
            self.format_stats["failures"] += 1
            return None
            
        # Small chunk handling for stream mode
        if self.stream_mode_active and len(audio_data) >= settings.small_chunk_buffer_threshold:
            # When in stream mode, attempt to process smaller chunks immediately
            # without buffering if they're above the minimum size threshold
            logger.debug(f"Stream mode active, processing {len(audio_data)} bytes directly")
            
            # Detect format based on previous successful chunk
            detected_format = self._detect_audio_format(audio_data)
            
            if detected_format != "unknown":
                # Try to process the chunk directly based on detected format
                pcm_array = await self._process_chunk_by_format(audio_data, detected_format)
                if pcm_array is not None:
                    return pcm_array
        
        # Regular processing with intelligent buffering
        # If not in stream mode or direct processing failed, use the regular buffer approach
        if len(self.chunk_buffer) > 0:
            # Append the new data to existing buffer
            self.chunk_buffer.extend(audio_data)
            
            # Process if buffer is large enough or we're in stream mode
            if len(self.chunk_buffer) >= self.buffer_size_threshold or self.stream_mode_active:
                # Copy buffer to avoid reference issues and clear buffer immediately
                buffer_copy = bytes(self.chunk_buffer)
                self.chunk_buffer.clear()
                
                # Multi-method fallback chain with prioritized formats
                pcm_array = await self._extract_pcm_from_format(buffer_copy)
                
                if pcm_array is not None:
                    # Stream mode becomes active after first successful decode
                    if not self.stream_mode_active:
                        self.stream_mode_active = True
                        logger.info("🔄 Stream mode activated - will process future chunks immediately")
                        
                    # Store buffer size for stats
                    self.last_processed_chunk_size = len(buffer_copy)
                    return pcm_array
                else:
                    # Processing failed, handle the failed chunk
                    logger.warning(f"All extraction methods failed for buffered chunk of {len(buffer_copy)} bytes")
                    self._handle_failed_chunk(buffer_copy)
                    self.format_stats["failures"] += 1
                    return None
        else:
            # First chunk or buffer was cleared, check if large enough to process directly
            if len(audio_data) >= self.buffer_size_threshold:
                pcm_array = await self._extract_pcm_from_format(audio_data)
                
                if pcm_array is not None:
                    # Stream mode becomes active after first successful decode
                    if not self.stream_mode_active:
                        self.stream_mode_active = True
                        logger.info("🔄 Stream mode activated - will process future chunks immediately")
                        
                    self.last_processed_chunk_size = len(audio_data)
                    return pcm_array
                else:
                    # Processing failed, handle the failed chunk
                    logger.warning(f"All extraction methods failed for direct chunk of {len(audio_data)} bytes")
                    self._handle_failed_chunk(audio_data)
                    self.format_stats["failures"] += 1
                    return None
            else:
                # Too small to process reliably, add to buffer
                self.chunk_buffer.extend(audio_data)
                logger.debug(f"Added {len(audio_data)} bytes to buffer (now {len(self.chunk_buffer)})")
                return None
    
    async def _process_chunk_by_format(self, audio_data: bytes, format_name: str) -> Optional[np.ndarray]:
        """Process a chunk directly based on detected format"""
        if format_name == "wav":
            logger.debug("Attempting to decode WAV chunk with soundfile")
            try:
                with io.BytesIO(audio_data) as f:
                    pcm_array, sr = sf.read(f, dtype='float32')
                if sr != self.sample_rate:
                    logger.warning(f"Input WAV sample rate {sr} does not match service rate {self.sample_rate}. Audio quality might be affected.")
                self.format_stats["wav_success"] += 1
                logger.debug("Successfully decoded WAV with soundfile.")
                return pcm_array
            except Exception as e:
                logger.warning(f"Soundfile WAV decoding failed: {e}. Falling back to FFmpeg.")
                pcm_array = await self._run_ffmpeg_conversion_internal_async(
                    audio_data, ['-f', 'wav']
                )
                if pcm_array is not None:
                    self.format_stats["wav_success"] += 1
                    return pcm_array
                
        elif format_name == "webm":
            # Try WebM
            logger.debug("Attempting to decode WebM chunk with FFmpeg")
            pcm_array = await self._run_ffmpeg_conversion_internal_async(
                audio_data, ['-f', 'webm']
            )
            if pcm_array is not None:
                self.format_stats["webm_success"] += 1
                return pcm_array
                
        elif format_name == "ogg":
            # Try Ogg
            logger.debug("Attempting to decode Ogg chunk with FFmpeg")
            pcm_array = await self._run_ffmpeg_conversion_internal_async(
                audio_data, ['-f', 'ogg']
            )
            if pcm_array is not None:
                self.format_stats["ogg_success"] += 1
                return pcm_array
                
        # Fall back to auto-detect
        logger.debug("Falling back to FFmpeg auto-detect for chunk")
        pcm_array = await self._run_ffmpeg_conversion_internal_async(
            audio_data, []  # No format specified, let FFmpeg auto-detect
        )
        if pcm_array is not None:
            self.format_stats["auto_success"] += 1
            return pcm_array
            
        return None
        
    def _handle_failed_chunk(self, chunk_data: bytes) -> None:
        """Store failed chunks for potential retry if they're not too large"""
        if len(self.failed_chunks) >= self.max_failed_chunks:
            # Remove oldest chunk if we've reached the limit
            if self.failed_chunks:
                oldest_chunk = self.failed_chunks.pop(0)
                self.failed_chunks_buffer_size -= len(oldest_chunk)
                
        # Check if adding this chunk would exceed the max buffer size
        if self.failed_chunks_buffer_size + len(chunk_data) <= self.max_failed_chunks_buffer_size:
            self.failed_chunks.append(chunk_data)
            self.failed_chunks_buffer_size += len(chunk_data)
            logger.debug(f"Stored failed chunk of {len(chunk_data)} bytes for retry (buffer: {len(self.failed_chunks)} chunks, {self.failed_chunks_buffer_size} bytes)")
        else:
            logger.warning(f"Failed chunk of {len(chunk_data)} bytes exceeds buffer limit, discarding")

    async def _extract_pcm_from_format(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        Enhanced PCM extraction with format prioritization and retries.
        
        Attempts multiple methods in a prioritized order:
        1. WAV (most reliable)
        2. WebM (common from browsers)
        3. Ogg (alternative container)
        4. Auto-detect
        5. Raw PCM (last resort)
        """
        if not audio_data:
            return None
            
        # Get detected format to help prioritize
        format_name = self._detect_audio_format(audio_data)
        logger.debug(f"Detected format: {format_name} for {len(audio_data)} bytes")
        
        # Method 1: Try WAV format first (generally most reliable)
        if format_name == "wav" or format_name == "unknown":
            logger.debug("Attempting to decode as WAV with soundfile")
            try:
                with io.BytesIO(audio_data) as f:
                    pcm_array, sr = sf.read(f, dtype='float32')
                if sr != self.sample_rate:
                    logger.warning(f"Input WAV sample rate {sr} does not match service rate {self.sample_rate}. Audio quality might be affected.")
                self.format_stats["wav_success"] += 1
                logger.debug("Successfully decoded WAV with soundfile.")
                return pcm_array
            except Exception as e:
                logger.warning(f"Soundfile WAV decoding failed: {e}. Trying FFmpeg for WAV.")
                pcm_array = await self._run_ffmpeg_conversion_internal_async(
                    audio_data, ['-f', 'wav']
                )
                if pcm_array is not None:
                    self.format_stats["wav_success"] += 1
                    return pcm_array
        
        # Method 2: Try WebM format (most common from browsers)
        if format_name == "webm" or format_name == "unknown":
            logger.debug("Attempting to decode as WebM with FFmpeg")
            pcm_array = await self._run_ffmpeg_conversion_internal_async(
                audio_data, ['-f', 'webm']
            )
            if pcm_array is not None:
                self.format_stats["webm_success"] += 1
                return pcm_array
                
        # Method 3: Try Ogg format
        if format_name == "ogg" or format_name == "unknown":
            logger.debug("Attempting to decode as Ogg with FFmpeg")
            pcm_array = await self._run_ffmpeg_conversion_internal_async(
                audio_data, ['-f', 'ogg']
            )
            if pcm_array is not None:
                self.format_stats["ogg_success"] += 1
                return pcm_array
        
        # Method 4: Try auto-detect format as a fallback
        logger.debug("Falling back to FFmpeg auto-detect")
        pcm_array = await self._run_ffmpeg_conversion_internal_async(
            audio_data, []  # No format specified, let FFmpeg auto-detect
        )
        if pcm_array is not None:
            self.format_stats["auto_success"] += 1
            return pcm_array
            
        # Method 5: Last resort - try as raw PCM
        if format_name == "raw_pcm" or format_name == "unknown":
            pcm_array = self.extract_pcm_from_raw(audio_data)
            if pcm_array is not None:
                self.format_stats["raw_success"] += 1
                return pcm_array
                
        # All methods failed
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