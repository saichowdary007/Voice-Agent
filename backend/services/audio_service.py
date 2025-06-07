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
            try: 
                process.terminate() # Try graceful termination
            except ProcessLookupError:
                pass # Process might have already exited
            except Exception as e:
                logger.error(f"Error terminating FFmpeg process: {e}")
                
            await asyncio.sleep(0.1) # Give it a moment
            if process.returncode is None: # Still running
                try:
                    process.kill() # Force kill
                except ProcessLookupError:
                    pass
                except Exception as e:
                    logger.error(f"Error killing FFmpeg process: {e}")
            return None
        except Exception as e:
            logger.error(f"FFmpeg async conversion with args {input_args} failed with exception: {e}", exc_info=True)
            if process.returncode is None:
                try:
                    process.kill() # Ensure process is killed on any error
                except Exception as kill_error:
                    logger.error(f"Error killing FFmpeg process after exception: {kill_error}")
            return None

    def _has_valid_header(self, audio_bytes: bytes) -> bool:
        """Check for basic valid headers for common formats."""
        if not audio_bytes: return False
        # WAV check
        if audio_bytes.startswith(b'RIFF') and len(audio_bytes) > 12 and audio_bytes[8:12] == b'WAVE':
            return True
        # WebM check (EBML header)
        if audio_bytes.startswith(b'\x1aE\xdf\xa3'):
            return True
        # Ogg check
        if audio_bytes.startswith(b'OggS'):
            return True
        return False
        
    def _detect_audio_format(self, audio_bytes: bytes) -> str:
        """Detects audio format based on magic bytes."""
        if not audio_bytes:
            return "unknown"
            
        # Basic magic byte detection
        if audio_bytes.startswith(b'RIFF') and len(audio_bytes) > 12 and audio_bytes[8:12] == b'WAVE':
            self.format_stats["total_attempts"] += 1
            return "wav"
        if audio_bytes.startswith(b'\x1aE\xdf\xa3'):
            self.format_stats["total_attempts"] += 1
            return "webm"
        if audio_bytes.startswith(b'OggS'):
            self.format_stats["total_attempts"] += 1
            return "ogg"
        
        # Check for raw Opus based on a reasonable byte length (heuristic)
        if len(audio_bytes) > 20 and len(audio_bytes) < 1500: # Typical Opus packet size
             # This is a weak check, but can be a hint. Stronger validation is in the decoder.
             return "opus"

        return "unknown"

    async def extract_pcm_smart_async(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        Intelligently tries to extract PCM from audio data using format detection and fallback mechanisms.
        This is the primary public method for converting incoming audio chunks to PCM.
        """
        if not audio_data:
            return None
            
        self.chunk_buffer.extend(audio_data)
        logger.debug(f"Received {len(audio_data)} bytes. Buffer size is now {len(self.chunk_buffer)} bytes.")

        # If buffer is too small, wait for more data, unless it's the end of a stream.
        if len(self.chunk_buffer) < 100: # Need at least a minimal amount for header detection
             logger.debug("Buffer too small, waiting for more data.")
             return None

        # Create a memory-copy of the buffer to work with
        current_chunk = bytes(self.chunk_buffer)

        # Attempt to process the chunk using format-priority processing
        pcm_data = await self._extract_pcm_from_format(current_chunk)

        if pcm_data is not None and pcm_data.size > 0:
            logger.info(f"Successfully processed {len(current_chunk)} bytes of audio into {pcm_data.size} PCM samples.")
            # Clear the buffer since we successfully processed it
            self.chunk_buffer.clear()
            self.last_processed_chunk_size = len(current_chunk)
            
            # Reset failed chunks buffer upon a success
            if self.failed_chunks:
                logger.info(f"Clearing {len(self.failed_chunks)} previously failed chunks after successful processing.")
                self.failed_chunks.clear()
                self.failed_chunks_buffer_size = 0
            return pcm_data
        else:
            logger.warning(f"Failed to extract PCM from chunk of size {len(current_chunk)}. Buffering and will retry.")
            # If processing failed, we keep the data in the buffer and wait for more.
            # This is crucial for formats like WebM that might send header and data separately.
            if len(self.chunk_buffer) > self.buffer_size_threshold:
                logger.error(f"Buffer size ({len(self.chunk_buffer)}) exceeds threshold ({self.buffer_size_threshold}) after failed decode. Handling as a failed chunk.")
                self._handle_failed_chunk(bytes(self.chunk_buffer))
                self.chunk_buffer.clear()
            return None

    async def _process_chunk_by_format(self, audio_data: bytes, format_name: str) -> Optional[np.ndarray]:
        """Helper to process a chunk with a specific format decoder."""
        logger.debug(f"Attempting to decode chunk as {format_name}...")
        start_time = time.monotonic()
        pcm_data = None
        
        try:
            if format_name == "wav":
                # Use soundfile for robust WAV parsing
                with io.BytesIO(audio_data) as bio:
                    data, sr = sf.read(bio)
                    # Ensure correct sample rate and convert to float32 if needed
                    if sr != self.sample_rate:
                        logger.warning(f"WAV file has sample rate {sr}, but service is configured for {self.sample_rate}. This may cause issues.")
                    pcm_data = data.astype(np.float32)
            elif format_name == "webm":
                # Use FFmpeg for WebM
                pcm_data = await self._run_ffmpeg_conversion_internal_async(audio_data, ['-f', 'webm'])
            elif format_name == "ogg":
                # Use FFmpeg for Ogg
                pcm_data = await self._run_ffmpeg_conversion_internal_async(audio_data, ['-f', 'ogg'])
            elif format_name == "opus":
                # Use internal Opus decoder
                pcm_data = self.decode_opus_frame(audio_data)
            elif format_name == "raw":
                # Fallback to raw PCM
                pcm_data = self.extract_pcm_from_raw(audio_data)
        except Exception as e:
            logger.error(f"Error decoding chunk as {format_name}: {e}", exc_info=False) # Keep log cleaner on expected failures
            return None

        if pcm_data is not None and pcm_data.size > 0:
            duration = time.monotonic() - start_time
            logger.info(f"Successfully decoded as {format_name} in {duration:.3f}s")
            self.format_stats[f"{format_name}_success"] += 1
            return pcm_data
        
        return None

    def _handle_failed_chunk(self, chunk_data: bytes) -> None:
        """Handles a chunk that could not be processed."""
        self.format_stats["failures"] += 1
        
        if len(self.failed_chunks) >= self.max_failed_chunks:
            logger.error(f"Max failed chunks ({self.max_failed_chunks}) reached. Dropping oldest failed chunk.")
            oldest_chunk = self.failed_chunks.pop(0)
            self.failed_chunks_buffer_size -= len(oldest_chunk)
        
        if self.failed_chunks_buffer_size + len(chunk_data) > self.max_failed_chunks_buffer_size:
            logger.error(f"Failed chunks buffer size exceeds limit ({self.max_failed_chunks_buffer_size} bytes). Dropping chunk.")
            return
            
        self.failed_chunks.append(chunk_data)
        self.failed_chunks_buffer_size += len(chunk_data)
        logger.warning(f"Stored failed chunk. Total failed chunks: {len(self.failed_chunks)}, buffer size: {self.failed_chunks_buffer_size} bytes.")

    async def _extract_pcm_from_format(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        Format-priority processing pipeline. Tries to decode audio by checking for common formats in a specific order.
        Order of attempts: WAV -> WebM -> Ogg -> Raw PCM (as fallback).
        """
        if not audio_data:
            return None

        # 1. Attempt WAV decoding (high-quality, common)
        if audio_data.startswith(b'RIFF'):
            pcm_data = await self._process_chunk_by_format(audio_data, "wav")
            if pcm_data is not None:
                return pcm_data

        # 2. Attempt WebM decoding (common for web-based clients)
        if audio_data.startswith(b'\x1aE\xdf\xa3'):
            pcm_data = await self._process_chunk_by_format(audio_data, "webm")
            if pcm_data is not None:
                return pcm_data
        
        # 3. Attempt Ogg decoding (another common web format)
        if audio_data.startswith(b'OggS'):
            pcm_data = await self._process_chunk_by_format(audio_data, "ogg")
            if pcm_data is not None:
                return pcm_data
        
        # 4. Fallback for Auto-detection (let FFmpeg figure it out, can be slow)
        # This is a broader attempt if specific headers aren't matched.
        logger.debug("Specific format checks failed, attempting auto-detection with FFmpeg.")
        pcm_data_auto = await self._run_ffmpeg_conversion_internal_async(audio_data, []) # No input args, let FFmpeg detect
        if pcm_data_auto is not None:
             self.format_stats["auto_success"] += 1
             return pcm_data_auto

        # 5. Final fallback: Try to interpret as raw PCM data
        logger.warning("All container format decoders failed. Attempting to interpret as raw PCM.")
        pcm_data_raw = await self._process_chunk_by_format(audio_data, "raw")
        if pcm_data_raw is not None:
            return pcm_data_raw
            
        logger.error(f"All decoding attempts failed for chunk of size {len(audio_data)}.")
        return None

    def get_last_processed_chunk_size(self) -> int:
        """Returns the size of the last chunk that was successfully processed."""
        return self.last_processed_chunk_size
        
    def get_buffered_chunk_size(self) -> int:
        """Returns the current size of the unprocessed chunk buffer."""
        return len(self.chunk_buffer)

    def has_failed_chunks(self) -> bool:
        """Returns True if there are any chunks that failed to process."""
        return len(self.failed_chunks) > 0

    def get_failed_chunks_info(self) -> Dict[str, Any]:
        """Returns statistics about the failed chunks."""
        return {
            "count": len(self.failed_chunks),
            "total_size": self.failed_chunks_buffer_size,
            "max_count": self.max_failed_chunks,
            "max_size": self.max_failed_chunks_buffer_size
        }
        
    def clear_failed_chunks(self) -> None:
        """Clears the buffer of failed chunks."""
        if self.failed_chunks:
            logger.info(f"Clearing {len(self.failed_chunks)} failed chunks ({self.failed_chunks_buffer_size} bytes).")
            self.failed_chunks.clear()
            self.failed_chunks_buffer_size = 0

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