import logging
import numpy as np
import asyncio
import io
from typing import Optional, Tuple, Dict, Any, List
import opuslib
import soundfile as sf
import subprocess
import shutil

from ..app.config import settings, get_sample_rate, get_channels # Relative import for app config

logger = logging.getLogger(__name__) # Use standard logger

class AudioService:
    """Audio processing service for Opus decoding, PCM conversion, and format management"""
    
    def __init__(self, sample_rate: Optional[int] = None, channels: Optional[int] = None):
        self.sample_rate = sample_rate or get_sample_rate()
        self.channels = channels or get_channels()
        self.opus_decoder = None
        self.opus_encoder = None
        self.is_available = False
        self.ffmpeg_available = False
        
        self.audio_buffer = bytearray()
        # Calculate frame_size based on configured audio_frame_ms
        self.frame_size_samples = int(self.sample_rate * settings.audio_frame_ms / 1000)
        # Frame size in bytes for 16-bit PCM
        self.frame_size_bytes = self.frame_size_samples * self.channels * 2 
        
    async def initialize(self):
        """Initialize audio processing components"""
        try:
            logger.info("Initializing audio processing service...")
            
            await self._check_ffmpeg_availability()
            
            try:
                self.opus_decoder = opuslib.Decoder(fs=self.sample_rate, channels=self.channels)
                logger.info(f"✅ Opus decoder initialized (Sample Rate: {self.sample_rate}Hz, Channels: {self.channels})")
            except Exception as e:
                logger.error(f"Failed to initialize Opus decoder: {e}")
                self.opus_decoder = None
            
            try:
                self.opus_encoder = opuslib.Encoder(
                    fs=self.sample_rate, 
                    channels=self.channels, 
                    application=opuslib.APPLICATION_AUDIO # Use AUDIO for broader compatibility
                )
                # Configure encoder for voice, but be resilient to errors if some options are not supported
                try:
                    self.opus_encoder.bitrate = 32000
                    self.opus_encoder.signal = opuslib.SIGNAL_VOICE 
                    self.opus_encoder.complexity = 5 
                    self.opus_encoder.packet_loss_perc = 1 # Robustness for potential packet loss
                    self.opus_encoder.dtx = True # Discontinuous transmission for silence
                except Exception as enc_opt_e:
                    logger.warning(f"Could not set all Opus encoder parameters: {enc_opt_e}")
                logger.info(f"✅ Opus encoder initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Opus encoder: {e}")
                self.opus_encoder = None
            
            self.is_available = (self.opus_decoder is not None or self.opus_encoder is not None or self.ffmpeg_available)
            
            if self.is_available:
                logger.info(f"✅ Audio service initialized. FFmpeg: {'Available' if self.ffmpeg_available else 'Not Available'}. Opus Decoder: {'Yes' if self.opus_decoder else 'No'}. Opus Encoder: {'Yes' if self.opus_encoder else 'No'}.")
            else:
                logger.error("❌ Audio service failed to initialize with any working components (Opus/FFmpeg).")
            
        except Exception as e:
            logger.error(f"Critical failure during audio service initialization: {e}", exc_info=True)
            self.is_available = False
    
    async def _check_ffmpeg_availability(self):
        """Check if FFmpeg is available and properly installed"""
        try:
            ffmpeg_path = shutil.which('ffmpeg')
            if not ffmpeg_path:
                logger.warning("FFmpeg command not found in PATH. FFmpeg-based conversions will be unavailable.")
                self.ffmpeg_available = False
                return
            
            process = await asyncio.create_subprocess_exec(
                ffmpeg_path, '-version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                version_info = stdout.decode('utf-8', errors='ignore').split('\n')[0]
                logger.info(f"FFmpeg is available: {version_info.strip()}")
                self.ffmpeg_available = True
            else:
                err_msg = stderr.decode('utf-8', errors='ignore').strip()
                logger.error(f"FFmpeg test execution failed (Code: {process.returncode}): {err_msg}")
                self.ffmpeg_available = False
                
        except FileNotFoundError:
            logger.error("FFmpeg command not found. Please ensure FFmpeg is installed and in system PATH.")
            self.ffmpeg_available = False
        except Exception as e:
            logger.error(f"FFmpeg availability check encountered an error: {e}", exc_info=True)
            self.ffmpeg_available = False
    
    def decode_opus_frame(self, opus_frame: bytes) -> Optional[np.ndarray]:
        """Decode a single Opus frame to PCM."""
        if not self.opus_decoder:
            logger.warning("Opus decoder not initialized, cannot decode frame.")
            return None
        try:
            # Opus decoder expects frame_size in samples per channel
            pcm_data = self.opus_decoder.decode(opus_frame, frame_size=self.frame_size_samples)
            audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            logger.error(f"Opus decoding failed: {e}", exc_info=True)
            return None
    
    def encode_pcm_to_opus(self, pcm_data: np.ndarray) -> Optional[bytes]:
        """Encode PCM data (float32 numpy array) to Opus format."""
        if not self.opus_encoder:
            logger.warning("Opus encoder not initialized, cannot encode PCM.")
            return None
        try:
            if pcm_data.ndim > 1 and pcm_data.shape[1] != self.channels:
                logger.error(f"PCM data has {pcm_data.shape[1]} channels, encoder expects {self.channels}")
                return None # Or attempt mono conversion if appropriate

            pcm_int16 = (pcm_data * 32768.0).astype(np.int16)
            pcm_bytes = pcm_int16.tobytes()
            
            # Opus encoder's encode method expects number of samples per channel for frame_size
            num_samples_per_channel = len(pcm_int16) // self.channels
            opus_frame = self.opus_encoder.encode(pcm_bytes, frame_size=num_samples_per_channel)
            return opus_frame
        except Exception as e:
            logger.error(f"Opus encoding failed: {e}", exc_info=True)
            return None
    
    def extract_pcm_from_webm(self, webm_data: bytes) -> Optional[np.ndarray]:
        """Extract PCM audio from WebM/Opus data using ffmpeg."""
        return self._run_ffmpeg_conversion_internal(webm_data, ['-f', 'matroska'])

    def extract_pcm_from_raw(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """Attempt to interpret raw audio bytes as 16-bit PCM (little-endian)."""
        try:
            if len(audio_bytes) == 0 or len(audio_bytes) % 2 != 0:
                logger.debug("Raw PCM extraction: Empty or odd length data, cannot be 16-bit PCM.")
                return None
            
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Basic validation: check for extreme values which might indicate wrong format
            if np.any(np.abs(audio_array) > 1.5): # Allow some leeway for slightly clipped audio
                 logger.warning("Raw PCM extraction: Data contains values outside typical float32 range, might be incorrect format.")
                 # return None # Option: be stricter
            
            if len(audio_array) > 0:
                logger.debug(f"Raw PCM extraction successful: {len(audio_array)} samples.")
                return audio_array
            return None
        except Exception as e:
            logger.error(f"Raw PCM extraction failed: {e}", exc_info=True)
            return None
            
    def extract_pcm_smart(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """Smart PCM extraction with multiple fallbacks including common audio formats."""
        if not audio_bytes:
            logger.debug("Smart extract: Received empty audio bytes.")
            return None
            
        is_webm = audio_bytes.startswith(b'\x1a\x45\xdf\xa3')
        is_wav = audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[:20]
        is_ogg = audio_bytes.startswith(b'OggS')

        pcm_array: Optional[np.ndarray] = None

        if is_webm:
            logger.debug("Smart extract: WebM magic bytes detected. Attempting WebM/Matroska decode.")
            pcm_array = self._run_ffmpeg_conversion_internal(audio_bytes, ['-f', 'matroska'])
            if pcm_array is not None: return pcm_array

        if is_wav:
            logger.debug("Smart extract: WAV magic bytes detected. Attempting WAV decode.")
            pcm_array = self._run_ffmpeg_conversion_internal(audio_bytes, ['-f', 'wav'])
            if pcm_array is not None: return pcm_array
        
        if is_ogg:
            logger.debug("Smart extract: Ogg magic bytes detected. Attempting Ogg decode.")
            pcm_array = self._run_ffmpeg_conversion_internal(audio_bytes, ['-f', 'ogg'])
            if pcm_array is not None: return pcm_array

        # Fallback 1: Try WebM/Matroska even if magic bytes are not exact (e.g. for partial chunks)
        if len(audio_bytes) > 300: # Arbitrary threshold for "enough data for WebM"
            logger.debug("Smart extract: Fallback attempt with WebM/Matroska format.")
            pcm_array = self._run_ffmpeg_conversion_internal(audio_bytes, ['-f', 'matroska'])
            if pcm_array is not None: return pcm_array
                
        # Fallback 2: Try interpreting as raw PCM
        logger.debug("Smart extract: Fallback attempt with raw PCM interpretation.")
        pcm_array = self.extract_pcm_from_raw(audio_bytes)
        if pcm_array is not None: return pcm_array

        # Fallback 3: General FFmpeg auto-detection (most flexible but can be slow or error-prone)
        if self.ffmpeg_available and len(audio_bytes) > 500: # Need enough data for auto-detect
            logger.debug("Smart extract: Final fallback with FFmpeg auto-detection.")
            pcm_array = self._run_ffmpeg_conversion_internal(audio_bytes, []) # No input format specified
            if pcm_array is not None: return pcm_array

        logger.warning(f"All PCM extraction methods failed for audio chunk of {len(audio_bytes)} bytes.")
        return None

    def _run_ffmpeg_conversion_internal(self, audio_data: bytes, input_args: list) -> Optional[np.ndarray]:
        """Internal helper to run FFmpeg conversion and return PCM as numpy array."""
        if not self.ffmpeg_available:
            logger.error("FFmpeg not available, cannot perform FFmpeg-based conversion.")
            return None
        if not audio_data:
            logger.debug("FFmpeg internal: Received empty audio data for conversion.")
            return None

        ffmpeg_cmd = [
            'ffmpeg',
            *input_args,      # e.g., ['-f', 'matroska'] or [] for auto-detect
            '-i', 'pipe:0',   # Input from stdin
            '-f', 's16le',    # Output format: signed 16-bit little-endian PCM
            '-ar', str(self.sample_rate),
            '-ac', str(self.channels),
            '-hide_banner',   # Suppress version and build info
            '-loglevel', 'warning', # Log warnings and errors from FFmpeg
            'pipe:1'          # Output to stdout
        ]
        
        logger.debug(f"Executing FFmpeg: {' '.join(ffmpeg_cmd)}")
        
        process = None
        try:
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            # Setting a timeout (e.g., 5 seconds) for FFmpeg to prevent indefinite blocking.
            # Adjust timeout based on expected conversion times for typical chunk sizes.
            pcm_bytes, stderr_bytes = process.communicate(input=audio_data, timeout=10.0) 
            
            if process.returncode != 0:
                stderr_str = stderr_bytes.decode('utf-8', errors='ignore').strip()
                logger.warning(f"FFmpeg conversion failed (Code: {process.returncode}) with args {input_args}. Stderr: {stderr_str}")
                return None
            if not pcm_bytes:
                logger.warning(f"FFmpeg conversion with args {input_args} produced no PCM data, though exited successfully.")
                return None
            
            audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            logger.debug(f"FFmpeg conversion success with args {input_args}: {len(pcm_bytes)} bytes PCM -> {len(audio_array)} samples.")
            return audio_array
                
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg conversion timed out after 10s for args {input_args}. Killing process.")
            if process: process.kill()
            # Ensure communicate is called again to free resources after kill
            try:
                 if process: process.communicate()
            except: pass # Ignore errors on second communicate
            return None
        except Exception as e:
            logger.error(f"FFmpeg conversion with args {input_args} failed with exception: {e}", exc_info=True)
            if process and process.poll() is None: # If process still running after exception
                process.kill()
                try: process.communicate()
                except: pass
            return None
    
    # Other methods like resample_audio, normalize_audio, apply_silence_detection, get_audio_stats, get_status
    # would remain largely the same as their logic is independent of these specific fixes.
    # For brevity, they are not repeated here unless a change was directly implied.

    def get_status(self) -> Dict[str, Any]:
        """Get audio service status"""
        return {
            "available": self.is_available,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frame_size_samples": self.frame_size_samples,
            "frame_size_bytes": self.frame_size_bytes,
            "opus_decoder_available": self.opus_decoder is not None,
            "opus_encoder_available": self.opus_encoder is not None,
            "ffmpeg_available": self.ffmpeg_available
        }

    async def cleanup(self):
        """Clean up audio service resources"""
        try:
            if self.opus_decoder:
                # opuslib objects don't have explicit del/close. Rely on GC.
                self.opus_decoder = None
            if self.opus_encoder:
                self.opus_encoder = None
                
            self.audio_buffer.clear()
            self.is_available = False
            logger.info("Audio service cleaned up.")
        except Exception as e:
            logger.error(f"Audio service cleanup error: {e}", exc_info=True)