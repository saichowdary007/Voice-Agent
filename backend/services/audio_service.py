import logging
import numpy as np
import asyncio
import io
from typing import Optional, Tuple, Dict, Any, List
import opuslib
import soundfile as sf
import subprocess
import shutil

from ..app.config import settings, get_sample_rate, get_channels

logger = logging.getLogger(__name__)

class AudioService:
    """Audio processing service for Opus decoding, PCM conversion, and format management"""
    
    def __init__(self, sample_rate: Optional[int] = None, channels: Optional[int] = None):
        # Use standardized configuration
        self.sample_rate = sample_rate or get_sample_rate()
        self.channels = channels or get_channels()
        self.opus_decoder = None
        self.opus_encoder = None
        self.is_available = False
        self.ffmpeg_available = False
        
        # Audio buffers
        self.audio_buffer = bytearray()
        self.frame_size = int(self.sample_rate * settings.audio_frame_ms / 1000)  # Frame size based on config
        
    async def initialize(self):
        """Initialize audio processing components"""
        try:
            logger.info("Initializing audio processing service...")
            
            # Check FFmpeg availability first
            await self._check_ffmpeg_availability()
            
            # Initialize Opus codec
            self.opus_decoder = opuslib.Decoder(fs=self.sample_rate, channels=self.channels)
            self.opus_encoder = opuslib.Encoder(fs=self.sample_rate, channels=self.channels, application=opuslib.APPLICATION_VOIP)
            
            # Set encoder parameters for low latency
            self.opus_encoder.bitrate = 32000  # 32 kbps for voice
            self.opus_encoder.signal = opuslib.SIGNAL_VOICE
            self.opus_encoder.complexity = 5  # Balance between quality and CPU
            
            self.is_available = True
            logger.info(f"✅ Audio service initialized successfully - Sample rate: {self.sample_rate}Hz, Channels: {self.channels}, FFmpeg: {self.ffmpeg_available}")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio service: {e}")
            self.is_available = False
    
    async def _check_ffmpeg_availability(self):
        """Check if FFmpeg is available and properly installed"""
        try:
            # Check if ffmpeg is in PATH
            ffmpeg_path = shutil.which('ffmpeg')
            if not ffmpeg_path:
                logger.error("FFmpeg not found in PATH. Please ensure FFmpeg is installed.")
                self.ffmpeg_available = False
                return
            
            # Test FFmpeg execution
            result = await asyncio.create_subprocess_exec(
                'ffmpeg', '-version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                version_info = stdout.decode('utf-8').split('\n')[0]
                logger.info(f"FFmpeg available: {version_info}")
                self.ffmpeg_available = True
            else:
                logger.error(f"FFmpeg test failed: {stderr.decode('utf-8')}")
                self.ffmpeg_available = False
                
        except Exception as e:
            logger.error(f"FFmpeg availability check failed: {e}")
            self.ffmpeg_available = False
    
    def decode_opus_frame(self, opus_frame: bytes) -> Optional[np.ndarray]:
        """
        Decode a single Opus frame to PCM
        
        Args:
            opus_frame: Opus-encoded audio frame
            
        Returns:
            PCM audio data as numpy array or None if decoding fails
        """
        if not self.is_available or not self.opus_decoder:
            return None
            
        try:
            # Decode Opus frame to PCM
            pcm_data = self.opus_decoder.decode(opus_frame, frame_size=self.frame_size)
            
            # Convert to numpy array
            audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Opus decoding failed: {e}")
            return None
    
    def encode_pcm_to_opus(self, pcm_data: np.ndarray) -> Optional[bytes]:
        """
        Encode PCM data to Opus format
        
        Args:
            pcm_data: PCM audio data as numpy array
            
        Returns:
            Opus-encoded bytes or None if encoding fails
        """
        if not self.is_available or not self.opus_encoder:
            return None
            
        try:
            # Convert numpy array to int16 bytes
            pcm_int16 = (pcm_data * 32768.0).astype(np.int16)
            pcm_bytes = pcm_int16.tobytes()
            
            # Encode to Opus
            opus_frame = self.opus_encoder.encode(pcm_bytes, frame_size=len(pcm_data))
            
            return opus_frame
            
        except Exception as e:
            logger.error(f"Opus encoding failed: {e}")
            return None
    
    def process_webm_chunk(self, webm_data: bytes) -> List[np.ndarray]:
        """
        Process WebM/Opus audio chunk and extract PCM frames
        
        Args:
            webm_data: WebM container data with Opus audio
            
        Returns:
            List of PCM audio arrays
        """
        audio_frames = []
        
        if not self.is_available or not self.ffmpeg_available:
            return audio_frames
            
        try:
            # Use ffmpeg to extract raw Opus frames from WebM
            process = subprocess.Popen([
                'ffmpeg', '-i', 'pipe:0', '-f', 'opus', '-c:a', 'copy', 'pipe:1'
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            opus_data, stderr = process.communicate(input=webm_data)
            
            if process.returncode == 0 and opus_data:
                # Extract individual Opus frames (simplified approach)
                # In a full implementation, you'd parse the Opus stream properly
                frame_size_bytes = 960 * 2  # Approximate size for 20ms at 48kHz
                
                for i in range(0, len(opus_data), frame_size_bytes):
                    frame = opus_data[i:i + frame_size_bytes]
                    if len(frame) == frame_size_bytes:
                        pcm_frame = self.decode_opus_frame(frame)
                        if pcm_frame is not None:
                            audio_frames.append(pcm_frame)
            
        except Exception as e:
            logger.debug(f"WebM processing failed: {e}")
            
        return audio_frames
    
    def convert_audio_format(self, audio_data: bytes, input_format: str, output_format: str) -> Optional[bytes]:
        """
        Convert audio between different formats using ffmpeg
        
        Args:
            audio_data: Input audio data
            input_format: Input format (e.g., 'webm', 'opus', 'wav')
            output_format: Output format (e.g., 'pcm_s16le', 'opus', 'wav')
            
        Returns:
            Converted audio data or None if conversion fails
        """
        if not self.ffmpeg_available:
            logger.error("FFmpeg not available for audio conversion")
            return None
            
        try:
            # Build ffmpeg command
            cmd = [
                'ffmpeg', '-f', input_format, '-i', 'pipe:0',
                '-f', output_format, '-ar', str(self.sample_rate),
                '-ac', str(self.channels), 'pipe:1'
            ]
            
            process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            
            stdout, stderr = process.communicate(input=audio_data)
            
            if process.returncode == 0:
                return stdout
            else:
                logger.error(f"Audio conversion failed: {stderr.decode()}")
                return None
                
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return None
    
    def extract_pcm_from_webm(self, webm_data: bytes) -> Optional[np.ndarray]:
        """
        Extract PCM audio from WebM/Opus data using ffmpeg
        
        Args:
            webm_data: Raw WebM bytes
            
        Returns:
            Numpy array of PCM audio samples or None if failed
        """
        try:
            if len(webm_data) == 0:
                return None
            
            # Use ffmpeg to convert WebM/Opus to raw PCM
            # Updated command to properly handle Opus streams
            cmd = [
                'ffmpeg', 
                '-f', 'matroska',  # Input format
                '-i', 'pipe:0',    # Input from stdin
                '-f', 's16le',     # Output format (signed 16-bit little endian)
                '-acodec', 'pcm_s16le',  # Audio codec
                '-ar', str(self.sample_rate),  # Sample rate
                '-ac', str(self.channels),     # Number of channels
                '-loglevel', 'quiet',          # Suppress logs
                'pipe:1'           # Output to stdout
            ]
            
            process = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            pcm_data, error = process.communicate(input=webm_data)
            
            if process.returncode != 0:
                # If direct conversion fails, try with format detection
                logger.debug(f"Direct conversion failed, trying format detection: {error.decode()}")
                
                cmd_alt = [
                    'ffmpeg',
                    '-i', 'pipe:0',    # Let ffmpeg detect format
                    '-f', 's16le',     # Output format
                    '-ar', str(self.sample_rate),
                    '-ac', str(self.channels),
                    '-acodec', 'pcm_s16le',
                    '-loglevel', 'quiet',
                    'pipe:1'
                ]
                
                process = subprocess.Popen(
                    cmd_alt,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                pcm_data, error = process.communicate(input=webm_data)
                
                if process.returncode != 0:
                    logger.debug(f"Alternative conversion also failed: {error.decode()}")
                    return None
            
            if len(pcm_data) == 0:
                return None
            
            # Convert bytes to numpy array
            audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Validate audio data
            if len(audio_array) == 0:
                return None
                
            return audio_array
            
        except Exception as e:
            logger.debug(f"PCM extraction failed: {e}")
            return None
    
    def resample_audio(self, audio_data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        """
        Resample audio to target sample rate
        
        Args:
            audio_data: Input audio data
            source_rate: Source sample rate
            target_rate: Target sample rate
            
        Returns:
            Resampled audio data
        """
        if source_rate == target_rate:
            return audio_data
            
        try:
            from scipy import signal
            
            # Calculate resampling ratio
            ratio = target_rate / source_rate
            num_samples = int(len(audio_data) * ratio)
            
            # Resample using scipy
            resampled = signal.resample(audio_data, num_samples)
            
            return resampled.astype(np.float32)
            
        except ImportError:
            logger.warning("scipy not available for resampling, using simple interpolation")
            # Simple linear interpolation fallback
            ratio = target_rate / source_rate
            indices = np.arange(0, len(audio_data), 1/ratio)
            return np.interp(indices, np.arange(len(audio_data)), audio_data)
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            return audio_data
    
    def normalize_audio(self, audio_data: np.ndarray, target_level: float = 0.8) -> np.ndarray:
        """
        Normalize audio levels
        
        Args:
            audio_data: Input audio data
            target_level: Target peak level (0.0 to 1.0)
            
        Returns:
            Normalized audio data
        """
        try:
            if len(audio_data) == 0:
                return audio_data
                
            # Find peak level
            peak = np.max(np.abs(audio_data))
            
            if peak > 0.001:  # Avoid division by zero
                # Calculate scaling factor
                scale = target_level / peak
                # Apply scaling, but cap at target level to prevent clipping
                scale = min(scale, target_level / peak)
                return audio_data * scale
            else:
                return audio_data
                
        except Exception as e:
            logger.error(f"Audio normalization failed: {e}")
            return audio_data
    
    def apply_silence_detection(self, audio_data: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Detect if audio contains speech or is mostly silence
        
        Args:
            audio_data: Audio data to analyze
            threshold: Energy threshold for speech detection
            
        Returns:
            True if speech detected, False if silence
        """
        try:
            if len(audio_data) == 0:
                return False
                
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            return rms > threshold
            
        except Exception as e:
            logger.error(f"Silence detection failed: {e}")
            return False
    
    def get_audio_stats(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Get statistics about audio data
        
        Args:
            audio_data: Audio data to analyze
            
        Returns:
            Dictionary with audio statistics
        """
        try:
            if len(audio_data) == 0:
                return {"error": "Empty audio data"}
                
            return {
                "duration": len(audio_data) / self.sample_rate,
                "samples": len(audio_data),
                "peak": float(np.max(np.abs(audio_data))),
                "rms": float(np.sqrt(np.mean(audio_data ** 2))),
                "mean": float(np.mean(audio_data)),
                "std": float(np.std(audio_data)),
                "sample_rate": self.sample_rate,
                "channels": self.channels
            }
            
        except Exception as e:
            logger.error(f"Audio stats calculation failed: {e}")
            return {"error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get audio service status"""
        return {
            "available": self.is_available,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frame_size": self.frame_size,
            "opus_decoder": self.opus_decoder is not None,
            "opus_encoder": self.opus_encoder is not None
        }
    
    def extract_pcm_from_raw(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """
        Extract PCM from raw audio bytes (fallback method)
        
        Args:
            audio_bytes: Raw audio bytes (assume 16-bit PCM)
            
        Returns:
            Numpy array of PCM audio samples or None if failed
        """
        try:
            if len(audio_bytes) == 0:
                return None
            
            # Try as 16-bit signed PCM
            if len(audio_bytes) % 2 == 0:  # Must be even for 16-bit
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Basic validation
                if len(audio_array) > 0 and np.max(np.abs(audio_array)) <= 1.0:
                    return audio_array
            
            # Try as unsigned 8-bit
            audio_array = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
            if len(audio_array) > 0:
                return audio_array
                
            return None
            
        except Exception as e:
            logger.debug(f"Raw PCM extraction failed: {e}")
            return None
            
    def extract_pcm_smart(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """
        Smart PCM extraction with multiple fallbacks
        
        Args:
            audio_bytes: Audio bytes in any format
            
        Returns:
            Numpy array of PCM audio samples or None if all methods fail
        """
        if len(audio_bytes) == 0:
            return None
            
        # Method 1: Try WebM conversion
        pcm_data = self.extract_pcm_from_webm(audio_bytes)
        if pcm_data is not None:
            return pcm_data
        
        # Method 2: Try raw audio processing
        pcm_data = self.extract_pcm_from_raw(audio_bytes)
        if pcm_data is not None:
            return pcm_data
        
        logger.debug("All PCM extraction methods failed")
        return None
    
    async def cleanup(self):
        """Clean up audio service resources"""
        try:
            if self.opus_decoder:
                del self.opus_decoder
                self.opus_decoder = None
                
            if self.opus_encoder:
                del self.opus_encoder
                self.opus_encoder = None
                
            self.audio_buffer.clear()
            self.is_available = False
            logger.info("Audio service cleaned up")
            
        except Exception as e:
            logger.error(f"Audio service cleanup error: {e}") 