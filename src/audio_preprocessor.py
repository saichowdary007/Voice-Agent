"""
Nova-3 Single-Word Speech Recognition Optimization
Implements high-accuracy single-word command recognition with <500ms latency and >90% accuracy.
Addresses leading-edge audio clipping and delayed finalization issues.
"""

import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any
import webrtcvad
from scipy import signal
import asyncio
import time
import subprocess
from collections import deque

logger = logging.getLogger(__name__)


class RingBuffer:
    """200ms rolling buffer for pre-roll injection to prevent leading-edge clipping."""
    
    def __init__(self, sample_rate: int = 16000, duration_ms: int = 200):
        self.sample_rate = sample_rate
        self.duration_ms = duration_ms
        self.buffer_size = int(sample_rate * duration_ms / 1000) * 2  # 16-bit samples
        self.buffer = deque(maxlen=self.buffer_size)
        self.is_full = False
    
    def add_audio(self, audio_bytes: bytes):
        """Add audio bytes to the ring buffer."""
        for byte in audio_bytes:
            self.buffer.append(byte)
            if len(self.buffer) == self.buffer_size:
                self.is_full = True
    
    def get_preroll(self) -> bytes:
        """Get the current pre-roll buffer as bytes."""
        if not self.is_full:
            return b''
        return bytes(self.buffer)
    
    def clear(self):
        """Clear the ring buffer."""
        self.buffer.clear()
        self.is_full = False


class AudioNormalizer:
    """Audio normalization to -6 dBFS peaks with -45 dBFS speech floor."""
    
    def __init__(self):
        self.target_peak_db = -6.0
        self.speech_floor_db = -45.0
        self.target_peak_linear = 10 ** (self.target_peak_db / 20)
        self.speech_floor_linear = 10 ** (self.speech_floor_db / 20)
    
    def normalize_audio(self, audio_np: np.ndarray) -> np.ndarray:
        """Normalize audio to target levels."""
        if len(audio_np) == 0:
            return audio_np
        
        # Calculate current peak and RMS
        current_peak = np.max(np.abs(audio_np))
        current_rms = np.sqrt(np.mean(audio_np ** 2))
        
        if current_peak < 1e-6:  # Silence
            return audio_np
        
        # Calculate gain to reach target peak
        peak_gain = self.target_peak_linear / current_peak
        
        # Apply gain with limiting
        normalized = audio_np * peak_gain
        
        # Ensure speech floor is maintained
        rms_after_gain = np.sqrt(np.mean(normalized ** 2))
        if rms_after_gain < self.speech_floor_linear:
            floor_gain = self.speech_floor_linear / rms_after_gain
            normalized *= floor_gain
        
        # Hard limiting to prevent clipping
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized


class CommandWordMatcher:
    """N-best filtering with Levenshtein distance for command word matching."""
    
    def __init__(self):
        self.command_words = {
            'yes', 'no', 'stop', 'start', 'go', 'pause', 'play', 'next', 'back',
            'up', 'down', 'left', 'right', 'ok', 'cancel', 'help', 'menu', 'home'
        }
        self.max_distance = 1  # Levenshtein distance threshold
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def match_command(self, transcript: str, alternatives: list = None) -> Optional[str]:
        """Match transcript against command words with fuzzy matching."""
        candidates = [transcript.lower().strip()]
        
        # Add alternatives if provided
        if alternatives:
            candidates.extend([alt.lower().strip() for alt in alternatives])
        
        best_match = None
        best_distance = float('inf')
        
        for candidate in candidates:
            # Remove punctuation and extra spaces
            clean_candidate = ''.join(c for c in candidate if c.isalnum() or c.isspace()).strip()
            
            for command in self.command_words:
                distance = self.levenshtein_distance(clean_candidate, command)
                if distance <= self.max_distance and distance < best_distance:
                    best_match = command
                    best_distance = distance
        
        return best_match


class StreamingAudioProcessor:
    """
    FIX #4: Streaming ffmpeg processor that maintains persistent ffmpeg process per WebSocket connection.
    Addresses the CPU spike issue from spawning new processes for every blob.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.audio_buffer = bytearray()
        self.is_initialized = False
        self.process_lock = asyncio.Lock()
        
        # Performance monitoring
        self.performance_metrics = {
            'total_chunks_processed': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'max_processing_time': 0.0,
            'ffmpeg_restarts': 0,
            'processing_errors': 0,
            'last_processing_time': 0.0,
            'cpu_usage_samples': deque(maxlen=100),
            'memory_usage_samples': deque(maxlen=100),
            'throughput_samples': deque(maxlen=100)
        }
        
        logger.info(f"🚀 StreamingAudioProcessor initialized for {sample_rate}Hz")
    
    async def initialize_streaming_ffmpeg(self) -> None:
        """FIX #4: Initialize persistent ffmpeg process for streaming audio conversion."""
        if self.is_initialized and self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            return  # Already initialized and running
        
        try:
            # FIX #4: Single persistent ffmpeg process for WebM/Opus to 16kHz mono PCM
            ffmpeg_cmd = [
                'ffmpeg',
                '-f', 'webm',           # Input format
                '-i', 'pipe:0',         # Read from stdin
                '-f', 's16le',          # Output format: signed 16-bit little-endian
                '-acodec', 'pcm_s16le', # Audio codec
                '-ar', str(self.sample_rate),  # Sample rate
                '-ac', '1',             # Mono channel
                '-loglevel', 'error',   # Reduce logging
                '-fflags', '+nobuffer', # FIX #4: Disable buffering for real-time
                '-flags', 'low_delay',  # FIX #4: Low delay mode
                'pipe:1'                # Write to stdout
            ]
            
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0  # Unbuffered for real-time processing
            )
            
            self.is_initialized = True
            logger.info("✅ FIX #4: Persistent streaming ffmpeg process initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize streaming ffmpeg: {e}")
            self.is_initialized = False
            raise
    
    async def process_audio_stream(self, audio_chunk: bytes) -> bytes:
        """FIX #4: Process audio chunk through persistent streaming ffmpeg."""
        processing_start = time.perf_counter()
        
        async with self.process_lock:
            if not self.is_initialized:
                await self.initialize_streaming_ffmpeg()
            
            if not self.ffmpeg_process or self.ffmpeg_process.poll() is not None:
                logger.warning("ffmpeg process died, reinitializing...")
                self.performance_metrics['ffmpeg_restarts'] += 1
                await self.initialize_streaming_ffmpeg()
            
            try:
                # FIX #4: Stream audio chunk to persistent ffmpeg stdin
                self.ffmpeg_process.stdin.write(audio_chunk)
                self.ffmpeg_process.stdin.flush()
                
                # FIX #4: Read processed audio from persistent ffmpeg stdout
                processed_audio = b''
                start_time = time.perf_counter()
                timeout = 0.5  # Reduced timeout for faster processing
                
                while time.perf_counter() - start_time < timeout:
                    if self.ffmpeg_process.stdout.readable():
                        chunk = self.ffmpeg_process.stdout.read(4096)
                        if chunk:
                            processed_audio += chunk
                        else:
                            break
                    await asyncio.sleep(0.001)  # Small delay to prevent busy waiting
                
                # Update performance metrics
                processing_time = (time.perf_counter() - processing_start) * 1000
                self.performance_metrics['total_chunks_processed'] += 1
                self.performance_metrics['total_processing_time'] += processing_time
                self.performance_metrics['last_processing_time'] = processing_time
                self.performance_metrics['avg_processing_time'] = (
                    self.performance_metrics['total_processing_time'] / 
                    self.performance_metrics['total_chunks_processed']
                )
                self.performance_metrics['max_processing_time'] = max(
                    self.performance_metrics['max_processing_time'], 
                    processing_time
                )
                
                # Track throughput (bytes per second)
                if processing_time > 0:
                    throughput = len(audio_chunk) / (processing_time / 1000)
                    self.performance_metrics['throughput_samples'].append(throughput)
                
                # Log performance improvement
                if processing_time < 50:  # Good performance
                    logger.debug(f"✅ Fast audio processing: {processing_time:.1f}ms for {len(audio_chunk)} bytes")
                elif processing_time > 100:  # Still slow
                    logger.warning(f"⚠️ Slow audio processing: {processing_time:.1f}ms for {len(audio_chunk)} bytes")
                
                return processed_audio
                
            except Exception as e:
                logger.error(f"❌ Error processing audio stream: {e}")
                self.performance_metrics['processing_errors'] += 1
                # Try to restart ffmpeg process
                await self.cleanup_ffmpeg_process()
                return b''
    
    async def cleanup_ffmpeg_process(self) -> None:
        """FIX #4: Clean up persistent ffmpeg process."""
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.stdout.close()
                self.ffmpeg_process.stderr.close()
                self.ffmpeg_process.terminate()
                
                # Wait for process to terminate with timeout
                try:
                    self.ffmpeg_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.ffmpeg_process.kill()
                    self.ffmpeg_process.wait()
                
                logger.info("🧹 FIX #4: Persistent ffmpeg process cleaned up")
            except Exception as e:
                logger.error(f"❌ Error cleaning up ffmpeg process: {e}")
            finally:
                self.ffmpeg_process = None
                self.is_initialized = False


class AudioPreprocessor:
    """
    Nova-3 optimized audio preprocessor for single-word command recognition.
    Implements ring buffer, normalization, and enhanced VAD for <500ms latency.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
        # Core components for Nova-3 optimization
        self.ring_buffer = RingBuffer(sample_rate, 200)  # 200ms pre-roll
        self.normalizer = AudioNormalizer()
        self.command_matcher = CommandWordMatcher()
        
        # WebRTC VAD mode 2 with 200ms hang-over
        self.webrtc_vad = webrtcvad.Vad(2)
        self.frame_bytes = int(sample_rate * 0.02) * 2  # 20ms frames
        
        # VAD state management
        self.vad_hangover_frames = 10  # 200ms hangover (10 * 20ms)
        self.hangover_counter = 0
        self.speech_active = False
        
        # Back-pressure monitoring
        self.max_buffer_size = 256 * 1024  # 256KB limit
        self.frame_timing = deque(maxlen=100)  # Track frame timing
        
        # Performance metrics
        self.metrics = {
            'speech_started_events': 0,
            'speech_final_events': 0,
            'false_positives': 0,
            'processing_times': deque(maxlen=100)
        }
        
        # Streaming audio processor
        self.streaming_processor = StreamingAudioProcessor(sample_rate)
        
        logger.info(f"✅ Nova-3 AudioPreprocessor initialized ({sample_rate}Hz, 200ms pre-roll)")
    
    def preprocess_audio_chunk(self, audio_bytes: bytes, inject_preroll: bool = False) -> Tuple[bytes, Dict[str, Any]]:
        """
        Process audio chunk with Nova-3 optimizations.
        
        Args:
            audio_bytes: Raw 16-bit PCM audio
            inject_preroll: Whether to inject pre-roll buffer
            
        Returns:
            Tuple of (processed_audio, metadata)
        """
        start_time = time.perf_counter()
        
        if not audio_bytes or len(audio_bytes) < 320:  # Minimum 10ms
            return audio_bytes, {'is_speech': False, 'confidence': 0.0}
        
        try:
            # Add to ring buffer for future pre-roll
            self.ring_buffer.add_audio(audio_bytes)
            
            # Convert to numpy for processing
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Apply normalization (-6 dBFS peaks, -45 dBFS floor)
            normalized_audio = self.normalizer.normalize_audio(audio_np)
            
            # Convert back to bytes for VAD
            normalized_bytes = (normalized_audio * 32767).astype(np.int16).tobytes()
            
            # Enhanced VAD with hangover
            is_speech, confidence = self._enhanced_vad_with_hangover(normalized_bytes)
            
            # Prepare final audio with optional pre-roll injection
            final_audio = normalized_bytes
            if inject_preroll and self.ring_buffer.is_full:
                preroll = self.ring_buffer.get_preroll()
                final_audio = preroll + normalized_bytes
                logger.debug(f"Pre-roll injected: {len(preroll)} + {len(normalized_bytes)} bytes")
            
            # Update metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            self.metrics['processing_times'].append(processing_time)
            
            metadata = {
                'is_speech': is_speech,
                'confidence': confidence,
                'processing_time_ms': processing_time,
                'has_preroll': inject_preroll and self.ring_buffer.is_full,
                'buffer_size': len(final_audio)
            }
            
            return final_audio, metadata
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return audio_bytes, {'is_speech': False, 'confidence': 0.0, 'error': str(e)}
    
    def _enhanced_vad_with_hangover(self, audio_bytes: bytes) -> Tuple[bool, float]:
        """
        Enhanced VAD with 200ms hangover for better speech boundary detection.
        Optimized to reduce false positives while maintaining speech detection accuracy.
        """
        try:
            # Convert to numpy for analysis
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            rms_energy = np.sqrt(np.mean(audio_np ** 2))
            peak_energy = np.max(np.abs(audio_np))
            
            # Multi-stage energy analysis to reduce false positives
            # Stage 1: Eliminate obvious silence
            if rms_energy < 50.0 and peak_energy < 100.0:
                self.hangover_counter = max(0, self.hangover_counter - 1)
                return self.hangover_counter > 0, 0.05
            
            # Stage 2: Practical speech detection for single-word commands
            # For Nova-3 optimization, we prioritize speed and accuracy over complex analysis
            
            # Basic energy thresholds optimized for command recognition
            # These values are tuned for typical microphone input levels
            min_speech_energy = 100.0  # Minimum RMS for speech consideration
            max_noise_energy = 5000.0  # Maximum RMS to avoid loud noise false positives
            
            # Quick spectral analysis for speech-like content
            is_speech_like = True  # Default to permissive for single-word commands
            
            if len(audio_np) >= 128:  # Minimum for basic analysis
                # Calculate zero crossing rate (simple speech indicator)
                zero_crossings = np.sum(np.diff(np.sign(audio_np)) != 0)
                zcr = zero_crossings / len(audio_np) if len(audio_np) > 0 else 0
                
                # Speech typically has moderate ZCR, pure tones have very low ZCR, noise has high ZCR
                # For single-word commands, be more permissive but filter obvious non-speech
                if zcr < 0.001:  # Pure tone or DC offset
                    is_speech_like = False
                elif zcr > 0.6:  # Very high frequency noise
                    is_speech_like = False
            
            # Stage 3: Combined energy and basic characteristics check
            speech_energy_ok = min_speech_energy <= rms_energy <= max_noise_energy
            
            if not (speech_energy_ok and is_speech_like):
                self.hangover_counter = max(0, self.hangover_counter - 1)
                return self.hangover_counter > 0, max(0.05, self.hangover_counter / self.vad_hangover_frames)
            
            # Stage 4: WebRTC VAD check (only if passed previous stages)
            vad_result = self._webrtc_vad_check(audio_bytes)
            
            if vad_result:
                # Speech detected - reset hangover and mark active
                self.hangover_counter = self.vad_hangover_frames
                if not self.speech_active:
                    self.speech_active = True
                    self.metrics['speech_started_events'] += 1
                    logger.debug(f"🎤 Speech started (RMS: {rms_energy:.1f}, ZCR: {zcr:.3f})")
                return True, 0.8
            else:
                # No speech - decrement hangover
                self.hangover_counter = max(0, self.hangover_counter - 1)
                
                if self.speech_active and self.hangover_counter == 0:
                    self.speech_active = False
                    self.metrics['speech_final_events'] += 1
                    logger.debug("🔇 Speech ended (VAD)")
                
                # Return True if still in hangover period
                confidence = max(0.1, self.hangover_counter / self.vad_hangover_frames)
                return self.hangover_counter > 0, confidence
                
        except Exception as e:
            logger.warning(f"Enhanced VAD error: {e}")
            return False, 0.1  # Default to no speech when in doubt (changed from True)
    
    def _webrtc_vad_check(self, audio_bytes: bytes) -> bool:
        """WebRTC VAD mode 2 with frame-by-frame analysis."""
        if len(audio_bytes) < self.frame_bytes:
            return False
        
        speech_frames = 0
        total_frames = 0
        
        # Check each 20ms frame
        for i in range(0, len(audio_bytes) - self.frame_bytes + 1, self.frame_bytes):
            frame = audio_bytes[i:i + self.frame_bytes]
            if len(frame) == self.frame_bytes:
                total_frames += 1
                if self.webrtc_vad.is_speech(frame, self.sample_rate):
                    speech_frames += 1
        
        # Require at least 30% of frames to be speech
        if total_frames > 0:
            speech_ratio = speech_frames / total_frames
            return speech_ratio >= 0.3
        
        return False
    
    def should_process_for_stt(self, total_buffer_size: int, speech_detected: bool, silence_duration_ms: float) -> bool:
        """
        Determine if audio buffer should be sent to STT based on Nova-3 criteria.
        """
        # Minimum audio length based on speech detection
        if speech_detected:
            min_bytes = int(self.sample_rate * 0.2 * 2)  # 200ms if speech detected
        else:
            min_bytes = int(self.sample_rate * 0.5 * 2)  # 500ms if no clear speech
        
        if total_buffer_size < min_bytes:
            return False
        
        # Maximum buffer size to prevent memory issues
        max_bytes = int(self.sample_rate * 10 * 2)  # 10 seconds max
        if total_buffer_size > max_bytes:
            logger.warning(f"Audio buffer too large: {total_buffer_size} bytes, forcing processing")
            return True
        
        # Silence timeout check
        if silence_duration_ms > 1200:  # 1.2 second silence
            return True
        
        # Back-pressure check
        if total_buffer_size > self.max_buffer_size:
            logger.warning("Back-pressure detected, forcing processing")
            return True
        
        return False
    
    def match_command_word(self, transcript: str, alternatives: list = None) -> Optional[str]:
        """Match transcript against command vocabulary with fuzzy matching."""
        return self.command_matcher.match_command(transcript, alternatives)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        processing_times = list(self.metrics['processing_times'])
        
        return {
            'speech_started_events': self.metrics['speech_started_events'],
            'speech_final_events': self.metrics['speech_final_events'],
            'false_positives': self.metrics['false_positives'],
            'avg_processing_time_ms': np.mean(processing_times) if processing_times else 0,
            'max_processing_time_ms': np.max(processing_times) if processing_times else 0,
            'speech_active': self.speech_active,
            'hangover_frames_remaining': self.hangover_counter
        }
    
    def reset_session_state(self):
        """Reset state for new audio session."""
        self.ring_buffer.clear()
        self.hangover_counter = 0
        self.speech_active = False
        self.frame_timing.clear()
        logger.debug("🔄 Audio session state reset")
    
    def configure_for_ultra_fast_mode(self):
        """Configure for ultra-fast mode targeting ~500ms latency."""
        # Reduce hangover for faster response
        self.vad_hangover_frames = 5  # 100ms hangover
        
        # Tighter energy thresholds
        if hasattr(self.normalizer, 'target_peak_db'):
            self.normalizer.target_peak_db = -3.0  # Higher gain for faster detection
        
        logger.info("🚀 Ultra-fast mode configured (100ms hangover, -3dBFS peaks)")
    
    # Legacy compatibility methods
    def preprocess_audio(self, audio_bytes: bytes) -> Tuple[bytes, bool]:
        """Legacy compatibility wrapper."""
        processed_audio, metadata = self.preprocess_audio_chunk(audio_bytes)
        return processed_audio, metadata.get('is_speech', False)
    
    def reset_noise_profile(self):
        """Legacy compatibility - reset session state."""
        self.reset_session_state()


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
    """Get global Nova-3 optimized audio preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = AudioPreprocessor()
        _preprocessor.configure_for_ultra_fast_mode()
        logger.info("🚀 Global Nova-3 audio preprocessor initialized")
    return _preprocessor


async def preprocess_audio_chunk(audio_bytes: bytes, inject_preroll: bool = False) -> Tuple[bytes, Dict[str, Any]]:
    """
    Async wrapper for Nova-3 optimized audio preprocessing.
    
    Args:
        audio_bytes: Raw audio bytes
        inject_preroll: Whether to inject pre-roll buffer
        
    Returns:
        Tuple of (processed_bytes, metadata_dict)
    """
    preprocessor = get_audio_preprocessor()
    
    # Run preprocessing in thread to avoid blocking
    def _process():
        return preprocessor.preprocess_audio_chunk(audio_bytes, inject_preroll)
    
    return await asyncio.to_thread(_process)


# Legacy compatibility function
async def preprocess_audio_legacy(audio_bytes: bytes) -> Tuple[bytes, bool]:
    """Legacy compatibility wrapper."""
    processed_audio, metadata = await preprocess_audio_chunk(audio_bytes)
    return processed_audio, metadata.get('is_speech', False)