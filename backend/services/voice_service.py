import asyncio
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, AsyncGenerator, Callable
from dataclasses import dataclass
import json

from backend.app.config import settings
from .vad_service import VADService
from .stt_service import STTService
from .llm_service import LLMService
from .tts_service import TTSService
from .audio_service import AudioService

logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Metrics for tracking voice processing performance"""
    vad_time: float = 0.0
    stt_time: float = 0.0
    llm_time: float = 0.0
    tts_time: float = 0.0
    total_time: float = 0.0
    audio_chunks_processed: int = 0
    speech_sessions: int = 0

class VoiceService:
    """
    Unified real-time voice agent service optimized for 500ms end-to-end latency
    
    Architecture:
    Audio Input (16kHz) → VAD (50ms) → STT (150ms) → LLM (200ms) → TTS (100ms) → Audio Output
    Total Target Latency: ~500ms
    """

    def __init__(self, 
                 chunk_duration_ms: int = 32,  # 32ms chunks for real-time processing
                 sample_rate: int = 16000,
                 vad_threshold: float = 0.7,
                 speech_timeout: float = 3.0):
        
        # Audio processing configuration
        self.chunk_duration_ms = chunk_duration_ms
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)  # 512 samples for 32ms
        self.speech_timeout = speech_timeout
        
        # Service instances
        self.vad_service = VADService(
            speech_threshold=vad_threshold,
            silence_threshold=0.4,
            min_speech_duration=0.3,
            min_silence_duration=0.5,
            sample_rate=sample_rate
        )
        self.stt_service = STTService(
            language="en-US",
            continuous_recognition=True,
            interim_results=True
        )
        self.llm_service = LLMService()
        self.tts_service = TTSService()
        self.audio_service = AudioService()
        
        # Service state
        self.is_available = False
        self.is_initialized = False
        
        # Real-time processing state
        self.is_processing_audio = False
        self.current_session_id = None
        self.speech_buffer = []
        self.is_user_speaking = False
        self.speech_start_time = None
        
        # Performance tracking
        self.processing_stats = ProcessingMetrics()
        self.latency_history = []
        self.session_count = 0
        
        # Audio processing queue for real-time handling
        self.audio_queue = asyncio.Queue(maxsize=100)
        self.processing_task = None
        
        # Callbacks for real-time events
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None
        self.on_partial_transcript: Optional[Callable] = None
        self.on_final_transcript: Optional[Callable] = None
        self.on_llm_response: Optional[Callable] = None
        self.on_audio_response: Optional[Callable] = None

    async def initialize(self):
        """Initialize all voice processing services for real-time operation"""
        try:
            logger.info("Initializing real-time voice service...")
            
            # Initialize all services in parallel for faster startup
            init_tasks = [
                self.vad_service.initialize(),
                self.stt_service.initialize(),
                self.llm_service.initialize(),
                self.tts_service.initialize(),
                self.audio_service.initialize()
            ]
            
            await asyncio.gather(*init_tasks, return_exceptions=True)
            
            # Check service availability
            services_ready = [
                self.vad_service.is_available,
                self.stt_service.is_available,
                self.llm_service.is_available,
                self.tts_service.is_available,
                self.audio_service.is_available
            ]
            
            if all(services_ready):
                self.is_available = True
                self.is_initialized = True
                
                # Setup STT callbacks for real-time processing
                self.stt_service.set_callbacks(
                    on_partial=self._handle_partial_transcript,
                    on_final=self._handle_final_transcript,
                    on_error=self._handle_stt_error
                )
                
                # Start background audio processing
                self.processing_task = asyncio.create_task(self._audio_processing_loop())
                
                logger.info("✅ Real-time voice service initialized successfully")
                logger.info(f"Target latency: 500ms | Chunk size: {self.chunk_size} samples ({self.chunk_duration_ms}ms)")
            else:
                logger.warning(f"Some services failed to initialize: VAD={services_ready[0]}, STT={services_ready[1]}, LLM={services_ready[2]}, TTS={services_ready[3]}, Audio={services_ready[4]}")
                self.is_available = any(services_ready)
                
        except Exception as e:
            logger.error(f"Failed to initialize voice service: {e}", exc_info=True)
            self.is_available = False

    async def start_session(self, session_id: str) -> bool:
        """Start a new voice processing session"""
        if not self.is_available:
            logger.error("Voice service not available")
            return False
            
        try:
            self.current_session_id = session_id
            self.session_count += 1
            self.speech_buffer.clear()
            self.is_user_speaking = False
            self.speech_start_time = None
            
            # Start STT session
            stt_session = await self.stt_service.start_continuous_recognition()
            if not stt_session:
                logger.error("Failed to start STT session")
                return False
                
            self.is_processing_audio = True
            logger.info(f"Voice session started: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start voice session: {e}")
            return False

    async def process_audio_frame(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Process real-time audio frame with target 50ms latency
        Expected: 16kHz, 16-bit PCM audio chunks
        """
        if not self.is_processing_audio:
            return {"status": "not_processing"}
            
        start_time = time.time()
        
        try:
            # Convert bytes to numpy array (16-bit PCM)
            audio_chunk = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Ensure correct chunk size (pad or truncate)
            if len(audio_chunk) != self.chunk_size:
                if len(audio_chunk) < self.chunk_size:
                    audio_chunk = np.pad(audio_chunk, (0, self.chunk_size - len(audio_chunk)))
                else:
                    audio_chunk = audio_chunk[:self.chunk_size]
            
            # Add to processing queue (non-blocking)
            try:
                self.audio_queue.put_nowait({
                    'audio': audio_chunk,
                    'timestamp': time.time()
                })
            except asyncio.QueueFull:
                logger.warning("Audio queue full, dropping frame")
                return {"status": "queue_full", "latency": 0}
            
            processing_time = (time.time() - start_time) * 1000
            self.processing_stats.audio_chunks_processed += 1
            
            return {
                "status": "queued",
                "latency": processing_time,
                "queue_size": self.audio_queue.qsize()
            }
            
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            return {"status": "error", "error": str(e)}

    async def _audio_processing_loop(self):
        """Background loop for real-time audio processing"""
        logger.info("Starting real-time audio processing loop")
        
        while True:
            try:
                # Get audio chunk from queue
                chunk_data = await self.audio_queue.get()
                audio_chunk = chunk_data['audio']
                chunk_timestamp = chunk_data['timestamp']
                
                # Process with VAD
                vad_start = time.time()
                vad_result = await self.vad_service.process_audio_chunk(audio_chunk)
                vad_time = (time.time() - vad_start) * 1000
                
                # Handle speech state changes
                if vad_result.get('speech_started'):
                    await self._handle_speech_start()
                elif vad_result.get('speech_ended'):
                    await self._handle_speech_end()
                
                # Send audio to STT if speech is detected
                if vad_result.get('speech_detected') and self.is_user_speaking:
                    # Convert back to bytes for STT
                    audio_bytes = (audio_chunk * 32768).astype(np.int16).tobytes()
                    
                    stt_start = time.time()
                    await self.stt_service.process_audio_chunk(audio_bytes)
                    stt_time = (time.time() - stt_start) * 1000
                    
                    self.processing_stats.stt_time = stt_time
                
                self.processing_stats.vad_time = vad_time
                
                # Track queue latency
                queue_latency = (time.time() - chunk_timestamp) * 1000
                if queue_latency > 100:  # Log if queue latency exceeds 100ms
                    logger.warning(f"High queue latency: {queue_latency:.1f}ms")
                
            except asyncio.CancelledError:
                logger.info("Audio processing loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
                await asyncio.sleep(0.001)  # Brief pause to prevent tight error loop

    async def _handle_speech_start(self):
        """Handle speech start event"""
        self.is_user_speaking = True
        self.speech_start_time = time.time()
        logger.info("Speech started")
        
        if self.on_speech_start:
            await self.on_speech_start()

    async def _handle_speech_end(self):
        """Handle speech end event"""
        if self.is_user_speaking:
            speech_duration = time.time() - (self.speech_start_time or time.time())
            logger.info(f"Speech ended (duration: {speech_duration:.2f}s)")
            
            self.is_user_speaking = False
            self.speech_start_time = None
            
            # Finalize STT recognition
            await self._finalize_recognition()
            
            if self.on_speech_end:
                await self.on_speech_end()

    async def _handle_partial_transcript(self, text: str):
        """Handle partial STT transcript"""
        if self.on_partial_transcript:
            await self.on_partial_transcript(text)

    async def _handle_final_transcript(self, text: str):
        """Handle final STT transcript and trigger LLM processing"""
        logger.info(f"Final transcript: '{text}'")
        
        if self.on_final_transcript:
            await self.on_final_transcript(text)
        
        # Process with LLM if we have meaningful text
        if text and len(text.strip()) > 2:
            await self._process_with_llm(text)

    async def _handle_stt_error(self, error: str):
        """Handle STT error"""
        logger.error(f"STT error: {error}")

    async def _finalize_recognition(self):
        """Finalize STT recognition with timeout"""
        try:
            final_text = await self.stt_service.finalize_recognition(timeout=2.0)
            if final_text:
                await self._handle_final_transcript(final_text)
        except Exception as e:
            logger.error(f"Error finalizing recognition: {e}")

    async def _process_with_llm(self, text: str):
        """Process text with LLM and generate response"""
        llm_start = time.time()
        
        try:
            logger.info(f"Processing with LLM: '{text[:50]}...'")
            response = await self.llm_service.process_text(text)
            
            llm_time = (time.time() - llm_start) * 1000
            self.processing_stats.llm_time = llm_time
            
            if response and response.strip():
                logger.info(f"LLM response: '{response[:50]}...' (took {llm_time:.1f}ms)")
                
                if self.on_llm_response:
                    await self.on_llm_response(response)
                
                # Generate TTS audio
                await self._generate_tts_response(response)
            else:
                logger.warning("Empty LLM response")
                
        except Exception as e:
            logger.error(f"Error processing with LLM: {e}")

    async def _generate_tts_response(self, text: str):
        """Generate TTS audio response"""
        tts_start = time.time()
        
        try:
            logger.info(f"Generating TTS for: '{text[:50]}...'")
            
            # Stream TTS audio chunks
            async for audio_chunk in self.tts_service.generate_speech_stream(text):
                if audio_chunk and self.on_audio_response:
                    await self.on_audio_response(audio_chunk)
            
            tts_time = (time.time() - tts_start) * 1000
            self.processing_stats.tts_time = tts_time
            
            logger.info(f"TTS generation completed (took {tts_time:.1f}ms)")
            
        except Exception as e:
            logger.error(f"Error generating TTS response: {e}")

    def set_callbacks(self,
                     on_speech_start: Optional[Callable] = None,
                     on_speech_end: Optional[Callable] = None,
                     on_partial_transcript: Optional[Callable] = None,
                     on_final_transcript: Optional[Callable] = None,
                     on_llm_response: Optional[Callable] = None,
                     on_audio_response: Optional[Callable] = None):
        """Set callback functions for real-time events"""
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_partial_transcript = on_partial_transcript
        self.on_final_transcript = on_final_transcript
        self.on_llm_response = on_llm_response
        self.on_audio_response = on_audio_response

    async def stop_session(self):
        """Stop the current voice processing session"""
        try:
            self.is_processing_audio = False
            
            if self.stt_service.is_listening:
                await self.stt_service.stop_continuous_recognition()
            
            # Clear audio queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            logger.info(f"Voice session stopped: {self.current_session_id}")
            self.current_session_id = None
            
        except Exception as e:
            logger.error(f"Error stopping voice session: {e}")

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all underlying services"""
        
        # Helper to get status safely
        def get_status_safe(service):
            if hasattr(service, 'get_status'):
                return service.get_status()
            elif hasattr(service, 'is_available'):
                return {"available": service.is_available}
            return {"available": False, "error": "Status method not found"}

        services = {
            "vad_service": self.vad_service,
            "stt_service": self.stt_service,
            "llm_service": self.llm_service,
            "tts_service": self.tts_service,
            "audio_service": self.audio_service
        }
        
        service_statuses = {name: get_status_safe(service) for name, service in services.items()}
        
        # Overall voice service status
        is_healthy = all(status.get('available', False) for status in service_statuses.values())
        
        return {
            "voice_service": {"available": self.is_available and is_healthy, "is_initialized": self.is_initialized},
            **service_statuses
        }

    async def cleanup(self):
        """Cleanup all voice processing resources"""
        logger.info("Cleaning up voice service...")
        
        # Stop processing
        self.is_processing_audio = False
        
        # Cancel background task
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup all services
        cleanup_tasks = [
            self.vad_service.cleanup() if hasattr(self.vad_service, 'cleanup') else asyncio.sleep(0),
            self.stt_service.cleanup() if hasattr(self.stt_service, 'cleanup') else asyncio.sleep(0),
            self.llm_service.cleanup() if hasattr(self.llm_service, 'cleanup') else asyncio.sleep(0),
            self.tts_service.cleanup() if hasattr(self.tts_service, 'cleanup') else asyncio.sleep(0),
            self.audio_service.cleanup() if hasattr(self.audio_service, 'cleanup') else asyncio.sleep(0)
        ]
        
        try:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        logger.info("Voice service cleanup completed")

    async def process_text(self, text: str) -> Optional[str]:
        """Process text input with LLM and return response"""
        try:
            if not self.llm_service or not self.llm_service.is_available:
                logger.warning("LLM service not available")
                return None
                
            return await self.llm_service.process_text(text)
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return None

    async def generate_speech(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate speech audio for text and yield audio chunks"""
        try:
            if not self.tts_service or not self.tts_service.is_available:
                logger.warning("TTS service not available")
                return
                
            async for audio_chunk in self.tts_service.generate_speech_stream(text):
                yield audio_chunk
                
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return

    async def process_audio_chunk(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """Process audio chunk with VAD"""
        try:
            return await self.vad_service.process_audio_chunk(audio_chunk)
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {"speech_detected": False, "error": str(e)}
            
    def get_vad_status(self) -> Dict[str, Any]:
        """Get current VAD status and diagnostics"""
        try:
            return self.vad_service.get_status()
        except Exception as e:
            logger.error(f"Error getting VAD status: {e}")
            return {"error": str(e), "available": False}
            
    async def end_user_speech(self) -> None:
        """Force end current user speech if any"""
        try:
            # Notify STT service to finalize any pending transcription
            await self.stt_service.stop_continuous_recognition()
            
            # Reset VAD state
            self.is_user_speaking = False
            
            logger.info("User speech ended by explicit request")
        except Exception as e:
            logger.error(f"Error ending user speech: {e}") 