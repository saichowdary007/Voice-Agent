import asyncio
import time
import logging
import os
from typing import Optional, Dict, Any, List, Callable
import azure.cognitiveservices.speech as speechsdk
import numpy as np
from dataclasses import dataclass
import structlog
import threading
import uuid

# Configuration from environment or settings
from backend.app.config import settings

logger = structlog.get_logger(__name__)

@dataclass
class STTResult:
    """Speech-to-text processing result"""
    partial_text: str = ""
    final_text: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    alternatives: List[Dict[str, Any]] = None

class STTService:
    """Optimized Speech-to-Text service using Azure Speech SDK for real-time processing"""

    def __init__(self, 
                 language: str = "en-US",
                 continuous_recognition: bool = True,
                 interim_results: bool = True,
                 profanity_filter: bool = False):
        
        # Azure Speech configuration
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION", "eastus")
        self.language = language
        self.continuous_recognition = continuous_recognition
        self.interim_results = interim_results
        self.profanity_filter = profanity_filter
        
        # Service state
        self.is_available = False
        self.is_listening = False
        self.is_processing = False
        self.use_mock = False
        
        # Recognition components
        self.speech_config = None
        self.audio_config = None
        self.speech_recognizer = None
        self.push_stream = None
        
        # Session management
        self.current_session_id = None
        self.recognition_result = ""
        self.interim_text = ""
        
        # Threading for async operations
        self.recognition_lock = asyncio.Lock()
        self.result_event = asyncio.Event()
        
        # Performance tracking
        self.processing_times = []
        self.recognition_count = 0
        
        # Callbacks
        self.on_partial_result: Optional[Callable] = None
        self.on_final_result: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

    async def initialize(self):
        """Initialize Azure Speech SDK with optimized settings for real-time processing"""
        try:
            # Check if we should use mock service
            if not self.speech_key and os.getenv("ENABLE_MOCK_SERVICES", "false").lower() == "true":
                logger.info("Using mock STT service for testing (AZURE_SPEECH_KEY not set but ENABLE_MOCK_SERVICES=true)")
                self.use_mock = True
                self.is_available = True
                return
                
            if not self.speech_key:
                logger.error("AZURE_SPEECH_KEY environment variable not set")
                self.is_available = False
                return

            logger.info(f"Initializing Azure Speech STT service for real-time processing...")
            logger.info(f"Language: {self.language}, Region: {self.speech_region}")

            # Configure Azure Speech SDK
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key, 
                region=self.speech_region
            )
            
            # Optimize for real-time streaming
            self.speech_config.speech_recognition_language = self.language
            self.speech_config.enable_dictation() if not self.profanity_filter else None
            
            # Set optimized properties for low latency
            self.speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "3000"
            )
            self.speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "500"
            )
            self.speech_config.set_property(
                speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "500"
            )
            
            # Enable streaming mode
            self.speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceResponse_RequestDetailedResultTrueFalse, "true"
            )
            
            # Setup push audio stream for real-time processing
            audio_format = speechsdk.audio.AudioStreamFormat(
                samples_per_second=16000,
                bits_per_sample=16,
                channels=1
            )
            
            self.push_stream = speechsdk.audio.PushAudioInputStream(audio_format)
            self.audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)
            
            # Create speech recognizer during initialization
            self.speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=self.audio_config
            )
            
            # Set up event handlers for real-time processing
            self._setup_recognition_handlers()

            self.is_available = True
            logger.info("✅ Azure Speech STT service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize STT service: {e}", exc_info=True)
            self.is_available = False

    async def start_continuous_recognition(self) -> str:
        """Start continuous speech recognition session"""
        if not self.is_available:
            logger.error("STT service not available")
            return ""
            
        # If using mock service, create a mock session
        if self.use_mock:
            self.current_session_id = f"mock_{str(uuid.uuid4())[:8]}"
            self.is_listening = True
            self.recognition_result = ""
            self.interim_text = ""
            logger.info(f"Mock STT session started: {self.current_session_id}")
            return self.current_session_id

        async with self.recognition_lock:
            try:
                # Create new session
                self.current_session_id = str(uuid.uuid4())[:8]
                logger.info(f"Starting speech recognition session: {self.current_session_id}")

                # Start continuous recognition using the already-configured recognizer
                await asyncio.get_event_loop().run_in_executor(
                    None, self.speech_recognizer.start_continuous_recognition
                )

                self.is_listening = True
                self.recognition_result = ""
                self.interim_text = ""
                
                logger.info(f"Continuous recognition started for session: {self.current_session_id}")
                return self.current_session_id

            except Exception as e:
                logger.error(f"Failed to start continuous recognition: {e}", exc_info=True)
                return ""

    def _setup_recognition_handlers(self):
        """Setup event handlers for speech recognition"""
        
        def recognizing_handler(evt):
            """Handle partial recognition results"""
            if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
                self.interim_text = evt.result.text
                if self.on_partial_result:
                    asyncio.create_task(self.on_partial_result(evt.result.text))
                logger.debug(f"RECOGNIZING: {evt.result.text}")

        def recognized_handler(evt):
            """Handle final recognition results"""
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                final_text = evt.result.text.strip()
                if final_text:
                    self.recognition_result = final_text
                    self.recognition_count += 1
                    if self.on_final_result:
                        asyncio.create_task(self.on_final_result(final_text))
                    logger.info(f"RECOGNIZED: {final_text}")
                else:
                    logger.info("RECOGNIZED: (empty)")
                    
                # Signal completion
                self.result_event.set()
                
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                logger.debug("No speech recognized")

        def session_started_handler(evt):
            """Handle session started event"""
            logger.debug(f"Speech recognition session started: {evt.session_id}")

        def session_stopped_handler(evt):
            """Handle session stopped event"""
            logger.debug(f"Speech recognition session stopped: {evt.session_id}")

        def canceled_handler(evt):
            """Handle recognition canceled event"""
            if evt.reason == speechsdk.CancellationReason.Error:
                logger.error(f"Speech recognition canceled due to error: {evt.error_details}")
                if self.on_error:
                    asyncio.create_task(self.on_error(evt.error_details))
            else:
                logger.info(f"Speech recognition canceled: {evt.reason}")

        # Connect handlers
        self.speech_recognizer.recognizing.connect(recognizing_handler)
        self.speech_recognizer.recognized.connect(recognized_handler)
        self.speech_recognizer.session_started.connect(session_started_handler)
        self.speech_recognizer.session_stopped.connect(session_stopped_handler)
        self.speech_recognizer.canceled.connect(canceled_handler)

    async def process_audio_chunk(self, audio_chunk: bytes) -> Dict[str, Any]:
        """Process audio chunk for speech recognition"""
        if not self.is_listening:
            return {"status": "not_listening", "text": "", "is_final": False}

        start_time = time.time()
        
        try:
            # Mock processing for testing
            if self.use_mock:
                processing_time = (time.time() - start_time) * 1000
                
                # Simulate random words being recognized based on audio energy
                if len(audio_chunk) > 100:
                    # Calculate audio energy (simplified)
                    energy = sum(abs(b) for b in audio_chunk[:100]) / 100
                    
                    # If there's significant energy, simulate speech recognition
                    if energy > 50:
                        # Set mock interim text if not set
                        if not self.interim_text:
                            mock_phrases = [
                                "hello", "testing", "one two three", 
                                "how are you", "what time is it",
                                "weather today", "set a timer"
                            ]
                            import random
                            self.interim_text = random.choice(mock_phrases)
                            
                            # Simulate final result after a few chunks
                            if random.random() > 0.7 and self.on_final_result:
                                self.recognition_result = self.interim_text
                                asyncio.create_task(self.on_final_result(self.recognition_result))
                                self.result_event.set()
                    
                    # Simulate partial results
                    if self.interim_text and self.on_partial_result:
                        asyncio.create_task(self.on_partial_result(self.interim_text))
                
                return {
                    "status": "processing",
                    "text": self.interim_text,
                    "is_final": False,
                    "processing_time": processing_time,
                    "is_mock": True
                }
            
            # Real processing with Azure
            # Write audio data to stream
            self.push_stream.write(audio_chunk)
            
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)

            return {
                "status": "processing",
                "text": self.interim_text,
                "is_final": False,
                "processing_time": processing_time
            }

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {"status": "error", "text": "", "is_final": False, "error": str(e)}

    async def finalize_recognition(self, timeout: float = 3.0) -> str:
        """Wait for final recognition result"""
        try:
            # For mock service, just return the current recognition result
            if self.use_mock:
                # If no recognition result yet, generate one
                if not self.recognition_result:
                    import random
                    mock_phrases = [
                        "Hello, how can I help you?",
                        "I heard you speaking.",
                        "The weather looks nice today.",
                        "It's time for a coffee break.",
                        "Is there anything I can do for you?"
                    ]
                    self.recognition_result = random.choice(mock_phrases)
                
                logger.info(f"Mock STT finalization complete. Final text: '{self.recognition_result}'")
                return self.recognition_result
                
            # Wait for recognition to complete
            await asyncio.wait_for(self.result_event.wait(), timeout=timeout)
            self.result_event.clear()
            
            final_text = self.recognition_result.strip()
            logger.info(f"STT finalization complete. Final text: '{final_text}'")
            
            return final_text
            
        except asyncio.TimeoutError:
            logger.warning("Recognition finalization timed out")
            return self.interim_text.strip()
        except Exception as e:
            logger.error(f"Error during recognition finalization: {e}")
            return ""

    async def stop_continuous_recognition(self):
        """Stop continuous speech recognition"""
        if not self.is_listening:
            return

        # For mock service, just reset state
        if self.use_mock:
            self.is_listening = False
            logger.info(f"Mock STT session stopped: {self.current_session_id}")
            return

        async with self.recognition_lock:
            try:
                if self.speech_recognizer:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.speech_recognizer.stop_continuous_recognition
                    )
                    
                if self.push_stream:
                    self.push_stream.close()
                    
                self.is_listening = False
                logger.info(f"Speech recognition stopped for session: {self.current_session_id}")
                
            except Exception as e:
                logger.error(f"Error stopping recognition: {e}")

    async def reset_session(self):
        """Reset recognition session for new input"""
        await self.stop_continuous_recognition()
        
        # Reset state
        self.recognition_result = ""
        self.interim_text = ""
        self.current_session_id = None
        
        # Recreate stream
        if self.is_available:
            audio_format = speechsdk.audio.AudioStreamFormat(
                samples_per_second=16000,
                bits_per_sample=16,
                channels=1
            )
            self.push_stream = speechsdk.audio.PushAudioInputStream(audio_format)
            self.audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)

    def set_callbacks(self, 
                     on_partial: Optional[Callable] = None,
                     on_final: Optional[Callable] = None, 
                     on_error: Optional[Callable] = None):
        """Set callback functions for recognition events"""
        self.on_partial_result = on_partial
        self.on_final_result = on_final
        self.on_error = on_error

    async def get_status(self) -> Dict[str, Any]:
        """Get current STT service status"""
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times) 
            if self.processing_times else 0.0
        )
        
        return {
            "available": self.is_available,
            "service_type": "Azure_Speech_STT",
            "is_listening": self.is_listening,
            "is_processing": self.is_processing,
            "language": self.language,
            "current_session": self.current_session_id,
            "recognition_count": self.recognition_count,
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "interim_text": self.interim_text[:50] + "..." if len(self.interim_text) > 50 else self.interim_text
        }

    async def cleanup(self):
        """Cleanup STT resources"""
        logger.info("Cleaning up STT service...")
        try:
            await self.stop_continuous_recognition()
            
            if self.speech_recognizer:
                self.speech_recognizer = None
            if self.push_stream:
                self.push_stream = None
            if self.audio_config:
                self.audio_config = None
                
            self.is_available = False
            logger.info("STT service cleanup completed") 
        except Exception as e:
            logger.error(f"Error during STT cleanup: {e}", exc_info=True) 