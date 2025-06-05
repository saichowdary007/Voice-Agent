import asyncio
import time
import logging
import os
from typing import Optional, Dict, Any, List
import azure.cognitiveservices.speech as speechsdk
import numpy as np
from dataclasses import dataclass
import structlog

# Configuration from environment or settings
from app.config import settings

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
    """Microsoft Azure Speech-to-Text service implementation"""
    
    def __init__(self, api_key: Optional[str] = None, region: Optional[str] = None):
        self.api_key = api_key or os.getenv('AZURE_SPEECH_KEY', settings.azure_speech_key)
        self.region = region or os.getenv('AZURE_SPEECH_REGION', settings.azure_speech_region)
        self.language = settings.azure_speech_language
        self.sample_rate = settings.sample_rate
        self.channels = settings.channels
        
        # Speech recognition state
        self.is_available = False
        self.is_initialized = False
        self.partial_result = ""
        self.latest_partial_result = ""
        self.final_result = ""
        
        # Azure Speech SDK components
        self.speech_config = None
        self.audio_config = None
        self.speech_recognizer = None
        self.push_stream = None
        
        # Session management
        self.is_listening = False
        self.current_session_id = None
        self.buffer = bytearray()
        self.lock = asyncio.Lock()  # Add a lock for thread safety
        self.last_activity_time = time.time()
        
        logger.info(f"Azure STT Service initialized with region: {self.region}, language: {self.language}")
        
    async def initialize(self):
        """Initialize the Azure STT service"""
        try:
            if not self.api_key:
                logger.error("Azure Speech API key not provided. Service will not be available.")
                self.is_available = False
                return
                
            # Create speech config with the specified Azure region
            self.speech_config = speechsdk.SpeechConfig(subscription=self.api_key, region=self.region)
            self.speech_config.speech_recognition_language = self.language
            
            # Configure timeouts for silence detection
            self.speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, 
                str(settings.azure_speech_initial_silence_timeout_ms)
            )
            self.speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, 
                str(settings.azure_speech_end_silence_timeout_ms)
            )
            
            # Create an audio configuration with a custom push stream
            self.push_stream = speechsdk.audio.PushAudioInputStream()
            self.audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)
            
            # Create speech recognizer
            self.speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config, 
                audio_config=self.audio_config
            )
            
            # Set up event handlers
            self.speech_recognizer.recognizing.connect(self._recognizing_callback)
            self.speech_recognizer.recognized.connect(self._recognized_callback)
            self.speech_recognizer.session_started.connect(self._session_started_callback)
            self.speech_recognizer.session_stopped.connect(self._session_stopped_callback)
            self.speech_recognizer.canceled.connect(self._canceled_callback)
            
            # Start continuous recognition
            await self._restart_continuous_recognition()
            
            self.is_initialized = True
            self.is_available = True
            logger.info(f"Azure STT service initialized successfully with language: {self.language}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure STT service: {e}", exc_info=True)
            self.is_available = False
    
    async def _restart_continuous_recognition(self):
        """Safely restart continuous recognition"""
        try:
            if self.speech_recognizer:
                if self.is_listening:
                    logger.debug("Stopping existing continuous recognition session")
                    self.speech_recognizer.stop_continuous_recognition_async()
                    await asyncio.sleep(0.2)  # Short wait to ensure it stops
                
                logger.debug("Starting continuous recognition session")
                self.speech_recognizer.start_continuous_recognition_async()
                self.is_listening = True
        except Exception as e:
            logger.error(f"Error restarting continuous recognition: {e}", exc_info=True)
            
    def _recognizing_callback(self, evt):
        """Callback for partial recognition results"""
        if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
            self.partial_result = evt.result.text
            self.latest_partial_result = evt.result.text
            self.last_activity_time = time.time()
            logger.debug(f"RECOGNIZING: {self.partial_result}")
        
    def _recognized_callback(self, evt):
        """Callback for final recognition results"""
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            self.final_result = evt.result.text  # Store the final result
            self.partial_result = ""  # Clear partial result
            self.last_activity_time = time.time()
            logger.info(f"RECOGNIZED: {self.final_result}")
            
    def _session_started_callback(self, evt):
        """Callback for session start events"""
        self.current_session_id = evt.session_id
        logger.info(f"Speech recognition session started: {self.current_session_id}")
        
    def _session_stopped_callback(self, evt):
        """Callback for session stop events"""
        logger.info(f"Speech recognition session stopped: {self.current_session_id}")
        self.current_session_id = None
        
    def _canceled_callback(self, evt):
        """Callback for cancellation events"""
        logger.warning(f"Speech recognition canceled: {evt.reason}")
        if evt.reason == speechsdk.CancellationReason.Error:
            logger.error(f"Error details: {evt.error_details}")
            # Try to automatically recover from error
            asyncio.create_task(self._restart_continuous_recognition())
            
    async def process_frame(self, audio_frame: bytes) -> STTResult:
        """Process a single audio frame for incremental speech recognition"""
        if not self.is_available or not self.is_initialized:
            logger.warning("Azure STT service not available or not initialized")
            return STTResult()
            
        try:
            async with self.lock:  # Use lock for thread safety
                # Add frame to buffer
                self.buffer.extend(audio_frame)
                
                # Push audio data to the stream
                if self.push_stream and self.is_listening:
                    self.push_stream.write(audio_frame)
                    
                # Return the latest partial result
                partial_text = self.latest_partial_result
                
                # Don't clear the latest partial result until finalization
                # This ensures we don't lose partial transcriptions between frames
                
                return STTResult(partial_text=partial_text)
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}", exc_info=True)
            return STTResult()
            
    async def finalize(self) -> STTResult:
        """Finalize speech recognition and get final transcript"""
        if not self.is_available or not self.is_initialized:
            logger.warning("Azure STT service not available or not initialized for finalization")
            return STTResult()
            
        try:
            async with self.lock:  # Use lock for thread safety
                # Signal end of stream
                if self.push_stream:
                    # Push a bit of silence to ensure recognition completes
                    silence = bytes([0] * 3200)  # 100ms of silence at 16kHz
                    self.push_stream.write(silence)
                    
                # Wait for final results
                await asyncio.sleep(0.5)
                
                # First try to get the final result from the callback
                final_text = self.final_result
                
                # If no final result, use the latest partial result
                if not final_text:
                    final_text = self.partial_result
                
                # Clear all results for the next utterance
                self.partial_result = ""
                self.latest_partial_result = ""
                self.final_result = ""
                
                # Reset recognizer to ensure a clean state
                await self._restart_continuous_recognition()
                
                # Clear buffer
                self.buffer.clear()
                
                logger.info(f"STT finalization complete. Final text: '{final_text}'")
                return STTResult(final_text=final_text)
        except Exception as e:
            logger.error(f"Error finalizing speech recognition: {e}", exc_info=True)
            return STTResult()
            
    async def reset_stream(self):
        """Reset the speech recognition stream"""
        try:
            async with self.lock:
                if self.is_initialized and self.speech_recognizer:
                    self.partial_result = ""
                    self.latest_partial_result = ""
                    self.final_result = ""
                    self.buffer.clear()
                    
                    # Restart recognition
                    await self._restart_continuous_recognition()
                    logger.info("Azure STT stream reset successfully")
        except Exception as e:
            logger.error(f"Error resetting Azure STT stream: {e}", exc_info=True)
            
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the STT service"""
        return {
            "available": self.is_available,
            "service": "Azure Speech Service",
            "sample_rate": self.sample_rate,
            "model_path": "Microsoft Azure Cloud",
            "streaming": True,
            "region": self.region
        }
        
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.is_initialized and self.speech_recognizer:
                self.speech_recognizer.stop_continuous_recognition_async()
                
            self.is_listening = False
            self.is_available = False
            self.is_initialized = False
            self.buffer.clear()
            self.partial_result = ""
            self.latest_partial_result = ""
            
            # Help GC by clearing references
            self.speech_recognizer = None
            self.audio_config = None
            self.speech_config = None
            self.push_stream = None
            
            logger.info("Azure STT service cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up Azure STT service: {e}", exc_info=True) 