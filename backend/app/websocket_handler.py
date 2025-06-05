import asyncio
import json
import time
import os
from typing import Dict, Any, Optional, Set
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect # Ensure WebSocketDisconnect is imported
from fastapi.websockets import WebSocketState

from .ai.context_manager import ContextManager
from .config import settings
from services.vad_service import VADResult  # Import VADResult class


class WebSocketHandler:
    """Handles WebSocket connections and message routing for voice agent"""
    
    def __init__(self, websocket: WebSocket, services: Dict, metrics: Any, logger: Any):
        self.websocket = websocket
        self.services = services
        self.metrics = metrics # Assuming metrics object is passed and has methods like record_latency
        self.logger = logger # This should be a session-specific logger instance
        
        self.session_id = f"session_{int(time.time() * 1000)}_{websocket.client.host}_{websocket.client.port}"
        # Re-bind if a generic logger was passed, to ensure session_id is included.
        # If logger is already session-specific, this re-bind might be redundant but harmless.
        self.session_logger = self.logger.bind(session_id=self.session_id, client_ip=websocket.client.host)
        
        self.session_logger.info(f"WebSocketHandler initialized for session.")
        self.is_muted = False
        self.is_speaking = False # Tracks if client VAD (user) is speaking
        self.audio_buffer = bytearray()
        self.context_manager = ContextManager() # Each handler gets its own context
        
        self.small_chunk_buffer = bytearray()
        self.failed_chunk_buffer = bytearray()
        self.failed_chunk_retry_count = 0
        self.max_failed_chunk_retries = settings.max_failed_chunk_retries
        
        self.speech_start_time: Optional[float] = None
        self.processing_start_time: Optional[float] = None
        
        self.tts_active = False # Tracks if backend is currently generating/sending TTS audio
        self.should_interrupt_tts = False # Flag to signal TTS interruption, e.g., for barge-in
        
        self.processing_tasks: Set[asyncio.Task] = set()
        self.last_audio_time: float = time.time() # For watchdog
        self.pipeline_running: bool = False # True if STT, AI, or TTS is active
        self.watchdog_task: Optional[asyncio.Task] = None
        
    async def handle_connection(self):
        """Main connection handler loop"""
        self.watchdog_task = asyncio.create_task(self._watchdog_timer())
        try:
            while self.websocket.client_state == WebSocketState.CONNECTED:
                message = await self.websocket.receive()
                
                if message["type"] == "websocket.disconnect":
                    self.session_logger.info("WebSocket disconnect message received by handler.")
                    break 
                    
                if "bytes" in message and message["bytes"] is not None:
                    self.session_logger.debug(f"Received binary message: {len(message['bytes'])} bytes")
                    await self.handle_audio_chunk(message["bytes"])
                elif "text" in message and message["text"] is not None:
                    self.session_logger.debug(f"Received text message: {message['text'][:100]}...")
                    try:
                        data = json.loads(message["text"])
                        await self.handle_control_message(data)
                        if data.get("type") == "control" and data.get("action") == "end_session":
                            # If end_session is processed, we might want to break the loop
                            # The actual closing is handled by main.py's finally block now
                            self.session_logger.info("end_session control message processed, handler loop will exit.")
                            break 
                    except json.JSONDecodeError as e:
                        self.session_logger.warning(f"Invalid JSON received: {message['text']}, error: {e}")
                else:
                    self.session_logger.warning(f"Unknown message format or null content: {message}")
                        
        except WebSocketDisconnect:
            self.session_logger.info("WebSocket client disconnected (WebSocketDisconnect caught in handler).")
        except ConnectionResetError:
            self.session_logger.warning("WebSocket connection reset by client (ConnectionResetError caught in handler).")
        except Exception as e:
            self.session_logger.error(f"Unexpected error in WebSocket handler_connection: {e}", exc_info=True)
        finally: 
            if self.watchdog_task and not self.watchdog_task.done():
                self.watchdog_task.cancel()
            self.session_logger.info("Exiting handle_connection loop in WebSocketHandler.")
            # Cleanup tasks are managed by the main endpoint's finally block calling handler.cleanup()

    async def handle_audio_chunk(self, audio_data: bytes):
        """Handle incoming audio chunk from WebSocket using AudioService."""
        try:
            self.session_logger.debug(f"Handling audio chunk: {len(audio_data)} bytes")
            
            if not audio_data or len(audio_data) < 100: # Basic validation
                self.session_logger.debug(f"Rejecting small/empty audio chunk: {len(audio_data) if audio_data else 0} bytes")
                return
                
            # Skip small chunk buffering - let AudioService handle it with its chunk buffering
            self.last_audio_time = time.time() # Update activity time

            # Add timeout and better error handling for audio conversion
            try:
                self.session_logger.info(f"Converting audio chunk of {len(audio_data)} bytes using AudioService")
                pcm_data = await asyncio.wait_for(
                    self._convert_audio_via_service(audio_data), 
                    timeout=settings.audio_conversion_timeout  # Use configured timeout
                )
                self.session_logger.info(f"Audio conversion completed, received {len(pcm_data) if pcm_data else 0} bytes of PCM data")
            except asyncio.TimeoutError:
                self.session_logger.warning(f"Audio conversion timed out for chunk of {len(audio_data)} bytes")
                # Don't send error to client for individual chunk failures
                return
            except Exception as e:
                self.session_logger.error(f"Audio conversion failed: {e}", exc_info=True)
                # Don't send error to client for individual chunk failures
                return

            if not pcm_data:
                # Error already logged by _convert_audio_via_service if it fails consistently
                self.session_logger.debug(f"Audio conversion by AudioService returned None for chunk of {len(audio_data)} bytes (may be buffering).")
                return

            if len(pcm_data) < (settings.sample_rate * settings.audio_frame_ms // 1000 * 2 * settings.channels / 2) : # Min e.g. 10ms
                self.session_logger.warning(f"PCM data too short after conversion: {len(pcm_data)} bytes, skipping processing for this chunk.")
                return

            # Add to buffer with error handling
            try:
                self.session_logger.debug(f"Extending audio buffer with {len(pcm_data)} bytes of PCM data")
                self.audio_buffer.extend(pcm_data)
            except Exception as e:
                self.session_logger.error(f"Error extending audio buffer: {e}", exc_info=True)
                self.audio_buffer = bytearray(pcm_data)  # Reset and try again

            # Process audio with timeout and error recovery
            try:
                self.session_logger.debug(f"Processing audio buffer of size {len(self.audio_buffer)} bytes")
                await asyncio.wait_for(self._process_audio_buffer(), timeout=settings.audio_processing_timeout)
                self.session_logger.debug(f"Audio buffer processing completed successfully")
            except asyncio.TimeoutError:
                self.session_logger.warning("Audio buffer processing timed out")
                # Clear buffer to prevent backlog
                self.audio_buffer.clear()
            except Exception as e:
                self.session_logger.error(f"Audio buffer processing failed: {e}", exc_info=True)
                # Clear buffer to prevent further issues
                self.audio_buffer.clear()

        except Exception as e:
            self.session_logger.error(f"Critical error in handle_audio_chunk: {e}", exc_info=True)
            # Don't re-raise to prevent WebSocket disconnection
            # Clear audio buffer to prevent issues with subsequent chunks
            self.audio_buffer.clear()

    async def _process_audio_buffer(self):
        """Process accumulated audio buffer with frame extraction."""
        frame_size_bytes = self._get_frame_size() 
        
        while len(self.audio_buffer) >= frame_size_bytes:
            frame_to_process = bytes(self.audio_buffer[:frame_size_bytes])
            self.audio_buffer = self.audio_buffer[frame_size_bytes:]
            
            self.session_logger.debug(f"Extracted frame of {len(frame_to_process)} bytes for processing. Buffer remaining: {len(self.audio_buffer)}")
            
            # Process frame with error handling to prevent task accumulation
            try:
                task = asyncio.create_task(self._process_audio_frame(frame_to_process))
                self.processing_tasks.add(task)
                task.add_done_callback(self.processing_tasks.discard) # Auto-remove task when done
            except Exception as e:
                self.session_logger.error(f"Failed to create audio processing task: {e}", exc_info=True)

    async def _process_audio_frame(self, frame: bytes):
        """Process a single PCM audio frame through VAD and potentially STT/AI pipeline."""
        self.pipeline_running = True # Indicate that some processing is happening
        try:
            self.session_logger.debug(f"Processing frame: {len(frame)} bytes. Current user speaking state: {self.is_speaking}")
            
            # Skip processing if muted
            if self.is_muted:
                self.session_logger.debug("Frame processing skipped - session is muted")
                self.pipeline_running = False
                return
            
            vad_service = self.services.get('vad')
            stt_service = self.services.get('stt')

            if not vad_service or not stt_service:
                self.session_logger.error("VAD or STT service not available.")
                await self._send_error("Voice processing services are currently unavailable.")
                self.pipeline_running = False
                return

            # Wrap VAD processing in try/except to prevent server crashes
            try:
                vad_result = await vad_service.process_frame(frame)
                self.session_logger.debug(f"VAD: speech={vad_result.is_speech}, end={vad_result.is_end_of_speech}, conf={vad_result.confidence:.2f}")
            except Exception as vad_e:
                self.session_logger.error(f"VAD processing error: {vad_e}", exc_info=True)
                # Create a default negative result if VAD fails
                vad_result = VADResult(is_speech=False, is_end_of_speech=False, confidence=0.0, timestamp=0.0)
            
            if vad_result.is_speech:
                if not self.is_speaking: # Start of user's speech segment
                    self.is_speaking = True
                    self.speech_start_time = time.time()
                    self.session_logger.info("User speech segment started (VAD).")
                    if self.tts_active: # AI is currently speaking
                        self.session_logger.info("Barge-in: User started speaking during TTS.")
                        await self._handle_barge_in()
                
                # Wrap STT processing in try/except to prevent server crashes
                try:        
                    stt_result = await stt_service.process_frame(frame)
                    if stt_result.partial_text:
                        await self._send_message({"type": "transcript", "partial": stt_result.partial_text})
                except Exception as stt_e:
                    self.session_logger.error(f"STT processing error: {stt_e}", exc_info=True)
                    # Continue without speech-to-text if it fails
            
            elif vad_result.is_end_of_speech and self.is_speaking: # End of user's speech segment
                self.session_logger.info("User speech segment ended (VAD). Finalizing STT.")
                self.is_speaking = False # Reset user speaking state
                
                final_stt_text = None
                try:
                    if hasattr(stt_service, 'finalize') and callable(stt_service.finalize):
                        final_result_obj = await stt_service.finalize()
                        if final_result_obj and final_result_obj.final_text:
                            final_stt_text = final_result_obj.final_text.strip()
                except Exception as finalize_e:
                    self.session_logger.error(f"STT finalization error: {finalize_e}", exc_info=True)
                
                if final_stt_text:
                    self.session_logger.info(f"STT Final: '{final_stt_text}'")
                    await self._send_message({"type": "transcript", "final": final_stt_text})
                    # Create a new task for AI processing to not block this audio frame loop
                    ai_task = asyncio.create_task(self._process_with_ai(final_stt_text))
                    self.processing_tasks.add(ai_task)
                    ai_task.add_done_callback(self.processing_tasks.discard)
                else:
                    self.session_logger.info("STT finalization yielded no text after VAD EOS.")
                    self.pipeline_running = False # No AI to run, pipeline idle for now
            
            # If it's silence but not end of a speech segment, and no TTS is active.
            elif not vad_result.is_speech and not self.is_speaking and not self.tts_active:
                 self.pipeline_running = False

        except Exception as e:
            self.session_logger.error(f"Error in _process_audio_frame: {e}", exc_info=True)
            await self._send_error("Error during voice processing.")
            self.pipeline_running = False # Reset on error
        # Note: pipeline_running is primarily set to False by AI/TTS completion or specific idle conditions.

    async def _handle_barge_in(self):
        """Handle user interrupting AI speech (barge-in)"""
        try:
            self.should_interrupt_tts = True
            
            # Stop TTS service if it has a stop method
            tts_service = self.services.get('tts')
            if tts_service and hasattr(tts_service, 'stop'):
                await tts_service.stop()
                
            # Notify client to stop audio playback
            await self._send_message({"type": "stop_audio"})
            
            self.tts_active = False
            self.session_logger.info("Barge-in handling completed")
            
        except Exception as e:
            self.session_logger.error(f"Error in barge-in handling: {e}")

    async def _process_with_ai(self, text: str):
        """Process transcript with AI and generate response."""
        try:
            self.session_logger.info(f"Processing transcript with AI: {text}")
            llm_service = self.services.get('llm')
            
            if not llm_service:
                self.session_logger.error("LLM service not available for AI processing")
                await self._send_error("AI response service is currently unavailable.")
                return
            
            # Add AI processing time measurement
            self.processing_start_time = time.time()
            
            messages = self.context_manager.get_messages_for_llm(text)
            
            # Create a task for AI response generation with timeout handling
            try:
                async def generate_ai_response():
                    async for chunk in llm_service.generate_response_stream(messages):
                        if self.should_interrupt_tts:
                            self.session_logger.info("AI response generation interrupted by client.")
                            break
                        
                        # Add chunk to context manager
                        self.context_manager.add_ai_response_chunk(chunk)
                        
                        # Send partial response to frontend
                        await self._send_message({
                            "type": "ai_response",
                            "text": chunk,
                            "final": False
                        })
                    
                    # Send TTS only after AI generation is complete and not interrupted
                    if not self.should_interrupt_tts:
                        full_response = self.context_manager.get_last_ai_response()
                        await self._start_tts_streaming(full_response)
                        
                        # Send final response
                        await self._send_message({
                            "type": "ai_response",
                            "text": full_response,
                            "final": True
                        })
                        
                        # Record total time if we track performance
                        if self.processing_start_time and self.metrics:
                            processing_time = time.time() - self.processing_start_time
                            self.metrics.record_latency('ai_response_time', processing_time)
                            self.session_logger.info(f"AI processing completed in {processing_time:.2f}s")
                
                # Use the configured timeout from settings
                await asyncio.wait_for(generate_ai_response(), timeout=settings.ai_response_timeout)
                
            except asyncio.TimeoutError:
                self.session_logger.error(f"AI response generation timed out after {settings.ai_response_timeout}s")
                await self._send_error("AI response generation timed out. Please try again.")
            except Exception as e:
                self.session_logger.error(f"Error during AI response generation: {e}", exc_info=True)
                await self._send_error("An error occurred while generating the AI response.")
        
        except Exception as e:
            self.session_logger.error(f"Error in AI processing pipeline: {e}", exc_info=True)
            await self._send_error("An error occurred during AI processing.")

    async def _start_tts_streaming(self, text: str):
        """Start text-to-speech generation and streaming"""
        try:
            self.tts_active = True
            await self._send_message({"type": "tts_start"})
            
            tts_service = self.services.get('tts')
            if not tts_service:
                await self._send_error("TTS service is currently unavailable")
                self.tts_active = False
                self.pipeline_running = False
                return
                
            # Generate and stream TTS audio
            async for audio_chunk in tts_service.generate_speech_stream(text):
                if self.should_interrupt_tts:
                    self.session_logger.info("TTS streaming interrupted")
                    break
                    
                # Send audio chunk to client - use base64 encoding for binary data
                import base64
                audio_base64 = base64.b64encode(audio_chunk).decode('utf-8')
                await self._send_message({
                    "type": "audio_chunk",
                    "audio_data": audio_base64,
                    "encoding": "base64"
                })
                
            # TTS complete
            await self._send_message({"type": "tts_complete"})
            self.tts_active = False
            
            # Reset interruption flag
            self.should_interrupt_tts = False
            
            # Record metrics if available
            if self.metrics and self.processing_start_time:
                total_time = time.time() - self.processing_start_time
                self.metrics.record_latency('total_processing', total_time)
                
            self.pipeline_running = False # Pipeline complete
            
        except Exception as e:
            self.session_logger.error(f"Error in TTS streaming: {e}", exc_info=True)
            await self._send_error("Error generating speech response")
            self.tts_active = False
            self.pipeline_running = False

    async def handle_control_message(self, data: Dict[str, Any]):
        """Handle control messages from frontend (mute, unmute, end_session, ping, eos)."""
        message_type = data.get("type")
        try:
            if message_type == "control":
                action = data.get("action")
                self.session_logger.info(f"Received control action: {action}")
                if action == "mute":
                    self.is_muted = True
                    await self._send_message({"type": "mute_status", "muted": True})
                elif action == "unmute":
                    self.is_muted = False
                    await self._send_message({"type": "mute_status", "muted": False})
                elif action == "end_session":
                    self.session_logger.info("Control: User requested end_session.")
                    # Signal to main loop to close by closing from here after sending confirmation
                    await self._send_message({"type": "control", "action": "session_ended"})
                    if self.websocket.client_state == WebSocketState.CONNECTED:
                         await self.websocket.close(code=1000, reason="Session ended by user")
                    # The handle_connection loop will break upon WebSocketDisconnect

            elif message_type == "mute":
                # Direct mute message format
                muted = data.get("muted", True)
                self.is_muted = muted
                self.session_logger.info(f"Received direct mute message: muted={muted}")
                await self._send_message({"type": "mute_status", "muted": muted})
                
                # If muting, pause watchdog timer by updating last activity
                if muted:
                    self.last_audio_time = time.time()

            elif message_type == "ping":
                self.last_audio_time = time.time() # Treat ping as activity
                pong_response = {"type": "pong"}
                if "timestamp" in data: pong_response["timestamp"] = data["timestamp"]
                await self._send_message(pong_response)
                self.session_logger.debug(f"Responded to ping with {pong_response}")

            elif message_type == "text_command":
                # Direct text command without going through STT
                text = data.get("text", "").strip()
                if text:
                    self.session_logger.info(f"Received direct text command: '{text}'")
                    await self._send_message({"type": "transcript", "final": text})
                    # Create a task for AI processing
                    ai_task = asyncio.create_task(self._process_with_ai(text))
                    self.processing_tasks.add(ai_task)
                    ai_task.add_done_callback(self.processing_tasks.discard)
                else:
                    self.session_logger.warning("Received empty text command")
                    await self._send_error("Empty text command received")

            elif message_type == "eos": # Client explicitly signals end of their speech
                self.session_logger.info("Client signaled EOS (End of Speech).")
                self.is_speaking = False # User has finished speaking based on client EOS

                stt_service = self.services.get('stt')
                if stt_service:
                    final_stt_result_obj = await stt_service.finalize()
                    if final_stt_result_obj and final_stt_result_obj.final_text and final_stt_result_obj.final_text.strip():
                        final_text = final_stt_result_obj.final_text.strip()
                        self.session_logger.info(f"STT Final (from client EOS): '{final_text}'")
                        await self._send_message({"type": "transcript", "final": final_text})
                        # Create a task for AI processing to avoid blocking
                        ai_task = asyncio.create_task(self._process_with_ai(final_text))
                        self.processing_tasks.add(ai_task)
                        ai_task.add_done_callback(self.processing_tasks.discard)
                    else:
                        self.session_logger.info("STT finalization on client EOS yielded no significant text.")
                        self.pipeline_running = False # No text, pipeline idle
                else:
                    self.session_logger.error("STT service not available for EOS finalization.")
                    self.pipeline_running = False
            else:
                self.session_logger.warning(f"Unknown message type received: {message_type}, data: {data}")
        except Exception as e:
            self.session_logger.error(f"Error handling control message (type: {message_type}): {e}", exc_info=True)
            await self._send_error(f"Failed to process your request: {str(e)}")
            
    async def _send_message(self, data: Dict[str, Any]):
        """Send JSON message to frontend, checking WebSocket state."""
        if self.websocket.client_state == WebSocketState.CONNECTED:
            try:
                await asyncio.wait_for(
                    self.websocket.send_text(json.dumps(data)), 
                    timeout=settings.message_send_timeout
                )
            except asyncio.TimeoutError:
                self.session_logger.error(f"Message send timed out (type: {data.get('type', 'Unknown')})")
                # Don't raise exception to prevent WebSocket disconnection
            except ConnectionResetError:
                self.session_logger.warning("Connection reset while sending message")
                # Connection is lost, stop trying to send
            except Exception as e: # Catch errors during send_text specifically
                self.session_logger.error(f"Failed to send message (type: {data.get('type', 'Unknown')}): {e}", exc_info=True)
                # Don't re-raise to prevent WebSocket disconnection
        else:
            self.session_logger.warning(f"Attempted to send message on a non-connected WebSocket. State: {self.websocket.client_state.name}. Data: {str(data)[:100]}")
            
    async def _send_error(self, error_message: str):
        """Send error message to frontend, checking WebSocket state."""
        self.session_logger.error(f"Sending error to client: {error_message}") # Log the error being sent
        await self._send_message({"type": "error", "message": error_message})
        
    def _get_frame_size(self) -> int:
        """Calculate frame size in bytes based on standardized configuration."""
        return settings.sample_rate * settings.audio_frame_ms // 1000 * settings.channels * 2
        
    async def cleanup(self):
        """Clean up resources when WebSocket connection is closed or handler is destroyed."""
        self.session_logger.info(f"Initiating cleanup for WebSocketHandler session.")
        
        if self.watchdog_task and not self.watchdog_task.done():
            self.watchdog_task.cancel()
            try: await self.watchdog_task # Allow cancellation to propagate
            except asyncio.CancelledError: self.session_logger.info("Watchdog task successfully cancelled.")
            except Exception as e: self.session_logger.error(f"Error awaiting watchdog task cancellation: {e}", exc_info=True)
        
        self.should_interrupt_tts = True # Signal any pending TTS to stop
        tts_service = self.services.get('tts')
        if self.tts_active and tts_service and hasattr(tts_service, 'stop'):
            self.session_logger.info("Stopping active TTS during cleanup.")
            try: await tts_service.stop()
            except Exception as e: self.session_logger.error(f"Error stopping TTS service during cleanup: {e}", exc_info=True)
        self.tts_active = False

        if self.processing_tasks:
            self.session_logger.info(f"Cancelling {len(self.processing_tasks)} active processing tasks.")
            for task in list(self.processing_tasks):
                if not task.done():
                    task.cancel()
            results = await asyncio.gather(*self.processing_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, asyncio.CancelledError):
                    self.session_logger.debug(f"Task {i} was cancelled.")
                elif isinstance(result, Exception):
                    self.session_logger.error(f"Error in gathered task {i} during cleanup: {result}", exc_info=result)
            self.processing_tasks.clear()
        
        # Reset states of services that maintain per-session/stream state
        stt_service = self.services.get('stt')
        if stt_service and hasattr(stt_service, 'reset_stream'):
            try: stt_service.reset_stream()
            except Exception as e: self.session_logger.error(f"Error resetting STT stream: {e}", exc_info=True)
        
        vad_service = self.services.get('vad')
        if vad_service and hasattr(vad_service, 'reset_state'):
            try: vad_service.reset_state()
            except Exception as e: self.session_logger.error(f"Error resetting VAD state: {e}", exc_info=True)

        # Reset audio service session state
        audio_service = self.services.get('audio')
        if audio_service and hasattr(audio_service, 'reset_session'):
            try: 
                audio_service.reset_session()
                self.session_logger.info("Audio service session state reset")
            except Exception as e: 
                self.session_logger.error(f"Error resetting audio service session: {e}", exc_info=True)

        self.pipeline_running = False
        self.is_speaking = False
        self.audio_buffer.clear()
        self.small_chunk_buffer.clear()
        self.failed_chunk_buffer.clear()
        self.failed_chunk_retry_count = 0
        
        self.session_logger.info(f"WebSocketHandler session cleanup completed.")

    async def _convert_audio_via_service(self, audio_data: bytes) -> Optional[bytes]:
        """Centralized audio conversion using AudioService with async processing."""
        try:
            audio_service = self.services.get('audio')
            if not audio_service or not hasattr(audio_service, 'extract_pcm_smart_async'):
                self.session_logger.error("AudioService is not available or misconfigured.")
                return None

            # Use the async version with chunk buffering
            pcm_array = await audio_service.extract_pcm_smart_async(audio_data)
            
            if pcm_array is not None:
                # Convert to 16-bit PCM bytes
                pcm_int16 = (pcm_array * 32768.0).astype(np.int16)
                return pcm_int16.tobytes()
            else:
                # None return is normal for buffering - don't log as error
                return None
            
        except Exception as e:
            self.session_logger.error(f"Critical error in _convert_audio_via_service: {e}", exc_info=True)
            return None

    async def _watchdog_timer(self):
        """Watchdog timer to detect inactivity and manage session lifecycle"""
        self.session_logger.info(f"Watchdog timer started for session {self.session_id}. Check interval: {settings.watchdog_check_interval}s, Inactivity timeout: {settings.watchdog_inactivity_timeout}s.")
        try:
            while True:
                await asyncio.sleep(settings.watchdog_check_interval)
                current_time = time.time()

                if self.websocket.client_state != WebSocketState.CONNECTED:
                    self.session_logger.info(f"Watchdog: WebSocket no longer connected (State: {self.websocket.client_state.name}). Stopping watchdog.")
                    break
                
                # Conditions for inactivity timeout:
                # 1. Time since last audio received exceeds watchdog_inactivity_timeout.
                # 2. Backend pipeline is NOT running (i.e., not in STT, AI, or TTS processing).
                # 3. User is NOT currently speaking (client-side VAD is not active).
                # 4. TTS is NOT currently active from backend.
                # 5. Session is NOT muted (muted sessions should not timeout due to inactivity)
                is_inactive = (current_time - self.last_audio_time > settings.watchdog_inactivity_timeout)
                
                if is_inactive and not self.pipeline_running and not self.is_speaking and not self.tts_active and not self.is_muted:
                    self.session_logger.warning(
                        f"Watchdog: Inactivity timeout. Last audio: {self.last_audio_time:.2f} ({(current_time - self.last_audio_time):.2f}s ago). "
                        f"Pipeline: {self.pipeline_running}, UserSpeaking(VAD): {self.is_speaking}, TTSActive: {self.tts_active}, Muted: {self.is_muted}. Closing WebSocket."
                    )
                    if self.websocket.client_state == WebSocketState.CONNECTED:
                        # Close with a specific code if desired, e.g., 4008 for Policy Violation if inactivity is against policy
                        # Or 1008 (Policy Violation) or custom codes. 1011 is "Server Error" which might be misleading.
                        # Using 1000 (Normal Closure) or 1001 (Going Away) if server is initiating due to policy.
                        # Let's use 1000 and a clear reason.
                        await self.websocket.close(code=1000, reason="Session inactivity timeout")
                    break # Exit watchdog loop
                elif is_inactive:
                    self.session_logger.debug(
                        f"Watchdog: Inactivity detected ({ (current_time - self.last_audio_time):.1f}s), but activity/mute active. "
                        f"Pipeline: {self.pipeline_running}, UserSpeaking(VAD): {self.is_speaking}, TTSActive: {self.tts_active}, Muted: {self.is_muted}."
                    )
                else:
                    self.session_logger.debug(f"Watchdog: Activity within timeout. Last audio { (current_time - self.last_audio_time):.1f}s ago.")

        except asyncio.CancelledError:
            self.session_logger.info("Watchdog timer task was cancelled.")
        except Exception as e:
            self.session_logger.error(f"Watchdog timer encountered an unhandled error: {e}", exc_info=True)
        finally:
            self.session_logger.info(f"Watchdog timer stopped for session {self.session_id}.")

    async def _generate_tts(self, text: str):
        """Generate TTS audio with proper error handling."""
        try:
            tts_service = self.services.get('tts')
            if not tts_service:
                self.session_logger.warning("TTS service not available")
                return

            self.tts_active = True
            await self._send_message({"type": "tts_start"})
            
            # Import base64 for encoding audio data
            import base64
            
            # Generate TTS audio
            async for audio_chunk in tts_service.generate_audio_stream(text):
                if self.should_interrupt_tts:
                    self.session_logger.info("TTS interrupted by user")
                    break
                    
                if audio_chunk:
                    # Encode binary audio data as base64 string
                    audio_base64 = base64.b64encode(audio_chunk).decode('utf-8')
                    await self._send_message({
                        "type": "audio_chunk",
                        "audio_data": audio_base64,
                        "encoding": "base64"
                    })
            
            if not self.should_interrupt_tts:
                await self._send_message({"type": "tts_complete"})
                
        except Exception as e:
            self.session_logger.error(f"TTS generation error: {e}", exc_info=True)
            await self._send_error("Audio generation failed")
        finally:
            self.tts_active = False
            self.should_interrupt_tts = False

    async def _check_connection_health(self):
        """Periodically check WebSocket connection health."""
        try:
            while self.websocket.client_state == WebSocketState.CONNECTED:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Send ping to check connection
                try:
                    await self._send_message({"type": "ping", "timestamp": time.time()})
                except Exception as e:
                    self.session_logger.warning(f"Health check failed: {e}")
                    break
                    
        except Exception as e:
            self.session_logger.error(f"Connection health check error: {e}", exc_info=True)