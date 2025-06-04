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
                
            if len(audio_data) < settings.small_chunk_buffer_threshold:
                self.small_chunk_buffer.extend(audio_data)
                self.session_logger.debug(f"Audio chunk too small ({len(audio_data)}B), buffered. Total buffer: {len(self.small_chunk_buffer)}B")
                return
            
            if self.small_chunk_buffer:
                audio_data = bytes(self.small_chunk_buffer) + audio_data
                self.small_chunk_buffer.clear()
                self.session_logger.debug(f"Combined with buffered data, new audio data size: {len(audio_data)} bytes")
            
            self.last_audio_time = time.time() # Update activity time

            pcm_data = await self._convert_audio_via_service(audio_data)

            if not pcm_data:
                # Error already logged by _convert_audio_via_service if it fails consistently
                # Send error to client only if it's a persistent issue (handled by retry logic in _convert)
                # If _convert_audio_via_service returns None after retries, it means it's unrecoverable for this stream of chunks
                self.session_logger.error(f"Audio conversion by AudioService failed definitively for chunk of {len(audio_data)} bytes.")
                await self._send_error("Audio processing failed: Unrecoverable format error.")
                return

            if len(pcm_data) < (settings.sample_rate * settings.audio_frame_ms // 1000 * 2 * settings.channels / 2) : # Min e.g. 10ms
                self.session_logger.warning(f"PCM data too short after conversion: {len(pcm_data)} bytes, skipping processing for this chunk.")
                return

            self.audio_buffer.extend(pcm_data)
            self.session_logger.debug(f"PCM data added to buffer. Current audio_buffer size: {len(self.audio_buffer)} bytes")
            
            frame_size_bytes = self._get_frame_size() 
            
            while len(self.audio_buffer) >= frame_size_bytes:
                frame_to_process = bytes(self.audio_buffer[:frame_size_bytes])
                self.audio_buffer = self.audio_buffer[frame_size_bytes:]
                
                self.session_logger.debug(f"Extracted frame of {len(frame_to_process)} bytes for processing. Buffer remaining: {len(self.audio_buffer)}")
                
                task = asyncio.create_task(self._process_audio_frame(frame_to_process))
                self.processing_tasks.add(task)
                task.add_done_callback(self.processing_tasks.discard) # Auto-remove task when done
                
        except Exception as e:
            self.session_logger.error(f"Error in handle_audio_chunk: {e}", exc_info=True)
            await self._send_error(f"Server error during audio handling: {str(e)}")

    async def _process_audio_frame(self, frame: bytes):
        """Process a single PCM audio frame through VAD and potentially STT/AI pipeline."""
        self.pipeline_running = True # Indicate that some processing is happening
        try:
            self.session_logger.debug(f"Processing frame: {len(frame)} bytes. Current user speaking state: {self.is_speaking}")
            
            vad_service = self.services.get('vad')
            stt_service = self.services.get('stt')

            if not vad_service or not stt_service:
                self.session_logger.error("VAD or STT service not available.")
                await self._send_error("Voice processing services are currently unavailable.")
                self.pipeline_running = False
                return

            vad_result = await vad_service.process_frame(frame)
            self.session_logger.debug(f"VAD: speech={vad_result.is_speech}, end={vad_result.is_end_of_speech}, conf={vad_result.confidence:.2f}")
            
            if vad_result.is_speech:
                if not self.is_speaking: # Start of user's speech segment
                    self.is_speaking = True
                    self.speech_start_time = time.time()
                    self.session_logger.info("User speech segment started (VAD).")
                    if self.tts_active: # AI is currently speaking
                        self.session_logger.info("Barge-in: User started speaking during TTS.")
                        await self._handle_barge_in()
                        
                stt_result = await stt_service.process_frame(frame)
                if stt_result.partial_text:
                    await self._send_message({"type": "transcript", "partial": stt_result.partial_text})
            
            elif vad_result.is_end_of_speech and self.is_speaking: # End of user's speech segment
                self.session_logger.info("User speech segment ended (VAD). Finalizing STT.")
                self.is_speaking = False # Reset user speaking state
                
                final_stt_text = None
                if hasattr(stt_service, 'finalize') and callable(stt_service.finalize):
                    final_result_obj = await stt_service.finalize()
                    if final_result_obj and final_result_obj.final_text:
                        final_stt_text = final_result_obj.final_text.strip()
                
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
        """Handle barge-in: user speaks while TTS is active."""
        self.session_logger.info("Barge-in detected. Attempting to stop TTS.")
        self.should_interrupt_tts = True # Signal to stop ongoing/pending TTS tasks
        
        tts_service = self.services.get('tts')
        if self.tts_active and tts_service and hasattr(tts_service, 'stop'):
            await tts_service.stop() # Tell TTS service to halt generation
        self.tts_active = False # Mark TTS as no longer active
            
        await self._send_message({"type": "control", "action": "stop_audio"}) # Tell client to stop playing audio
            
        if self.metrics and hasattr(self.metrics, 'increment_counter'):
            self.metrics.increment_counter('barge_ins_total')
        
    async def _process_with_ai(self, text: str):
        """Process user text with LLM and stream TTS response."""
        if not text or not text.strip():
            self.session_logger.info("Empty text received for AI processing, skipping.")
            self.pipeline_running = False # Nothing to process
            return

        self.pipeline_running = True # AI processing is part of the pipeline
        self.processing_start_time = time.time()
        self.session_logger.info(f"AI processing started for text: '{text[:70]}...'")

        llm_service = self.services.get('llm')
        if not llm_service:
            self.session_logger.error("LLM service not available for AI processing.")
            await self._send_error("AI service is currently unavailable.")
            self.pipeline_running = False
            return

        self.context_manager.add_user_message(text)
        self.should_interrupt_tts = False # Reset for new AI response

        full_ai_response_text = ""
        try:
            async for token in llm_service.generate_streaming(text): # Assuming generate_streaming takes plain text
                if self.should_interrupt_tts: # Check if barge-in occurred during LLM streaming
                    self.session_logger.info("LLM streaming interrupted by barge-in.")
                    break
                full_ai_response_text += token
                await self._send_message({"type": "ai_response", "token": token})
            
            if not self.should_interrupt_tts and full_ai_response_text.strip():
                await self._send_message({"type": "ai_response", "complete": True})
                self.context_manager.add_ai_message(full_ai_response_text)
                
                # Start TTS for the complete response
                # _start_tts_streaming will handle the tts_active and pipeline_running flags internally
                tts_task = asyncio.create_task(self._start_tts_streaming(full_ai_response_text))
                self.processing_tasks.add(tts_task)
                tts_task.add_done_callback(self.processing_tasks.discard)
            elif full_ai_response_text.strip() and self.should_interrupt_tts:
                 # If interrupted, we still add the partial AI response to context for consistency.
                 self.context_manager.add_ai_message(full_ai_response_text)
                 self.pipeline_running = False # Interrupted, so main pipeline flow stops here.
            else: # No text from LLM or interrupted before any text
                self.session_logger.info("No significant AI response generated or stream was empty/interrupted early.")
                self.pipeline_running = False # No TTS to play, pipeline idle.

            if self.speech_start_time and self.metrics and hasattr(self.metrics, 'record_latency'):
                latency = time.time() - self.speech_start_time
                self.metrics.record_latency('end_to_end_latency_ms', latency * 1000)
                self.session_logger.info(f"End-to-end latency: {latency*1000:.0f} ms")

        except Exception as e:
            self.session_logger.error(f"AI processing or TTS initiation error: {e}", exc_info=True)
            await self._send_error(f"AI conversation failed: {str(e)}")
            self.pipeline_running = False # Error, so pipeline stops
        # finally:
            # pipeline_running is set to False within _start_tts_streaming or if no TTS happens
            # self.session_logger.debug(f"AI processing block finished. Pipeline: {self.pipeline_running}, TTS Active: {self.tts_active}")


    async def _start_tts_streaming(self, text: str):
        """Stream TTS audio for the given text. Manages tts_active and pipeline_running flags."""
        if not text or not text.strip():
            self.session_logger.info("Empty text for TTS, skipping TTS generation.")
            # If AI produced no text, the overall pipeline for this interaction might be considered done.
            self.pipeline_running = False 
            return

        tts_service = self.services.get('tts')
        if not tts_service or not tts_service.is_available:
            self.session_logger.error("TTS service not available, cannot play AI response.")
            await self._send_error("Text-to-speech service is unavailable.")
            self.pipeline_running = False
            return

        if self.should_interrupt_tts:
            self.session_logger.info("TTS streaming initiation skipped due to prior interruption signal.")
            self.pipeline_running = False # No TTS will run
            return
            
        self.tts_active = True
        # self.pipeline_running should already be true from _process_with_ai
        self.session_logger.info(f"TTS streaming started for: '{text[:70]}...'")
        await self._send_message({"type": "tts_start"})

        try:
            # generate_speech_stream from TTSService now yields a single, complete audio buffer
            async for complete_audio_chunk in tts_service.generate_speech_stream(text):
                if self.should_interrupt_tts: # Check if barge-in occurred during synthesis
                    self.session_logger.info("TTS synthesis interrupted by barge-in signal.")
                    # TTSService.stop() should have been called by _handle_barge_in
                    break 
                
                if complete_audio_chunk:
                    self.session_logger.debug(f"Sending complete TTS audio buffer of {len(complete_audio_chunk)} bytes.")
                    await self.websocket.send_bytes(complete_audio_chunk)
                else:
                    self.session_logger.warning("TTS service yielded no audio data for a non-empty text.")

            # Signal TTS completion only if not interrupted
            if not self.should_interrupt_tts:
                 await self._send_message({"type": "tts_complete"})
            
        except Exception as e:
            self.session_logger.error(f"Error during TTS streaming: {e}", exc_info=True)
            await self._send_error(f"Text-to-speech playback failed: {str(e)}")
        finally:
            self.tts_active = False
            self.pipeline_running = False # TTS part of pipeline (and thus the whole user turn) is now complete or errored
            self.session_logger.info(f"TTS streaming finished or errored. Pipeline reset. TTS Active: {self.tts_active}")
            
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

            elif message_type == "ping":
                self.last_audio_time = time.time() # Treat ping as activity
                pong_response = {"type": "pong"}
                if "timestamp" in data: pong_response["timestamp"] = data["timestamp"]
                await self._send_message(pong_response)
                self.session_logger.debug(f"Responded to ping with {pong_response}")

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
                await self.websocket.send_text(json.dumps(data))
            except Exception as e: # Catch errors during send_text specifically
                self.session_logger.error(f"Failed to send message (type: {data.get('type', 'Unknown')}): {e}", exc_info=True)
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

        self.pipeline_running = False
        self.is_speaking = False
        self.audio_buffer.clear()
        self.small_chunk_buffer.clear()
        self.failed_chunk_buffer.clear()
        self.failed_chunk_retry_count = 0
        
        self.session_logger.info(f"WebSocketHandler session cleanup completed.")

    async def _convert_audio_via_service(self, audio_data: bytes) -> Optional[bytes]:
        """Centralized audio conversion using AudioService with retry logic for failed chunks."""
        try:
            audio_service = self.services.get('audio')
            if not audio_service or not hasattr(audio_service, 'extract_pcm_smart'):
                self.session_logger.error("AudioService is not available or misconfigured.")
                return None

            pcm_array = audio_service.extract_pcm_smart(audio_data)
            
            if pcm_array is not None:
                if self.failed_chunk_buffer: # If there was a previously failed buffer, it's now cleared by this success
                    self.session_logger.info(f"Successfully processed audio after prior failures, clearing {len(self.failed_chunk_buffer)}B failed buffer.")
                    self.failed_chunk_buffer.clear()
                self.failed_chunk_retry_count = 0
                pcm_int16 = (pcm_array * 32768.0).astype(np.int16)
                return pcm_int16.tobytes()

            # If initial conversion failed, accumulate and retry
            self.session_logger.warning(f"AudioService initial conversion failed for {len(audio_data)}B chunk. Buffering.")
            
            max_buffer_size = settings.failed_chunk_buffer_max_size
            if len(self.failed_chunk_buffer) + len(audio_data) > max_buffer_size:
                # Trim buffer from the beginning to make space
                amount_to_trim = (len(self.failed_chunk_buffer) + len(audio_data)) - max_buffer_size
                self.failed_chunk_buffer = self.failed_chunk_buffer[amount_to_trim:]
                self.session_logger.warning(f"Failed chunk buffer trimmed by {amount_to_trim}B to prevent overflow. New size: {len(self.failed_chunk_buffer)}B.")
            
            self.failed_chunk_buffer.extend(audio_data)
            
            if self.failed_chunk_retry_count >= self.max_failed_chunk_retries:
                self.session_logger.error(f"Max retry attempts ({self.max_failed_chunk_retries}) for accumulated audio ({len(self.failed_chunk_buffer)}B) reached. Clearing buffer.")
                self.failed_chunk_buffer.clear()
                self.failed_chunk_retry_count = 0
                return None 

            # Attempt to process accumulated buffer if it's large enough
            # A larger buffer might provide more context for FFmpeg.
            # Use a multiple of expected small chunk sizes, e.g., 5KB
            retry_buffer_threshold = max(settings.small_chunk_buffer_threshold * 10, 5000) 
            if len(self.failed_chunk_buffer) >= retry_buffer_threshold:
                self.session_logger.info(f"Attempting to process accumulated failed buffer: {len(self.failed_chunk_buffer)}B (Retry {self.failed_chunk_retry_count + 1}/{self.max_failed_chunk_retries})")
                self.failed_chunk_retry_count += 1
                
                accumulated_pcm_array = audio_service.extract_pcm_smart(bytes(self.failed_chunk_buffer))
                
                if accumulated_pcm_array is not None:
                    self.session_logger.info(f"Successfully processed accumulated failed buffer. Cleared {len(self.failed_chunk_buffer)}B.")
                    self.failed_chunk_buffer.clear()
                    self.failed_chunk_retry_count = 0 # Reset on success
                    pcm_int16 = (accumulated_pcm_array * 32768.0).astype(np.int16)
                    return pcm_int16.tobytes()
                else:
                    self.session_logger.warning(f"Processing accumulated buffer also failed (Retry {self.failed_chunk_retry_count}). Buffer size: {len(self.failed_chunk_buffer)}B.")
            else:
                self.session_logger.debug(f"Accumulated failed buffer ({len(self.failed_chunk_buffer)}B) not yet large enough for retry ({retry_buffer_threshold}B).")

            return None # No PCM data successfully converted in this attempt
            
        except Exception as e:
            self.session_logger.error(f"Critical error in _convert_audio_via_service: {e}", exc_info=True)
            # Reset buffer and count on unexpected errors to prevent bad state
            self.failed_chunk_buffer.clear()
            self.failed_chunk_retry_count = 0
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
                is_inactive = (current_time - self.last_audio_time > settings.watchdog_inactivity_timeout)
                
                if is_inactive and not self.pipeline_running and not self.is_speaking and not self.tts_active:
                    self.session_logger.warning(
                        f"Watchdog: Inactivity timeout. Last audio: {self.last_audio_time:.2f} ({(current_time - self.last_audio_time):.2f}s ago). "
                        f"Pipeline: {self.pipeline_running}, UserSpeaking(VAD): {self.is_speaking}, TTSActive: {self.tts_active}. Closing WebSocket."
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
                        f"Watchdog: Inactivity detected ({ (current_time - self.last_audio_time):.1f}s), but pipeline/speech active. "
                        f"Pipeline: {self.pipeline_running}, UserSpeaking(VAD): {self.is_speaking}, TTSActive: {self.tts_active}."
                    )
                else:
                    self.session_logger.debug(f"Watchdog: Activity within timeout. Last audio { (current_time - self.last_audio_time):.1f}s ago.")

        except asyncio.CancelledError:
            self.session_logger.info("Watchdog timer task was cancelled.")
        except Exception as e:
            self.session_logger.error(f"Watchdog timer encountered an unhandled error: {e}", exc_info=True)
        finally:
            self.session_logger.info(f"Watchdog timer stopped for session {self.session_id}.")