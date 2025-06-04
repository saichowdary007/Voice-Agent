import asyncio
import json
import time
import os
from typing import Dict, Any, Optional, Set
import numpy as np
# Remove ffmpeg import if direct calls are fully removed:
# import ffmpeg  # This is ffmpeg-python package
from fastapi import WebSocket
from fastapi.websockets import WebSocketState

from .ai.context_manager import ContextManager
from .config import settings


class WebSocketHandler:
    """Handles WebSocket connections and message routing for voice agent"""
    
    def __init__(self, websocket: WebSocket, services: Dict, metrics: Any, logger: Any):
        self.websocket = websocket
        self.services = services
        self.metrics = metrics
        self.logger = logger
        
        self.session_id = f"session_{int(time.time() * 1000)}_{websocket.client.host}"
        self.session_logger = self.logger.bind(session_id=self.session_id)
        self.session_logger.info(f"Created session {self.session_id}")
        self.is_muted = False
        self.is_speaking = False # Tracks if client VAD thinks user is speaking
        self.audio_buffer = bytearray()
        self.context_manager = ContextManager()
        
        self.small_chunk_buffer = bytearray()
        self.failed_chunk_buffer = bytearray()
        self.failed_chunk_retry_count = 0
        self.max_failed_chunk_retries = settings.max_failed_chunk_retries
        
        self.speech_start_time = None
        self.processing_start_time = None
        
        self.tts_active = False # Tracks if backend is currently generating/sending TTS audio
        self.should_interrupt_tts = False
        
        self.processing_tasks: Set[asyncio.Task] = set()
        # self.active_subprocesses is removed as ffmpeg calls are now in AudioService
        self.last_audio_time = time.time()
        self.pipeline_running = False
        self.watchdog_task = None
        
        self.session_logger.info(f"WebSocket handler initialized for session {self.session_id}")
        
    async def handle_connection(self):
        """Main connection handler loop"""
        try:
            self.watchdog_task = asyncio.create_task(self._watchdog_timer())

            while True:
                try:
                    message = await self.websocket.receive()
                except WebSocketDisconnect: # Specific handling for disconnect
                    self.session_logger.info("WebSocket client disconnected (received disconnect message type).")
                    break # Exit loop on graceful disconnect
                except ConnectionResetError: # Handle abrupt disconnections
                    self.session_logger.warning("WebSocket connection reset by client.")
                    break # Exit loop
                except Exception as e: # Catch other receive errors
                    self.session_logger.error(f"Error receiving WebSocket message: {e}", exc_info=True)
                    # Consider if we should break or continue. Breaking is safer.
                    break 
                
                if message["type"] == "websocket.disconnect": # Explicit disconnect message from FastAPI
                    self.session_logger.info("FastAPI signaled WebSocket disconnect.")
                    break
                    
                try:
                    if "bytes" in message and message["bytes"] is not None:
                        self.session_logger.debug(f"Received binary message: {len(message['bytes'])} bytes")
                        await self.handle_audio_chunk(message["bytes"])
                    elif "text" in message and message["text"] is not None:
                        self.session_logger.debug(f"Received text message: {message['text'][:100]}...")
                        try:
                            data = json.loads(message["text"])
                            await self.handle_control_message(data)
                        except json.JSONDecodeError as e:
                            self.session_logger.warning(f"Invalid JSON received: {message['text']}, error: {e}")
                    else:
                        self.session_logger.warning(f"Unknown message format or null content: {message}")
                except Exception as e:
                    self.session_logger.error(f"Error processing message: {e}", exc_info=True)
                    # Continue processing other messages unless it's a critical error
                        
        except Exception as e: # Catch errors in the main while loop or setup
            self.session_logger.error(f"WebSocket handler connection error: {e}", exc_info=True)
            # We don't re-raise here, cleanup will be handled by the caller's finally block
        finally: 
            if self.watchdog_task and not self.watchdog_task.done():
                self.watchdog_task.cancel()
            # Cleanup is now primarily handled by the finally block in main.py's websocket_endpoint
            self.session_logger.info("Exiting handle_connection loop.")

    async def handle_audio_chunk(self, audio_data: bytes):
        """Handle incoming audio chunk from WebSocket using AudioService."""
        try:
            self.session_logger.debug(f"Received binary message for audio chunk: {len(audio_data)} bytes")
            
            if not audio_data or len(audio_data) < 100:
                self.session_logger.debug(f"Rejecting invalid audio chunk: {len(audio_data) if audio_data else 0} bytes")
                return
                
            if len(audio_data) < settings.small_chunk_buffer_threshold:
                self.session_logger.debug(f"Audio chunk too small ({len(audio_data)} bytes), buffering for next chunk")
                self.small_chunk_buffer.extend(audio_data)
                return
            
            if hasattr(self, 'small_chunk_buffer') and len(self.small_chunk_buffer) > 0:
                combined_data = bytes(self.small_chunk_buffer) + audio_data
                self.small_chunk_buffer.clear()
                audio_data = combined_data
                self.session_logger.debug(f"Combined with buffered data, new size: {len(audio_data)} bytes")
            
            self.session_logger.debug(f"Processing audio chunk: {len(audio_data)} bytes")
            self.last_audio_time = time.time()
            self.session_logger.debug(f"Updated last_audio_time to {self.last_audio_time}")

            pcm_data = await self._convert_audio_via_service(audio_data)

            if not pcm_data:
                error_msg = "Audio processing failed - invalid format or corrupted data. Please check your microphone or browser settings."
                self.session_logger.warning(f"Audio conversion failed for chunk of {len(audio_data)} bytes by AudioService.")
                self.failed_chunk_buffer.clear()
                self.failed_chunk_retry_count = 0
                await self._send_error(error_msg)
                return

            if len(pcm_data) < 160:  # Minimum for 10ms at 16kHz
                self.session_logger.warning(f"PCM data too short after conversion: {len(pcm_data)} bytes, skipping")
                return

            self.audio_buffer.extend(pcm_data)
            self.session_logger.debug(f"Audio buffer now has {len(self.audio_buffer)} bytes")
            
            frame_size = self._get_frame_size() 
            self.session_logger.debug(f"Frame size: {frame_size} bytes")
            
            while len(self.audio_buffer) >= frame_size:
                frame = bytes(self.audio_buffer[:frame_size])
                self.audio_buffer = self.audio_buffer[frame_size:]
                self.session_logger.debug(f"Processing frame of {len(frame)} bytes, buffer remaining: {len(self.audio_buffer)}")
                
                samples_in_frame = len(frame) // 2
                self.session_logger.debug(f"Frame contains {samples_in_frame} samples (expected: {frame_size // 2})")
                
                task = asyncio.create_task(self._process_audio_frame(frame))
                self.processing_tasks.add(task)
                task.add_done_callback(self.processing_tasks.discard)
                
        except Exception as e:
            self.session_logger.error(f"Audio chunk processing error: {e}", exc_info=True)
            await self._send_error(f"Critical audio processing error: {str(e)}")
            # Decide if to re-raise or just log and send error

    async def _process_audio_frame(self, frame: bytes):
        """Process a single PCM audio frame through the pipeline"""
        try:
            self.session_logger.debug(f"Starting audio frame processing: {len(frame)} bytes")
            self.pipeline_running = True 
            
            vad_result = await self.services['vad'].process_frame(frame)
            self.session_logger.debug(f"VAD result: is_speech={vad_result.is_speech}, is_end_of_speech={vad_result.is_end_of_speech}, confidence={vad_result.confidence:.2f}")
            
            if vad_result.is_speech: # This means VAD confirms active speech
                self.session_logger.debug("Speech detected by VAD")
                if not self.is_speaking: # If we weren't previously in a speaking state (user speaking)
                    self.is_speaking = True # Mark that user is now speaking
                    self.speech_start_time = time.time()
                    self.session_logger.debug(f"User speech started at {self.speech_start_time}")
                    
                    if self.tts_active: # If AI is currently speaking
                        self.session_logger.info("Barge-in detected during TTS")
                        await self._handle_barge_in()
                        
                stt_result = await self.services['stt'].process_frame(frame)
                self.session_logger.debug(f"STT result: partial='{stt_result.partial_text}', final='{stt_result.final_text}', is_final={stt_result.is_final}")
                
                if stt_result.partial_text:
                    self.session_logger.debug(f"Sending partial transcript: {stt_result.partial_text}")
                    await self._send_message({
                        "type": "transcript",
                        "partial": stt_result.partial_text
                    })
                    
                # Note: STT's is_final from process_frame is usually for segments, not full EOS.
                # Full finalization is better handled by VAD's is_end_of_speech or explicit EOS message.

            elif vad_result.is_end_of_speech and self.is_speaking: # End of user speech detected by VAD
                self.session_logger.info("End of speech detected by VAD")
                self.is_speaking = False # Mark that user has stopped speaking
                
                final_result = await self.services['stt'].finalize()
                if final_result and final_result.final_text and final_result.final_text.strip():
                    self.session_logger.info(f"STT finalized with text after VAD EOS: {final_result.final_text}")
                    await self._send_message({
                        "type": "transcript",
                        "final": final_result.final_text
                    })
                    # Process AI only if there's substantial text
                    if len(final_result.final_text.strip()) > 0:
                         # Create task to not block VAD/STT pipeline
                        ai_task = asyncio.create_task(self._process_with_ai(final_result.final_text))
                        self.processing_tasks.add(ai_task)
                        ai_task.add_done_callback(self.processing_tasks.discard)
                    else:
                        self.pipeline_running = False # No AI to run, pipeline idle
                else:
                    self.session_logger.info("STT finalization after VAD EOS returned no significant text.")
                    self.pipeline_running = False # No text, pipeline idle
            # If not is_speech and not is_end_of_speech, it's silence, pipeline might still be considered running if waiting for speech.
            # The pipeline_running flag will be set to False once AI and TTS are complete.
            if not vad_result.is_speech and not self.is_speaking and not self.tts_active:
                self.pipeline_running = False


        except Exception as e:
            self.session_logger.error(f"Audio frame processing error: {e}", exc_info=True)
            self.pipeline_running = False # Ensure reset on error
        # finally: # pipeline_running is managed by the AI/TTS flow now
            # pass
            
    async def _handle_barge_in(self):
        """Handle barge-in when user speaks over TTS"""
        self.session_logger.info("Barge-in detected, stopping TTS")
        self.should_interrupt_tts = True # Signal TTS to stop
        
        if self.tts_active: # If TTS is truly active
            if hasattr(self.services['tts'], 'stop'):
                 await self.services['tts'].stop() # Call stop method on TTS service
            self.tts_active = False 
            
            await self._send_message({
                "type": "control", # Ensure frontend knows to stop audio
                "action": "stop_audio" 
            })
            
        if self.metrics:
            self.metrics.increment_counter('barge_ins_total')
        
    async def _process_with_ai(self, text: str):
        """Process user text with Gemini AI"""
        if not text.strip():
            self.session_logger.info("Empty text received for AI processing, skipping.")
            self.pipeline_running = False
            return
        try:
            self.pipeline_running = True
            self.processing_start_time = time.time()
            
            self.context_manager.add_user_message(text)
            
            # Prepare for TTS, reset interruption flag
            self.should_interrupt_tts = False 
            
            response_text_accumulator = ""
            first_token_received = False

            # Use a timeout for the LLM response generation loop
            llm_timeout = 30.0  # 30 seconds timeout for LLM to respond
            start_llm_time = time.time()

            async for token in self.services['llm'].generate_streaming(
                self.context_manager.get_messages()[-1].content if self.context_manager.get_messages() else text
            ):
                if time.time() - start_llm_time > llm_timeout:
                    self.session_logger.error("LLM response generation timed out.")
                    await self._send_error("AI response timed out.")
                    break # Exit token loop

                if self.should_interrupt_tts: # Check for barge-in before processing token
                    self.session_logger.info("AI response generation interrupted by barge-in.")
                    break 

                response_text_accumulator += token
                
                await self._send_message({ "type": "ai_response", "token": token })
                
                # Start TTS early if we have enough tokens and TTS is not already active from a previous call
                if not first_token_received and len(response_text_accumulator) > 20 and not self.tts_active : # Heuristic: 20 chars
                    first_token_received = True
                    # Important: Do not await _start_tts_streaming directly if it sends many chunks.
                    # The new _start_tts_streaming sends one big chunk.
                    if not self.should_interrupt_tts: # Double check before starting TTS
                        # Create task to allow LLM token processing to continue if TTS is slow
                        tts_task = asyncio.create_task(self._start_tts_streaming(response_text_accumulator))
                        self.processing_tasks.add(tts_task)
                        tts_task.add_done_callback(self.processing_tasks.discard)
                        response_text_accumulator = "" # Reset accumulator as this part is sent for TTS
            
            # If barge-in happened, response_text_accumulator might be partial or empty
            # and tts_active might have been set to false by _handle_barge_in
            if self.should_interrupt_tts:
                 self.session_logger.info("AI processing concluded early due to barge-in.")
            else:
                # Handle any remaining text that wasn't sent for early TTS
                if response_text_accumulator and not self.tts_active and not first_token_received:
                    self.session_logger.debug(f"Starting TTS for remaining text: {response_text_accumulator[:50]}...")
                    tts_task = asyncio.create_task(self._start_tts_streaming(response_text_accumulator))
                    self.processing_tasks.add(tts_task)
                    tts_task.add_done_callback(self.processing_tasks.discard)

            # Mark AI response as complete, only if not interrupted badly
            if not self.should_interrupt_tts :
                 await self._send_message({ "type": "ai_response", "complete": True })
                 self.context_manager.add_ai_message(response_text_accumulator) # Add the full or remaining AI response
            
            if self.speech_start_time and self.metrics:
                latency = time.time() - self.speech_start_time
                self.metrics.record_latency('end_to_end_latency', latency)
                
        except Exception as e:
            self.session_logger.error(f"AI processing error: {e}", exc_info=True)
            await self._send_error(f"AI conversation failed: {str(e)}")
        finally:
            # Pipeline is considered not running only if TTS is also not active.
            # _start_tts_streaming will set pipeline_running to False when it's done.
            if not self.tts_active:
                 self.pipeline_running = False
            self.session_logger.info(f"AI processing finished. Pipeline running: {self.pipeline_running}")


    async def _start_tts_streaming(self, text: str):
        """Start TTS streaming for AI response by receiving one complete audio buffer."""
        if not text.strip():
            self.session_logger.info("Empty text for TTS, skipping.")
            self.tts_active = False # Ensure tts_active is false if no TTS happens
            self.pipeline_running = False # No TTS, so pipeline part is done
            return

        try:
            if self.should_interrupt_tts: # Check for barge-in signal
                self.session_logger.info("TTS streaming skipped due to pre-existing interruption signal.")
                self.tts_active = False
                self.pipeline_running = False
                return
                
            self.tts_active = True
            # self.pipeline_running = True # Already set by caller (_process_with_ai or EOS)
            
            full_audio_buffer = b""
            # The TTSService's generate_speech_stream now yields a single, complete audio buffer.
            async for audio_chunk in self.services['tts'].generate_speech_stream(text):
                if self.should_interrupt_tts: # Check for interruption during (potentially long) synthesis
                    self.session_logger.info("TTS synthesis/streaming interrupted mid-stream by barge-in.")
                    if hasattr(self.services.get('tts'), 'stop'):
                        await self.services['tts'].stop()
                    full_audio_buffer = b"" # Do not send partial/interrupted audio
                    break 
                full_audio_buffer += audio_chunk 

            if full_audio_buffer and not self.should_interrupt_tts: # Send only if not interrupted
                self.session_logger.debug(f"Sending complete TTS audio chunk of {len(full_audio_buffer)} bytes.")
                await self.websocket.send_bytes(full_audio_buffer)
                await self._send_message({"type": "tts_complete"}) # Signal TTS completion
            elif self.should_interrupt_tts:
                 self.session_logger.info("TTS audio sending skipped due to interruption.")
            else:
                self.session_logger.warning("No full TTS audio buffer to send, though no interruption was signaled.")
            
        except Exception as e:
            self.session_logger.error(f"TTS streaming error: {e}", exc_info=True)
            await self._send_error(f"Text-to-speech failed: {str(e)}")
        finally:
            self.tts_active = False 
            self.pipeline_running = False # TTS part of pipeline is now complete or errored
            self.session_logger.info("TTS streaming finished or errored. Pipeline running set to False.")
            
    async def handle_control_message(self, data: Dict[str, Any]):
        """Handle control messages from frontend"""
        try:
            message_type = data.get("type")
            
            if message_type == "control":
                action = data.get("action")
                if action == "mute":
                    self.is_muted = True
                    self.session_logger.info("Session muted")
                elif action == "unmute":
                    self.is_muted = False
                    self.session_logger.info("Session unmuted")
                elif action == "end_session":
                    self.session_logger.info("Session ended by user")
                    await self._send_message({"type": "control", "action": "session_ended"})
                    # Closing the WebSocket should be handled by the main endpoint after this returns
                    # For now, we can signal to the main loop to break or close directly.
                    if self.websocket.client_state == WebSocketState.CONNECTED:
                        await self.websocket.close(code=1000, reason="Session ended by user")
                    return # Important to return to allow main loop to break
                    
            elif message_type == "ping":
                self.session_logger.debug(f"Received ping: {data}") 
                self.last_audio_time = time.time()
                pong_response = {"type": "pong"}
                if "timestamp" in data: pong_response["timestamp"] = data["timestamp"]
                elif "t" in data: pong_response["timestamp"] = data["t"]
                await self._send_message(pong_response)
                self.session_logger.debug("Sent pong response")

            elif message_type == "eos": 
                self.session_logger.info("EOS received from client")
                # No need to send "processing" status, AI processing will send its own status.
                
                # self.pipeline_running = True # Set by _process_audio_frame or _process_with_ai

                try:
                    final_stt_result = None
                    # Process any remaining audio buffer through VAD and STT finalize
                    if self.audio_buffer:
                        self.session_logger.info(f"Processing remaining audio buffer of {len(self.audio_buffer)} bytes after client EOS")
                        # Simulate processing the last frame through VAD to ensure STT finalizes correctly
                        # This part is tricky as client EOS means no more audio, VAD might not naturally hit is_end_of_speech
                        # The most robust way is to directly call STT finalize.
                        self.audio_buffer.clear() # Clear buffer as we are finalizing

                    final_stt_result = await self.services['stt'].finalize()
                    self.session_logger.debug(f"STT finalization result after client EOS: {final_stt_result}")
                    
                    if final_stt_result and final_stt_result.final_text and final_stt_result.final_text.strip():
                        final_text = final_stt_result.final_text.strip()
                        self.session_logger.info(f"Final text from STT after client EOS: {final_text}")
                        
                        if not self.context_manager.is_duplicate_user_message(final_text):
                            await self._send_message({"type": "transcript", "final": final_text})
                            ai_task = asyncio.create_task(self._process_with_ai(final_text))
                            self.processing_tasks.add(ai_task)
                            ai_task.add_done_callback(self.processing_tasks.discard)
                        else:
                            self.session_logger.info("Duplicate text detected on client EOS, not reprocessing.")
                            self.pipeline_running = False 
                    else:
                        self.session_logger.info("No final STT result after client EOS processing.")
                        self.pipeline_running = False 
                        
                except Exception as e:
                    self.session_logger.error(f"Error during client EOS processing: {e}", exc_info=True)
                    self.pipeline_running = False 
                    await self._send_error(f"Processing failed after EOS: {str(e)}")
            else:
                self.session_logger.warning(f"Unknown control message type: {message_type}")
                
        except Exception as e:
            self.session_logger.error(f"Control message handling error: {e}", exc_info=True)
            await self._send_error(f"Message processing failed: {str(e)}")
            
    async def _send_message(self, data: Dict[str, Any]):
        """Send JSON message to frontend, checking WebSocket state."""
        try:
            if self.websocket.client_state == WebSocketState.CONNECTED:
                await self.websocket.send_text(json.dumps(data))
            else:
                self.session_logger.warning(f"Attempted to send message on a non-connected WebSocket. State: {self.websocket.client_state.name}. Data: {str(data)[:100]}")
        except Exception as e:
            self.session_logger.error(f"Failed to send message: {data.get('type', 'Unknown type')}. Error: {e}")
            
    async def _send_error(self, error_message: str):
        """Send error message to frontend, checking WebSocket state."""
        try:
            # Avoid "error" recursion if _send_message itself fails critically
            if self.websocket.client_state == WebSocketState.CONNECTED:
                 await self.websocket.send_text(json.dumps({
                    "type": "error",
                    "message": error_message
                }))
            else:
                self.session_logger.warning(f"Attempted to send error message on a non-connected WebSocket. State: {self.websocket.client_state.name}, Error: {error_message}")
        except Exception as e: 
            self.session_logger.error(f"CRITICAL: Failed to send error message '{error_message}' due to: {e}")

        
    def _get_frame_size(self) -> int:
        """Calculate frame size based on standardized configuration"""
        sample_rate = settings.sample_rate
        frame_ms = settings.audio_frame_ms
        channels = settings.channels
        bytes_per_sample = 2  # 16-bit audio
        return sample_rate * frame_ms // 1000 * bytes_per_sample * channels
        
    async def cleanup(self):
        """Clean up resources when WebSocket connection is closed"""
        self.session_logger.info(f"Initiating cleanup for session: {self.session_id}")
        if self.watchdog_task and not self.watchdog_task.done():
            self.watchdog_task.cancel()
            try:
                await self.watchdog_task
            except asyncio.CancelledError:
                self.session_logger.info("Watchdog task cancelled.")
        try:
            # Signal TTS to stop immediately
            self.should_interrupt_tts = True
            if self.tts_active and hasattr(self.services.get('tts'), 'stop'):
                self.session_logger.info("Stopping active TTS during cleanup.")
                await self.services['tts'].stop()
                self.tts_active = False

            # Cancel all other processing tasks
            if self.processing_tasks:
                self.session_logger.info(f"Cancelling {len(self.processing_tasks)} processing tasks.")
                for task in list(self.processing_tasks): # Iterate over a copy
                    if not task.done():
                        task.cancel()
                # Wait for tasks to acknowledge cancellation
                # Note: Gather can re-raise CancelledError if not handled inside tasks
                await asyncio.gather(*self.processing_tasks, return_exceptions=True)
                self.processing_tasks.clear()
            
            # Reset service states if applicable
            if hasattr(self.services.get('stt'), 'reset_stream'):
                self.services['stt'].reset_stream()
            if hasattr(self.services.get('vad'), 'reset_state'):
                self.services['vad'].reset_state()

            # Ensure pipeline flag is reset
            self.pipeline_running = False
            self.is_speaking = False # Reset user speaking state
            
            self.session_logger.info(f"Session {self.session_id} cleaned up successfully.")
            
        except Exception as e:
            self.session_logger.error(f"Cleanup error for session {self.session_id}: {e}", exc_info=True)

    async def _convert_audio_via_service(self, audio_data: bytes) -> Optional[bytes]:
        """
        Centralized audio conversion using AudioService with retry logic.
        """
        try:
            # Ensure 'audio' service key matches what's in main.py services dict
            audio_service = self.services.get('audio') 
            if not audio_service:
                self.session_logger.error("AudioService not found in services dictionary.")
                return None

            pcm_array = audio_service.extract_pcm_smart(audio_data)
            
            if pcm_array is not None:
                if self.failed_chunk_buffer:
                    self.session_logger.info(f"Successfully processed audio, clearing accumulated failed buffer of {len(self.failed_chunk_buffer)} bytes.")
                    self.failed_chunk_buffer.clear()
                self.failed_chunk_retry_count = 0
                pcm_int16 = (pcm_array * 32768.0).astype(np.int16)
                return pcm_int16.tobytes()

            self.session_logger.warning("AudioService initial conversion failed, buffering chunk for accumulation and retry.")
            
            max_buffer_size = settings.failed_chunk_buffer_max_size
            if len(self.failed_chunk_buffer) + len(audio_data) > max_buffer_size:
                excess = len(self.failed_chunk_buffer) + len(audio_data) - max_buffer_size
                self.failed_chunk_buffer = self.failed_chunk_buffer[excess:]
                self.session_logger.warning(f"Failed chunk buffer trimmed. New size: {len(self.failed_chunk_buffer)} bytes")
            
            self.failed_chunk_buffer.extend(audio_data)
            
            if self.failed_chunk_retry_count >= self.max_failed_chunk_retries:
                self.session_logger.error(f"Max retry attempts ({self.max_failed_chunk_retries}) reached for accumulated audio. Clearing buffer.")
                self.failed_chunk_buffer.clear()
                self.failed_chunk_retry_count = 0
                return None

            buffer_threshold = settings.small_chunk_buffer_threshold * 10 # e.g., 1000 bytes
            if len(self.failed_chunk_buffer) >= buffer_threshold:
                self.session_logger.info(f"Trying to process accumulated buffer: {len(self.failed_chunk_buffer)} bytes (attempt {self.failed_chunk_retry_count + 1})")
                self.failed_chunk_retry_count += 1
                
                accumulated_pcm_array = audio_service.extract_pcm_smart(bytes(self.failed_chunk_buffer))
                
                if accumulated_pcm_array is not None:
                    self.session_logger.info(f"Successfully processed accumulated buffer. PCM length: {len(accumulated_pcm_array)} samples.")
                    self.failed_chunk_buffer.clear()
                    self.failed_chunk_retry_count = 0
                    pcm_int16 = (accumulated_pcm_array * 32768.0).astype(np.int16)
                    return pcm_int16.tobytes()
                else:
                    self.session_logger.warning(f"Accumulated buffer processing failed (attempt {self.failed_chunk_retry_count}). Will retry on next chunk if buffer not full.")
            return None
            
        except Exception as e:
            self.session_logger.error(f"Centralized audio conversion failed with exception: {e}", exc_info=True)
            self.failed_chunk_buffer.clear()
            self.failed_chunk_retry_count = 0
            return None

    async def _watchdog_timer(self):
        """Watchdog timer to detect inactivity and manage session lifecycle"""
        self.session_logger.info("Watchdog timer started.")
        try:
            while True:
                await asyncio.sleep(settings.watchdog_check_interval)
                current_time = time.time()

                if self.websocket.client_state != WebSocketState.CONNECTED:
                    self.session_logger.info(f"Watchdog: WebSocket no longer connected (State: {self.websocket.client_state.name}). Stopping watchdog.")
                    break

                # Inactivity timeout logic
                # Closes if no audio for configured timeout AND no pipeline (STT/AI/TTS) is actively running.
                # self.is_speaking refers to client-side VAD indicating user speech.
                # self.pipeline_running refers to backend STT/AI/TTS processing.
                if (current_time - self.last_audio_time > settings.watchdog_inactivity_timeout) and \
                   not self.pipeline_running and not self.is_speaking and not self.tts_active:
                    self.session_logger.warning(
                        f"Watchdog: Inactivity detected. Last audio: {self.last_audio_time:.2f}, Current: {current_time:.2f}, "
                        f"Timeout: {settings.watchdog_inactivity_timeout}s. Pipeline running: {self.pipeline_running}, "
                        f"User speaking (VAD): {self.is_speaking}, TTS active: {self.tts_active}. Closing connection."
                    )
                    if self.websocket.client_state == WebSocketState.CONNECTED:
                        await self.websocket.close(code=1011, reason="WebSocket inactivity timeout")
                    break 
        except asyncio.CancelledError:
            self.session_logger.info("Watchdog timer cancelled.")
        except Exception as e:
            self.session_logger.error(f"Watchdog timer error: {e}", exc_info=True)
        finally:
            self.session_logger.info("Watchdog timer stopped.")