import asyncio
import json
import time
import os
from typing import Dict, Any, Optional, Set
import numpy as np
from fastapi import WebSocket
from fastapi.websockets import WebSocketState
import ffmpeg  # This is ffmpeg-python package
# import asyncio.subprocess # Already imported with asyncio

from .ai.context_manager import ContextManager
from .config import settings


class WebSocketHandler:
    """Handles WebSocket connections and message routing for voice agent"""
    
    def __init__(self, websocket: WebSocket, services: Dict, metrics: Any, logger: Any):
        self.websocket = websocket
        self.services = services
        self.metrics = metrics
        self.logger = logger
        
        # Session state - use a more reliable session ID
        self.session_id = f"session_{int(time.time() * 1000)}_{websocket.client.host}"
        self.session_logger = self.logger.bind(session_id=self.session_id)
        self.session_logger.info(f"Created session {self.session_id}")
        self.is_muted = False
        self.is_speaking = False
        self.audio_buffer = bytearray()
        self.context_manager = ContextManager()
        
        # Audio chunk buffers for improved processing - using config settings
        self.small_chunk_buffer = bytearray()    # For small chunks that need to be combined
        self.failed_chunk_buffer = bytearray()   # For chunks that failed initial processing
        self.failed_chunk_retry_count = 0       # Track retry attempts to prevent infinite loops
        self.max_failed_chunk_retries = settings.max_failed_chunk_retries
        
        # Timing tracking
        self.speech_start_time = None
        self.processing_start_time = None
        
        # Audio streaming state for TTS
        self.tts_active = False
        self.should_interrupt_tts = False
        
        # Tasks and subprocess management
        self.processing_tasks: Set[asyncio.Task] = set()
        self.active_subprocesses: Set[asyncio.subprocess.Process] = set()
        self.last_audio_time = time.time() # Watchdog
        self.pipeline_running = False # Watchdog - Tracks if STT/AI/TTS is active
        self.watchdog_task = None # Watchdog
        
        self.session_logger.info(f"WebSocket handler initialized for session {self.session_id}")
        
    async def handle_connection(self):
        """Main connection handler loop"""
        try:
            # Start watchdog timer
            self.watchdog_task = asyncio.create_task(self._watchdog_timer())

            while True:
                # Receive message
                try:
                    message = await self.websocket.receive()
                except Exception as e:
                    self.session_logger.error(f"Error receiving WebSocket message: {e}")
                    break
                
                if message["type"] == "websocket.disconnect":
                    self.session_logger.info("WebSocket disconnect message received")
                    break
                    
                try:
                    if "bytes" in message:
                        # Binary audio data
                        self.session_logger.debug(f"Received binary message: {len(message['bytes'])} bytes")
                        await self.handle_audio_chunk(message["bytes"])
                    elif "text" in message:
                        # JSON control message
                        self.session_logger.debug(f"Received text message: {message['text'][:100]}...")
                        try:
                            data = json.loads(message["text"])
                            await self.handle_control_message(data)
                        except json.JSONDecodeError as e:
                            self.session_logger.warning(f"Invalid JSON received: {message['text']}, error: {e}")
                    else:
                        self.session_logger.warning(f"Unknown message format: {message}")
                except Exception as e:
                    self.session_logger.error(f"Error processing message: {e}", exc_info=True)
                    # Continue processing other messages instead of breaking
                        
        except Exception as e:
            self.session_logger.error(f"WebSocket handler connection error: {e}", exc_info=True)
            raise
        finally: # Watchdog
            if self.watchdog_task: # Watchdog
                self.watchdog_task.cancel() # Watchdog
            
    async def handle_audio_chunk(self, audio_data: bytes):
        """Handle incoming audio chunk from WebSocket with improved audio processing"""
        try:
            self.session_logger.debug(f"Received binary message: {len(audio_data)} bytes")
            
            # Validate audio input early - reject empty or too small chunks
            if not audio_data or len(audio_data) < 100:
                self.session_logger.debug(f"Rejecting invalid audio chunk: {len(audio_data) if audio_data else 0} bytes")
                return
                
            # Minimum chunk size check - using config setting
            if len(audio_data) < settings.small_chunk_buffer_threshold:
                self.session_logger.debug(f"Audio chunk too small ({len(audio_data)} bytes), buffering for next chunk")
                self.small_chunk_buffer.extend(audio_data)
                return
            
            # If we have buffered small chunks, combine them
            if hasattr(self, 'small_chunk_buffer') and len(self.small_chunk_buffer) > 0:
                combined_data = bytes(self.small_chunk_buffer) + audio_data
                self.small_chunk_buffer.clear()
                audio_data = combined_data
                self.session_logger.debug(f"Combined with buffered data, new size: {len(audio_data)} bytes")
            
            self.session_logger.debug(f"Processing audio chunk: {len(audio_data)} bytes")
                
            self.last_audio_time = time.time() # Watchdog
            self.session_logger.debug(f"Updated last_audio_time to {self.last_audio_time}")

            # Use AudioService for centralized audio conversion
            if 'audio_service' in self.services:
                pcm_data = await self._convert_audio_via_service(audio_data)
            else:
                # Fallback to direct FFmpeg processing
                pcm_data = await self._convert_audio_direct(audio_data)

            # Send error to frontend if all processing failed
            if not pcm_data:
                error_msg = "Audio processing failed - invalid format or corrupted data"
                self.session_logger.warning(f"Audio conversion failed for chunk of {len(audio_data)} bytes")
                await self._send_error(error_msg)
                return

            # Validate PCM data quality
            if len(pcm_data) < 160:  # Minimum for 10ms at 16kHz
                self.session_logger.warning(f"PCM data too short: {len(pcm_data)} bytes, skipping")
                return

            # Add PCM data to buffer for frame processing
            self.audio_buffer.extend(pcm_data)
            self.session_logger.debug(f"Audio buffer now has {len(self.audio_buffer)} bytes")
            
            # Check if we have enough data to process
            frame_size = self._get_frame_size() 
            self.session_logger.debug(f"Frame size: {frame_size} bytes")
            
            while len(self.audio_buffer) >= frame_size:
                # Extract frame
                frame = bytes(self.audio_buffer[:frame_size])
                self.audio_buffer = self.audio_buffer[frame_size:]
                self.session_logger.debug(f"Processing frame of {len(frame)} bytes, buffer remaining: {len(self.audio_buffer)}")
                
                # DEBUG: Verify frame size calculation
                samples_in_frame = len(frame) // 2  # 16-bit samples
                self.session_logger.debug(f"Frame contains {samples_in_frame} samples (expected: {frame_size // 2})")
                
                # Process frame asynchronously
                task = asyncio.create_task(self._process_audio_frame(frame))
                self.processing_tasks.add(task)
                task.add_done_callback(self.processing_tasks.discard)
                
        except Exception as e:
            # Log with full traceback for better debugging
            self.session_logger.error(f"Audio chunk processing error: {e}", exc_info=True)
            raise
            
    async def _process_audio_frame(self, frame: bytes):
        """Process a single PCM audio frame through the pipeline"""
        try:
            self.session_logger.debug(f"Starting audio frame processing: {len(frame)} bytes")
            self.pipeline_running = True # Watchdog
            
            # VAD check
            self.session_logger.debug("Calling VAD service process_frame...")
            vad_result = await self.services['vad'].process_frame(frame)
            self.session_logger.debug(f"VAD result: is_speech={vad_result.is_speech}, is_end_of_speech={vad_result.is_end_of_speech}")
            
            if vad_result.is_speech:
                self.session_logger.debug("Speech detected by VAD")
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_start_time = time.time()
                    self.session_logger.debug(f"Speech started at {self.speech_start_time}")
                    
                    if self.tts_active:
                        self.session_logger.debug("Barge-in detected during TTS")
                        await self._handle_barge_in()
                        
                # STT processing
                self.session_logger.debug("Calling STT service process_frame...")
                stt_result = await self.services['stt'].process_frame(frame)
                self.session_logger.debug(f"STT result: partial='{stt_result.partial_text}', final='{stt_result.final_text}', is_final={stt_result.is_final}")
                
                if stt_result.partial_text:
                    # Send partial transcript
                    self.session_logger.debug(f"Sending partial transcript: {stt_result.partial_text}")
                    await self._send_message({
                        "type": "transcript",
                        "partial": stt_result.partial_text
                    })
                    
                if stt_result.is_final:
                    # Send final transcript and process with AI
                    self.session_logger.debug(f"Sending final transcript: {stt_result.final_text}")
                    await self._send_message({
                        "type": "transcript", 
                        "final": stt_result.final_text
                    })
                    
                    # Process with Gemini
                    self.session_logger.debug("Starting AI processing...")
                    await self._process_with_ai(stt_result.final_text)
                    
            elif vad_result.is_end_of_speech and self.is_speaking:
                # End of speech detected
                self.session_logger.debug("End of speech detected by VAD")
                self.is_speaking = False
                
                # Finalize STT
                self.session_logger.debug("Finalizing STT service...")
                final_result = await self.services['stt'].finalize()
                if final_result and final_result.final_text:
                    self.session_logger.debug(f"STT finalized with text: {final_result.final_text}")
                    await self._send_message({
                        "type": "transcript",
                        "final": final_result.final_text
                    })
                    await self._process_with_ai(final_result.final_text)
                else:
                    self.session_logger.debug("STT finalization returned no text")
                    
        except Exception as e:
            self.session_logger.error(f"Audio frame processing error: {e}", exc_info=True)
        finally: # Watchdog
            # self.pipeline_running = False # This will be set to False after AI and TTS are done.
            pass
            
    async def _handle_barge_in(self):
        """Handle barge-in when user speaks over TTS"""
        self.session_logger.info("Barge-in detected, stopping TTS")
        
        # Stop TTS immediately
        if self.tts_active:
            await self.services['tts'].stop()
            self.tts_active = False
            self.should_interrupt_tts = True
            
            # Notify frontend to stop audio
            await self._send_message({
                "type": "control",
                "action": "stop_audio"
            })
            
        # Track barge-in metrics
        if self.metrics:
            self.metrics.increment_counter('barge_ins_total')
        
    async def _process_with_ai(self, text: str):
        """Process user text with Gemini AI"""
        try:
            # self.pipeline_running = True # Already set by _process_audio_frame or EOS handler
            self.processing_start_time = time.time()
            
            # Update context
            self.context_manager.add_user_message(text)
            
            # Generate AI response
            response_text = ""
            async for token in self.services['llm'].generate_streaming(
                self.context_manager.get_messages()[-1].content if self.context_manager.get_messages() else text
            ):
                response_text += token
                
                # Send streaming token
                await self._send_message({
                    "type": "ai_response",
                    "token": token
                })
                
                # Start TTS early if we have enough tokens
                if len(response_text) > 50 and not self.tts_active:
                    asyncio.create_task(self._start_tts_streaming(response_text))
                    
            # Mark response as complete
            await self._send_message({
                "type": "ai_response",
                "complete": True
            })
            
            # Update context with AI response
            self.context_manager.add_ai_message(response_text)
            
            # Track processing latency
            if self.speech_start_time and self.metrics:
                latency = time.time() - self.speech_start_time
                self.metrics.record_latency('end_to_end_latency', latency)
                
        except Exception as e:
            self.session_logger.error(f"AI processing error: {e}")
            await self._send_error(f"AI conversation failed: {str(e)}")
        finally: # Watchdog
            # self.pipeline_running will be set to False after TTS is done or if no TTS
            if not self.tts_active:
                 self.pipeline_running = False
            
    async def _start_tts_streaming(self, text: str):
        """Start TTS streaming for AI response"""
        try:
            if self.should_interrupt_tts:
                self.should_interrupt_tts = False
                self.session_logger.info("TTS streaming skipped due to interruption.")
                return
                
            self.tts_active = True
            self.pipeline_running = True
            
            # Generate TTS audio chunks
            async for audio_chunk in self.services['tts'].generate_speech_stream(text):
                if self.should_interrupt_tts:
                    self.session_logger.info("TTS streaming interrupted mid-stream.")
                    if hasattr(self.services.get('tts'), 'stop'):
                        await self.services['tts'].stop()
                    break
                    
                await self.websocket.send_bytes(audio_chunk)
                
            self.tts_active = False
            self.pipeline_running = False
            
        except Exception as e:
            self.session_logger.error(f"TTS streaming error: {e}", exc_info=True)
            await self._send_error(f"Text-to-speech failed: {str(e)}")
        finally:
            self.tts_active = False
            self.pipeline_running = False
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
                    await self._send_message({
                        "type": "control",
                        "action": "session_ended"
                    })
                    await self.websocket.close(code=1000, reason="Session ended")
                    
            elif message_type == "ping": # Keep-alive ping
                self.session_logger.debug(f"Received ping: {data}") 
                self.last_audio_time = time.time() # Update activity time on ping
                # Echo back the timestamp for latency calculation
                pong_response = {"type": "pong"}
                if "timestamp" in data:
                    pong_response["timestamp"] = data["timestamp"]
                elif "t" in data:  # Support legacy format
                    pong_response["timestamp"] = data["t"]
                await self.websocket.send_json(pong_response)
                self.session_logger.debug("Sent pong response")

            elif message_type == "eos": # End of stream
                self.session_logger.info("EOS received from client")
                await self.websocket.send_json({"type": "status", "status": "processing"})
                self.session_logger.info("Sent status: processing")

                self.pipeline_running = True # Watchdog - Start of pipeline after EOS
                try:
                    # Simplified EOS handling: always finalize STT and process any remaining audio
                    final_stt_result = None
                    
                    # Process any remaining audio buffer first
                    if self.audio_buffer:
                        self.session_logger.info(f"Processing remaining audio buffer of {len(self.audio_buffer)} bytes after EOS")
                        
                        # Process remaining buffer through VAD to signal end of speech
                        try:
                            vad_result = await self.services['vad'].process_frame(bytes(self.audio_buffer))
                            self.audio_buffer.clear()
                            self.session_logger.debug(f"VAD processed remaining buffer: speech={vad_result.is_speech}")
                        except Exception as vad_error:
                            self.session_logger.warning(f"VAD processing failed on remaining buffer: {vad_error}")
                    
                    # Always finalize STT to get any pending transcription
                    try:
                        final_stt_result = await self.services['stt'].finalize()
                        self.session_logger.debug(f"STT finalization result: {final_stt_result}")
                    except Exception as stt_error:
                        self.session_logger.error(f"STT finalization failed: {stt_error}")
                    
                    # Process the final result if we got text
                    if final_stt_result and final_stt_result.final_text and final_stt_result.final_text.strip():
                        final_text = final_stt_result.final_text.strip()
                        self.session_logger.info(f"Final text from STT after EOS: {final_text}")
                        
                        # Check for duplicate to prevent reprocessing
                        if not self.context_manager.is_duplicate_user_message(final_text):
                            await self._send_message({"type": "transcript", "final": final_text})
                            # Start AI processing (will set pipeline_running to False when complete)
                            asyncio.create_task(self._process_with_ai(final_text))
                        else:
                            self.session_logger.info("Duplicate text detected, not reprocessing with AI.")
                            self.pipeline_running = False # Watchdog
                    else:
                        self.session_logger.info("No final STT result after EOS processing.")
                        self.pipeline_running = False # Watchdog
                        
                except Exception as e:
                    self.session_logger.error(f"Error during EOS processing: {e}")
                    self.pipeline_running = False # Watchdog - Ensure pipeline_running is reset on error
                    await self._send_error(f"Processing failed: {str(e)}")

            else:
                self.session_logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            self.session_logger.error(f"Control message error: {e}")
            await self._send_error(f"Message processing failed: {str(e)}")
            
    async def _send_message(self, data: Dict[str, Any]):
        """Send JSON message to frontend"""
        try:
            await self.websocket.send_text(json.dumps(data))
        except Exception as e:
            self.session_logger.error(f"Failed to send message: {e}")
            
    async def _send_error(self, error_message: str):
        """Send error message to frontend"""
        await self._send_message({
            "type": "error",
            "message": error_message
        })
        
    def _get_frame_size(self) -> int:
        """Calculate frame size based on standardized configuration"""
        sample_rate = settings.sample_rate
        frame_ms = settings.audio_frame_ms
        channels = settings.channels
        bytes_per_sample = 2  # 16-bit audio
        
        return sample_rate * frame_ms // 1000 * bytes_per_sample * channels
        
    async def cleanup(self):
        """Clean up resources when WebSocket connection is closed"""
        self.session_logger.info(f"Cleaning up session: {self.session_id}")
        if self.watchdog_task: # Watchdog
            self.watchdog_task.cancel() # Watchdog
        try:
            # Cancel all processing tasks
            for task in self.processing_tasks:
                if not task.done():
                    task.cancel()
                    
            # Wait for tasks to complete
            if self.processing_tasks:
                await asyncio.gather(*self.processing_tasks, return_exceptions=True)
            
            # Terminate any active subprocesses
            if self.active_subprocesses:
                self.session_logger.info(f"Terminating {len(self.active_subprocesses)} active subprocesses")
                for proc in self.active_subprocesses.copy():
                    try:
                        if proc.returncode is None:  # Process still running
                            proc.terminate()
                            try:
                                await asyncio.wait_for(proc.wait(), timeout=2.0)
                            except asyncio.TimeoutError:
                                self.session_logger.warning(f"Subprocess did not terminate gracefully, killing it")
                                proc.kill()
                                await proc.wait()
                        self.active_subprocesses.discard(proc)
                    except Exception as e:
                        self.session_logger.error(f"Error terminating subprocess: {e}")
                
            # Stop any ongoing TTS
            if self.tts_active:
                await self.services['tts'].stop()
                
            self.session_logger.info(f"Session {self.session_id} cleaned up")
            
        except Exception as e:
            self.session_logger.error(f"Cleanup error: {e}") 

    async def _convert_audio_via_service(self, audio_data: bytes) -> Optional[bytes]:
        """Convert audio using the centralized AudioService"""
        try:
            audio_service = self.services['audio_service']
            
            # Try different extraction methods in order of preference
            pcm_array = None
            
            # Method 1: Smart PCM extraction (handles multiple formats)
            pcm_array = audio_service.extract_pcm_smart(audio_data)
            
            if pcm_array is None:
                # Method 2: Direct WebM extraction
                pcm_array = audio_service.extract_pcm_from_webm(audio_data)
            
            if pcm_array is None:
                # Method 3: Raw bytes extraction
                pcm_array = audio_service.extract_pcm_from_raw(audio_data)
            
            if pcm_array is not None:
                # Convert numpy array to bytes (s16le format expected by services)
                pcm_int16 = (pcm_array * 32768.0).astype(np.int16)
                return pcm_int16.tobytes()
            
            return None
            
        except Exception as e:
            self.session_logger.warning(f"AudioService conversion failed: {e}")
            return None

    async def _convert_audio_direct(self, audio_data: bytes) -> Optional[bytes]:
        """Direct FFmpeg audio conversion with improved subprocess management"""
        try:
            # Get target sample rate from standardized configuration
            target_sample_rate = settings.sample_rate
            target_channels = settings.channels
            
            # Enhanced format detection
            is_webm = audio_data.startswith(b'\x1a\x45\xdf\xa3')  # WebM magic bytes
            is_wav = audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:20]  # WAV magic bytes
            is_ogg = audio_data.startswith(b'OggS')  # Ogg magic bytes
            
            # Try different FFmpeg approaches based on data characteristics
            pcm_data = None
            
            # Method 1: Handle WAV format first (often more reliable)
            if is_wav or (len(audio_data) > 44 and not is_webm and not is_ogg):
                pcm_data = await self._run_ffmpeg_conversion(
                    audio_data, 
                    ['-f', 'wav'], 
                    target_sample_rate, 
                    target_channels,
                    "WAV format"
                )
            
            # Method 2: Try as WebM container (most common from browsers)
            if pcm_data is None and (is_webm or len(audio_data) > 500):
                pcm_data = await self._run_ffmpeg_conversion(
                    audio_data, 
                    ['-f', 'matroska'], 
                    target_sample_rate, 
                    target_channels,
                    "WebM format"
                )
            
            # Method 3: Try as Ogg/Opus
            if pcm_data is None and (is_ogg or len(audio_data) > 500):
                pcm_data = await self._run_ffmpeg_conversion(
                    audio_data, 
                    ['-f', 'ogg'], 
                    target_sample_rate, 
                    target_channels,
                    "Ogg format"
                )
            
            # Method 4: Try auto-detect format
            if pcm_data is None and len(audio_data) > 1000:
                pcm_data = await self._run_ffmpeg_conversion(
                    audio_data, 
                    [], 
                    target_sample_rate, 
                    target_channels,
                    "auto-detect format"
                )
            
            # Method 5: Handle failed chunks with retry logic
            if pcm_data is None:
                pcm_data = await self._handle_failed_chunk(audio_data, target_sample_rate, target_channels)
            
            return pcm_data
            
        except Exception as e:
            self.session_logger.error(f"Direct audio conversion failed: {e}")
            return None

    async def _run_ffmpeg_conversion(self, audio_data: bytes, input_args: list, 
                                   target_sample_rate: int, target_channels: int, format_name: str) -> Optional[bytes]:
        """Run FFmpeg conversion with proper subprocess management"""
        try:
            ffmpeg_cmd = [
                'ffmpeg',
                *input_args,
                '-i', 'pipe:0',
                '-f', 's16le',
                '-ar', str(target_sample_rate),
                '-ac', str(target_channels),
                '-hide_banner',
                '-loglevel', 'error',
                'pipe:1'
            ]
            
            self.session_logger.debug(f"FFmpeg {format_name} command: {' '.join(ffmpeg_cmd)}")

            proc = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Track subprocess for cleanup
            self.active_subprocesses.add(proc)
            
            try:
                pcm_data, stderr_bytes = await proc.communicate(input=audio_data)
                
                if proc.returncode != 0:
                    stderr_str = stderr_bytes.decode('utf-8', errors='ignore').strip()
                    self.session_logger.warning(f"FFmpeg {format_name} failed (return code {proc.returncode}): {stderr_str}")
                    return None
                elif not pcm_data:
                    self.session_logger.warning(f"FFmpeg {format_name} produced no PCM data")
                    return None
                else:
                    # DEBUG: Analyze the PCM data produced by FFmpeg
                    samples_produced = len(pcm_data) // 2  # 16-bit samples
                    duration_ms = (samples_produced / target_sample_rate) * 1000
                    self.session_logger.debug(f"FFmpeg {format_name} success: {len(pcm_data)} bytes = {samples_produced} samples = {duration_ms:.1f}ms @ {target_sample_rate}Hz")
                    self.session_logger.debug(f"Successfully converted audio to PCM via {format_name}. PCM data length: {len(pcm_data)}")
                    return pcm_data
                    
            finally:
                # Remove from tracking set
                self.active_subprocesses.discard(proc)
                
        except Exception as e:
            self.session_logger.warning(f"FFmpeg {format_name} failed with exception: {e}")
            return None

    async def _handle_failed_chunk(self, audio_data: bytes, target_sample_rate: int, target_channels: int) -> Optional[bytes]:
        """Handle failed audio chunks with improved buffering and retry logic"""
        try:
            self.session_logger.warning("All initial FFmpeg methods failed, buffering chunk for accumulation")
            
            # Prevent unbounded growth by limiting buffer size - using config setting
            max_buffer_size = settings.failed_chunk_buffer_max_size
            if len(self.failed_chunk_buffer) + len(audio_data) > max_buffer_size:
                # Keep only the most recent data
                excess = len(self.failed_chunk_buffer) + len(audio_data) - max_buffer_size
                self.failed_chunk_buffer = self.failed_chunk_buffer[excess:]
                self.session_logger.warning(f"Failed chunk buffer trimmed to prevent excessive memory usage")
            
            self.failed_chunk_buffer.extend(audio_data)
            
            # Check retry count to prevent infinite loops
            if self.failed_chunk_retry_count >= self.max_failed_chunk_retries:
                self.session_logger.error(f"Maximum retry attempts ({self.max_failed_chunk_retries}) reached, clearing failed buffer")
                self.failed_chunk_buffer.clear()
                self.failed_chunk_retry_count = 0
                return None
            
            # Try processing accumulated buffer if it's large enough
            buffer_threshold = 8000  # 8KB threshold
            if len(self.failed_chunk_buffer) > buffer_threshold:
                self.session_logger.info(f"Trying to process accumulated buffer: {len(self.failed_chunk_buffer)} bytes (attempt {self.failed_chunk_retry_count + 1})")
                self.failed_chunk_retry_count += 1
                
                # Try WebM first on accumulated buffer
                pcm_data = await self._run_ffmpeg_conversion(
                    bytes(self.failed_chunk_buffer),
                    ['-f', 'matroska'],
                    target_sample_rate,
                    target_channels,
                    "accumulated buffer WebM"
                )
                
                if pcm_data:
                    self.session_logger.info(f"Successfully processed accumulated buffer. PCM data length: {len(pcm_data)}")
                    self.failed_chunk_buffer.clear()
                    self.failed_chunk_retry_count = 0
                    return pcm_data
                else:
                    # Try fallback auto-detect on buffer
                    pcm_data = await self._run_ffmpeg_conversion(
                        bytes(self.failed_chunk_buffer),
                        [],
                        target_sample_rate,
                        target_channels,
                        "accumulated buffer auto-detect"
                    )
                    
                    if pcm_data:
                        self.session_logger.info("Accumulated buffer processed with fallback auto-detect")
                        self.failed_chunk_buffer.clear()
                        self.failed_chunk_retry_count = 0
                        return pcm_data
                    else:
                        self.session_logger.warning(f"Accumulated buffer processing failed (attempt {self.failed_chunk_retry_count})")
            
            return None
            
        except Exception as e:
            self.session_logger.error(f"Failed chunk handling error: {e}")
            self.failed_chunk_buffer.clear()
            self.failed_chunk_retry_count = 0
            return None

    async def _watchdog_timer(self):
        """Watchdog timer to detect inactivity and manage session lifecycle"""
        while True:
            await asyncio.sleep(settings.watchdog_check_interval)
            current_time = time.time()

            # Stop if client disconnected first, to avoid trying to close an already closed socket
            if self.websocket.client_state == WebSocketState.DISCONNECTED:
                 self.session_logger.info("Client disconnected, stopping watchdog.")
                 break

            # Close if no audio for configured timeout AND no pipeline (STT/AI/TTS) running
            if (current_time - self.last_audio_time > settings.watchdog_inactivity_timeout) and not self.pipeline_running:
                self.session_logger.warning(f"WebSocket inactivity detected ({settings.watchdog_inactivity_timeout}s audio, no pipeline). Last audio: {self.last_audio_time}, Current: {current_time}, Pipeline: {self.pipeline_running}. Closing connection.")
                if self.websocket.client_state == WebSocketState.CONNECTED:
                    await self.websocket.close(code=1011, reason="WebSocket inactivity timeout") # Using 1011 for server error, or a custom code
                break  # Exit watchdog loop after attempting to close or if already disconnected

async def run_pipeline_wrapper(handler_instance, buffer_data):
    # This is a conceptual wrapper. The actual logic is now integrated into EOS.
    # It would involve:
    # 1. Processing buffer_data through STT (including finalization)
    # 2. Taking STT output and running it through the AI (_process_with_ai)
    # This function might not be strictly necessary if EOS handler directly calls the sequence.
    pass 