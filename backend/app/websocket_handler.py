import asyncio
import json
import time
import os
from typing import Dict, Any, Optional
import numpy as np
from fastapi import WebSocket
from fastapi.websockets import WebSocketState
import ffmpeg  # This is ffmpeg-python package
# import asyncio.subprocess # Already imported with asyncio

from .ai.context_manager import ContextManager


class WebSocketHandler:
    """Handles WebSocket connections and message routing for voice agent"""
    
    def __init__(self, websocket: WebSocket, engines: Dict, metrics: Any, logger: Any):
        self.websocket = websocket
        self.engines = engines
        self.metrics = metrics
        self.logger = logger
        
        # Session state - use a more reliable session ID
        self.session_id = f"session_{int(time.time() * 1000)}"
        self.logger.info(f"Created session {self.session_id}")
        self.is_muted = False
        self.is_speaking = False
        self.audio_buffer = bytearray()
        self.context_manager = ContextManager()
        
        # Audio chunk buffers for improved processing
        self.small_chunk_buffer = bytearray()    # For small chunks that need to be combined
        self.failed_chunk_buffer = bytearray()   # For chunks that failed initial processing
        
        # Timing tracking
        self.speech_start_time = None
        self.processing_start_time = None
        
        # Audio streaming state
        self.tts_playing = False
        self.should_interrupt = False
        
        # Tasks
        self.processing_tasks = set()
        self.last_audio_time = time.time() # Watchdog
        self.pipeline_running = False # Watchdog - Tracks if STT/AI/TTS is active
        self.watchdog_task = None # Watchdog
        
        self.logger.info(f"WebSocket handler initialized for session {self.session_id}")
        
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
                    self.logger.error(f"Error receiving WebSocket message: {e}")
                    break
                
                if message["type"] == "websocket.disconnect":
                    self.logger.info("WebSocket disconnect message received")
                    break
                    
                try:
                    if "bytes" in message:
                        # Binary audio data
                        self.logger.debug(f"Received binary message: {len(message['bytes'])} bytes")
                        await self.handle_audio_chunk(message["bytes"])
                    elif "text" in message:
                        # JSON control message
                        self.logger.debug(f"Received text message: {message['text'][:100]}...")
                        try:
                            data = json.loads(message["text"])
                            await self.handle_control_message(data)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Invalid JSON received: {message['text']}, error: {e}")
                    else:
                        self.logger.warning(f"Unknown message format: {message}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}", exc_info=True)
                    # Continue processing other messages instead of breaking
                        
        except Exception as e:
            self.logger.error(f"WebSocket handler connection error: {e}", exc_info=True)
            raise
        finally: # Watchdog
            if self.watchdog_task: # Watchdog
                self.watchdog_task.cancel() # Watchdog
            
    async def handle_audio_chunk(self, audio_data: bytes):
        """Handle incoming audio chunk from WebSocket"""
        try:
            self.logger.debug(f"Received binary message: {len(audio_data)} bytes")
            
            if len(audio_data) == 0:
                self.logger.debug("Received empty audio chunk, skipping")
                return
                
            # Minimum chunk size check - avoid processing very small chunks
            if len(audio_data) < 100:  # Less than 100 bytes is likely not valid audio
                self.logger.debug(f"Audio chunk too small ({len(audio_data)} bytes), buffering for next chunk")
                self.small_chunk_buffer.extend(audio_data)
                return
            
            # If we have buffered small chunks, combine them
            if hasattr(self, 'small_chunk_buffer') and len(self.small_chunk_buffer) > 0:
                combined_data = bytes(self.small_chunk_buffer) + audio_data
                self.small_chunk_buffer.clear()
                audio_data = combined_data
                self.logger.debug(f"Combined with buffered data, new size: {len(audio_data)} bytes")
            
            self.logger.debug(f"Processing audio chunk: {len(audio_data)} bytes")
                
            self.last_audio_time = time.time() # Watchdog
            self.logger.debug(f"Updated last_audio_time to {self.last_audio_time}")

            # Check if this looks like valid WebM data
            is_webm = audio_data.startswith(b'\x1a\x45\xdf\xa3')  # WebM magic bytes
            
            # Convert incoming audio (WebM/Opus or Ogg/Opus) to raw PCM s16le
            # Get target sample rate from one of the engines
            target_sample_rate = 16000 # Default
            self.logger.debug(f"Checking engines for sample rate: {list(self.engines.keys())}")
            
            if self.engines and 'stt' in self.engines and hasattr(self.engines['stt'], 'sample_rate'):
                target_sample_rate = self.engines['stt'].sample_rate
                self.logger.debug(f"Using STT sample rate: {target_sample_rate}")
            elif self.engines and 'vad' in self.engines and hasattr(self.engines['vad'], 'sample_rate'):
                target_sample_rate = self.engines['vad'].sample_rate
                self.logger.debug(f"Using VAD sample rate: {target_sample_rate}")

            # Try different FFmpeg approaches based on data characteristics
            pcm_data = None
            
            # Method 1: Try as WebM container first (most common from browsers)
            if is_webm or len(audio_data) > 500:
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-f', 'matroska',      # Force WebM/Matroska input format
                    '-i', 'pipe:0',        
                    '-f', 's16le',         
                    '-ar', str(target_sample_rate), 
                    '-ac', '1',            
                    '-hide_banner',
                    '-loglevel', 'error',
                    'pipe:1'               
                ]
                
                self.logger.debug(f"FFmpeg WebM command: {' '.join(ffmpeg_cmd)}")

                try:
                    proc = await asyncio.create_subprocess_exec(
                        *ffmpeg_cmd,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    pcm_data, stderr_bytes = await proc.communicate(input=audio_data)
                    
                    if proc.returncode != 0:
                        stderr_str = stderr_bytes.decode('utf-8', errors='ignore').strip()
                        self.logger.warning(f"FFmpeg WebM format failed (return code {proc.returncode}): {stderr_str}")
                        pcm_data = None
                    elif not pcm_data:
                        self.logger.warning("FFmpeg WebM format produced no PCM data")
                        pcm_data = None
                    else:
                        self.logger.debug(f"Successfully converted audio to PCM via WebM format. PCM data length: {len(pcm_data)}")
                        
                except Exception as e:
                    self.logger.warning(f"FFmpeg WebM format failed with exception: {e}")
                    pcm_data = None
            
            # Method 2: Try auto-detect format if WebM failed
            if pcm_data is None and len(audio_data) > 1000:
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-i', 'pipe:0',        # Input from stdin (auto-detect format)
                    '-f', 's16le',         # Output format: signed 16-bit little-endian PCM
                    '-ar', str(target_sample_rate), # Output sample rate
                    '-ac', '1',            # Output channels: mono
                    '-hide_banner',
                    '-loglevel', 'error',
                    'pipe:1'               # Output to stdout
                ]
                
                self.logger.debug(f"FFmpeg auto-detect command: {' '.join(ffmpeg_cmd)}")

                try:
                    proc = await asyncio.create_subprocess_exec(
                        *ffmpeg_cmd,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    self.logger.debug("FFmpeg process created successfully")
                    
                    pcm_data, stderr_bytes = await proc.communicate(input=audio_data)
                    self.logger.debug(f"FFmpeg completed with return code: {proc.returncode}")

                    if proc.returncode != 0:
                        stderr_str = stderr_bytes.decode('utf-8', errors='ignore').strip()
                        self.logger.warning(f"FFmpeg auto-detect failed (return code {proc.returncode}): {stderr_str}")
                        pcm_data = None
                    elif not pcm_data:
                        self.logger.warning("FFmpeg auto-detect produced no PCM data")
                        pcm_data = None
                    else:
                        self.logger.debug(f"Successfully converted audio to PCM via auto-detect. PCM data length: {len(pcm_data)}")
                        
                except Exception as e:
                    self.logger.warning(f"FFmpeg auto-detect failed with exception: {e}")
                    pcm_data = None
            
            # Method 3: Try as Ogg/Opus if still no success
            if pcm_data is None and len(audio_data) > 500:
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-f', 'ogg',           # Force Ogg input format
                    '-i', 'pipe:0',        
                    '-f', 's16le',         
                    '-ar', str(target_sample_rate), 
                    '-ac', '1',            
                    '-hide_banner',
                    '-loglevel', 'error',
                    'pipe:1'               
                ]
                
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *ffmpeg_cmd,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    pcm_data, stderr_bytes = await proc.communicate(input=audio_data)
                    
                    if proc.returncode != 0:
                        stderr_str = stderr_bytes.decode('utf-8', errors='ignore').strip()
                        self.logger.warning(f"FFmpeg Ogg format failed (return code {proc.returncode}): {stderr_str}")
                        pcm_data = None
                    elif not pcm_data:
                        self.logger.warning("FFmpeg Ogg format produced no PCM data")
                        pcm_data = None
                    else:
                        self.logger.debug(f"Successfully converted audio to PCM via Ogg format. PCM data length: {len(pcm_data)}")
                        
                except Exception as e:
                    self.logger.warning(f"FFmpeg Ogg format failed with exception: {e}")
                    pcm_data = None
            
            # If all methods failed, buffer the chunk for later processing with improved logic
            if pcm_data is None:
                self.logger.warning("All FFmpeg methods failed, buffering chunk for accumulation")
                if not hasattr(self, 'failed_chunk_buffer'):
                    self.failed_chunk_buffer = bytearray()
                
                # Prevent unbounded growth by limiting buffer size
                max_buffer_size = 50000  # 50KB maximum
                if len(self.failed_chunk_buffer) + len(audio_data) > max_buffer_size:
                    # Keep only the most recent data
                    excess = len(self.failed_chunk_buffer) + len(audio_data) - max_buffer_size
                    self.failed_chunk_buffer = self.failed_chunk_buffer[excess:]
                    self.logger.warning(f"Failed chunk buffer trimmed to prevent excessive memory usage")
                
                self.failed_chunk_buffer.extend(audio_data)
                
                # Try processing accumulated buffer if it's large enough
                buffer_threshold = 8000  # 8KB threshold (increased from 5KB)
                if len(self.failed_chunk_buffer) > buffer_threshold:
                    self.logger.info(f"Trying to process accumulated buffer: {len(self.failed_chunk_buffer)} bytes")
                    
                    # Try WebM first on accumulated buffer
                    ffmpeg_cmd = [
                        'ffmpeg',
                        '-f', 'matroska',
                        '-i', 'pipe:0',        
                        '-f', 's16le',         
                        '-ar', str(target_sample_rate), 
                        '-ac', '1',            
                        '-hide_banner',
                        '-loglevel', 'error',
                        'pipe:1'               
                    ]
                    
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            *ffmpeg_cmd,
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        
                        pcm_data, stderr_bytes = await proc.communicate(input=bytes(self.failed_chunk_buffer))
                        
                        if proc.returncode == 0 and pcm_data:
                            self.logger.info(f"Successfully processed accumulated buffer. PCM data length: {len(pcm_data)}")
                            self.failed_chunk_buffer.clear()
                        else:
                            stderr_str = stderr_bytes.decode('utf-8', errors='ignore').strip()
                            self.logger.error(f"Accumulated buffer processing failed: {stderr_str}")
                            # Try fallback auto-detect on buffer
                            try:
                                fallback_cmd = [
                                    'ffmpeg', '-i', 'pipe:0', '-f', 's16le', 
                                    '-ar', str(target_sample_rate), '-ac', '1', 
                                    '-hide_banner', '-loglevel', 'error', 'pipe:1'
                                ]
                                
                                fallback_proc = await asyncio.create_subprocess_exec(
                                    *fallback_cmd,
                                    stdin=asyncio.subprocess.PIPE,
                                    stdout=asyncio.subprocess.PIPE,
                                    stderr=asyncio.subprocess.PIPE
                                )
                                
                                pcm_data, _ = await fallback_proc.communicate(input=bytes(self.failed_chunk_buffer))
                                if fallback_proc.returncode == 0 and pcm_data:
                                    self.logger.info("Accumulated buffer processed with fallback auto-detect")
                                    self.failed_chunk_buffer.clear()
                                else:
                                    self.logger.error("All buffer processing methods failed, clearing buffer")
                                    self.failed_chunk_buffer.clear()
                            except Exception as fallback_e:
                                self.logger.error(f"Fallback processing failed: {fallback_e}")
                                self.failed_chunk_buffer.clear()
                                
                    except Exception as e:
                        self.logger.error(f"Accumulated buffer processing failed with exception: {e}")
                        self.failed_chunk_buffer.clear()
                
                # If still no PCM data, just return without processing
                if pcm_data is None:
                    return

            # Send error to frontend if all processing failed
            if not pcm_data:
                await self._send_error("Audio processing failed - invalid format or corrupted data")
                return

            # Add PCM data to buffer for frame processing
            self.audio_buffer.extend(pcm_data)
            self.logger.debug(f"Audio buffer now has {len(self.audio_buffer)} bytes")
            
            # Check if we have enough data to process
            frame_size = self._get_frame_size() 
            self.logger.debug(f"Frame size: {frame_size} bytes")
            
            while len(self.audio_buffer) >= frame_size:
                # Extract frame
                frame = bytes(self.audio_buffer[:frame_size])
                self.audio_buffer = self.audio_buffer[frame_size:]
                self.logger.debug(f"Processing frame of {len(frame)} bytes, buffer remaining: {len(self.audio_buffer)}")
                
                # Process frame asynchronously
                task = asyncio.create_task(self._process_audio_frame(frame))
                self.processing_tasks.add(task)
                task.add_done_callback(self.processing_tasks.discard)
                
        except Exception as e:
            # Log with full traceback for better debugging
            self.logger.error(f"Audio chunk processing error: {e}", exc_info=True)
            raise
            
    async def _process_audio_frame(self, frame: bytes):
        """Process a single PCM audio frame through the pipeline"""
        try:
            self.logger.debug(f"Starting audio frame processing: {len(frame)} bytes")
            self.pipeline_running = True # Watchdog
            
            # VAD check
            self.logger.debug("Calling VAD process_frame...")
            vad_result = await self.engines['vad'].process_frame(frame)
            self.logger.debug(f"VAD result: is_speech={vad_result.is_speech}, is_end_of_speech={vad_result.is_end_of_speech}")
            
            if vad_result.is_speech:
                self.logger.debug("Speech detected by VAD")
                # Speech detected
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_start_time = time.time()
                    self.logger.debug(f"Speech started at {self.speech_start_time}")
                    
                    # Implement barge-in
                    if self.tts_playing:
                        self.logger.debug("Barge-in detected during TTS")
                        await self._handle_barge_in()
                        
                # STT processing
                self.logger.debug("Calling STT process_frame...")
                stt_result = await self.engines['stt'].process_frame(frame)
                self.logger.debug(f"STT result: partial='{stt_result.partial_text}', final='{stt_result.final_text}', is_final={stt_result.is_final}")
                
                if stt_result.partial_text:
                    # Send partial transcript
                    self.logger.debug(f"Sending partial transcript: {stt_result.partial_text}")
                    await self._send_message({
                        "type": "transcript",
                        "partial": stt_result.partial_text
                    })
                    
                if stt_result.is_final:
                    # Send final transcript and process with AI
                    self.logger.debug(f"Sending final transcript: {stt_result.final_text}")
                    await self._send_message({
                        "type": "transcript", 
                        "final": stt_result.final_text
                    })
                    
                    # Process with Gemini
                    self.logger.debug("Starting AI processing...")
                    await self._process_with_ai(stt_result.final_text)
                    
            elif vad_result.is_end_of_speech and self.is_speaking:
                # End of speech detected
                self.logger.debug("End of speech detected by VAD")
                self.is_speaking = False
                
                # Finalize STT
                self.logger.debug("Finalizing STT...")
                final_result = await self.engines['stt'].finalize()
                if final_result and final_result.final_text:
                    self.logger.debug(f"STT finalized with text: {final_result.final_text}")
                    await self._send_message({
                        "type": "transcript",
                        "final": final_result.final_text
                    })
                    await self._process_with_ai(final_result.final_text)
                else:
                    self.logger.debug("STT finalization returned no text")
                    
        except Exception as e:
            self.logger.error(f"Audio frame processing error: {e}", exc_info=True)
        finally: # Watchdog
            # self.pipeline_running = False # This will be set to False after AI and TTS are done.
            pass
            
    async def _handle_barge_in(self):
        """Handle barge-in when user speaks over TTS"""
        self.logger.info("Barge-in detected, stopping TTS")
        
        # Stop TTS immediately
        if self.tts_playing:
            await self.engines['tts'].stop()
            self.tts_playing = False
            self.should_interrupt = True
            
            # Notify frontend to stop audio
            await self._send_message({
                "type": "control",
                "action": "stop_audio"
            })
            
        # Track barge-in metrics
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
            async for token in self.engines['gemini'].generate_response(
                self.context_manager.get_messages()
            ):
                response_text += token
                
                # Send streaming token
                await self._send_message({
                    "type": "ai_response",
                    "token": token
                })
                
                # Start TTS early if we have enough tokens
                if len(response_text) > 50 and not self.tts_playing:
                    asyncio.create_task(self._start_tts_streaming(response_text))
                    
            # Mark response as complete
            await self._send_message({
                "type": "ai_response",
                "complete": True
            })
            
            # Update context with AI response
            self.context_manager.add_ai_message(response_text)
            
            # Track processing latency
            if self.speech_start_time:
                latency = time.time() - self.speech_start_time
                self.metrics.record_latency('end_to_end_latency', latency)
                
        except Exception as e:
            self.logger.error(f"AI processing error: {e}")
            await self._send_error(f"AI conversation failed: {str(e)}")
        finally: # Watchdog
            # self.pipeline_running will be set to False after TTS is done or if no TTS
            if not self.tts_playing:
                 self.pipeline_running = False
            
    async def _start_tts_streaming(self, text: str):
        """Start TTS streaming for AI response"""
        try:
            # self.pipeline_running = True # Should be true already
            if self.should_interrupt:
                # Skip TTS if interrupted
                self.should_interrupt = False
                return
                
            self.tts_playing = True
            
            # Generate TTS audio chunks
            async for audio_chunk in self.engines['tts'].generate_speech_stream(text):
                if self.should_interrupt:
                    # Stop if interrupted
                    break
                    
                # Send audio chunk to frontend
                await self.websocket.send_bytes(audio_chunk)
                
            self.tts_playing = False
            self.pipeline_running = False # Watchdog - Pipeline ends after TTS
            
        except Exception as e:
            self.logger.error(f"TTS streaming error: {e}")
            await self._send_error(f"Text-to-speech failed: {str(e)}")
            self.tts_playing = False
            self.pipeline_running = False # Watchdog - Pipeline ends on TTS error
            
    async def handle_control_message(self, data: Dict[str, Any]):
        """Handle control messages from frontend"""
        try:
            message_type = data.get("type")
            
            if message_type == "control":
                action = data.get("action")
                
                if action == "mute":
                    self.is_muted = True
                    self.logger.info("Session muted")
                    
                elif action == "unmute":
                    self.is_muted = False
                    self.logger.info("Session unmuted")
                    
                elif action == "end_session":
                    self.logger.info("Session ended by user")
                    await self._send_message({
                        "type": "control",
                        "action": "session_ended"
                    })
                    await self.websocket.close(code=1000, reason="Session ended")
                    
            elif message_type == "ping": # Keep-alive ping
                self.logger.info(f"Received ping: {data}") # Keep-alive ping
                self.last_audio_time = time.time() # Update activity time on ping
                await self.websocket.send_json({"type": "pong", "t": data.get("t")}) # Keep-alive ping
                self.logger.info("Sent pong") # Keep-alive ping

            elif message_type == "eos": # End of stream
                self.logger.info("EOS received from client")
                await self.websocket.send_json({"type": "status", "status": "processing"})
                self.logger.info("Sent status: processing")

                self.pipeline_running = True # Watchdog - Start of pipeline after EOS
                try:
                    # Simplified EOS handling: always finalize STT and process any remaining audio
                    final_stt_result = None
                    
                    # Process any remaining audio buffer first
                    if self.audio_buffer:
                        self.logger.info(f"Processing remaining audio buffer of {len(self.audio_buffer)} bytes after EOS")
                        
                        # Process remaining buffer through VAD to signal end of speech
                        try:
                            vad_result = await self.engines['vad'].process_frame(bytes(self.audio_buffer))
                            self.audio_buffer.clear()
                            self.logger.debug(f"VAD processed remaining buffer: speech={vad_result.is_speech}")
                        except Exception as vad_error:
                            self.logger.warning(f"VAD processing failed on remaining buffer: {vad_error}")
                    
                    # Always finalize STT to get any pending transcription
                    try:
                        final_stt_result = await self.engines['stt'].finalize()
                        self.logger.debug(f"STT finalization result: {final_stt_result}")
                    except Exception as stt_error:
                        self.logger.error(f"STT finalization failed: {stt_error}")
                    
                    # Process the final result if we got text
                    if final_stt_result and final_stt_result.final_text and final_stt_result.final_text.strip():
                        final_text = final_stt_result.final_text.strip()
                        self.logger.info(f"Final text from STT after EOS: {final_text}")
                        
                        # Check for duplicate to prevent reprocessing
                        if not self.context_manager.is_duplicate_user_message(final_text):
                            await self._send_message({"type": "transcript", "final": final_text})
                            # Start AI processing (will set pipeline_running to False when complete)
                            asyncio.create_task(self._process_with_ai(final_text))
                        else:
                            self.logger.info("Duplicate text detected, not reprocessing with AI.")
                            self.pipeline_running = False # Watchdog
                    else:
                        self.logger.info("No final STT result after EOS processing.")
                        self.pipeline_running = False # Watchdog
                        
                except Exception as e:
                    self.logger.error(f"Error during EOS processing: {e}")
                    self.pipeline_running = False # Watchdog - Ensure pipeline_running is reset on error
                    await self._send_error(f"Processing failed: {str(e)}")

            else:
                self.logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            self.logger.error(f"Control message error: {e}")
            await self._send_error(f"Message processing failed: {str(e)}")
            
    async def _send_message(self, data: Dict[str, Any]):
        """Send JSON message to frontend"""
        try:
            await self.websocket.send_text(json.dumps(data))
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            
    async def _send_error(self, error_message: str):
        """Send error message to frontend"""
        await self._send_message({
            "type": "error",
            "message": error_message
        })
        
    def _get_frame_size(self) -> int:
        """Calculate frame size based on configuration"""
        sample_rate = int(os.getenv('SAMPLE_RATE', '16000'))
        frame_ms = int(os.getenv('AUDIO_FRAME_MS', '120'))
        bytes_per_sample = 2  # 16-bit audio
        
        return sample_rate * frame_ms // 1000 * bytes_per_sample
        
    async def cleanup(self):
        """Clean up resources when WebSocket connection is closed"""
        self.logger.info(f"Cleaning up session: {self.session_id}")
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
                
            # Stop any ongoing TTS
            if self.tts_playing:
                await self.engines['tts'].stop()
                
            self.logger.info(f"Session {self.session_id} cleaned up")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}") 

    async def _watchdog_timer(self):
        """Watchdog timer to detect inactivity"""
        while True:
            await asyncio.sleep(1)  # Check every second
            current_time = time.time()

            # Stop if client disconnected first, to avoid trying to close an already closed socket
            if self.websocket.client_state == WebSocketState.DISCONNECTED:
                 self.logger.info("Client disconnected, stopping watchdog.")
                 break

            # Close if no audio for 10s AND no pipeline (STT/AI/TTS) running
            if (current_time - self.last_audio_time > 10) and not self.pipeline_running:
                self.logger.warning(f"WebSocket inactivity detected (10s audio, no pipeline). Last audio: {self.last_audio_time}, Current: {current_time}, Pipeline: {self.pipeline_running}. Closing connection.")
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