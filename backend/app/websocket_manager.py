import asyncio
import json
import time
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import structlog
import numpy as np

from backend.app.config import settings

logger = structlog.get_logger(__name__)

@dataclass
class WebSocketSession:
    """WebSocket session data"""
    websocket: WebSocket
    session_id: str
    created_at: float
    last_activity: float
    is_active: bool = True

class WebSocketManager:
    """Manages WebSocket connections and sessions"""
    
    def __init__(self, voice_service, metrics=None):
        self.voice_service = voice_service
        self.metrics = metrics
        self.sessions: Dict[str, WebSocketSession] = {}
        self.max_sessions = settings.max_concurrent_sessions
        self.session_timeout = getattr(settings, 'session_timeout', 300)  # 5 minutes default
        
    async def handle_connection(self, websocket: WebSocket):
        """Handle a new WebSocket connection"""
        session_id = self._generate_session_id(websocket)
        session_logger = logger.bind(session_id=session_id)
        
        try:
            # Check if voice service is available
            if not self.voice_service.is_available:
                # Check which specific services are unavailable
                health = await self.voice_service.get_health_status()
                unavailable_services = []
                
                for service_name, status in health.items():
                    if service_name != "voice_service" and isinstance(status, dict) and status.get("available") is False:
                        unavailable_services.append(service_name)
                
                error_msg = f"Voice services unavailable: {', '.join(unavailable_services)}"
                session_logger.error(error_msg)
                await websocket.close(code=1013, reason=error_msg)
                return
                
            # Check session limits
            if len(self.sessions) >= self.max_sessions:
                session_logger.warning("Max sessions reached, rejecting connection")
                await websocket.close(code=1013, reason="Server overloaded")
                return
            
            # Accept connection
            await websocket.accept()
            session_logger.info("WebSocket connection accepted")
            
            # Create session
            session = WebSocketSession(
                websocket=websocket,
                session_id=session_id,
                created_at=time.time(),
                last_activity=time.time()
            )
            self.sessions[session_id] = session
            
            # Send initial status
            await self._send_message(websocket, {
                "type": "status",
                "session_id": session_id,
                "ready": True,
                "config": {
                    "sample_rate": settings.sample_rate,
                    "channels": settings.channels,
                    "frame_duration_ms": settings.frame_duration_ms
                }
            })
            
            # Handle session
            await self._handle_session(session, session_logger)
            
        except WebSocketDisconnect:
            session_logger.info("Client disconnected")
        except Exception as e:
            session_logger.error(f"Connection error: {e}", exc_info=True)
        finally:
            # Cleanup session
            if session_id in self.sessions:
                del self.sessions[session_id]
                session_logger.info(f"Session cleaned up. Active sessions: {len(self.sessions)}")
    
    async def _handle_session(self, session: WebSocketSession, session_logger):
        """Handle messages for a WebSocket session"""
        websocket = session.websocket
        
        # Start session handler
        session_handler = WebSocketSessionHandler(
            session=session,
            voice_service=self.voice_service,
            logger=session_logger,
            metrics=self.metrics
        )
        
        try:
            while websocket.client_state == WebSocketState.CONNECTED:
                # Receive message with timeout
                try:
                    message = await asyncio.wait_for(
                        websocket.receive(),
                        timeout=30.0  # 30 second timeout
                    )
                except asyncio.TimeoutError:
                    # Send ping to check connection
                    await self._send_message(websocket, {"type": "ping"})
                    continue
                
                # Update activity time
                session.last_activity = time.time()
                
                # Handle disconnect
                if message["type"] == "websocket.disconnect":
                    session_logger.info("Disconnect message received")
                    break
                
                # Handle message
                if "bytes" in message and message["bytes"]:
                    await session_handler.handle_audio(message["bytes"])
                elif "text" in message and message["text"]:
                    try:
                        data = json.loads(message["text"])
                        await session_handler.handle_control(data)
                    except json.JSONDecodeError as e:
                        session_logger.warning(f"Invalid JSON: {e}")
                        await self._send_error(websocket, "Invalid JSON format")
                        
        except Exception as e:
            session_logger.error(f"Session error: {e}", exc_info=True)
        finally:
            await session_handler.cleanup()
    
    async def _send_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """Send message to WebSocket client"""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps(data))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    async def _send_error(self, websocket: WebSocket, error: str):
        """Send error message to client"""
        error_message = error if error else "Unknown server error"
        await self._send_message(websocket, {
            "type": "error",
            "message": error_message,
            "timestamp": time.time()
        })
    
    def _generate_session_id(self, websocket: WebSocket) -> str:
        """Generate unique session ID"""
        timestamp = int(time.time() * 1000)
        host = websocket.client.host
        port = websocket.client.port
        return f"session_{timestamp}_{host}_{port}"
    
    async def cleanup(self):
        """Cleanup all sessions"""
        logger.info("Cleaning up WebSocket manager")
        
        # Close all active sessions
        for session_id, session in list(self.sessions.items()):
            try:
                if session.websocket.client_state == WebSocketState.CONNECTED:
                    await session.websocket.close()
            except Exception as e:
                logger.error(f"Error closing session {session_id}: {e}")
        
        self.sessions.clear()
        logger.info("WebSocket manager cleanup complete")

class WebSocketSessionHandler:
    """Handles individual WebSocket session logic"""
    
    def __init__(self, session: WebSocketSession, voice_service, logger, metrics=None):
        self.session = session
        self.voice_service = voice_service
        self.logger = logger
        self.metrics = metrics
        
        # Audio processing state
        self.audio_buffer = bytearray()
        self.is_processing = False
        self.processing_tasks: Set[asyncio.Task] = set()
        
        # Voice activity state
        self.is_speaking = False
        self.speech_start_time: Optional[float] = None
        
    async def handle_audio(self, audio_data: bytes):
        """Handle incoming audio data"""
        try:
            if not audio_data or len(audio_data) < 100:
                return
            
            self.logger.debug(f"Received audio chunk: {len(audio_data)} bytes")
            
            # Detect audio format and convert if needed
            is_webm = audio_data.startswith(b'\x1a\x45\xdf\xa3')  # WebM magic bytes
            is_wav = audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:20]  # WAV magic bytes
            is_ogg = audio_data.startswith(b'OggS')  # Ogg magic bytes
            
            # Debug log for first few audio chunks
            if not hasattr(self, '_audio_debug_counter'):
                self._audio_debug_counter = 0
            
            if self._audio_debug_counter < 3:
                format_name = "unknown"
                if is_webm:
                    format_name = "WebM"
                elif is_wav:
                    format_name = "WAV"
                elif is_ogg:
                    format_name = "OGG"
                else:
                    # Try to detect PCM by checking byte patterns
                    pcm_bytes = audio_data[:20]
                    if all(0 <= b <= 255 for b in pcm_bytes) and any(b > 0 for b in pcm_bytes):
                        # Get first few bytes for debug
                        first_bytes = ", ".join([f"{b:02x}" for b in pcm_bytes[:10]])
                        format_name = f"Likely PCM (first bytes: {first_bytes})"
                
                self.logger.info(f"Audio format detection: {format_name}, size: {len(audio_data)} bytes")
                self._audio_debug_counter += 1
            
            if is_webm or is_ogg:
                self.logger.debug(f"Detected {'WebM' if is_webm else 'Ogg'} format, will convert to PCM")
                # For WebM and Ogg formats, we need to accumulate data
                # Add to buffer directly and return
                self.audio_buffer.extend(audio_data)
                if len(self.audio_buffer) >= 1024:  # Wait for at least 1KB of data
                    pcm_data = await self._convert_to_pcm(bytes(self.audio_buffer))
                    if pcm_data:
                        # Reset buffer and process the converted PCM
                        self.audio_buffer = bytearray()
                        # Process PCM data frame by frame
                        frame_size = self._get_frame_size()
                        for i in range(0, len(pcm_data), frame_size):
                            frame = pcm_data[i:i+frame_size]
                            if len(frame) == frame_size:
                                await self._process_audio_frame(frame)
                            elif len(frame) > 0:
                                # Pad last frame if needed
                                padded = frame + bytes(frame_size - len(frame))
                                await self._process_audio_frame(padded)
                return
            elif is_wav:
                # For WAV, extract PCM data from WAV header
                if len(audio_data) > 44:  # WAV header is 44 bytes
                    pcm_data = audio_data[44:]  # Skip WAV header
                    # Process PCM directly
                    frame_size = self._get_frame_size()
                    for i in range(0, len(pcm_data), frame_size):
                        frame = pcm_data[i:i+frame_size]
                        if len(frame) == frame_size:
                            await self._process_audio_frame(frame)
                        elif len(frame) > 0:
                            # Pad last frame if needed
                            padded = frame + bytes(frame_size - len(frame))
                            await self._process_audio_frame(padded)
                    return
            
            # Default handling for PCM data or unknown format
            self.audio_buffer.extend(audio_data)
            
            # Process if we have enough data
            frame_size = self._get_frame_size()
            if len(self.audio_buffer) >= frame_size:
                await self._process_audio_buffer()
                
        except Exception as e:
            error_msg = str(e) or "Unknown audio processing error"
            self.logger.error(f"Audio handling error: {error_msg}", exc_info=True)
            await self._send_error(f"Audio processing failed: {error_msg}")
    
    async def _convert_to_pcm(self, audio_data: bytes) -> Optional[bytes]:
        """Convert WebM/Ogg audio to raw PCM data"""
        try:
            import tempfile
            import subprocess
            import os
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as in_file, \
                 tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as out_file:
                
                in_path = in_file.name
                out_path = out_file.name
                
                # Write input data
                in_file.write(audio_data)
                in_file.flush()
                
                # Convert using ffmpeg
                cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output
                    '-i', in_path,  # Input file
                    '-ar', '16000',  # Sample rate
                    '-ac', '1',  # Mono
                    '-f', 's16le',  # 16-bit PCM
                    out_path  # Output file
                ]
                
                # Run ffmpeg
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                _, stderr = await process.communicate()
                
                if process.returncode != 0:
                    error = stderr.decode() if stderr else f"FFmpeg error code: {process.returncode}"
                    self.logger.error(f"FFmpeg conversion error: {error}")
                    return None
                
                # Read output PCM data
                with open(out_path, 'rb') as f:
                    pcm_data = f.read()
                
                # Clean up temp files
                try:
                    os.unlink(in_path)
                    os.unlink(out_path)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp files: {e}")
                
                return pcm_data
                
        except Exception as e:
            self.logger.error(f"Audio conversion error: {e}", exc_info=True)
            return None
    
    async def handle_control(self, data: Dict[str, Any]):
        """Handle control messages"""
        try:
            message_type = data.get("type")
            
            if message_type == "ping":
                await self._send_message({"type": "pong"})
            elif message_type == "mute":
                await self._handle_mute(data.get("muted", False))
            elif message_type == "end_speech":
                await self._handle_end_speech()
            elif message_type == "eos":
                await self._handle_end_of_stream()
            elif message_type == "config":
                await self._handle_config_message(data)
            else:
                self.logger.warning(f"Unknown control message: {message_type}")
                await self._send_error(f"Unknown control message type: {message_type}")
                
        except Exception as e:
            error_msg = str(e) or "Unknown control message error"
            self.logger.error(f"Control message error: {error_msg}", exc_info=True)
            await self._send_error(f"Failed to process message: {error_msg}")
            
    async def _handle_config_message(self, data: Dict[str, Any]):
        """Handle configuration message from client"""
        try:
            # Enable diagnostic mode
            if data.get("diagnostics") is True:
                self.logger.info("Diagnostic mode enabled by client")
                
                # Log client info if provided
                client_info = data.get("client_info", {})
                if client_info:
                    self.logger.info(f"Client info: {client_info}")
                    
                # Set verbose logging
                if data.get("verbose_logging") is True:
                    self.logger.info("Verbose logging enabled")
                    
                # Send acknowledgment
                await self._send_message({
                    "type": "config_response",
                    "diagnostics_enabled": True,
                    "server_info": {
                        "frame_size": self._get_frame_size(),
                        "sample_rate": 16000,
                        "channels": 1,
                        "bit_depth": 16
                    }
                })
                
        except Exception as e:
            self.logger.error(f"Error handling config message: {e}")
            await self._send_error(f"Failed to apply configuration: {str(e)}")
    
    async def _process_audio_buffer(self):
        """Process accumulated audio buffer"""
        frame_size = self._get_frame_size()
        
        while len(self.audio_buffer) >= frame_size:
            # Extract frame
            frame = bytes(self.audio_buffer[:frame_size])
            self.audio_buffer = self.audio_buffer[frame_size:]
            
            # Process frame asynchronously
            task = asyncio.create_task(self._process_audio_frame(frame))
            self.processing_tasks.add(task)
            task.add_done_callback(self.processing_tasks.discard)
    
    async def _process_audio_frame(self, frame: bytes):
        """Process a single audio frame with VAD and send to speech-to-text if speech is detected"""
        try:
            # Convert frame to numpy array for processing
            frame_array = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32767.0
            
            if len(frame_array) == 0:
                self.logger.warning("Empty audio frame received")
                return
                
            # Process with VAD service
            vad_result = await self.voice_service.process_audio_chunk(frame_array)
            
            # Add diagnostic info
            vad_debug_info = {
                "speech_prob": vad_result.get("confidence", 0),
                "speech_frames": vad_result.get("consecutive_speech_frames", 0),
                "silence_frames": vad_result.get("consecutive_silence_frames", 0),
                "frame_size": len(frame_array)
            }
            
            # Log voice activity transitions with more detailed diagnostics
            if vad_result.get("speech_started"):
                self.logger.info(f"VAD: Speech started. Diagnostics: {vad_debug_info}")
                await self._send_message({"type": "vad_status", "status": "speech_started", "diagnostics": vad_debug_info})
                
            if vad_result.get("speech_ended"):
                self.logger.info(f"VAD: Speech ended. Diagnostics: {vad_debug_info}")
                await self._send_message({"type": "vad_status", "status": "speech_ended", "diagnostics": vad_debug_info})
                
            # Forward audio to voice service for processing if needed
            if vad_result.get("speech_detected"):
                await self.voice_service.process_audio(frame)
                
        except Exception as e:
            self.logger.error(f"Audio frame processing error: {e}", exc_info=True)
    
    async def _handle_end_of_stream(self):
        """Handle explicit end-of-stream marker from client"""
        self.logger.debug("Received end-of-stream signal from client")
        
        try:
            # Get latest VAD state for diagnostics
            vad_status = self.voice_service.get_vad_status()
            self.logger.info(f"VAD status at end-of-stream: {vad_status}")
            
            # Force end speech processing
            await self.voice_service.end_user_speech()
            
            # Acknowledge EOS to client
            await self._send_message({
                "type": "eos_ack",
                "vad_status": vad_status
            })
        except Exception as e:
            self.logger.error(f"Error handling end-of-stream: {e}", exc_info=True)
    
    async def _handle_mute(self, muted: bool):
        """Handle mute/unmute"""
        self.logger.info(f"Session {'muted' if muted else 'unmuted'}")
        # Could implement muting logic here
    
    async def _handle_end_speech(self):
        """Handle end of speech signal"""
        if self.is_speaking:
            self.is_speaking = False
            await self._send_message({"type": "speech_end"})
            self.logger.info("Speech ended by client signal")
    
    async def _send_message(self, data: Dict[str, Any]):
        """Send message to client"""
        try:
            websocket = self.session.websocket
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps(data))
        except Exception as e:
            self.logger.error(f"Failed to send message: {str(e)}")
    
    async def _send_error(self, error: str):
        """Send error message to client"""
        error_message = error if error else "Unknown server error"
        await self._send_message({
            "type": "error",
            "message": error_message,
            "timestamp": time.time()
        })
    
    def _get_frame_size(self) -> int:
        """Get audio frame size in bytes"""
        # 16-bit PCM, mono, for frame duration
        samples_per_frame = settings.sample_rate * settings.frame_duration_ms // 1000
        return samples_per_frame * 2  # 2 bytes per sample (16-bit)
    
    async def cleanup(self):
        """Cleanup session handler"""
        self.logger.info("Cleaning up session handler")
        
        # Cancel processing tasks
        for task in list(self.processing_tasks):
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        self.processing_tasks.clear()
        self.audio_buffer.clear()
        
        self.logger.info("Session handler cleanup complete") 