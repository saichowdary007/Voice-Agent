import asyncio
import json
import time
import base64
from typing import Dict, Set, Optional, Any, Callable
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
    timeout_task: Optional[asyncio.Task] = None

class WebSocketManager:
    """Manages WebSocket connections and sessions"""
    
    def __init__(self, voice_service, audio_service, metrics=None):
        self.voice_service = voice_service
        self.audio_service = audio_service
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
            
            # Start voice service session
            if not await self.voice_service.start_session(session_id):
                session_logger.error("Failed to start voice service session")
                await self._send_error(websocket, "Failed to start voice processing session")
                await websocket.close(code=1011, reason="Failed to start voice processing session")
                return
            
            # Set up voice service callbacks
            self.voice_service.set_callbacks(
                on_speech_start=lambda: self._send_message(websocket, {"type": "vad_status", "status": "speech_started"}),
                on_speech_end=lambda: self._send_message(websocket, {"type": "vad_status", "status": "speech_ended"}),
                on_partial_transcript=lambda txt: self._send_message(websocket, {"type": "transcript", "partial": True, "text": txt}),
                on_final_transcript=lambda txt: self._send_message(websocket, {"type": "transcript", "final": True, "text": txt}),
                on_llm_response=lambda txt: self._send_message(websocket, {"type": "ai_response", "text": txt}),
                on_audio_response=lambda chunk: self._send_message(websocket, {"type": "audio_chunk", "audio_data": base64.b64encode(chunk).decode()})
            )
            
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
            
            # Start timeout monitor
            session.timeout_task = asyncio.create_task(
                self._monitor_session_timeout(session, session_logger)
            )
            
            # Handle session
            await self._handle_session(session, session_logger)
            
        except WebSocketDisconnect:
            session_logger.info("Client disconnected")
        except Exception as e:
            session_logger.error(f"Connection error: {e}", exc_info=True)
        finally:
            # Stop the voice service session
            if hasattr(self.voice_service, 'stop_session'):
                try:
                    await self.voice_service.stop_session()
                except Exception as e:
                    session_logger.error(f"Error stopping voice session: {e}")
            
            # Cancel timeout task if it exists
            if session_id in self.sessions and self.sessions[session_id].timeout_task:
                self.sessions[session_id].timeout_task.cancel()
                
            # Cleanup session
            if session_id in self.sessions:
                del self.sessions[session_id]
                session_logger.info(f"Session cleaned up. Active sessions: {len(self.sessions)}")
    
    async def _monitor_session_timeout(self, session: WebSocketSession, session_logger):
        """Monitor session for inactivity timeout"""
        try:
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds
                current_time = time.time()
                if current_time - session.last_activity > self.session_timeout:
                    session_logger.info(f"Session timed out after {self.session_timeout} seconds of inactivity")
                    
                    # Notify client that session is ending
                    await self._send_message(session.websocket, {
                        "type": "control", 
                        "action": "session_ended",
                        "reason": "timeout"
                    })
                    
                    # Close the connection
                    if session.websocket.client_state == WebSocketState.CONNECTED:
                        await session.websocket.close(code=1000, reason="Session timeout")
                    break
        except asyncio.CancelledError:
            # Task was cancelled, cleanup is handled by the parent
            pass
        except Exception as e:
            session_logger.error(f"Error in session timeout monitor: {e}")
    
    async def _handle_session(self, session: WebSocketSession, session_logger):
        """Handle messages for a WebSocket session"""
        websocket = session.websocket
        
        # Start session handler
        session_handler = WebSocketSessionHandler(
            session=session,
            voice_service=self.voice_service,
            audio_service=self.audio_service,
            logger=session_logger,
            metrics=self.metrics,
            send_message=self._send_message,
            send_error=self._send_error
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
                # Cancel timeout tasks
                if session.timeout_task:
                    session.timeout_task.cancel()
                    
                # Send session ended notification
                if session.websocket.client_state == WebSocketState.CONNECTED:
                    await self._send_message(session.websocket, {
                        "type": "control",
                        "action": "session_ended",
                        "reason": "server_shutdown"
                    })
                    await session.websocket.close()
            except Exception as e:
                logger.error(f"Error closing session {session_id}: {e}")
        
        self.sessions.clear()
        logger.info("WebSocket manager cleanup complete")

class WebSocketSessionHandler:
    """Handles individual WebSocket session logic"""
    
    def __init__(self, session: WebSocketSession, voice_service, audio_service, logger, 
                 metrics=None, send_message=None, send_error=None):
        self.session = session
        self.voice_service = voice_service
        self.audio_service = audio_service
        self.logger = logger
        self.metrics = metrics
        
        # Message sending functions
        self._send_message = send_message
        self._send_error = send_error
        
        # Audio processing state
        self.audio_buffer = bytearray()
        self.is_processing = False
        self.processing_tasks: Set[asyncio.Task] = set()
        
        # Voice activity state
        self.is_speaking = False
        self.speech_start_time: Optional[float] = None
        
    async def handle_audio(self, audio_data: bytes):
        """Handle incoming audio data by passing it to the advanced audio service."""
        try:
            if not audio_data:
                return

            self.logger.debug(f"Received audio chunk: {len(audio_data)} bytes. Processing with AudioService.")

            # Use the robust audio service to convert any format to raw PCM
            # Run in executor to avoid blocking the event loop with CPU-intensive work
            pcm_audio_array = await asyncio.to_thread(
                self._process_audio_data, audio_data
            )

            if pcm_audio_array is not None and pcm_audio_array.size > 0:
                # Convert float32 numpy array back to 16-bit PCM bytes for the rest of the pipeline
                pcm_bytes = (pcm_audio_array * 32767).astype(np.int16).tobytes()
                
                self.logger.debug(f"Successfully converted audio to {len(pcm_bytes)} bytes of PCM data.")
                
                # Add the processed frame to the audio buffer for processing
                self.audio_buffer.extend(pcm_bytes)
                
                # Process buffer if it contains enough data for a frame
                frame_size = self._get_frame_size()
                while len(self.audio_buffer) >= frame_size:
                    frame = self.audio_buffer[:frame_size]
                    del self.audio_buffer[:frame_size]
                    
                    # Offload the actual processing to avoid blocking the audio reception loop
                    task = asyncio.create_task(self._process_audio_frame(frame))
                    self.processing_tasks.add(task)
                    task.add_done_callback(self.processing_tasks.discard)
            
            elif self.audio_service.has_failed_chunks():
                # Log information about chronically failing chunks if they exist
                failed_info = self.audio_service.get_failed_chunks_info()
                self.logger.warning(f"AudioService reported failed chunks: {failed_info['count']} chunks, {failed_info['total_size']} bytes.")
                # Optional: Clear failed chunks buffer if it gets too large
                if failed_info['count'] > 10:
                    self.logger.warning("Clearing accumulated failed audio chunks.")
                    self.audio_service.clear_failed_chunks()

        except Exception as e:
            self.logger.error(f"Error in handle_audio: {e}", exc_info=True)
    
    def _process_audio_data(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Process audio data in a separate thread to avoid blocking the event loop"""
        try:
            return self.audio_service.extract_pcm_smart_async(audio_data)
        except Exception as e:
            self.logger.error(f"Error processing audio data: {e}")
            return None

    async def handle_control(self, data: Dict[str, Any]):
        """Handle control messages from the client"""
        msg_type = data.get("type")
        self.logger.info(f"Received control message: {msg_type}")

        try:
            if msg_type == "config":
                await self._handle_config_message(data)
            elif msg_type == "endOfStream":
                await self._handle_end_of_stream()
            elif msg_type == "mute":
                await self._handle_mute(data.get("muted", False))
            elif msg_type == "endSpeech":
                await self._handle_end_speech()
            elif msg_type == "end_session":
                await self._handle_end_session()
            elif msg_type == "ping":
                # Handle ping from client
                await self._send_message({
                    "type": "pong",
                    "timestamp": time.time()
                })
            else:
                self.logger.warning(f"Unknown control message type: {msg_type}")
                await self._send_error(f"Unknown control message type: {msg_type}")
        except Exception as e:
            self.logger.error(f"Error handling control message: {e}", exc_info=True)
            await self._send_error(f"Error processing control message: {str(e)}")

    async def _handle_config_message(self, data: Dict[str, Any]):
        """Handle client configuration message"""
        config = data.get("config", {})
        self.logger.info(f"Client configuration received: {config}")
        
        # Example: update audio settings if provided
        sample_rate = config.get("sampleRate")
        if sample_rate and self.voice_service.sample_rate != sample_rate:
            self.logger.warning(f"Client sample rate ({sample_rate}) differs from server ({self.voice_service.sample_rate}). Mismatches may cause issues.")
        
        # Reset audio service session to clear any buffered data from previous configs
        if self.audio_service:
            self.audio_service.reset_session()
            self.logger.info("Audio service session state has been reset.")

        # Acknowledge config
        await self._send_message({
            "type": "status",
            "message": "Configuration received",
            "timestamp": time.time()
        })

    async def _process_audio_frame(self, frame: bytes):
        """Process a single audio frame"""
        try:
            # Send frame to voice service for processing (VAD, STT)
            result = await self.voice_service.process_audio_frame(frame)
            
            # Update metrics if enabled
            if self.metrics and 'latency' in result:
                await self.metrics.record_audio_processing_latency(result['latency'])
            
            # Handle VAD events if implemented in voice service
            # This part is simplified as VoiceService now manages VAD internally
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}", exc_info=True)

    async def _handle_end_of_stream(self):
        """Handle end of audio stream from client"""
        self.logger.info("End of stream received")
        
        # Process any remaining audio in the buffer
        if len(self.audio_buffer) > 0:
            frame_size = self._get_frame_size()
            frame = self.audio_buffer[:]
            if len(frame) < frame_size:
                # Pad the final frame
                frame += bytes(frame_size - len(frame))
            
            await self._process_audio_frame(frame)
            self.audio_buffer.clear()
        
        # Notify voice service that user speech has ended
        await self.voice_service.end_user_speech()
        
        # Acknowledge end of stream to client
        await self._send_message({
            "type": "eos_ack",
            "timestamp": time.time()
        })

    async def _handle_mute(self, muted: bool):
        """Handle mute/unmute commands from client"""
        self.logger.info(f"Setting mute status to: {muted}")
        
        # Send acknowledgment to client
        await self._send_message({
            "type": "mute_status",
            "muted": muted,
            "timestamp": time.time()
        })

    async def _handle_end_speech(self):
        """Handle explicit end-of-speech command from client"""
        self.logger.info("Client explicitly ended speech")
        await self.voice_service.end_user_speech()
        
        # Acknowledge to client
        await self._send_message({
            "type": "status",
            "message": "Speech ended",
            "timestamp": time.time()
        })
    
    async def _handle_end_session(self):
        """Handle explicit end-session command from client"""
        self.logger.info("Client explicitly ended session")
        
        # Notify voice service to stop session
        if hasattr(self.voice_service, 'stop_session'):
            await self.voice_service.stop_session()
        
        # Send confirmation to client before closing
        await self._send_message({
            "type": "control",
            "action": "session_ended",
            "reason": "client_request",
            "timestamp": time.time()
        })
        
        # Close the WebSocket connection
        if self.session.websocket.client_state == WebSocketState.CONNECTED:
            try:
                await self.session.websocket.close(code=1000, reason="Session ended by client")
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")

    def _get_frame_size(self) -> int:
        """Calculate the audio frame size in bytes"""
        # 16000 sample rate, 16-bit depth, 1 channel
        # Frame duration from settings
        return int(16000 * (settings.frame_duration_ms / 1000) * 2)

    async def cleanup(self):
        """Clean up session resources"""
        self.logger.info("Cleaning up session handler")
        # Cancel any outstanding processing tasks
        for task in self.processing_tasks:
            task.cancel()
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        self.logger.info("Session handler cleanup complete")