"""
This module contains handler functions for the WebSocket endpoint in server.py.
"""
import asyncio
import base64
import json
import logging
import time
from datetime import datetime
from typing import AsyncGenerator, Optional, Dict

from fastapi import WebSocket

from src.conversation import ConversationManager
from src.llm import LLM
from src.voice_agent import DeepgramVoiceAgent
from src.config import USE_DEEPGRAM_AGENT

logger = logging.getLogger(__name__)


class WebSocketSessionManager:
    """Manages WebSocket sessions and their associated VAD instances."""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}
        self.session_lock = asyncio.Lock()
    
    async def create_session(self, websocket: WebSocket) -> str:
        """Create a new session for a WebSocket connection."""
        session_id = f"session_{id(websocket)}_{int(time.time())}"
        
        async with self.session_lock:
            self.active_sessions[session_id] = {
                'websocket': websocket,
                'agent': None,
                'created_at': time.time(),
                'last_activity': time.time(),
                'speech_state': {
                    'is_speaking': False,
                    'speech_start_time': None,
                    'silence_start_time': None,
                    'total_audio_duration': 0.0,
                    'chunk_count': 0
                }
            }
            
            logger.info(f"✅ Created WebSocket session: {session_id}")
            return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data by session ID."""
        async with self.session_lock:
            session = self.active_sessions.get(session_id)
            if session:
                session['last_activity'] = time.time()
            return session
    
    async def get_session_by_websocket(self, websocket: WebSocket) -> Optional[Dict]:
        """Get session data by WebSocket instance."""
        async with self.session_lock:
            for session_id, session_data in self.active_sessions.items():
                if session_data['websocket'] == websocket:
                    session_data['last_activity'] = time.time()
                    return session_data
            return None
    
    async def cleanup_session(self, session_id: str) -> None:
        """Clean up a session and its resources."""
        async with self.session_lock:
            session = self.active_sessions.pop(session_id, None)
            if session:
                # Stop agent
                if session.get('agent') is not None:
                    try:
                        await session['agent'].stop()
                    except Exception:
                        pass
                
                logger.info(f"🧹 Cleaned up WebSocket session: {session_id}")
    
    async def cleanup_session_by_websocket(self, websocket: WebSocket) -> None:
        """Clean up session by WebSocket instance."""
        session_id = None
        async with self.session_lock:
            for sid, session_data in self.active_sessions.items():
                if session_data['websocket'] == websocket:
                    session_id = sid
                    break
        
        if session_id:
            await self.cleanup_session(session_id)
    
    async def cleanup_inactive_sessions(self, max_idle_time: float = 3600) -> None:
        """Clean up sessions that have been inactive for too long."""
        current_time = time.time()
        inactive_sessions = []
        
        async with self.session_lock:
            for session_id, session_data in self.active_sessions.items():
                if current_time - session_data['last_activity'] > max_idle_time:
                    inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            await self.cleanup_session(session_id)
            logger.info(f"🧹 Cleaned up inactive session: {session_id}")


# Global session manager instance
session_manager = WebSocketSessionManager()


class WebSocketErrorHandler:
    """Handles WebSocket error classification and recovery strategies."""
    
    ERROR_CODES = {
        1000: "Normal closure",
        1001: "Going away",
        1002: "Protocol error", 
        1003: "Unsupported data",
        1006: "Connection lost",
        1007: "Invalid frame payload data",
        1008: "Authentication failed",
        1009: "Message too big",
        1010: "Mandatory extension",
        1011: "Unexpected server condition",
        1015: "TLS handshake failure"
    }
    
    @classmethod
    async def handle_connection_error(cls, error_code: int, websocket: WebSocket, reason: str = "") -> None:
        """Handle connection errors with appropriate recovery strategies."""
        error_description = cls.ERROR_CODES.get(error_code, f"Unknown error code: {error_code}")
        logger.error(f"WebSocket error {error_code}: {error_description} - {reason}")
        
        # Clean up session resources
        await session_manager.cleanup_session_by_websocket(websocket)
        
        # Log specific error handling
        if error_code == 1011:
            logger.error("1011 error detected - likely ping timeout or server overload")
        elif error_code == 1008:
            logger.error("Authentication failed - token may be expired")
        elif error_code == 1006:
            logger.error("Connection lost - network issue or server restart")
    
    @classmethod
    def should_attempt_reconnect(cls, error_code: int) -> bool:
        """Determine if reconnection should be attempted based on error code."""
        # Don't reconnect for authentication failures or protocol errors
        no_reconnect_codes = {1008, 1002, 1003, 1007}
        return error_code not in no_reconnect_codes
    
    @classmethod
    async def cleanup_failed_connection(cls, websocket: WebSocket) -> None:
        """Clean up resources for a failed connection."""
        try:
            await session_manager.cleanup_session_by_websocket(websocket)
            logger.info("✅ Cleaned up failed connection resources")
        except Exception as e:
            logger.error(f"❌ Error cleaning up failed connection: {e}")


def get_websocket_protocol(websocket: WebSocket) -> Optional[str]:
    """Get the negotiated WebSocket sub-protocol."""
    # In FastAPI, the negotiated protocol is available via the websocket object
    return getattr(websocket, 'subprotocol', None)


def is_binary_protocol(websocket: WebSocket) -> bool:
    """Check if the WebSocket is using binary protocol."""
    protocol = get_websocket_protocol(websocket)
    return protocol == "binary"


def is_stream_audio_protocol(websocket: WebSocket) -> bool:
    """Check if the WebSocket is using stream-audio protocol."""
    protocol = get_websocket_protocol(websocket)
    return protocol == "stream-audio"


class LatencyTracker:
    """Tracks latency events for a single voice interaction pipeline."""
    
    def __init__(self):
        self.events: dict[str, float] = {}
        self.start_time = time.perf_counter()
    
    def mark(self, event_name: str):
        """Mark a timestamp for an event."""
        self.events[event_name] = time.perf_counter() - self.start_time
    
    def get_duration(self, start_event: str, end_event: str) -> Optional[float]:
        """Get duration between two events in milliseconds."""
        if start_event in self.events and end_event in self.events:
            return (self.events[end_event] - self.events[start_event]) * 1000
        return None
    
    def get_total_latency(self) -> float:
        """Get total latency from start to last event in milliseconds."""
        if not self.events:
            return 0.0
        return max(self.events.values()) * 1000
    
    def log_summary(self):
        """Log a summary of all timing events."""
        if not self.events:
            return
        
        logger.info("=== Latency Summary ===")
        sorted_events = sorted(self.events.items(), key=lambda x: x[1])
        
        prev_time = 0.0
        for event, timestamp in sorted_events:
            duration_from_prev = (timestamp - prev_time) * 1000
            total_elapsed = timestamp * 1000
            logger.info(f"  {event}: +{duration_from_prev:.1f}ms (total: {total_elapsed:.1f}ms)")
            prev_time = timestamp
        
        total_latency = self.get_total_latency()
        logger.info(f"  TOTAL LATENCY: {total_latency:.1f}ms")
        
        # Log key segment durations
        key_durations = [
            ("audio_received", "stt_complete"),
            ("stt_complete", "llm_complete"), 
            ("llm_complete", "tts_complete"),
            ("tts_complete", "audio_sent")
        ]
        
        for start, end in key_durations:
            duration = self.get_duration(start, end)
            if duration is not None:
                logger.info(f"  {start} -> {end}: {duration:.1f}ms")


def get_latency_tracker(websocket: WebSocket) -> LatencyTracker:
    """Get or create latency tracker for a websocket connection."""
    if not hasattr(websocket, "_latency_tracker"):
        websocket._latency_tracker = LatencyTracker()
    return websocket._latency_tracker


async def _process_and_respond(
    websocket: WebSocket,
    user_text: str,
    language: str,
    conversation_mgr: Optional[ConversationManager],
    llm_interface: LLM,
    tts_engine,
):
    """Helper to process user text and send AI response."""
    history = []
    profile_facts = []

    if conversation_mgr:
        history, profile_facts = await asyncio.gather(
            conversation_mgr.get_context_for_llm(user_text),
            conversation_mgr.get_user_profile(),
        )

    if llm_interface and tts_engine:
        latency_tracker = get_latency_tracker(websocket)
        ai_response = await stream_text_to_audio_pipeline(
            user_text=user_text,
            history=history,
            profile_facts=profile_facts,
            llm_interface=llm_interface,
            tts_engine=tts_engine,
            websocket=websocket,
            latency_tracker=latency_tracker,
        )

        response_payload = {
            "type": "text_response",
            "text": ai_response,
            "language": language,
            "timestamp": datetime.utcnow().isoformat(),
        }
        await websocket.send_json(response_payload)

        if conversation_mgr:
            await conversation_mgr.add_message("user", user_text)
            await conversation_mgr.add_message("model", ai_response)

            new_facts = await llm_interface.extract_facts(
                f"User: {user_text}\nAI: {ai_response}"
            )
            if new_facts:
                await conversation_mgr.update_user_profile(new_facts)

            await conversation_mgr.handle_user_turn(
                user_text, ai_response, llm_interface
            )
    else:
        ai_response = "Sorry, the AI service is currently unavailable."
        response_payload = {
            "type": "text_response",
            "text": ai_response,
            "language": language,
            "timestamp": datetime.utcnow().isoformat(),
        }
        await websocket.send_json(response_payload)


async def handle_text_message(
    websocket: WebSocket,
    message: dict,
    conversation_mgr: Optional[ConversationManager],
    llm_interface: LLM,
    tts_engine,
):
    """Handles a text message from the client."""
    user_text = message.get("text", "")
    language = message.get("language", "en")

    if not user_text.strip():
        await websocket.send_json(
            {"type": "error", "message": "Empty message received"}
        )
        return

    try:
        await _process_and_respond(
            websocket,
            user_text,
            language,
            conversation_mgr,
            llm_interface,
            tts_engine,
        )
    except Exception as e:
        logger.error(f"Error processing text message: {e}")
        await websocket.send_json(
            {"type": "error", "message": f"Processing error: {str(e)}"}
        )


async def handle_audio_chunk(
    websocket: WebSocket,
    message: dict,
    conversation_mgr: Optional[ConversationManager],
    llm_interface: LLM,
    tts_engine,
):
    """Handles an audio chunk from the client with enhanced speech boundary detection."""
    # Agent path does not require legacy STT
    logger.debug(f"🎵 Received audio chunk: {len(message.get('data', ''))} bytes (base64), is_final: {message.get('is_final', False)}")

    try:
        encoded_audio = message.get("data")
        is_final = message.get("is_final", False)

        if not encoded_audio:
            raise ValueError("Missing audio data")

        audio_bytes = base64.b64decode(encoded_audio)
        audio_format = message.get("format")  # e.g., ogg_opus or webm_opus

        # Initialize minimal state for agent path only
        if not hasattr(websocket, "_audio_buffer"):
            websocket._audio_buffer = bytearray()
            websocket._is_processing = False
            websocket._last_final_time = 0
            websocket._speech_detected = False
            websocket._silence_start_time = None
            websocket._total_audio_duration = 0
            websocket._speech_started_time = None
            websocket._frame_count = 0

        # Prevent duplicate processing
        if is_final and websocket._is_processing:
            logger.debug("Skipping duplicate final audio processing")
            return

        # Assume raw PCM16 (frontend is configured to send PCM); if compressed is sent, ignore legacy decode path
        decoded_pcm = audio_bytes

        # Ensure agent is started (once per connection) when enabled
        if USE_DEEPGRAM_AGENT and not hasattr(websocket, "_dg_agent_started"):
            try:
                from src.config import DEEPGRAM_API_KEY
                # Schedule WebSocket sends safely from SDK background threads
                loop = asyncio.get_running_loop()

                def send_json_threadsafe(payload: dict) -> None:
                    try:
                        asyncio.run_coroutine_threadsafe(websocket.send_json(payload), loop)
                    except Exception as send_err:
                        logger.error(f"Thread-safe send_json failed: {send_err}")

                # Create and start agent; store on websocket for simplicity
                websocket._dg_agent = DeepgramVoiceAgent(
                    api_key=DEEPGRAM_API_KEY,
                    sample_rate_input=16000,
                    sample_rate_output=24000,
                    on_audio_ready=lambda audio_bytes, mime: send_json_threadsafe({
                        "type": "tts_audio",
                        "data": base64.b64encode(audio_bytes).decode("ascii"),
                        "mime": mime,
                    }),
                    on_text=lambda role, content: send_json_threadsafe({
                        "type": "agent_text",
                        "role": role,
                        "content": content,
                    }),
                    on_error=lambda err: send_json_threadsafe({
                        "type": "error",
                        "message": f"Agent error: {err}",
                    }),
                )
                started = await websocket._dg_agent.start()
                websocket._dg_agent_started = started
                if started:
                    logger.info("✅ Deepgram Voice Agent started for this WebSocket")
                else:
                    logger.error("❌ Failed to start Deepgram Voice Agent; continuing without agent")
            except Exception as e:
                logger.error(f"Agent start exception: {e}")
                websocket._dg_agent_started = False

        # Agent-first path: forward PCM directly to Deepgram Agent and return
        if USE_DEEPGRAM_AGENT and getattr(websocket, "_dg_agent_started", False):
            try:
                # Ultra-low latency: forward even smaller chunks (320 bytes = 10ms at 16kHz)
                if decoded_pcm and len(decoded_pcm) >= 320:
                    websocket._dg_agent.send_audio(decoded_pcm)
                    if is_final:
                        # Push minimal trailing silence for faster utterance detection
                        try:
                            trailing_silence = b"\x00" * int(0.1 * 16000 * 2)  # 100ms silence
                            websocket._dg_agent.send_audio(trailing_silence)
                        except Exception:
                            pass
                # Reduce feedback frequency for lower overhead
                if is_final or len(decoded_pcm) % 3200 == 0:  # Every 100ms or final
                    await websocket.send_json({
                        "type": "audio_processed",
                        "status": "receiving" if not is_final else "final",
                        "buffer_size": len(decoded_pcm),
                        "is_final": is_final,
                    })
            except Exception as e:
                logger.debug(f"Agent direct-forward error: {e}")
            # Short-circuit legacy VAD/STT/LLM pipeline entirely when agent is enabled
            return

        # Legacy pipeline removed when agent is enabled by design. If you reach here without agent, respond gracefully.
        await websocket.send_json({
            "type": "error",
            "message": "Deepgram Agent not active; enable USE_DEEPGRAM_AGENT or update client to agent mode"
        })
        return
        
        # Get sample rate from message or default to 16kHz
        sample_rate = message.get("sample_rate", 16000)
        
        # Track total audio duration based on actual sample rate using processed PCM bytes
        # Use processed_audio length to avoid miscalculations if input was compressed
        chunk_duration_ms = (len(processed_audio) / 2) / sample_rate * 1000
        websocket._total_audio_duration += chunk_duration_ms
        
        # Store sample rate for duration calculations
        if not hasattr(websocket, "_sample_rate"):
            websocket._sample_rate = sample_rate
            logger.info(f"🎵 Audio sample rate detected: {sample_rate}Hz")
        
        # Update speech detection state from preprocessor
        current_speech_detected = metadata.get('is_speech', False)
        confidence = metadata.get('confidence', 0.0)
        
        logger.debug(f"Audio chunk: {len(audio_bytes)} bytes ({audio_format or 'pcm16'}), processed: {len(processed_audio)} bytes, "
                    f"buffer: {len(websocket._audio_buffer)} bytes, duration: {websocket._total_audio_duration:.1f}ms, "
                    f"speech: {current_speech_detected}, confidence: {confidence:.2f}, is_final: {is_final}")

        latency_tracker = get_latency_tracker(websocket)

        # Stream decoded PCM to agent for real-time responses (including final to flush remainder)
        if processed_audio and len(processed_audio) >= 640:
            if USE_DEEPGRAM_AGENT and getattr(websocket, "_dg_agent_started", False):
                try:
                    websocket._dg_agent.send_audio(processed_audio)
                    # On final chunks, also send a short trailing silence to help agent segment the utterance
                    if is_final:
                        try:
                            trailing_silence = b"\x00" * int(0.2 * 16000 * 2)  # 200ms of PCM16 silence @16kHz
                            websocket._dg_agent.send_audio(trailing_silence)
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug(f"Agent send error: {e}")
        
        # Enhanced speech boundary detection using Nova-3 preprocessor
        if current_speech_detected and not websocket._speech_detected:
            websocket._speech_detected = True
            websocket._speech_started_time = time.time()
            websocket._silence_start_time = None
            logger.info(f"🎤 Speech started (Nova-3 VAD, confidence: {confidence:.2f})")
            
            # Send speech_started event to client
            await websocket.send_json({
                "type": "speech_started",
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        elif not current_speech_detected and websocket._speech_detected:
            if websocket._silence_start_time is None:
                websocket._silence_start_time = time.time()
            
            # Check if we should end speech based on silence duration (optimized for low latency)
            silence_duration = time.time() - websocket._silence_start_time
            if silence_duration > 0.4:  # 400ms silence timeout for faster response
                logger.info(f"🔇 Speech ended after {silence_duration:.2f}s silence")
                is_final = True  # Force final processing
                
                # Send speech_final event to client
                await websocket.send_json({
                    "type": "speech_final",
                    "silence_duration_ms": silence_duration * 1000,
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        # Process transcription with improved logic
        user_text = None
        should_process = False
        too_short_audio = False
        
        if is_final:
            # Set processing flag to prevent duplicates
            websocket._is_processing = True
            current_time = time.time()
            
            # Reduced debounce time for better responsiveness
            if current_time - websocket._last_final_time < 0.3:
                logger.debug("Skipping duplicate final processing due to debounce")
                websocket._is_processing = False
                return
            
            websocket._last_final_time = current_time
            
            # More intelligent minimum audio length based on speech detection and actual sample rate
            effective_sample_rate = getattr(websocket, "_sample_rate", 16000)
            
            if websocket._speech_detected:
                # If speech was detected, use shorter minimum (200ms)
                min_speech_bytes = int(effective_sample_rate * 0.2 * 2)  # 200ms at actual sample rate
            else:
                # If no speech detected, require slightly longer audio to avoid false positives (350ms)
                min_speech_bytes = int(effective_sample_rate * 0.35 * 2)  # 350ms at actual sample rate
            
            logger.debug(f"Minimum audio check: buffer={len(websocket._audio_buffer)} bytes, "
                        f"required={min_speech_bytes} bytes, sample_rate={effective_sample_rate}Hz")
            
            if len(websocket._audio_buffer) < min_speech_bytes:
                logger.debug(f"Audio buffer too short: {len(websocket._audio_buffer)} < {min_speech_bytes} bytes")
                websocket._is_processing = False
                # Defer any state resets until after sending feedback to avoid duplicate speech_started events
                # Don't return; allow final handling to send user feedback
                too_short_audio = True
            
            if not too_short_audio:
                # Additional validation using VAD
                if vad_instance and hasattr(vad_instance, 'should_process_audio'):
                    should_process = vad_instance.should_process_audio(
                        bytes(websocket._audio_buffer), 
                        force_final=True
                    )
                else:
                    should_process = True
                
                if should_process:
                    logger.info(f"Finalizing after {websocket._total_audio_duration:.1f}ms")
                    
                    # Start latency tracking
                    latency_tracker.mark("audio_received")
                    
                    # Finalize agent if enabled (no more audio to send right now)
                    if USE_DEEPGRAM_AGENT and getattr(websocket, "_dg_agent_started", False):
                        try:
                            # No special finalize call; agent generates responses asynchronously
                            pass
                        except Exception:
                            pass
                    user_text = None
                else:
                    logger.info("VAD determined audio should not be processed (likely silence)")
            else:
                logger.info("Skipping STT: audio below minimum length threshold")
            
            # Defer buffer reset until after we send result/feedback so metrics are intact
            # Only clear the processing flag here to allow the final-result block to run
            websocket._is_processing = False
        # Handle final processing results
        if is_final and not websocket._is_processing:
            # Snapshot metrics BEFORE any reset so feedback/debug use real values
            pre_reset_buffer_bytes = len(websocket._audio_buffer)
            pre_reset_duration_ms = websocket._total_audio_duration
            pre_reset_speech_detected = websocket._speech_detected
            pre_reset_frame_count = websocket._frame_count
            pre_reset_sample_rate = getattr(websocket, "_sample_rate", 16000)
            
            if user_text and user_text.strip():
                language = message.get("language", "en")
                logger.info(f"✅ Processing user speech: '{user_text}'")
                await _process_and_respond(
                    websocket,
                    user_text,
                    language,
                    conversation_mgr,
                    llm_interface,
                    tts_engine,
                )
            else:
                # Send helpful feedback when no speech is detected
                logger.info("No transcript detected, providing user feedback")
                
                # Determine the most likely cause based on audio characteristics
                feedback_message = "No clear speech detected."
                effective_sample_rate = pre_reset_sample_rate
                
                if pre_reset_duration_ms < 500:
                    feedback_message = "Audio too short. Try speaking for at least half a second."
                elif not pre_reset_speech_detected:
                    feedback_message = "No speech activity detected. Try speaking louder or closer to the microphone."
                else:
                    feedback_message = "Speech detected but not clear enough to transcribe. Try speaking more clearly."
                
                # Add debug information to help troubleshoot
                debug_info = {
                    "buffer_bytes": pre_reset_buffer_bytes,
                    "duration_ms": pre_reset_duration_ms,
                    "sample_rate": effective_sample_rate,
                    "speech_detected": pre_reset_speech_detected,
                    "frame_count": pre_reset_frame_count
                }
                
                logger.info(f"Audio processing debug: {debug_info}")
                
                await websocket.send_json({
                    "type": "stt_result",
                    "transcript": "",
                    "is_final": True,
                    "message": feedback_message,
                    "debug_info": debug_info
                })
            
            # Now safe to reset state for the next utterance (after sending result/feedback)
            websocket._audio_buffer = bytearray()
            websocket._is_processing = False
            websocket._speech_detected = False
            websocket._silence_start_time = None
            websocket._total_audio_duration = 0
            # Reset additional counters for next utterance
            websocket._frame_count = 0
            websocket._speech_started_time = None
        elif not is_final:
            # Minimal periodic feedback for ongoing audio (agent path)
            if len(websocket._audio_buffer) % 8192 == 0:
                await websocket.send_json({
                    "type": "audio_processed",
                    "status": "receiving",
                    "buffer_size": len(websocket._audio_buffer)
                })
                
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        await websocket.send_json(
            {"type": "error", "message": f"Audio processing failed: {str(e)}"}
        )


async def handle_ping(websocket: WebSocket, message: dict):
    """Handles a ping message from the client."""
    await websocket.send_json(
        {"type": "pong", "timestamp": message.get("timestamp")}
    )


async def handle_vad_status(websocket: WebSocket, message: dict):
    """Handles a VAD status message from the client."""
    is_active = message.get("isActive", False)
    await websocket.send_json(
        {
            "type": "vad_status",
            "isActive": is_active,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


async def handle_start_listening(websocket: WebSocket, message: dict):
    """Handles a start listening message from the client."""
    await websocket.send_json(
        {
            "type": "listening_status",
            "status": "started",
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


async def handle_stop_listening(websocket: WebSocket, message: dict):
    """Handles a stop listening message from the client."""
    await websocket.send_json(
        {
            "type": "listening_status",
            "status": "stopped",
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


async def handle_heartbeat(websocket: WebSocket, message: dict):
    """Handles a heartbeat message from the client."""
    # Check connection state before sending
    from starlette.websockets import WebSocketState
    if websocket.client_state == WebSocketState.CONNECTED:
        await websocket.send_json(
            {"type": "heartbeat_ack", "timestamp": datetime.utcnow().isoformat()}
        )


async def handle_connection_message(websocket: WebSocket, message: dict):
    """Handles connection acknowledgment messages from the client."""
    # This is just an acknowledgment, no response needed
    logger.debug(f"Connection message received: {message}")
    # Optionally send back a confirmation
    await websocket.send_json(
        {"type": "connection_ack", "message": "Connection acknowledged"}
    )


async def handle_unknown_message(websocket: WebSocket, message: dict):
    """Handles unknown message types."""
    message_type = message.get('type', 'undefined')
    # Don't send error for connection messages - they're expected
    if message_type == 'connection':
        await handle_connection_message(websocket, message)
        return
    
    await websocket.send_json(
        {"type": "error", "message": f"Unknown message type: {message_type}"}
    )


async def stream_text_to_audio_pipeline(
    user_text: str,
    history: list,
    profile_facts: list,
    llm_interface: LLM,
    tts_engine,
    websocket: WebSocket,
    latency_tracker: LatencyTracker
) -> str:
    """
    Stream the entire LLM->TTS pipeline for ultra-low latency.
    Collects LLM tokens and synthesizes audio when complete.
    """
    try:
        latency_tracker.mark("llm_start")
        
        # Stream LLM response and collect tokens
        llm_stream = llm_interface.generate_stream(
            user_text=user_text,
            conversation_history=history,
            user_profile=profile_facts
        )
        
        # Collect the full response from the stream (fix async generator bug)
        tokens = []
        async for token in llm_stream:
            if token:
                tokens.append(token)
                logger.debug(f"LLM token received: '{token}'")
        
        full_response = ''.join(tokens).strip()
        logger.info(f"LLM full response collected: '{full_response}' (length: {len(full_response)})")
        
        latency_tracker.mark("llm_complete")
        
        if not full_response.strip():
            logger.warning("LLM response was empty, using fallback")
            full_response = "I'm not sure how to respond to that."
        
        latency_tracker.mark("tts_start")
        
        # TTS is handled by Deepgram Agent in agent mode; send text to client as a fallback
        await websocket.send_json({
            "type": "text_response",
            "text": full_response
        })
        
        latency_tracker.mark("pipeline_complete")
        latency_tracker.log_summary()
        
        return full_response
        
    except Exception as e:
        logger.error(f"Streaming pipeline error: {e}")
        # Fallback to non-streaming
        if llm_interface:
            response = await llm_interface.generate_response(user_text, history, profile_facts)
            await websocket.send_json({
                "type": "text_response",
                "text": response
            })
            return response
        return "Sorry, I'm having trouble processing that right now."


async def handle_streaming_stt_chunk(*args, **kwargs):
    return None
