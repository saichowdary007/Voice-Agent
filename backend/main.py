import os
import asyncio
import json
import logging
import time
import base64
from typing import Dict, Any, Optional, Union
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import numpy as np

# Import our enhanced services
from services.vad_service import VADService
from services.stt_service import STTService
from services.tts_service import TTSService
from services.llm_service import LLMService
from services.audio_service import AudioService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ultra-Fast Voice Agent Backend",
    description="Sub-500ms voice assistant with Gemini Flash, sherpa-ncnn, and Piper TTS",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
vad_service: Optional[VADService] = None
stt_service: Optional[STTService] = None
tts_service: Optional[TTSService] = None
llm_service: Optional[LLMService] = None
audio_service: Optional[AudioService] = None

# Configuration
CONFIG = {
    "sample_rate": 16000,
    "frame_duration_ms": 120,  # 120ms frames as per spec
    "vad_threshold": float(os.getenv("VAD_THRESHOLD", "0.6")),
    "max_concurrent_sessions": int(os.getenv("MAX_CONCURRENT_SESSIONS", "3")),
    "enable_metrics": os.getenv("ENABLE_METRICS", "true").lower() == "true",
    "tts_speed": float(os.getenv("TTS_SPEED", "1.0")),
}

class SessionManager:
    """Manage WebSocket sessions with state tracking"""
    
    def __init__(self):
        self.sessions: Dict[WebSocket, Dict[str, Any]] = {}
        self.max_sessions = CONFIG["max_concurrent_sessions"]
    
    async def create_session(self, websocket: WebSocket) -> bool:
        """Create a new session"""
        if len(self.sessions) >= self.max_sessions:
            logger.warning(f"Max sessions ({self.max_sessions}) reached")
            return False
        
        session_id = f"session_{int(time.time() * 1000)}"
        
        self.sessions[websocket] = {
            "id": session_id,
            "created_at": time.time(),
            "audio_buffer": bytearray(),
            "is_recording": False,
            "is_processing": False,
            "is_speaking": False,
            "conversation_started": False,
            "total_frames": 0,
            "speech_frames": 0,
            "metrics": {
                "vad_time": 0,
                "stt_time": 0,
                "llm_time": 0,
                "tts_time": 0,
                "total_latency": 0,
                "utterance_count": 0
            },
            "keep_alive_task": None,
        }
        
        logger.info(f"Created session {session_id}")
        return True
    
    def get_session(self, websocket: WebSocket) -> Optional[Dict[str, Any]]:
        """Get session data"""
        return self.sessions.get(websocket)
    
    def remove_session(self, websocket: WebSocket):
        """Remove session"""
        session = self.sessions.pop(websocket, None)
        if session:
            logger.info(f"Removed session {session['id']}")
    
    def get_active_sessions(self) -> int:
        """Get number of active sessions"""
        return len(self.sessions)

session_manager = SessionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup"""
    global vad_service, stt_service, tts_service, llm_service, audio_service
    
    logger.info("🚀 Starting Ultra-Fast Voice Agent Backend...")
    
    try:
        # Initialize services in parallel for faster startup
        logger.info("Initializing services...")
        
        # Create service instances
        vad_service = VADService(threshold=CONFIG["vad_threshold"])
        stt_service = STTService()
        tts_service = TTSService()
        llm_service = LLMService()
        audio_service = AudioService(sample_rate=CONFIG["sample_rate"])
        
        # Initialize all services
        init_tasks = [
            vad_service.initialize(),
            stt_service.initialize(),
            tts_service.initialize(),
            llm_service.initialize(),
            audio_service.initialize()
        ]
        
        await asyncio.gather(*init_tasks)
        
        # Check service status
        services_status = {
            "VAD": vad_service.is_available,
            "STT": stt_service.is_available,
            "TTS": tts_service.is_available,
            "LLM": llm_service.is_available,
            "Audio": audio_service.is_available
        }
        
        available_services = sum(services_status.values())
        total_services = len(services_status)
        
        logger.info(f"✅ Services initialized: {available_services}/{total_services}")
        for service, status in services_status.items():
            logger.info(f"  {service}: {'✅' if status else '❌'}")
        
        if available_services < total_services:
            logger.warning("Some services failed to initialize - running in degraded mode")
        
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Ultra-Fast Voice Agent Backend",
        "version": "2.0.0",
        "status": "healthy",
        "latency_target": "< 500ms",
        "active_sessions": session_manager.get_active_sessions()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    services_status = {}
    
    if vad_service:
        services_status["vad"] = vad_service.get_status()
    if stt_service:
        services_status["stt"] = stt_service.get_status()
    if tts_service:
        services_status["tts"] = tts_service.get_status()
    if llm_service:
        services_status["llm"] = llm_service.get_status()
    if audio_service:
        services_status["audio"] = audio_service.get_status()
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_sessions": session_manager.get_active_sessions(),
        "config": CONFIG,
        "services": services_status
    }

async def websocket_receiver(websocket: WebSocket):
    """Receives both text and binary messages from WebSocket."""
    session = session_manager.get_session(websocket)
    if not session:
        logger.error(f"Session not found for websocket {websocket} during receive")
        return

    try:
        while True:
            message = await websocket.receive()
            message_type = message.get("type")
            
            if message_type == "websocket.receive":
                # Handle different message types
                if "text" in message:
                    # Text message (JSON control messages)
                    try:
                        data = json.loads(message["text"])
                        await handle_websocket_message(websocket, data)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode JSON message: {message['text']}")
                elif "bytes" in message:
                    # Binary message (audio data)
                    await handle_audio_frame_bytes(websocket, message["bytes"])
                else:
                    logger.warning(f"Received message without text or bytes: {message}")
            elif message_type == "websocket.disconnect":
                logger.info("WebSocket disconnect message received")
                break
            else:
                logger.warning(f"Unknown message type: {message_type}")
    except Exception as e:
        logger.error(f"Error in websocket_receiver: {e}", exc_info=True)
        raise

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for voice interaction"""
    
    await websocket.accept()
    if not await session_manager.create_session(websocket):
        await websocket.close(code=1013, reason="Server overloaded")
        return
    
    session = session_manager.get_session(websocket)
    session_id = session["id"]
    
    logger.info(f"WebSocket connected: {session_id}")
    
    session["keep_alive_task"] = asyncio.create_task(send_keep_alive_pings(websocket))

    try:
        await send_message(websocket, {
            "type": "status",
            "session_id": session_id,
            "ready": True,
            "config": {
                "sample_rate": CONFIG["sample_rate"],
                "frame_duration_ms": CONFIG["frame_duration_ms"]
            }
        })
        
        # Use the new receiver that handles mixed message types
        await websocket_receiver(websocket)
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except ConnectionResetError:
        logger.info(f"WebSocket connection reset: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error in {session_id}: {str(e)}", exc_info=True)
        try:
            await send_error(websocket, f"Internal server error: {str(e)}")
        except:
            pass  # Connection might already be closed
    finally:
        if session and session.get("keep_alive_task"):
            session["keep_alive_task"].cancel()
        session_manager.remove_session(websocket)

async def send_keep_alive_pings(websocket: WebSocket):
    """Send ping messages to keep the WebSocket connection alive."""
    try:
        while True:
            await asyncio.sleep(10) # Send ping every 10 seconds
            try:
                await websocket.send_json({"type": "ping", "timestamp": time.time()})
            except WebSocketDisconnect:
                logger.info(f"Client disconnected, stopping ping for {websocket.scope.get('client')}")
                break
            except Exception as e:
                logger.warning(f"Could not send ping, connection might be closed: {e}")
                break # Assume connection is problematic
    except asyncio.CancelledError:
        logger.info("Keep-alive ping task cancelled.")
    except Exception as e:
        logger.error(f"Error in keep-alive ping task: {e}")

async def handle_websocket_message(websocket: WebSocket, data: Dict[str, Any]):
    session = session_manager.get_session(websocket)
    if not session:
        logger.warning("Message received for non-existent session.")
        return

    try:
        message_type = data.get("type")
        logger.debug(f"JSON data received: {message_type} for session {session['id']}")
        
        if message_type == "eos": # End of speech from client
            logger.info(f"EOS received from {session['id']}")
            session["is_recording"] = False # Ensure recording stops
            if session.get("audio_buffer") and len(session["audio_buffer"]) > 0:
                session["is_processing"] = True # Set processing state
                await send_message(websocket, {"type": "status", "status": "processing", "session_id": session['id']})
                await process_speech_utterance(websocket)
            else:
                # No audio in buffer, but EOS received. Could be VAD misfire or already processed.
                logger.info(f"EOS received for session {session['id']} but audio buffer is empty.")
                session["is_processing"] = False # Ensure processing is false if nothing to process
                await send_message(websocket, {"type": "status", "status": "idle", "session_id": session['id']}) # Or appropriate status

        elif message_type == "start_recording":
            await handle_start_recording(websocket)
        elif message_type == "stop_recording":
            await handle_stop_recording(websocket)
        elif message_type == "mute":
            await handle_mute(websocket, data.get("muted", True))
        elif message_type == "end_session":
            await handle_end_session(websocket)
        elif message_type == "ping":
            await send_message(websocket, {"type": "pong", "timestamp": time.time()})
        else:
            logger.warning(f"Unknown message type: {message_type}")

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)

async def handle_audio_frame(websocket: WebSocket, data: Dict[str, Any]):
    """Process incoming audio frame with enhanced VAD and STT pipeline"""
    session = session_manager.get_session(websocket)
    
    if not session:
        return
    
    start_time = time.time()
    
    try:
        # Decode audio data
        audio_data_b64 = data.get("audio_data")
        if not audio_data_b64:
            return
        
        audio_bytes = base64.b64decode(audio_data_b64)
        await handle_audio_frame_bytes(websocket, audio_bytes)
        
    except Exception as e:
        logger.error(f"Audio frame (JSON) processing error: {e}")

async def handle_audio_frame_bytes(websocket: WebSocket, audio_bytes: bytes):
    """Process raw audio bytes (e.g., from ArrayBuffer)"""
    session = session_manager.get_session(websocket)
    if not session:
        logger.warning(f"Audio frame received for non-existent session from {websocket.client}")
        return

    # The client VAD controls MediaRecorder start/stop.
    # The server buffers all incoming audio for the active session until an 'eos' message is received.
    # The original check `if not session.get("is_recording", True):` is removed to ensure all data is buffered
    # before 'eos' triggers processing.

    try:
        # Log the first few bytes in hex for easier debugging of format
        session_id = session.get("id", "unknown_session")
        logger.info(f"Binary data received (first 4 bytes hex): {audio_bytes[:4].hex()}")
        logger.debug(f"Received binary audio frame for session {session_id}: {len(audio_bytes)} bytes")

        # Directly append raw audio_bytes (expected to be WebM) to buffer.
        # The audio_service.extract_pcm_smart is designed to handle this.
        session["audio_buffer"].extend(audio_bytes)
        session["total_frames"] += 1 # General counter for frames in the session.

        # VAD logic is now primarily on client. Server accumulates based on client signals.
        # Partial transcript logic could still run if desired, but relies on STT service capabilities
        # and STT service being able to process parts of the current buffer.
        # Example:
        # if session["total_frames"] % 15 == 0: # Roughly every 1.8 seconds if 120ms frames
        #     await process_partial_transcript(websocket)

    except Exception as e:
        session_id = session.get("id", "unknown_session") # Ensure session_id is available for error logging
        logger.error(f"Error processing raw audio frame for session {session_id}: {e}", exc_info=True)
        # Consider sending an error message to the client if this failure is critical
        # await send_error(websocket, "Internal server error processing audio frame.")

async def process_speech_utterance(websocket: WebSocket):
    """Process complete speech utterance with enhanced STT and response generation"""
    session = session_manager.get_session(websocket)
    
    if not session or session["is_processing"]:
        return
    
    session["is_processing"] = True
    utterance_start = time.time()
    
    try:
        logger.info(f"Processing utterance in {session['id']}")
        
        # Get audio buffer
        audio_buffer = bytes(session["audio_buffer"])
        session["audio_buffer"].clear()
        
        if len(audio_buffer) == 0:
            logger.info("No audio data to process")
            return
        
        # Convert to PCM for STT
        stt_start = time.time()
        transcript = ""
        confidence = 0.0
        
        if stt_service and stt_service.is_available:
            # Use smart extraction method with fallbacks
            if audio_service and audio_service.is_available:
                pcm_data = audio_service.extract_pcm_smart(audio_buffer)
                if pcm_data is not None:
                    # Convert to bytes for STT service
                    audio_int16 = (pcm_data * 32768.0).astype(np.int16)
                    result = await stt_service.transcribe_audio(audio_int16.tobytes())
                    transcript = result.get("transcript", "")
                    confidence = result.get("confidence", 0.95)
                else:
                    logger.error(f"STT Error: pcm_data is None after smart extraction in session {session['id']}. Cannot transcribe.")
                    transcript = "" # Or handle as an error state
                    confidence = 0.0
                    # Consider sending an error to the client if STT fails critically
                    await send_message(websocket, {"type": "error", "message": "STT failed: Could not process audio."})
        
        stt_latency = time.time() - stt_start
        
        session["metrics"]["stt_time"] += stt_latency * 1000
        
        # Check if we got a valid transcript
        if not transcript.strip():
            logger.info("No speech detected in utterance")
            await send_message(websocket, {
                "type": "transcript",
                "final": "",
                "confidence": 0.0,
                "error": "No speech detected"
            })
            return
        
        # Clean up transcript
        transcript = transcript.strip()
        logger.info(f"Transcript: '{transcript}' (confidence: {confidence:.2f})")
        
        # Send final transcript
        await send_message(websocket, {
            "type": "transcript",
            "final": transcript,
            "confidence": confidence
        })
        
        # Generate AI response
        llm_start = time.time()
        response_text = ""
        
        if llm_service and llm_service.is_available:
            try:
                # Stream LLM response
                await send_message(websocket, {"type": "ai_response_start"})
                
                async for token in llm_service.generate_streaming(transcript):
                    response_text += token
                    await send_message(websocket, {
                        "type": "ai_response",
                        "token": token,
                        "complete": False
                    })
                
                await send_message(websocket, {
                    "type": "ai_response",
                    "complete": True,
                    "full_text": response_text
                })
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                response_text = f"I heard you say: '{transcript}'. (LLM processing failed)"
                await send_message(websocket, {
                    "type": "ai_response",
                    "token": response_text,
                    "complete": True,
                    "full_text": response_text
                })
        else:
            # Fallback response
            response_text = f"I heard you say: '{transcript}'"
            await send_message(websocket, {
                "type": "ai_response",
                "token": response_text,
                "complete": True,
                "full_text": response_text
            })
        
        session["metrics"]["llm_time"] += (time.time() - llm_start) * 1000
        
        # Text-to-Speech
        if response_text.strip():
            await generate_and_stream_tts(websocket, response_text)
        
        # Update metrics
        session["metrics"]["utterance_count"] += 1
        total_latency = (time.time() - utterance_start) * 1000
        session["metrics"]["total_latency"] = total_latency
        
        # Reset VAD state for next utterance
        if vad_service:
            vad_service.reset_state()
        
        logger.info(f"Utterance processed in {total_latency:.1f}ms")
        
        # Send metrics if enabled
        if CONFIG["enable_metrics"]:
            await send_message(websocket, {
                "type": "session_metrics", 
                "metrics": {
                    "utterance_latency": total_latency,
                    "stt_time": session["metrics"]["stt_time"],
                    "llm_time": session["metrics"]["llm_time"],
                    "tts_time": session["metrics"]["tts_time"],
                    "vad_time": session["metrics"]["vad_time"]
                }
            })
        
    except Exception as e:
        logger.error(f"Speech processing error: {e}")
        await send_error(websocket, f"Speech processing failed: {str(e)}")
    finally:
        session["is_processing"] = False

async def process_partial_transcript(websocket: WebSocket):
    """Process partial transcript for real-time feedback"""
    session = session_manager.get_session(websocket)
    
    if not session or not stt_service or not stt_service.is_available:
        return
    
    try:
        # Get recent audio
        audio_buffer = bytes(session["audio_buffer"][-8000:])  # Last ~0.5 seconds
        
        if len(audio_buffer) > 0:
            # Quick transcription
            if audio_service and audio_service.is_available:
                pcm_data = audio_service.extract_pcm_from_webm(audio_buffer)
            else:
                pcm_data = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            
            if pcm_data is not None and len(pcm_data) > 0:
                audio_int16 = (pcm_data * 32768.0).astype(np.int16)
                result = await stt_service.transcribe_streaming(audio_int16.tobytes())
                
                if result.get("partial"):
                    await send_message(websocket, {
                        "type": "transcript",
                        "partial": result["partial"]
                    })
    
    except Exception as e:
        logger.debug(f"Partial transcript error: {e}")

async def generate_and_stream_tts(websocket: WebSocket, text: str):
    """Generate and stream TTS audio"""
    session = session_manager.get_session(websocket)
    
    if not session or not tts_service or not tts_service.is_available:
        return
    
    tts_start = time.time()
    session["is_speaking"] = True
    
    try:
        await send_message(websocket, {"type": "tts_start"})
        
        # Generate TTS with streaming
        async for audio_chunk in tts_service.synthesize_streaming(text, speed=CONFIG["tts_speed"]):
            # Check for barge-in
            if not session["is_speaking"]:
                break
            
            # Convert to base64 and send
            audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
            await send_message(websocket, {
                "type": "audio_chunk",
                "audio_data": audio_b64,
                "format": "wav"
            })
        
        await send_message(websocket, {"type": "tts_complete"})
        
        session["metrics"]["tts_time"] += (time.time() - tts_start) * 1000
        
    except Exception as e:
        logger.error(f"TTS streaming error: {e}")
    finally:
        session["is_speaking"] = False

async def handle_start_recording(websocket: WebSocket):
    """Handle start recording command"""
    session = session_manager.get_session(websocket)
    if session:
        session["is_recording"] = True
        await send_message(websocket, {"type": "recording_started"})

async def handle_stop_recording(websocket: WebSocket):
    """Handle stop recording command"""
    session = session_manager.get_session(websocket)
    if session:
        session["is_recording"] = False
        await send_message(websocket, {"type": "recording_stopped"})
        
        # Process any remaining audio
        if len(session["audio_buffer"]) > 0:
            await process_speech_utterance(websocket)

async def handle_mute(websocket: WebSocket, muted: bool):
    """Handle mute/unmute"""
    session = session_manager.get_session(websocket)
    if session:
        if muted:
            session["is_recording"] = False
            session["audio_buffer"].clear()
            # Stop TTS playback
            session["is_speaking"] = False
            await send_message(websocket, {"type": "stop_audio"})
        
        await send_message(websocket, {"type": "mute_status", "muted": muted})

async def handle_end_session(websocket: WebSocket):
    """Handle end session command"""
    session = session_manager.get_session(websocket)
    if session:
        # Clear conversation history
        if llm_service:
            llm_service.clear_history()
        
        # Send session metrics if enabled
        if CONFIG["enable_metrics"]:
            await send_message(websocket, {
                "type": "session_metrics",
                "metrics": session["metrics"]
            })
        
        await send_message(websocket, {"type": "session_ended"})

async def send_message(websocket: WebSocket, message: Dict[str, Any]):
    """Send JSON message to WebSocket"""
    try:
        await websocket.send_text(json.dumps(message))
    except Exception as e:
        logger.error(f"Failed to send message: {e}")

async def send_error(websocket: WebSocket, error_message: str):
    """Send error message to WebSocket"""
    await send_message(websocket, {
        "type": "error",
        "message": error_message,
        "timestamp": time.time()
    })

# Graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Voice Agent Backend...")
    
    # Close all active sessions
    for websocket in list(session_manager.sessions.keys()):
        try:
            await send_message(websocket, {"type": "server_shutdown"})
            await websocket.close()
        except:
            pass
    
    logger.info("✅ Shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
        access_log=True
    ) 