"""
This module contains handler functions for the WebSocket endpoint in server.py.
"""
import asyncio
import base64
import json
import logging
import time
from datetime import datetime
from typing import AsyncGenerator, Optional

from fastapi import WebSocket

from src.conversation import ConversationManager
from src.llm import LLM
from src.stt import STT
from src.tts import TTS
from src.vad import VAD

logger = logging.getLogger(__name__)


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
    tts_engine: TTS,
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
    tts_engine: TTS,
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
    stt_instance: STT,
    llm_interface: LLM,
    tts_engine: TTS,
    vad_instance: VAD,
):
    """Handles an audio chunk from the client."""
    if not stt_instance:
        await websocket.send_json(
            {"type": "error", "message": "STT service not available"}
        )
        return

    try:
        encoded_audio = message.get("data")
        is_final = message.get("is_final", False)

        if not encoded_audio:
            raise ValueError("Missing audio data")

        audio_bytes = base64.b64decode(encoded_audio)

        # Skip VAD check for small chunks to prevent excessive filtering
        if vad_instance and len(audio_bytes) >= 640 and not vad_instance.is_speech(audio_bytes) and not is_final:
            await websocket.send_json(
                {
                    "type": "vad_status",
                    "isActive": False,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return

        latency_tracker = get_latency_tracker(websocket)
        user_text = await handle_streaming_stt_chunk(
            websocket=websocket,
            audio_chunk=audio_bytes,
            is_final=is_final,
            stt_instance=stt_instance,
            latency_tracker=latency_tracker,
        )

        if user_text and is_final:
            language = message.get("language", "en")

            await _process_and_respond(
                websocket,
                user_text,
                language,
                conversation_mgr,
                llm_interface,
                tts_engine,
            )
        elif not is_final:
            await websocket.send_json(
                {
                    "type": "audio_processed",
                    "status": "receiving",
                    "message": "Audio chunk received",
                }
            )
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
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
    await websocket.send_json(
        {"type": "heartbeat_ack", "timestamp": datetime.utcnow().isoformat()}
    )


async def handle_unknown_message(websocket: WebSocket, message: dict):
    """Handles unknown message types."""
    await websocket.send_json(
        {"type": "error", "message": f"Unknown message type: {message.get('type', 'undefined')}"}
    )


async def stream_text_to_audio_pipeline(
    user_text: str,
    history: list,
    profile_facts: list,
    llm_interface: LLM,
    tts_engine: TTS,
    websocket: WebSocket,
    latency_tracker: LatencyTracker
) -> str:
    """
    Stream the entire LLM->TTS pipeline for ultra-low latency.
    Starts TTS as soon as LLM begins producing tokens.
    """
    try:
        latency_tracker.mark("llm_start")
        
        # Stream LLM response and feed directly to TTS
        llm_stream = llm_interface.generate_stream(
            user_text=user_text,
            conversation_history=history,
            user_profile=profile_facts
        )
        
        latency_tracker.mark("tts_start")
        
        # Stream TTS synthesis from LLM stream
        audio_stream = tts_engine.stream_synthesize(llm_stream)
        
        first_audio_sent = False
        full_response = ""
        
        async for audio_chunk in audio_stream:
            if audio_chunk:
                if not first_audio_sent:
                    latency_tracker.mark("first_audio_sent")
                    first_audio_sent = True
                
                # Send audio chunk to client immediately
                await websocket.send_json({
                    "type": "audio_stream",
                    "data": base64.b64encode(audio_chunk).decode("ascii"),
                    "mime": "audio/mp3"
                })
        
        latency_tracker.mark("tts_complete")
        
        # Collect full response for conversation history (re-generate for now)
        full_response = await llm_interface.generate_response(user_text, history, profile_facts)
        
        latency_tracker.mark("pipeline_complete")
        latency_tracker.log_summary()
        
        return full_response
        
    except Exception as e:
        logger.error(f"Streaming pipeline error: {e}")
        # Fallback to non-streaming
        if llm_interface:
            response = await llm_interface.generate_response(user_text, history, profile_facts)
            audio_bytes = await tts_engine.synthesize(response)
            if audio_bytes:
                await websocket.send_json({
                    "type": "tts_audio",
                    "data": base64.b64encode(audio_bytes).decode("ascii"),
                    "mime": "audio/mp3"
                })
            return response
        return "Sorry, I'm having trouble processing that right now."


async def handle_streaming_stt_chunk(
    websocket: WebSocket,
    audio_chunk: bytes,
    is_final: bool,
    stt_instance: STT,
    latency_tracker: LatencyTracker
) -> Optional[str]:
    """
    Handle a single STT audio chunk with streaming processing.
    Returns transcript if is_final=True, None otherwise.
    """
    try:
        if not is_final:
            latency_tracker.mark("audio_received")
        
        # Validate audio chunk before processing
        if not audio_chunk or len(audio_chunk) < 320:  # Skip very small chunks
            logger.debug(f"Skipping small audio chunk: {len(audio_chunk)} bytes")
            return None
        
        # Process with streaming STT
        transcript = await stt_instance.stream_transcribe_chunk(audio_chunk, is_final=is_final)
        
        if transcript and transcript.strip():
            if is_final:
                latency_tracker.mark("stt_complete")
                # Send final transcription result
                await websocket.send_json({
                    "type": "stt_result",
                    "transcript": transcript,
                    "is_final": True
                })
                return transcript
            else:
                # Send partial result
                await websocket.send_json({
                    "type": "stt_partial",
                    "transcript": transcript,
                    "is_final": False
                })
        
        return None
        
    except Exception as e:
        logger.error(f"STT chunk processing error: {e}")
        return None
