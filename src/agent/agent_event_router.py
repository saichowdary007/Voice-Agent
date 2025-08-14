"""
AgentEventRouter
Normalizes Deepgram Agent events and prepares client-facing messages.
"""
import base64
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from src.config import AUDIO_OUTPUT_SAMPLE_RATE

logger = logging.getLogger(__name__)


class AgentEventRouter:
    """Transforms Deepgram Agent events to client-facing payloads."""

    def transform(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform JSON event from Deepgram to a client payload.
        Returns None if the event should be dropped.
        """
        msg_type = data.get("type", "")
        if msg_type == "Welcome":
            return {
                "type": "connection_ack",
                "message": "Connected to Deepgram Voice Agent",
                "request_id": data.get("request_id"),
            }
        if msg_type == "SettingsApplied":
            return {"type": "settings_applied", "timestamp": datetime.utcnow().isoformat()}
        if msg_type == "ConversationText":
            return {
                "type": "agent_text",
                "role": data.get("role", "assistant"),
                "content": data.get("content", ""),
                "timestamp": datetime.utcnow().isoformat(),
            }
        if msg_type == "UserStartedSpeaking":
            return {"type": "user_started_speaking", "timestamp": datetime.utcnow().isoformat()}
        if msg_type == "AgentThinking":
            return {"type": "agent_thinking", "timestamp": datetime.utcnow().isoformat()}
        if msg_type == "AgentAudioDone":
            return {"type": "agent_audio_done", "timestamp": datetime.utcnow().isoformat()}
        # Be tolerant to variant event names like AgentAudioCompleted, AgentTtsDone, etc.
        try:
            normalized = str(msg_type or "").lower()
            if "audio" in normalized and "done" in normalized:
                return {"type": "agent_audio_done", "timestamp": datetime.utcnow().isoformat()}
        except Exception:
            pass
        if msg_type in ["AgentErrors", "AgentWarnings", "Error", "Warning"]:
            return {
                "type": "error",
                "message": data.get("message") or data.get("description") or "Agent error occurred",
                "timestamp": datetime.utcnow().isoformat(),
            }
        # Pass-through unknowns for debugging
        return data

    def wrap_audio(self, pcm16_bytes: bytes) -> Dict[str, Any]:
        """Convert raw PCM to WAV and base64 for browser compatibility."""
        wav = self._add_wav_header(pcm16_bytes)
        audio_b64 = base64.b64encode(wav).decode("ascii")
        return {"type": "tts_audio", "data": audio_b64, "mime": "audio/wav"}

    def _add_wav_header(self, pcm_data: bytes) -> bytes:
        sample_rate = AUDIO_OUTPUT_SAMPLE_RATE
        channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * channels * (bits_per_sample // 8)
        block_align = channels * (bits_per_sample // 8)
        data_size = len(pcm_data)

        header = bytearray(44)
        header[0:4] = b"RIFF"
        header[4:8] = (36 + data_size).to_bytes(4, "little")
        header[8:12] = b"WAVE"
        header[12:16] = b"fmt "
        header[16:20] = (16).to_bytes(4, "little")
        header[20:22] = (1).to_bytes(2, "little")
        header[22:24] = (channels).to_bytes(2, "little")
        header[24:28] = (sample_rate).to_bytes(4, "little")
        header[28:32] = (byte_rate).to_bytes(4, "little")
        header[32:34] = (block_align).to_bytes(2, "little")
        header[34:36] = (bits_per_sample).to_bytes(2, "little")
        header[36:40] = b"data"
        header[40:44] = (data_size).to_bytes(4, "little")
        return bytes(header) + pcm_data
