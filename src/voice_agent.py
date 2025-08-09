"""
Deepgram Voice Agent integration (WebSocket V1)
Minimal wrapper to start an Agent session, stream PCM16 audio, and emit
callbacks for conversation text and synthesized audio.
"""

import asyncio
import base64
import logging
from typing import Callable, Optional
import os

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    AgentWebSocketEvents,
    AgentKeepAlive,
)
from deepgram.clients.agent.v1.websocket.options import SettingsOptions

logger = logging.getLogger(__name__)


class DeepgramVoiceAgent:
    def __init__(
        self,
        api_key: str,
        sample_rate_input: int = 16000,
        sample_rate_output: int = 24000,
        on_audio_ready: Optional[Callable[[bytes, str], None]] = None,
        on_text: Optional[Callable[[str, str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._client = DeepgramClient(api_key, DeepgramClientOptions(options={"keepalive": "true"}))
        self._conn = None
        self._audio_buffer = bytearray()
        self._keepalive_task: Optional[asyncio.Task] = None
        self._on_audio_ready = on_audio_ready
        self._on_text = on_text
        self._on_error = on_error
        self._sr_in = sample_rate_input
        self._sr_out = sample_rate_output

    def _register_handlers(self) -> None:
        assert self._conn is not None

        def on_audio_data(_self, data, **kwargs):
            try:
                self._audio_buffer.extend(data)
            except Exception as e:
                logger.warning(f"Agent audio buffer error: {e}")

        def on_agent_audio_done(_self, agent_audio_done, **kwargs):
            try:
                if not self._audio_buffer:
                    logger.debug("No audio buffer data for TTS response")
                    return
                # Assemble a WAV file (PCM16 mono) for client playback
                wav_bytes = self._build_wav(bytes(self._audio_buffer), sample_rate=self._sr_out)
                logger.info(f"🔊 Generated TTS audio: {len(wav_bytes)} bytes WAV file")
                self._audio_buffer.clear()
                if self._on_audio_ready:
                    self._on_audio_ready(wav_bytes, "audio/wav")
                    logger.info("✅ TTS audio sent to client")
                else:
                    logger.warning("No audio callback registered")
            except Exception as e:
                logger.error(f"Agent audio finalize error: {e}")

        def on_conversation_text(_self, **kwargs):
            try:
                message = kwargs.get('message')
                if not message:
                    logger.debug("No message in conversation text event")
                    return
                role = getattr(message, "role", "") or "agent"
                content = getattr(message, "content", "") or ""
                logger.info(f"💬 Agent conversation text: {role}: {content}")
                if content and self._on_text:
                    self._on_text(role, str(content))
            except Exception as e:
                logger.warning(f"Agent text handler error: {e}")

        def on_error(_self, error, **kwargs):
            if self._on_error:
                self._on_error(str(error))
            logger.error(f"Agent error: {error}")

        def on_close(_self, close, **kwargs):
            logger.info(f"Agent closed: {close}")

        self._conn.on(AgentWebSocketEvents.AudioData, on_audio_data)
        self._conn.on(AgentWebSocketEvents.AgentAudioDone, on_agent_audio_done)
        self._conn.on(AgentWebSocketEvents.ConversationText, on_conversation_text)
        self._conn.on(AgentWebSocketEvents.Error, on_error)
        self._conn.on(AgentWebSocketEvents.Close, on_close)

    async def start(self) -> bool:
        try:
            self._conn = self._client.agent.websocket.v("1")
            self._register_handlers()

            options = SettingsOptions()
            # Ultra-low latency configuration (<500ms target)
            options.audio.input.encoding = "linear16"
            options.audio.input.sample_rate = self._sr_in
            options.audio.output.encoding = "linear16"
            options.audio.output.sample_rate = self._sr_out
            options.audio.output.container = "none"  # Remove container overhead
            options.audio.output.bitrate = 64000  # Higher bitrate for faster processing
            
            # Ultra-low latency STT settings
            options.agent.listen.interim_results = True  # Enable streaming results
            options.agent.listen.endpointing = 300  # 300ms silence detection (faster)
            options.agent.listen.utterance_end_ms = 800  # Shorter utterance timeout
            options.agent.listen.vad_turnoff = 300  # Quick VAD turnoff
            # Align language with backend config default
            try:
                from src.config import DEEPGRAM_STT_LANGUAGE, DEEPGRAM_TTS_MODEL
                agent_language = DEEPGRAM_STT_LANGUAGE or "en-US"
                tts_model = DEEPGRAM_TTS_MODEL or "aura-asteria-en"
            except Exception:
                agent_language = "en-US"
                tts_model = "aura-asteria-en"

            options.agent.language = agent_language

            # Listen provider (Deepgram STT)
            options.agent.listen.provider.type = "deepgram"
            options.agent.listen.provider.model = "nova-3"

            # Think provider selection (configurable via env, with sensible fallbacks)
            env_provider = os.getenv("DG_THINK_PROVIDER", "").strip().lower()
            env_model = os.getenv("DG_THINK_MODEL", "").strip()

            # Ultra-low latency optimization - use fastest available models
            if os.getenv("OPENAI_API_KEY"):
                options.agent.think.provider.type = "open_ai"
                # Force fastest model regardless of env override for ultra-low latency
                options.agent.think.provider.model = "gpt-4o-mini"  # Fastest OpenAI model
                logger.info("Using ultra-low latency OpenAI provider (gpt-4o-mini)")
            elif os.getenv("GEMINI_API_KEY"):
                options.agent.think.provider.type = "google"
                # Force fastest Gemini model for minimal latency
                options.agent.think.provider.model = "gemini-2.0-flash"  # Optimized for speed
                logger.info("Using ultra-low latency Google provider (gemini-2.0-flash)")
            else:
                logger.error("No THINK provider API key detected. Set OPENAI_API_KEY or GEMINI_API_KEY.")
                return False

            # Speak provider (Deepgram TTS)
            options.agent.speak.provider.type = "deepgram"
            options.agent.speak.provider.model = tts_model
            options.agent.greeting = ""

            if not self._conn.start(options):
                logger.error("Failed to start Deepgram Agent WS with configured THINK provider/model")
                # Fallbacks for common misconfigs
                try:
                    # If Google Gemini 2.0 model was set, fall back to 1.5-flash
                    if getattr(options.agent.think.provider, 'type', '') == 'google':
                        current_model = getattr(options.agent.think.provider, 'model', '')
                        if current_model.startswith('gemini-2.0'):
                            logger.info("Falling back THINK model to gemini-2.0-flash")
                            options.agent.think.provider.model = 'gemini-2.0-flash'
                            if self._conn.start(options):
                                self._keepalive_task = asyncio.create_task(self._run_keepalive())
                                return True
                    # If still failing and OpenAI key is available, switch provider to OpenAI
                    if os.getenv('OPENAI_API_KEY'):
                        logger.info("Switching THINK provider to open_ai:gpt-4o-mini as fallback")
                        options.agent.think.provider.type = 'open_ai'
                        options.agent.think.provider.model = 'gpt-4o-mini'
                        if self._conn.start(options):
                            self._keepalive_task = asyncio.create_task(self._run_keepalive())
                            return True
                except Exception as _:
                    pass
                return False

            # Keepalive every 8 seconds
            self._keepalive_task = asyncio.create_task(self._run_keepalive())
            return True
        except Exception as e:
            logger.error(f"Agent start failed: {e}")
            return False

    async def _run_keepalive(self) -> None:
        try:
            while self._conn is not None:
                await asyncio.sleep(8)
                try:
                    self._conn.send(str(AgentKeepAlive()))
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    def send_audio(self, pcm16_bytes: bytes) -> None:
        if self._conn is None:
            logger.warning("Cannot send audio: agent connection is None")
            return
        try:
            self._conn.send(pcm16_bytes)
            logger.debug(f"📤 Sent {len(pcm16_bytes)} bytes of audio to Deepgram Agent")
        except Exception as e:
            logger.warning(f"Agent send error: {e}")

    async def stop(self) -> None:
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        if self._conn:
            try:
                await asyncio.to_thread(self._conn.finish)
            except Exception:
                pass
            finally:
                self._conn = None

    @staticmethod
    def _build_wav(pcm: bytes, sample_rate: int, bits_per_sample: int = 16, channels: int = 1) -> bytes:
        byte_rate = sample_rate * channels * (bits_per_sample // 8)
        block_align = channels * (bits_per_sample // 8)

        data_size = len(pcm)
        riff_chunk_size = 36 + data_size

        header = bytearray(44)
        header[0:4] = b"RIFF"
        header[4:8] = (riff_chunk_size).to_bytes(4, "little")
        header[8:12] = b"WAVE"
        header[12:16] = b"fmt "
        header[16:20] = (16).to_bytes(4, "little")  # PCM
        header[20:22] = (1).to_bytes(2, "little")  # AudioFormat PCM
        header[22:24] = (channels).to_bytes(2, "little")
        header[24:28] = (sample_rate).to_bytes(4, "little")
        header[28:32] = (byte_rate).to_bytes(4, "little")
        header[32:34] = (block_align).to_bytes(2, "little")
        header[34:36] = (bits_per_sample).to_bytes(2, "little")
        header[36:40] = b"data"
        header[40:44] = (data_size).to_bytes(4, "little")

        return bytes(header) + pcm


