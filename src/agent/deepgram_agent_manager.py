"""
DeepgramAgentManager
Manages lifecycle of Deepgram Voice Agent WebSocket connections.
"""
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Optional, Tuple, Union

import websockets

from src.config import (
    DEEPGRAM_API_KEY,
    DEEPGRAM_AGENT_ENDPOINT,
    AUDIO_INPUT_ENCODING,
    AUDIO_INPUT_SAMPLE_RATE,
    AUDIO_OUTPUT_ENCODING,
    AUDIO_OUTPUT_SAMPLE_RATE,
    LLM_PROVIDER_TYPE,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    LLM_ENDPOINT_URL,
    LLM_ENDPOINT_HEADERS,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentSettings:
    language: str = "en"
    input_encoding: str = AUDIO_INPUT_ENCODING
    input_sample_rate: int = AUDIO_INPUT_SAMPLE_RATE
    output_encoding: str = AUDIO_OUTPUT_ENCODING
    output_sample_rate: int = AUDIO_OUTPUT_SAMPLE_RATE
    listen_model: str = "nova-3"
    speak_model: str = "aura-asteria-en"
    think_provider_type: Optional[str] = LLM_PROVIDER_TYPE
    think_model: Optional[str] = LLM_MODEL
    think_temperature: Optional[float] = LLM_TEMPERATURE


class DeepgramAgentManager:
    """Manages lifecycle of a Deepgram Agent connection and I/O."""

    def __init__(
        self,
        api_key: str = DEEPGRAM_API_KEY,
        endpoint: str = DEEPGRAM_AGENT_ENDPOINT,
        keepalive_interval: int = 8,
        connection_timeout: int = 30,
        max_retries: int = 3,
        settings: Optional[AgentSettings] = None,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._keepalive_interval = keepalive_interval
        self._connection_timeout = connection_timeout
        self._max_retries = max_retries
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._running = False
        self._settings = settings or AgentSettings()
        self.last_settings_latency_ms: Optional[float] = None
        self._pending_settings_started_at: Optional[float] = None
        self.awaiting_client_settings: bool = False

    async def start(self) -> bool:
        """Connect to Deepgram. Try default Settings, but if they fail, defer to client Settings.

        We previously failed hard when default Settings were invalid (e.g., no LLM provider keys),
        which prevented the client from sending its own Settings right after connection. This
        implementation connects first, attempts to apply defaults, and if that fails, keeps the
        connection running and awaits client-provided Settings.
        """
        backoffs = [1, 2, 4]
        attempt = 0
        while attempt <= self._max_retries:
            try:
                # 1) Connect to Deepgram Agent endpoint
                await self._connect()

                # 2) Try sending default Settings. If it works, we're fully ready.
                try:
                    loop = asyncio.get_running_loop()
                    t0 = loop.time()
                    await self._send_settings()
                    ok = await self._wait_for_settings_applied(timeout=self._connection_timeout)
                    if ok:
                        self.last_settings_latency_ms = (loop.time() - t0) * 1000.0
                        self._running = True
                        self.awaiting_client_settings = False
                        self._keepalive_task = asyncio.create_task(self._send_keepalive_loop())
                        logger.info("✅ Deepgram Agent ready (SettingsApplied)")
                        return True
                    # If timeout waiting for SettingsApplied, fall through to defer-to-client path
                    logger.warning("Initial SettingsApplied timeout; deferring to client Settings")
                except Exception as settings_error:
                    # Most common root cause: missing/invalid LLM provider configuration.
                    logger.warning(
                        "Initial default Settings failed: %s. Deferring to client-provided Settings.",
                        settings_error,
                    )

                # 3) Defer-to-client path: start keepalive and allow client to send Settings.
                self._running = True
                self.awaiting_client_settings = True
                self._keepalive_task = asyncio.create_task(self._send_keepalive_loop())
                logger.info("🕓 Deepgram Agent connected; awaiting client Settings")
                return True

            except Exception as e:
                attempt += 1
                logger.error(f"DeepgramAgentManager start attempt {attempt} failed: {e}")
                if attempt > self._max_retries:
                    break
                await asyncio.sleep(backoffs[min(attempt - 1, len(backoffs) - 1)])
        return False

    async def _connect(self) -> None:
        # Deepgram expects lowercase 'token' prefix per docs
        if not self._api_key:
            raise ValueError("DEEPGRAM_API_KEY is not set. Please configure it in your environment.")
        # Authenticate via token subprotocol per Deepgram Agent docs.
        # Avoid passing extra_headers to maintain compatibility with some websocket client versions.
        self._ws = await websockets.connect(
            self._endpoint,
            subprotocols=["token", self._api_key],
            ping_interval=None,  # we send our own keepalives
            close_timeout=5,
        )
        logger.info("Connected to Deepgram Agent endpoint")

    async def _send_settings(self) -> None:
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        settings = self._build_settings_message(self._settings)
        await self._ws.send(json.dumps(settings))
        logger.debug("Sent Settings to Deepgram Agent")

    async def _wait_for_settings_applied(self, timeout: int = 30) -> bool:
        assert self._ws is not None
        try:
            async def waiter() -> bool:
                while True:
                    msg = await self._ws.recv()
                    if isinstance(msg, (bytes, bytearray)):
                        # ignore early audio frames during negotiation
                        continue
                    try:
                        data = json.loads(msg)
                        etype = data.get("type")
                        if etype in ("Error", "AgentErrors", "AgentWarnings"):
                            # Surface agent-side error to caller
                            raise RuntimeError(data.get("message") or str(data))
                        if data.get("type") == "SettingsApplied":
                            return True
                        # Allow Welcome pass-through
                    except json.JSONDecodeError:
                        continue
            return await asyncio.wait_for(waiter(), timeout=timeout)
        except asyncio.TimeoutError:
            return False

    async def apply_settings(self, new_settings: AgentSettings) -> bool:
        """Dynamically apply new settings without contending for recv().

        We avoid awaiting _wait_for_settings_applied() because another coroutine
        (events loop) is reading from the same websocket, which would cause
        'cannot call recv while another coroutine is already running recv'.
        Latency is computed in events() when SettingsApplied arrives.
        """
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        self._settings = new_settings
        loop = asyncio.get_running_loop()
        self._pending_settings_started_at = loop.time()
        await self._ws.send(json.dumps(self._build_settings_message(new_settings)))
        # Fire-and-forget; success will be reflected by a SettingsApplied event
        return True

    def set_keepalive_interval(self, seconds: int) -> None:
        """Update keepalive interval for subsequent pings."""
        self._keepalive_interval = max(1, int(seconds))

    def _build_settings_message(self, s: AgentSettings) -> Dict[str, Any]:
        return {
            "type": "Settings",
            "audio": {
                "input": {
                    "encoding": s.input_encoding,
                    "sample_rate": s.input_sample_rate,
                },
                "output": {
                    "encoding": s.output_encoding,
                    "sample_rate": s.output_sample_rate,
                    "container": "none",
                },
            },
            "agent": {
                "language": s.language,
                "listen": {
                    "provider": {"type": "deepgram", "model": s.listen_model, "smart_format": False}
                },
                "think": self._think_provider_block(s),
                "speak": {
                    "provider": {"type": "deepgram", "model": s.speak_model}
                },
            },
        }

    def _think_provider_block(self, s: AgentSettings) -> Dict[str, Any]:
        t = (s.think_provider_type or "").lower()
        model = s.think_model or LLM_MODEL or "gemini-2.0-flash-exp"
        temp = s.think_temperature if s.think_temperature is not None else 0.7

        # Build provider block first
        if t.startswith("open") and OPENAI_API_KEY:
            provider = {"type": "open_ai", "model": model, "temperature": temp}
        elif t.startswith("anth") and ANTHROPIC_API_KEY:
            provider = {"type": "anthropic", "model": model, "temperature": temp}
        elif t.startswith("google") and GEMINI_API_KEY:
            provider = {"type": "google", "model": model, "temperature": temp}
        else:
            # Fallbacks
            if OPENAI_API_KEY:
                logger.warning("Falling back to OpenAI for think provider")
                provider = {"type": "open_ai", "model": "gpt-4o-mini", "temperature": 0.7}
            else:
                raise ValueError("No valid LLM provider configured")

        think_block: Dict[str, Any] = {"provider": provider}

        # Attach optional endpoint and headers at the think level (not inside provider)
        if LLM_ENDPOINT_URL:
            endpoint_obj: Dict[str, Any] = {"url": LLM_ENDPOINT_URL}
            # If headers provided via config, forward them
            if isinstance(LLM_ENDPOINT_HEADERS, dict) and LLM_ENDPOINT_HEADERS:
                endpoint_obj["headers"] = LLM_ENDPOINT_HEADERS
            # For Google/Gemini, if no explicit auth header provided, include API key header
            if provider.get("type") == "google" and GEMINI_API_KEY and (
                not isinstance(LLM_ENDPOINT_HEADERS, dict) or
                "x-goog-api-key" not in LLM_ENDPOINT_HEADERS
            ):
                endpoint_obj.setdefault("headers", {})["x-goog-api-key"] = GEMINI_API_KEY
            # Deepgram requirement: when using custom Google endpoints, omit model in provider
            if provider.get("type") == "google":
                provider.pop("model", None)
            think_block["endpoint"] = endpoint_obj

        return think_block

    async def _send_keepalive_loop(self) -> None:
        assert self._ws is not None
        try:
            while self._running:
                await asyncio.sleep(self._keepalive_interval)
                try:
                    await self._ws.send(json.dumps({"type": "KeepAlive"}))
                except Exception as e:
                    logger.warning(f"KeepAlive send failed: {e}")
                    break
        finally:
            logger.debug("Keepalive loop exited")

    async def send_audio(self, pcm16_bytes: bytes) -> None:
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        await self._ws.send(pcm16_bytes)

    async def events(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Yields Deepgram events as normalized dicts.
        For binary audio, yields {"type": "AudioData", "data": <bytes>}.
        """
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        while self._running:
            msg = await self._ws.recv()
            if isinstance(msg, (bytes, bytearray)):
                yield {"type": "AudioData", "data": bytes(msg)}
            else:
                try:
                    data = json.loads(msg)
                    # Capture SettingsApplied latency if we initiated a change
                    if isinstance(data, dict) and data.get("type") == "SettingsApplied":
                        if self._pending_settings_started_at is not None:
                            loop = asyncio.get_running_loop()
                            self.last_settings_latency_ms = (loop.time() - self._pending_settings_started_at) * 1000.0
                            self._pending_settings_started_at = None
                    yield data
                except json.JSONDecodeError:
                    logger.debug(f"Non-JSON message from agent ignored: {msg}")

    async def close(self) -> None:
        self._running = False
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.error(f"Error closing agent WebSocket: {e}")
            self._ws = None
