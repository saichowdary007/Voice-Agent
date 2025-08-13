"""
Deepgram Voice Agent WebSocket Proxy
Handles client connections and proxies them to Deepgram's Voice Agent API
"""
import asyncio
import base64
import json
import logging
import time
import websockets
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import WebSocket
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
    DEEPGRAM_TTS_MODEL,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GROQ_API_KEY,
    GEMINI_API_KEY,
    USE_FUNCTION_CALLS,
)
from src.agent import DeepgramAgentManager, AgentSettings, AgentEventRouter
from src.agent.agent_conversation_bridge import AgentConversationBridge

logger = logging.getLogger(__name__)


class DeepgramAgentProxy:
    """
    Proxies WebSocket connections between client and Deepgram Voice Agent API
    """
    
    def __init__(self, client_websocket: WebSocket, user_id: str | None = None):
        self.client_ws = client_websocket
        self.agent = DeepgramAgentManager()
        self.router = AgentEventRouter()
        self.bridge = AgentConversationBridge(user_id)
        self.running = False
        self.tasks = []
         
    async def start(self):
        """Start the proxy connection to Deepgram Voice Agent"""
        try:
            ok = await self.agent.start()
            if not ok:
                return False
            # Start proxy tasks
            self.running = True
            self.tasks = [
                asyncio.create_task(self._proxy_client_to_agent()),
                asyncio.create_task(self._proxy_agent_to_client()),
            ]
            return True
         
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram Voice Agent: {e}")
            return False
    
    async def _proxy_client_to_agent(self):
        """Proxy messages from client to Deepgram"""
        try:
            while self.running:
                try:
                    # Receive from client
                    message = await self.client_ws.receive()
                    
                    if message["type"] == "websocket.receive":
                        if "bytes" in message:
                            # Binary audio data -> forward as PCM16 to agent
                            await self.agent.send_audio(message["bytes"])
                        elif "text" in message:
                            # JSON message - handle a few client-side message types
                            try:
                                data = json.loads(message["text"])
                                msg_type = data.get("type", "")
                                
                                # Handle pings/heartbeats locally
                                if msg_type == "ping":
                                    await self.client_ws.send_json({
                                        "type": "pong",
                                        "timestamp": data.get("timestamp", datetime.utcnow().isoformat())
                                    })
                                    continue
                                if msg_type == "heartbeat":
                                    await self.client_ws.send_json({
                                        "type": "heartbeat_ack",
                                        "timestamp": datetime.utcnow().isoformat()
                                    })
                                    continue

                                # Connection ack from client (ignore to keep protocol tidy)
                                if msg_type == "connection":
                                    await self.client_ws.send_json({
                                        "type": "connection_ack",
                                        "message": "Connection acknowledged",
                                        "timestamp": datetime.utcnow().isoformat()
                                    })
                                    continue

                                # Handle telemetry acks to avoid client warnings
                                if msg_type in ("stt_client_info", "stt_metrics"):
                                    await self.client_ws.send_json({
                                        "type": f"{msg_type}_ack",
                                        "timestamp": datetime.utcnow().isoformat()
                                    })
                                    continue

                                # Dynamic settings update
                                if msg_type in ("settings", "Settings"):
                                    try:
                                        new_settings = self._parse_settings_message(data)
                                        # Avoid race with agent.events() recv; fire-and-forget apply
                                        ok = await self.agent.apply_settings(new_settings)
                                        payload = {
                                            # Reflect that settings were sent; final ack comes from agent
                                            "type": "settings_sent" if ok else "settings_error",
                                            "timestamp": datetime.utcnow().isoformat(),
                                        }
                                        if ok and self.agent.last_settings_latency_ms is not None:
                                            payload["latency_ms"] = round(self.agent.last_settings_latency_ms, 2)
                                            logger.info(
                                                "SettingsApplied latency(ms)=%s",
                                                payload["latency_ms"],
                                            )
                                        await self.client_ws.send_json(payload)
                                    except Exception as se:
                                        await self.client_ws.send_json({
                                            "type": "settings_error",
                                            "message": str(se),
                                            "timestamp": datetime.utcnow().isoformat(),
                                        })
                                    continue
                                # Audio chunk (PCM16 base64) from client → forward to agent
                                if msg_type in ("audio_chunk", "AudioData"):
                                    try:
                                        b64 = data.get("data")
                                        if isinstance(b64, str) and b64:
                                            audio_bytes = base64.b64decode(b64)
                                            await self.agent.send_audio(audio_bytes)
                                            # Acknowledge to avoid client warnings
                                            await self.client_ws.send_json({
                                                "type": "audio_ack",
                                                "timestamp": datetime.utcnow().isoformat(),
                                            })
                                        continue
                                    except Exception as ae:
                                        await self.client_ws.send_json({
                                            "type": "error",
                                            "message": f"Invalid audio payload: {ae}",
                                            "timestamp": datetime.utcnow().isoformat(),
                                        })
                                        continue

                                # Unknown text messages: keep protocol strict in agent-proxy mode
                                await self.client_ws.send_json({
                                    "type": "error",
                                    "message": f"Unknown message type: {msg_type}",
                                    "timestamp": datetime.utcnow().isoformat(),
                                })
                                continue
                            
                            except json.JSONDecodeError:
                                # Ignore non-JSON text
                                continue
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Client WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error(f"Error proxying client to Deepgram: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Client to Deepgram proxy error: {e}")
        finally:
            self.running = False
    
    async def _proxy_agent_to_client(self):
        """Proxy messages from Deepgram to client"""
        try:
            while self.running:
                try:
                    # Stream events from agent manager
                    async for event in self.agent.events():
                        # Function calling scaffold (server-side handling) - optional
                        if USE_FUNCTION_CALLS and isinstance(event, dict):
                            etype = event.get("type", "")
                            if etype in ("ToolRequest", "FunctionCall", "FunctionCallRequest"):
                                name = event.get("name") or event.get("tool") or "unknown"
                                args = event.get("arguments") or event.get("args") or {}
                                try:
                                    result = await self._handle_function_call(name, args)
                                    await self.client_ws.send_json({
                                        "type": "function_result",
                                        "name": name,
                                        "result": result,
                                        "timestamp": datetime.utcnow().isoformat(),
                                    })
                                except Exception as fe:
                                    await self.client_ws.send_json({
                                        "type": "function_error",
                                        "name": name,
                                        "message": str(fe),
                                        "timestamp": datetime.utcnow().isoformat(),
                                    })
                                # Skip forwarding to client as a normal agent message
                                continue
                        if isinstance(event, dict) and event.get("type") == "AudioData":
                            await self.client_ws.send_json(self.router.wrap_audio(event["data"]))
                        else:
                            transformed = self.router.transform(event if isinstance(event, dict) else {})
                            if transformed:
                                # Add latency metric when settings applied
                                if transformed.get("type") == "settings_applied" and self.agent.last_settings_latency_ms is not None:
                                    transformed["latency_ms"] = round(self.agent.last_settings_latency_ms, 2)

                                await self.client_ws.send_json(transformed)

                                # Persist conversation text when available
                                if transformed.get("type") == "agent_text":
                                    role = transformed.get("role", "assistant")
                                    content = transformed.get("content", "")
                                    try:
                                        await self.bridge.record_conversation_text(role, content)
                                    except Exception:
                                        pass
                        # Break loop iteration if manager stopped
                        if not self.running:
                            break
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Deepgram WebSocket connection closed")
                    try:
                        await self.client_ws.send_json({
                            "type": "error",
                            "message": "Agent disconnected",
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                    finally:
                        break
                except Exception as e:
                    logger.error(f"Error proxying Deepgram to client: {e}")
                    try:
                        await self.client_ws.send_json({
                            "type": "error",
                            "message": "Agent error occurred",
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                    finally:
                        break
                    
        except Exception as e:
            logger.error(f"Deepgram to client proxy error: {e}")
        finally:
            self.running = False
    
    def _parse_settings_message(self, data: Dict[str, Any]) -> AgentSettings:
        """Parse client 'settings' message into AgentSettings and apply keepalive if provided."""
        agent = data.get("agent", {}) or {}
        audio = data.get("audio", {}) or {}

        # Language
        language = agent.get("language", "en-US")

        # Listen provider/model
        listen = agent.get("listen", {}) or {}
        listen_provider = (listen.get("provider", {}) or {})
        listen_model = listen_provider.get("model") or "nova-3"

        # Speak provider/model
        speak = agent.get("speak", {}) or {}
        speak_provider = (speak.get("provider", {}) or {})
        speak_model = speak_provider.get("model") or DEEPGRAM_TTS_MODEL

        # Think provider
        think_provider = (agent.get("think", {}).get("provider", {}) or {})
        think_type = think_provider.get("type")
        think_model = think_provider.get("model")
        think_temp = think_provider.get("temperature")

        # Audio config
        input_cfg = audio.get("input", {}) or {}
        output_cfg = audio.get("output", {}) or {}

        # Optional keepalive interval tuning
        keepalive = data.get("keepalive_interval")
        if isinstance(keepalive, int) and keepalive > 0:
            self.agent.set_keepalive_interval(keepalive)

        return AgentSettings(
            language=language,
            input_encoding=input_cfg.get("encoding", AUDIO_INPUT_ENCODING),
            input_sample_rate=int(input_cfg.get("sample_rate", AUDIO_INPUT_SAMPLE_RATE)),
            output_encoding=output_cfg.get("encoding", AUDIO_OUTPUT_ENCODING),
            output_sample_rate=int(output_cfg.get("sample_rate", AUDIO_OUTPUT_SAMPLE_RATE)),
            listen_model=listen_model,
            speak_model=speak_model,
            think_provider_type=think_type or LLM_PROVIDER_TYPE,
            think_model=think_model or LLM_MODEL,
            think_temperature=think_temp if think_temp is not None else LLM_TEMPERATURE,
        )
    
    async def stop(self):
        """Stop the proxy and clean up resources"""
        self.running = False
        
        # Cancel tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Close Agent connection
        try:
            await self.agent.close()
        except Exception as e:
            logger.error(f"Error closing agent connection: {e}")
            await self.client_ws.send_json({
                "type": "error",
                "message": f"Connection error: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
            })
            
        logger.info("🛑 Deepgram Agent proxy stopped")
    
    async def _handle_function_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Simple function calling registry. Expand as needed. Returns result dict."""
        registry = {
            "get_time": self._fn_get_time,
        }
        fn = registry.get(name)
        if not fn:
            raise ValueError(f"Unknown function: {name}")
        return await fn(args)
    
    async def _fn_get_time(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {"now": datetime.utcnow().isoformat()}


# Global proxy instances
active_proxies: Dict[WebSocket, DeepgramAgentProxy] = {}


async def handle_websocket_connection(websocket: WebSocket, user_id: str | None = None):
    """Handle a new WebSocket connection by creating a Deepgram proxy"""
    proxy = DeepgramAgentProxy(websocket, user_id=user_id)
    active_proxies[websocket] = proxy
    
    try:
        success = await proxy.start()
        if not success:
            await websocket.send_json({
                "type": "error",
                "message": "Failed to connect to Deepgram Voice Agent"
            })
            return
        
        # Wait for proxy to finish
        await asyncio.gather(*proxy.tasks, return_exceptions=True)
        
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        await websocket.send_json({
            "type": "error", 
            "message": f"Connection error: {str(e)}"
        })
    finally:
        # Clean up
        if websocket in active_proxies:
            await active_proxies[websocket].stop()
            del active_proxies[websocket]


async def cleanup_websocket_connection(websocket: WebSocket):
    """Clean up a WebSocket connection"""
    if websocket in active_proxies:
        await active_proxies[websocket].stop()
        del active_proxies[websocket]


# Legacy handler functions for compatibility
async def handle_text_message(websocket: WebSocket, message: dict, *args, **kwargs):
    """Legacy handler - not used in proxy mode"""
    await websocket.send_json({
        "type": "error",
        "message": "Text messages not supported in Voice Agent mode. Use audio streaming."
    })


async def handle_audio_chunk(websocket: WebSocket, message: dict, *args, **kwargs):
    """Legacy handler - not used in proxy mode"""
    await websocket.send_json({
        "type": "error", 
        "message": "Legacy audio handling not supported. Use direct WebSocket streaming."
    })


async def handle_settings(websocket: WebSocket, message: dict):
    """Legacy handler - not used in proxy mode"""
    await websocket.send_json({
        "type": "error",
        "message": "Settings handled automatically in Voice Agent mode"
    })


async def handle_ping(websocket: WebSocket, message: dict):
    """Handle ping messages"""
    await websocket.send_json({
        "type": "pong",
        "timestamp": message.get("timestamp", datetime.utcnow().isoformat())
    })


async def handle_heartbeat(websocket: WebSocket, message: dict):
    """Handle heartbeat messages"""
    await websocket.send_json({
        "type": "heartbeat_ack",
        "timestamp": datetime.utcnow().isoformat()
    })


async def handle_connection_message(websocket: WebSocket, message: dict):
    """Handle connection messages"""
    await websocket.send_json({
        "type": "connection_ack",
        "message": "Connection acknowledged"
    })


async def handle_unknown_message(websocket: WebSocket, message: dict):
    """Handle unknown message types"""
    message_type = message.get('type', 'undefined')
    await websocket.send_json({
        "type": "error",
        "message": f"Unknown message type: {message_type}"
    })


# Compatibility functions
async def handle_vad_status(websocket: WebSocket, message: dict):
    """Legacy VAD handler"""
    pass


async def handle_start_listening(websocket: WebSocket, message: dict):
    """Legacy listening handler"""
    pass


async def handle_stop_listening(websocket: WebSocket, message: dict):
    """Legacy listening handler"""
    pass