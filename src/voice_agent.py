"""
Deepgram Voice Agent integration (WebSocket V1)
Minimal wrapper to start an Agent session, stream PCM16 audio, and emit
callbacks for conversation text and synthesized audio.
"""

import asyncio
import base64
import logging
from typing import Callable, Optional, Dict, Any
import json
import os

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    AgentWebSocketEvents,
    AgentKeepAlive,
)
from deepgram.clients.agent.v1.websocket.options import (
    SettingsOptions,
    FunctionCallResponse,
    Function as DGFunction,
)

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
        # Registry for client-side executable functions
        self._function_registry: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._function_descriptions: Dict[str, str] = {}

    def _register_handlers(self) -> None:
        assert self._conn is not None

        def on_audio_data(_self, data, **kwargs):
            try:
                logger.info(f"🎵 AudioData event received: {len(data)} bytes")
                self._audio_buffer.extend(data)
            except Exception as e:
                logger.warning(f"Agent audio buffer error: {e}")

        def on_agent_audio_done(_self, agent_audio_done, **kwargs):
            try:
                logger.info(f"🔊 Agent audio done event: {agent_audio_done}")
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

        def on_conversation_text(_self, conversation_text, **kwargs):
            try:
                logger.info(f"🎯 ConversationText event received: {conversation_text}")
                
                # Handle the conversation_text object directly
                if hasattr(conversation_text, 'role') and hasattr(conversation_text, 'content'):
                    role = conversation_text.role
                    content = conversation_text.content
                elif isinstance(conversation_text, dict):
                    role = conversation_text.get('role', 'agent')
                    content = conversation_text.get('content', '')
                else:
                    logger.warning(f"Unexpected conversation_text format: {type(conversation_text)}")
                    return
                
                logger.info(f"💬 Agent conversation text: {role}: {content}")
                if content and self._on_text:
                    self._on_text(role, str(content))
            except Exception as e:
                logger.error(f"Agent text handler error: {e}")
                logger.error(f"conversation_text object: {conversation_text}")
                logger.error(f"kwargs: {kwargs}")

        def on_error(_self, error, **kwargs):
            if self._on_error:
                self._on_error(str(error))
            logger.error(f"Agent error: {error}")

        def on_close(_self, close, **kwargs):
            logger.info(f"Agent closed: {close}")

        def on_user_started_speaking(_self, user_started_speaking, **kwargs):
            logger.info(f"🎤 User started speaking: {user_started_speaking}")

        def on_agent_thinking(_self, agent_thinking, **kwargs):
            logger.info(f"🤔 Agent is thinking: {agent_thinking}")

        def on_welcome(_self, welcome, **kwargs):
            logger.info(f"👋 Agent welcome received: {welcome}")

        def on_settings_applied(_self, settings_applied, **kwargs):
            logger.info(f"⚙️ Agent settings applied: {settings_applied}")
            
        def on_speech_started(_self, speech_started, **kwargs):
            logger.info(f"🗣️ Speech started: {speech_started}")
            
        def on_utterance_end(_self, utterance_end, **kwargs):
            logger.info(f"🔚 Utterance end: {utterance_end}")
            
        def on_transcript(_self, transcript, **kwargs):
            logger.info(f"📝 Transcript: {transcript}")
            
        def on_metadata(_self, metadata, **kwargs):
            logger.info(f"📊 Metadata: {metadata}")
        
        def on_unhandled(_self, unhandled, **kwargs):
            try:
                # Deepgram SDK emits Unhandled with raw JSON. Detect History messages to avoid noisy warnings.
                raw = getattr(unhandled, "raw", None)
                if isinstance(raw, str):
                    try:
                        data = json.loads(raw)
                        mtype = data.get("type")
                        if mtype == "History":
                            logger.debug("🧾 History event received and ignored (handled locally)")
                            return
                    except Exception:
                        pass
                logger.debug(f"Unhandled message passthrough: {unhandled}")
            except Exception as e:
                logger.debug(f"Unhandled handler error: {e}")
        
        def on_function_call_request(_self, function_call_request, **kwargs):
            try:
                logger.info(f"🛠️ FunctionCallRequest received: {function_call_request}")
                if not getattr(function_call_request, "functions", None):
                    logger.warning("FunctionCallRequest has no functions list")
                    return
                # Iterate over requested functions and respond individually
                for func in function_call_request.functions:
                    try:
                        name = getattr(func, "name", None)
                        fid = getattr(func, "id", None)
                        args_raw = getattr(func, "arguments", "{}")
                        client_side = bool(getattr(func, "client_side", False))
                        # Default content if something goes wrong
                        content: str = "{}"
                        if client_side:
                            # Execute locally if registered
                            if name in self._function_registry:
                                try:
                                    args: Dict[str, Any] = {}
                                    if isinstance(args_raw, str) and args_raw.strip():
                                        try:
                                            args = json.loads(args_raw)
                                        except Exception:
                                            # Deepgram may already provide JSON-serialized string
                                            args = {"_raw": args_raw}
                                    elif isinstance(args_raw, dict):
                                        args = args_raw
                                    result = self._function_registry[name](args)
                                    # Ensure string content payload
                                    if isinstance(result, (dict, list)):
                                        content = json.dumps(result)
                                    else:
                                        content = str(result)
                                except Exception as exec_err:
                                    logger.error(f"Function '{name}' execution error: {exec_err}")
                                    content = json.dumps({"error": f"execution failed: {exec_err}"})
                            else:
                                logger.warning(f"Requested client-side function not registered: {name}")
                                content = json.dumps({"error": f"unknown function: {name}"})
                        else:
                            # Not client-side: let server handle via URL/endpoint; we do not execute here
                            logger.info(f"Skipping non-client-side function '{name}' per request flag")
                            continue

                        # Send FunctionCallResponse back to agent
                        try:
                            response = FunctionCallResponse(id=fid or "", name=name or "", content=content)
                            self._conn.send(str(response))
                            logger.info(f"📨 Sent FunctionCallResponse for '{name}' (id={fid})")
                        except Exception as send_err:
                            logger.error(f"Failed sending FunctionCallResponse for '{name}': {send_err}")
                    except Exception as one_err:
                        logger.error(f"Error handling one function call: {one_err}")
            except Exception as e:
                logger.error(f"FunctionCallRequest handler error: {e}")

        # Register all event handlers
        self._conn.on(AgentWebSocketEvents.AudioData, on_audio_data)
        self._conn.on(AgentWebSocketEvents.AgentAudioDone, on_agent_audio_done)
        self._conn.on(AgentWebSocketEvents.ConversationText, on_conversation_text)
        self._conn.on(AgentWebSocketEvents.Error, on_error)
        self._conn.on(AgentWebSocketEvents.Close, on_close)
        
        # Additional event handlers for better debugging
        try:
            self._conn.on(AgentWebSocketEvents.UserStartedSpeaking, on_user_started_speaking)
            self._conn.on(AgentWebSocketEvents.AgentThinking, on_agent_thinking)
            self._conn.on(AgentWebSocketEvents.Welcome, on_welcome)
            self._conn.on(AgentWebSocketEvents.SettingsApplied, on_settings_applied)
            self._conn.on(AgentWebSocketEvents.FunctionCallRequest, on_function_call_request)
            self._conn.on(AgentWebSocketEvents.Unhandled, on_unhandled)
        except AttributeError as e:
            logger.debug(f"Some event handlers not available: {e}")

    def register_function(
        self,
        name: str,
        func: Callable[[Dict[str, Any]], Any],
        description: str = "",
    ) -> None:
        """
        Register a client-side function that can be executed upon FunctionCallRequest.

        The callable receives a single dict of arguments and may return any JSON-serializable value
        or a string. Returned values are sent as the 'content' field in FunctionCallResponse.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Function name must be a non-empty string")
        if not callable(func):
            raise ValueError("func must be callable")
        self._function_registry[name] = func
        self._function_descriptions[name] = description or name

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
            
            # Adjusted STT settings for better speech detection
            options.agent.listen.interim_results = True  # Enable streaming results
            options.agent.listen.endpointing = 100  # ms of silence to mark endpoint (more sensitive)
            options.agent.listen.utterance_end_ms = 500  # shorter utterance timeout for faster detection
            options.agent.listen.vad_turnoff = 100  # more sensitive VAD turnoff
            options.agent.listen.smart_format = True  # Enable smart formatting for better transcripts
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

            # Think provider configuration using new LLM config
            from src.config import (
                LLM_PROVIDER_TYPE, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
                LLM_ENDPOINT_URL, LLM_ENDPOINT_HEADERS,
                OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, GEMINI_API_KEY
            )

            # Configure LLM provider based on config
            options.agent.think.provider.type = LLM_PROVIDER_TYPE
            options.agent.think.provider.temperature = LLM_TEMPERATURE
            
            # Add a system prompt to guide the agent's responses
            options.agent.think.prompt = "You are a helpful AI voice assistant. Respond naturally and conversationally to user questions. Keep your responses concise but informative. Always respond when the user speaks to you."
            
            # Set up custom endpoint for providers that require it
            if LLM_ENDPOINT_URL and LLM_ENDPOINT_HEADERS:
                options.agent.think.endpoint = {
                    "url": LLM_ENDPOINT_URL,
                    "headers": LLM_ENDPOINT_HEADERS
                }
                logger.info(f"Using custom endpoint for {LLM_PROVIDER_TYPE}: {LLM_ENDPOINT_URL}")
                # For Google with custom endpoint, model is in URL - use a placeholder or empty model
                if LLM_PROVIDER_TYPE == "google":
                    logger.info("Google custom endpoint detected - model specified in URL, not in provider settings")
                    # Don't set model at all for Google custom endpoints
                else:
                    options.agent.think.provider.model = LLM_MODEL
            else:
                # No custom endpoint, set model normally
                options.agent.think.provider.model = LLM_MODEL
            
            # Validate API keys for the selected provider
            provider_valid = False
            if LLM_PROVIDER_TYPE == "open_ai" and OPENAI_API_KEY:
                provider_valid = True
            elif LLM_PROVIDER_TYPE == "anthropic" and ANTHROPIC_API_KEY:
                provider_valid = True
            elif LLM_PROVIDER_TYPE == "google" and GEMINI_API_KEY:
                # Try Google provider with native Deepgram management (like OpenAI/Anthropic)
                provider_valid = True
                logger.info(f"Google provider configured with native Deepgram management")
            elif LLM_PROVIDER_TYPE == "groq" and GROQ_API_KEY:
                # Groq requires explicit endpoint with headers per API spec
                provider_valid = bool(LLM_ENDPOINT_URL and LLM_ENDPOINT_HEADERS)
            
            # If provider not valid, try fallbacks
            if not provider_valid:
                logger.warning(f"Provider '{LLM_PROVIDER_TYPE}' not properly configured, trying fallbacks...")
                
                # Try OpenAI first (even with test key to see error)
                if OPENAI_API_KEY:
                    logger.info("Falling back to OpenAI GPT-4o-mini")
                    options.agent.think.provider.type = "open_ai"
                    options.agent.think.provider.model = "gpt-4o-mini"
                    # Clear custom endpoint for OpenAI
                    if hasattr(options.agent.think, 'endpoint'):
                        delattr(options.agent.think, 'endpoint')
                    provider_valid = True
                # Try Anthropic as second fallback
                elif ANTHROPIC_API_KEY:
                    logger.info("Falling back to Anthropic Claude")
                    options.agent.think.provider.type = "anthropic"
                    options.agent.think.provider.model = "claude-3-haiku-20240307"
                    # Clear custom endpoint for Anthropic
                    if hasattr(options.agent.think, 'endpoint'):
                        delattr(options.agent.think, 'endpoint')
                    provider_valid = True
                else:
                    logger.error(f"No valid API key found for LLM provider: {LLM_PROVIDER_TYPE} and no fallback available")
                    return False

            # Log the provider configuration
            model_info = getattr(options.agent.think.provider, 'model', 'not set')
            logger.info(f"Using LLM provider: {options.agent.think.provider.type} with model: {model_info}")

            # Speak provider (Deepgram TTS)
            options.agent.speak.provider.type = "deepgram"
            options.agent.speak.provider.model = tts_model
            options.agent.greeting = ""

            # Register client-side functions as tool definitions for the think provider
            if self._function_registry:
                funcs = []
                for fname, desc in self._function_descriptions.items():
                    try:
                        # Client-side functions: omit endpoint/method/url to mark as client-side
                        # Keep strict snake_case keys to avoid payload validation errors
                        func_def = {
                            "name": fname,
                            "description": desc or fname,
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                        }
                        funcs.append(func_def)
                    except Exception as e:
                        logger.warning(f"Failed to append function definition for {fname}: {e}")
                if funcs:
                    options.agent.think.functions = funcs
                    try:
                        # Debug the exact JSON that will be sent for functions to ensure no camelCase fields
                        payload_preview = options.to_dict()
                        think_funcs = (
                            payload_preview.get("agent", {})
                            .get("think", {})
                            .get("functions", [])
                        )
                        logger.debug("Think.functions payload preview: %s", json.dumps(think_funcs))
                    except Exception as _log_err:
                        logger.debug(f"Think.functions payload preview failed: {_log_err}")

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
            logger.info(f"📤 Sent {len(pcm16_bytes)} bytes of audio to Deepgram Agent")
        except Exception as e:
            logger.error(f"Agent send error: {e}")

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


