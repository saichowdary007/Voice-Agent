#!/usr/bin/env python3
"""
Single-file FastAPI backend that bridges the frontend WebSocket to Deepgram's
Voice Agent API. Deepgram handles STT, TTS, and LLM orchestration.

Runtime: uvicorn server:app --host 0.0.0.0 --port 8000

Env Vars:
  - DEEPGRAM_API_KEY      (required)   e.g. dg_xxx
  - LLM_PROVIDER          (optional)   e.g. gemini, openai (we use gemini)
  - LLM_API_KEY           (optional)   if using provider above
  - GEMINI_API_KEY        (optional)   when using gemini provider (recommended)
  - LLM_MODEL             (optional)   e.g. gpt-4o-mini or gemini-2.0-flash
  - DG_STT_MODEL          (optional)   default nova-3
  - DG_TTS_MODEL          (optional)   default aura-2-thalia-en
  - AGENT_GREETING        (optional)   greeting string
  - SUPABASE_URL          (optional)
  - SUPABASE_ANON_KEY     (optional)
  - SUPABASE_TABLE        (optional)

HTTP Endpoints:
  - GET  /health               → status
  - GET  /healthz              → {"ok": true}
  - POST /api/auth/debug-signin → returns a guest token for local dev

WebSocket Endpoints:
  - WS /ws/{token}             → existing frontend compatibility
  - WS /ws?user_id=<uuid>      → acceptance criteria compatibility

Frontend contract preserved: connection + settings_applied + tts_wav + heartbeats.
"""
import os
import asyncio
import json
import base64
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_AGENT_ENDPOINT = os.getenv("DEEPGRAM_AGENT_ENDPOINT", "wss://agent.deepgram.com/v1")
AUDIO_OUTPUT_SAMPLE_RATE = int(os.getenv("AUDIO_OUTPUT_SAMPLE_RATE", 24000))

if not DEEPGRAM_API_KEY:
    logger.warning("DEEPGRAM_API_KEY is not set. Set it in your environment before running.")


# -----------------------------------------------------------------------------
# Minimal utilities
# -----------------------------------------------------------------------------
def wav_from_pcm16_mono(pcm_data: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM16 mono data into a WAV container."""
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


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="Voice Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    user: Dict[str, str]
    expires_at: int


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "message": "Voice Agent is running",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    }


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/api/auth/debug-signin", response_model=AuthResponse)
async def debug_sign_in(body: Dict[str, str]):
    """Simple local dev auth that returns a guest token the frontend can use."""
    email = body.get("email") if isinstance(body, dict) else "guest@example.com"
    token = f"guest_{int(time.time())}"
    expires_at = int((datetime.utcnow() + timedelta(hours=1)).timestamp())
    return {
        "access_token": token,
        "refresh_token": "",
        "user": {"id": token, "email": email or "guest@example.com", "created_at": datetime.utcnow().isoformat()},
        "expires_at": expires_at,
    }


# -----------------------------------------------------------------------------
# Deepgram Agent proxy (inline, single-file)
# -----------------------------------------------------------------------------
class AgentProxy:
    def __init__(self, client_ws: WebSocket):
        self.client_ws = client_ws
        self.agent_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.running = False
        self.tasks = []
        self._tts_pcm = bytearray()
        self._last_pcm_ms: Optional[float] = None
        self._flush_task: Optional[asyncio.Task] = None
        self._agent_ready: bool = False  # becomes True after SettingsApplied

    async def start(self) -> bool:
        if not DEEPGRAM_API_KEY:
            await self.client_ws.send_json({"type": "error", "message": "DEEPGRAM_API_KEY not set"})
            return False
        try:
            self.agent_ws = await websockets.connect(
                DEEPGRAM_AGENT_ENDPOINT,
                subprotocols=["token", DEEPGRAM_API_KEY],
                ping_interval=None,
                close_timeout=5,
            )
            self.running = True
            self.tasks = [
                asyncio.create_task(self._client_to_agent()),
                asyncio.create_task(self._agent_to_client()),
            ]
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram Agent: {e}")
            return False

    async def stop(self):
        self.running = False
        for t in self.tasks:
            if not t.done():
                t.cancel()
        try:
            if self.agent_ws:
                await self.agent_ws.close()
        except Exception:
            pass

    async def _client_to_agent(self):
        try:
            while self.running:
                msg = await self.client_ws.receive()
                if msg.get("type") != "websocket.receive":
                    continue
                if "bytes" in msg and msg["bytes"] is not None:
                    # Drop early audio until SettingsApplied to avoid agent errors
                    if not self._agent_ready:
                        continue
                    if self.agent_ws:
                        await self.agent_ws.send(msg["bytes"])
                elif "text" in msg and msg["text"]:
                    try:
                        data = json.loads(msg["text"])  # pass-through JSON
                    except json.JSONDecodeError:
                        continue
                    mtype = data.get("type", "")
                    # app-level heartbeats
                    if mtype == "heartbeat":
                        await self.client_ws.send_json({"type": "heartbeat_ack", "timestamp": datetime.utcnow().isoformat()})
                        continue
                    if mtype == "ping":
                        await self.client_ws.send_json({"type": "pong", "timestamp": data.get("timestamp", datetime.utcnow().isoformat())})
                        continue
                    if mtype == "connection":
                        await self.client_ws.send_json({"type": "connection_ack", "message": "OK", "timestamp": datetime.utcnow().isoformat()})
                        continue
                    if mtype in ("audio_chunk", "AudioData"):
                        b64 = data.get("data")
                        if isinstance(b64, str) and b64:
                            raw = base64.b64decode(b64)
                            if self.agent_ws:
                                await self.agent_ws.send(raw)
                            await self.client_ws.send_json({"type": "audio_ack", "timestamp": datetime.utcnow().isoformat()})
                        continue
                    # Intercept Settings to ensure sensible defaults
                    if mtype in ("settings", "Settings"):
                        try:
                            agent = data.setdefault("agent", {})
                            # Greeting default
                            if "greeting" not in agent:
                                agent["greeting"] = "Hello, I'm your AI assistant."

                            # Ensure speak is a list; add optional cartesia fallback when present
                            speak = agent.get("speak")
                            if not isinstance(speak, list):
                                # normalize single object to list or create default primary
                                if isinstance(speak, dict):
                                    speak_list = [speak]
                                else:
                                    speak_list = [{"provider": {"type": "deepgram", "model": "aura-2-thalia-en"}}]
                                agent["speak"] = speak_list
                            # Append cartesia fallback when configured
                            if os.getenv("CARTESIA_API_KEY"):
                                if not any((isinstance(p, dict) and p.get("provider", {}).get("type") == "cartesia") for p in agent["speak"]):
                                    agent["speak"].append({"provider": {"type": "cartesia", "model": "sonic-english"}})

                            # Prefer Google Gemini provider if requested; rely on Deepgram's native integration
                            think = agent.setdefault("think", {})
                            provider = think.setdefault("provider", {})
                            ptype = str(provider.get("type", "")).lower()
                            # Do NOT inject custom endpoint; Deepgram handles Gemini payload/transport.
                        except Exception:
                            # Non-fatal; forward original
                            pass
                    # Forward only supported message types to agent
                    if self.agent_ws and mtype in ("settings", "Settings", "Speak"):
                        await self.agent_ws.send(json.dumps(data))
        except Exception as e:
            logger.debug(f"client_to_agent end: {e}")
        finally:
            self.running = False

    async def _flush_after_quiet(self):
        try:
            quiet_ms = 350
            while self.running:
                await asyncio.sleep(0.15)
                if self._last_pcm_ms is None or not self._tts_pcm:
                    continue
                if (time.time() * 1000.0) - self._last_pcm_ms >= quiet_ms:
                    await self._emit_wav()
                    break
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def _emit_wav(self):
        try:
            wav = wav_from_pcm16_mono(bytes(self._tts_pcm), AUDIO_OUTPUT_SAMPLE_RATE)
            b64 = base64.b64encode(wav).decode("ascii")
            await self.client_ws.send_json({"type": "tts_wav", "data": b64, "sample_rate": AUDIO_OUTPUT_SAMPLE_RATE})
        finally:
            self._tts_pcm.clear()
            self._last_pcm_ms = None
            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()

    async def _agent_to_client(self):
        try:
            assert self.agent_ws is not None
            while self.running:
                msg = await self.agent_ws.recv()
                if isinstance(msg, (bytes, bytearray)):
                    self._tts_pcm.extend(bytes(msg))
                    self._last_pcm_ms = time.time() * 1000.0
                    if self._flush_task is None or self._flush_task.done():
                        self._flush_task = asyncio.create_task(self._flush_after_quiet())
                    continue
                # JSON
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                etype = data.get("type", "")
                if etype == "Welcome":
                    await self.client_ws.send_json({"type": "connection_ack", "message": "Connected to Deepgram Voice Agent", "request_id": data.get("request_id")})
                    continue
                if etype == "SettingsApplied":
                    self._agent_ready = True
                    await self.client_ws.send_json({"type": "settings_applied", "timestamp": datetime.utcnow().isoformat()})
                    continue
                if etype == "ConversationText":
                    await self.client_ws.send_json({
                        "type": "agent_text",
                        "role": data.get("role", "assistant"),
                        "content": data.get("content", ""),
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                    continue
                if etype in ("AgentAudioDone", "AgentAudioCompleted", "AgentTtsDone"):
                    if self._tts_pcm:
                        await self._emit_wav()
                    else:
                        await self.client_ws.send_json({"type": "agent_audio_done", "timestamp": datetime.utcnow().isoformat()})
                    continue
                if etype in ("AgentErrors", "AgentWarnings", "Error", "Warning"):
                    # Surface provider errors clearly (e.g., Gemini JSON payload issues)
                    await self.client_ws.send_json({
                        "type": "error",
                        "message": data.get("message") or data.get("description") or "Agent error",
                        "provider": data.get("provider"),
                        "code": data.get("code") or data.get("status"),
                        "debug": data.get("details") or data.get("error"),
                    })
                    continue
                # pass-through unknowns for debugging
                try:
                    await self.client_ws.send_json(data)
                except Exception:
                    pass
        except websockets.exceptions.ConnectionClosed:
            logger.info("Deepgram Agent connection closed")
        except Exception as e:
            logger.debug(f"agent_to_client end: {e}")
        finally:
            self.running = False


# -----------------------------------------------------------------------------
# WS endpoint
# -----------------------------------------------------------------------------
@app.websocket("/ws/{token}")
async def ws_endpoint(ws: WebSocket, token: str):
    try:
        await ws.accept()
        if not token:
            await ws.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        # initial connection message (client expects it before sending Settings)
        try:
            await ws.send_json({"type": "connection", "message": "Voice Agent server connected", "timestamp": datetime.utcnow().isoformat()})
        except Exception:
            pass

        proxy = AgentProxy(ws)
        ok = await proxy.start()
        if not ok:
            await ws.send_json({"type": "error", "message": "Failed to connect to Deepgram Voice Agent"})
            await ws.close(code=status.WS_1011_INTERNAL_ERROR)
            return

        await asyncio.gather(*proxy.tasks, return_exceptions=True)
    except WebSocketDisconnect:
        logger.info("Client WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.websocket("/ws")
async def ws_endpoint_query(ws: WebSocket, user_id: Optional[str] = Query(default=None)):
    """Alternate WS endpoint accepting ?user_id= for compatibility."""
    try:
        await ws.accept()
        # initial message
        try:
            await ws.send_json({"type": "connection", "message": "Voice Agent server connected", "timestamp": datetime.utcnow().isoformat()})
        except Exception:
            pass

        proxy = AgentProxy(ws)
        ok = await proxy.start()
        if not ok:
            await ws.send_json({"type": "error", "message": "Failed to connect to Deepgram Voice Agent"})
            await ws.close(code=status.WS_1011_INTERNAL_ERROR)
            return
        await asyncio.gather(*proxy.tasks, return_exceptions=True)
    except WebSocketDisconnect:
        logger.info("Client WebSocket disconnected (/ws query)")
    except Exception as e:
        logger.error(f"WebSocket error (/ws query): {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=os.getenv("RELOAD", "false").lower() == "true",
        ws_ping_interval=None,
        ws_ping_timeout=None,
        ws_max_size=2 * 1024 * 1024,
    )
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix multiprocessing issues on macOS
import platform
if platform.system() == "Darwin":
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)

import asyncio
import logging
import io
import base64
import time
from typing import Optional, Union, Dict, List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, status, APIRouter
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, field_validator
import uvicorn
import json
from datetime import datetime
from starlette.websockets import WebSocketState
try:
    from fastapi.websockets import ConnectionClosedError
except ImportError:
    # Fallback for older FastAPI versions
    from starlette.websockets import WebSocketDisconnect as ConnectionClosedError

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import WebSocket handlers from dedicated module

from src.config import USE_DEEPGRAM_AGENT

stt_instance = None
tts_engine = None

from src.config import (
    USE_SUPABASE, 
    ALLOWED_ORIGINS, 
    DEBUG_MODE, 
    IS_PRODUCTION, 
    MAX_MESSAGE_LENGTH,
    MAX_REQUESTS_PER_MINUTE
)

# Initialize Voice Agent components
try:
    from src.llm import LLM
    llm_interface = LLM()
    logger.info("✅ LLM interface initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize LLM interface: {e}")
    llm_interface = None

# TTS handled by Deepgram Agent when enabled

# Import conversation manager
try:
    from src.conversation import ConversationManager
    logger.info("✅ ConversationManager imported")
except Exception as e:
    logger.error(f"❌ Failed to import ConversationManager: {e}")

# Import auth manager
try:
    from src.auth import AuthManager
    auth_manager = AuthManager()
    logger.info("✅ Auth manager initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize Auth manager: {e}")
    auth_manager = None

# VAD is handled by Deepgram Agent internally

# Application lifecycle management using modern lifespan pattern
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    # Startup
    logger.info("🚀 Voice Agent API starting up...")
    
    # Validate critical services
    # STT/TTS handled by Deepgram Agent; legacy engines disabled
    if not llm_interface:
        logger.warning("⚠️ LLM engine not available")
    # TTS handled by Deepgram Agent
    if not auth_manager:
        logger.warning("⚠️ Auth manager not available")
    
    logger.info("✅ Voice Agent API startup complete")

    # VAD is handled by Deepgram Agent internally
    logger.info("✅ Voice Agent startup complete - VAD handled by Deepgram Agent")
    
    yield
    
    # Shutdown
    logger.info("🛑 Voice Agent API shutting down...")
    
    try:
        # Clean up WebSocket connections
        await connection_manager.cleanup_all()
        
        # Clean up LLM session
        if llm_interface:
            await llm_interface.cleanup_session()
            
        logger.info("✅ Resource cleanup complete")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")
    
    logger.info("👋 Voice Agent API shutdown complete")

# FastAPI app
app = FastAPI(
    title="Voice Agent API",
    description="Real-time voice interaction API with 3D audio visualization",
    version="1.0.0",
    lifespan=lifespan
)

# API Router
api_router = APIRouter(prefix="/api")

# --- [BEGIN Deepgram Agent wiring] ---
from deepgram import DeepgramClient
from src.deepgram_settings import build_settings

dg_router = APIRouter(prefix="/api/dg", tags=["deepgram"])

class GrantTokenBody(BaseModel):
    ttl_seconds: Optional[int] = 30

@dg_router.post("/token")
def grant_token(body: GrantTokenBody):
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="DEEPGRAM_API_KEY not set")
    try:
        client = DeepgramClient(api_key=api_key)
        resp = client.auth.v("1").grant_token(ttl_seconds=body.ttl_seconds or 30)
        return {"access_token": resp.access_token, "expires_in": resp.expires_in}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Deepgram grant failed: {e}")

@dg_router.get("/settings")
def get_settings():
    return build_settings()

app.include_router(dg_router)
# --- [END Deepgram Agent wiring] ---

# CORS middleware - SECURITY FIX: Use environment-configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://voice-agent.vercel.app",   # prod
                   "http://localhost:3000",           # dev (React default)
                   "http://localhost:3001",           # dev (React alt port)
                   "http://127.0.0.1:3000",          # dev (alternative localhost)
                   "http://127.0.0.1:3001"],         # dev (alternative localhost)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models with improved validation
class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    version: str

class MessageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH, description="Message text")
    language: str = Field(default="en", pattern="^[a-z]{2}(-[A-Z]{2})?$", description="Language code")
    user_id: Optional[str] = Field(None, max_length=100)

    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        # Basic sanitization - remove null bytes and control characters
        cleaned = ''.join(char for char in v if ord(char) >= 32 or char in '\n\r\t')
        return cleaned

class MessageResponse(BaseModel):
    text: str
    language: str
    timestamp: str

# Authentication models with validation
class SignUpRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128, description="Password must be 8-128 characters")

class SignInRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)

class AuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    user: dict
    expires_at: int

class UserResponse(BaseModel):
    id: str
    email: str
    created_at: str

class RefreshRequest(BaseModel):
    refresh_token: str

# -----------------------------
# Helpers
# -----------------------------

def _get_expires_at(session_obj):
    """Return epoch-seconds expiry for demo or Supabase session objects."""
    if not session_obj:
        return int(datetime.utcnow().timestamp()) + 3600

    exp = getattr(session_obj, "expires_at", None)
    if exp is None:
        return int(datetime.utcnow().timestamp()) + 3600

    # If already numeric
    if isinstance(exp, (int, float)):
        return int(exp)

    # If datetime
    try:
        from datetime import datetime as _dt
        if isinstance(exp, _dt):
            return int(exp.timestamp())
    except Exception:
        pass

    return int(datetime.utcnow().timestamp()) + 3600

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return current user."""
    if not auth_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available"
        )
    
    token = credentials.credentials
    try:
        user_obj = auth_manager.verify_token(token)
        if not user_obj:
            raise ValueError("Invalid token")
        return {"id": getattr(user_obj, "id", "unknown"), "email": getattr(user_obj, "email", "unknown")}
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# In-memory storage for active WebSocket connections and their conversation managers
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, str] = {}  # websocket -> user_id
        self.conversation_managers: Dict[str, ConversationManager] = {}  # user_id -> conversation_manager
        self._cleanup_lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, user_id: str):
        """Connect a websocket and initialize conversation manager if needed."""
        # NOTE: WebSocket should already be accepted before calling this method
        self.active_connections[websocket] = user_id
        
        # Initialize conversation manager if Supabase is enabled
        if USE_SUPABASE and user_id not in self.conversation_managers:
            try:
                self.conversation_managers[user_id] = ConversationManager(user_id=user_id)
            except Exception as e:
                logger.error(f"Failed to create ConversationManager for {user_id}: {e}")

    def disconnect(self, websocket: WebSocket):
        """Disconnect a websocket and clean up resources."""
        user_id = self.active_connections.pop(websocket, None)
        if user_id:
            # Deepgram Agent handles state management internally
            logger.debug(f"🔄 WebSocket disconnected for user {user_id}")
            
            # Clean up conversation manager if no other connections for this user
            user_connections = [ws for ws, uid in self.active_connections.items() if uid == user_id]
            if not user_connections and user_id in self.conversation_managers:
                del self.conversation_managers[user_id]
                logger.info(f"Cleaned up ConversationManager for user {user_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific websocket."""
        try:
            # Check if websocket is still connected before sending
            if websocket.client_state != WebSocketState.CONNECTED:
                logger.warning("Attempted to send message to disconnected websocket")
                return False
                
            await websocket.send_text(message)
            logger.debug(f"Successfully sent message to websocket")
            return True
        except ConnectionClosedError:
            logger.warning("WebSocket connection closed while sending message")
            self.disconnect(websocket)
            return False
        except WebSocketDisconnect:
            logger.warning("WebSocket disconnected while sending message")
            self.disconnect(websocket)
            return False
        except Exception as e:
            logger.error(f"Failed to send message to websocket: {e}")
            self.disconnect(websocket)
            return False

    async def send_to_websocket(self, websocket: WebSocket, message: dict):
        """Send a JSON message to a specific websocket."""
        try:
            # Check if websocket is still connected before sending
            if websocket.client_state != WebSocketState.CONNECTED:
                logger.warning("Attempted to send message to disconnected websocket")
                return False
                
            await websocket.send_json(message)
            logger.debug(f"Successfully sent JSON message to websocket")
            return True
        except ConnectionClosedError:
            logger.warning("WebSocket connection closed while sending message")
            self.disconnect(websocket)
            return False
        except WebSocketDisconnect:
            logger.warning("WebSocket disconnected while sending message")
            self.disconnect(websocket)
            return False
        except Exception as e:
            logger.error(f"Failed to send JSON message to websocket: {e}")
            self.disconnect(websocket)
            return False

    async def broadcast(self, message: str):
        """Send a message to all connected websockets."""
        if not self.active_connections:
            return
        
        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to websocket: {e}")
                disconnected.append(websocket)
        
        # Clean up disconnected websockets
        for websocket in disconnected:
            self.disconnect(websocket)

    def get_conversation_manager(self, websocket: WebSocket) -> Optional[ConversationManager]:
        """Get the conversation manager for a specific websocket."""
        user_id = self.active_connections.get(websocket)
        if user_id and USE_SUPABASE:
            return self.conversation_managers.get(user_id)
        return None

    async def cleanup_all(self):
        """Clean up all connections and resources."""
        async with self._cleanup_lock:
            # Close all websockets
            for websocket in list(self.active_connections.keys()):
                try:
                    await websocket.close(code=status.WS_1012_SERVICE_RESTART)
                except Exception as e:
                    logger.error(f"Error closing websocket: {e}")
            
            # Clear all data
            self.active_connections.clear()
            self.conversation_managers.clear()
            logger.info("All WebSocket connections closed.")

connection_manager = ConnectionManager()

# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Provides a health check endpoint for monitoring."""
    return {
        "status": "ok",
        "message": "Voice Agent is running",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@api_router.post("/auth/signup", response_model=Union[AuthResponse, Dict[str, str]])
async def sign_up(request: SignUpRequest):
    """
    Register a new user using Supabase or demo mode.
    """
    if not auth_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available"
        )
    
    try:
        session = auth_manager.sign_up(request.email, request.password)
        
        if session:
            logger.info(f"User registered successfully: {request.email}")

            # If we already have valid tokens (demo mode), return them right away
            if getattr(session, "access_token", None):
                expires_at = _get_expires_at(session)
                return AuthResponse(
                    access_token=session.access_token,
                    refresh_token=session.refresh_token or "",
                    user={
                        "id": session.user.id,
                        "email": session.user.email,
                        "created_at": session.user.created_at
                    },
                    expires_at=expires_at
                )

            return {"message": "User registered successfully. Please check your email for verification."}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Registration failed - user may already exist or invalid credentials"
            )
    except Exception as e:
        error_message = str(e).lower()
        logger.error(f"Registration error: {e}")
        
        # Handle specific Supabase errors
        if "rate limit" in error_message:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many signup attempts. Please wait a few minutes and try again."
            )
        elif "user already registered" in error_message or "already exists" in error_message:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email already exists. Please try signing in instead."
            )
        elif "invalid email" in error_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please provide a valid email address."
            )
        elif "password" in error_message and ("weak" in error_message or "short" in error_message):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters long."
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to sign up: {str(e)}"
            )

@api_router.post("/auth/signin", response_model=AuthResponse)
async def sign_in(request: SignInRequest):
    """
    Authenticate a user and return a JWT session.
    """
    if not auth_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available"
        )
    
    try:
        session = auth_manager.sign_in(request.email, request.password)
        
        if session and session.user:
            # Use real expiration if available
            expires_at = _get_expires_at(session)
            
            return AuthResponse(
                access_token=session.access_token,
                refresh_token=session.refresh_token or "",
                user={
                    "id": session.user.id,
                    "email": session.user.email,
                    "created_at": session.user.created_at
                },
                expires_at=expires_at
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
    except Exception as e:
        error_message = str(e).lower()
        logger.error(f"Authentication error: {e}")
        
        # Handle specific authentication errors
        if "rate limit" in error_message:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many login attempts. Please wait a few minutes and try again."
            )
        elif "email not confirmed" in error_message or "not verified" in error_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please verify your email address before signing in."
            )
        elif "invalid" in error_message and ("email" in error_message or "password" in error_message):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"An unexpected error occurred during sign-in: {e}"
            )

@api_router.post("/auth/signout")
async def sign_out(current_user: dict = Depends(get_current_user)):
    """
    Sign out the current user by invalidating their session.
    """
    if not auth_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available"
        )
    
    try:
        auth_manager.sign_out()
        logger.info(f"User signed out: {current_user.get('email', 'unknown')}")
        return {"message": "Successfully signed out"}
    except Exception as e:
        logger.error(f"Sign out error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during sign out: {e}"
        )

@api_router.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Fetches information about the currently authenticated user."""
    return current_user

@api_router.post("/auth/refresh", response_model=AuthResponse)
async def refresh_token_endpoint(request: RefreshRequest):
    """
    Refresh an expired access token using a refresh token.
    """
    if not auth_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available"
        )

    try:
        session = auth_manager.refresh_session(request.refresh_token)

        if session and session.user:
            expires_at = _get_expires_at(session)
            return AuthResponse(
                access_token=session.access_token,
                refresh_token=session.refresh_token or "",
                user={
                    "id": session.user.id,
                    "email": session.user.email,
                    "created_at": session.user.created_at
                },
                expires_at=expires_at
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during token refresh: {e}"
        )

@api_router.post("/auth/debug-signin", response_model=AuthResponse)
async def debug_sign_in(request: SignInRequest):
    """
    A debug endpoint to get a valid session without Supabase.
    """
    # SECURITY FIX: Only allow in debug mode
    if not DEBUG_MODE:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Endpoint not found"
        )
    
    if IS_PRODUCTION:
        logger.warning(f"Debug endpoint accessed in production environment from {request.email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Debug endpoints disabled in production"
        )
    
    try:
        # Create a mock session for development
        from src.auth import MockUser, MockSession
        
        demo_user = MockUser(request.email)
        demo_session = MockSession(demo_user)
        
        expires_at = int(demo_session.expires_at.timestamp())
        
        logger.info(f"Debug auth session created for: {request.email}")
        
        return AuthResponse(
            access_token=demo_session.access_token,
            refresh_token=demo_session.refresh_token,
            user={
                "id": demo_user.id,
                "email": demo_user.email,
                "created_at": demo_user.created_at
            },
            expires_at=expires_at
        )
        
    except Exception as e:
        logger.error(f"Debug authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Debug authentication failed"
        )

@api_router.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest, current_user: dict = Depends(get_current_user)):
    """
    Processes a text message and returns a spoken response.
    This is a simplified endpoint for non-streaming interaction.
    """
    try:
        if not llm_interface:
            raise HTTPException(status_code=503, detail="LLM service not available")
            
        user_id = current_user.get('id', 'unknown')
        
        # Get conversation context if available
        history = []
        profile_facts = []
        
        if USE_SUPABASE:
            try:
                conversation_mgr = ConversationManager(user_id=user_id)
                history, profile_facts = await asyncio.gather(
                    conversation_mgr.get_context_for_llm(request.text),
                    conversation_mgr.get_user_profile()
                )
            except Exception as e:
                logger.warning(f"Failed to get conversation context: {e}")
        
        # Generate AI response using actual LLM
        ai_response = await llm_interface.generate_response(
            user_text=request.text,
            conversation_history=history,
            user_profile=profile_facts
        )
        
        # Save conversation if Supabase is enabled
        if USE_SUPABASE:
            try:
                conversation_mgr = ConversationManager(user_id=user_id)
                await conversation_mgr.add_message("user", request.text)
                await conversation_mgr.add_message("model", ai_response)
                
                # Extract and update user profile facts
                new_facts = await llm_interface.extract_facts(f"User: {request.text}\nAI: {ai_response}")
                if new_facts:
                    await conversation_mgr.update_user_profile(new_facts)
                
                # Handle personal fact storage pipeline
                await conversation_mgr.handle_user_turn(request.text, ai_response, llm_interface)
            except Exception as e:
                logger.warning(f"Failed to save conversation: {e}")
        
        return MessageResponse(
            text=ai_response,
            language=request.language,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@api_router.get("/status")
async def get_status():
    """
    Provides the current status of the agent, including active connections.
    """
    auth_mode = "disabled"
    auth_details = {}
    
    if auth_manager:
        if getattr(auth_manager, 'demo_mode', False):
            auth_mode = "demo"
            auth_details = {
                "mode": "demo",
                "supabase_configured": False,
                "note": "Running in demo mode for testing"
            }
        else:
            auth_mode = "supabase"
            auth_details = {
                "mode": "supabase",
                "supabase_configured": True,
                "database_enabled": USE_SUPABASE
            }
    
    return {
        "status": "ready",
        "active_connections": len(connection_manager.active_connections),
        "services": {
            "agent": True,
            "llm": llm_interface is not None,
            "database": USE_SUPABASE,
            "auth": auth_mode
        },
        "authentication": auth_details,
        "endpoints": {
            "auth_signup": "/auth/signup",
            "auth_signin": "/auth/signin", 
            "auth_debug": "/auth/debug-signin",
            "health": "/health",
            "docs": "/docs"
        },
        "tts_engine_status": "ready" if tts_engine else "error",
        "auth_manager_status": "ready" if auth_manager else "error",
    }

app.include_router(api_router)

# Import handlers for compatibility (not used in proxy mode)
from src.websocket_handlers import (
    handle_ping,
    handle_heartbeat,
    handle_connection_message,
    handle_unknown_message,
)

# WebSocket endpoint for real-time voice interaction
@app.websocket("/ws/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str):
    """
    WebSocket endpoint for Deepgram Voice Agent proxy.
    Proxies client connections directly to Deepgram's Voice Agent API.
    """
    try:
        logger.info(f"🔌 New WebSocket connection attempt with token: {token[:20]}...")
        
        # Accept WebSocket connection
        await websocket.accept()
        logger.info("✅ WebSocket connection accepted")
        
        # -----------------------------------
        # Authenticate user (token path param)
        # -----------------------------------
        if token.startswith("guest_"):
            user_id = token
            logger.info(f"🎭 Guest connection established: {user_id}")
        else:
            try:
                user = await get_current_user(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))
                user_id = user["id"]
                logger.info(f"🔐 Authenticated connection for user: {user_id}")
            except HTTPException as e:
                logger.error(f"WebSocket authentication failed for token: {token[:20]}... Error: {e}")
                
                # Try auth manager fallback
                if auth_manager:
                    try:
                        auth_user = auth_manager.verify_token(token)
                        if auth_user:
                            user_id = auth_user.id
                            logger.info(f"WebSocket authentication succeeded via auth_manager fallback for user: {auth_user.email}")
                        else:
                            logger.error("WebSocket authentication failed: auth_manager.verify_token returned None")
                            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                            return
                    except Exception as fallback_error:
                        logger.error(f"WebSocket authentication fallback failed: {fallback_error}")
                        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                        return
                else:
                    logger.error("WebSocket authentication failed: auth_manager is None")
                    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                    return

        # -----------------------------------
        # Register the connection
        # -----------------------------------
        await connection_manager.connect(websocket, user_id)
        logger.info(f"✅ WebSocket connected successfully for user_id: {user_id}")
        
        # -----------------------------------
        # Notify client connection established so it can send Settings
        # -----------------------------------
        try:
            await websocket.send_json({
                "type": "connection",
                "message": "Voice Agent server connected",
                "timestamp": datetime.utcnow().isoformat(),
            })
        except Exception:
            pass

        # -----------------------------------
        # Start Deepgram Voice Agent proxy (feature-flagged)
        # -----------------------------------
        from src.websocket_handlers import handle_websocket_connection
        if USE_DEEPGRAM_AGENT:
            await handle_websocket_connection(websocket, user_id=user_id)
        else:
            # Legacy path not supported in this build
            await websocket.send_json({
                "type": "error",
                "message": "Deepgram agent disabled (USE_DEEPGRAM_AGENT=false) and no legacy pipeline available.",
                "timestamp": datetime.utcnow().isoformat(),
            })
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            return
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up connection
        connection_manager.disconnect(websocket)
        from src.websocket_handlers import cleanup_websocket_connection
        await cleanup_websocket_connection(websocket)

# Server startup
if __name__ == "__main__":
    import multiprocessing
    # Fix multiprocessing issues on macOS
    multiprocessing.freeze_support()
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting Voice Agent API server on {host}:{port}")
    
    # FIX #2: Disable ping timeouts to prevent 1011 errors during processing
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=os.getenv("RELOAD", "false").lower() == "true",
        ws_ping_interval=None,  # Disable automatic pings
        ws_ping_timeout=None,   # Disable ping timeouts
        ws_max_size=2 * 1024 * 1024  # FIX #6: 2MB max message size for audio bursts
    ) 

# -----------------------------------------------------------------------------
# Compatibility alias: /voice-agent/{user_id}
# Treats {user_id} as the token for authentication/backcompat.
# -----------------------------------------------------------------------------
@app.websocket("/voice-agent/{user_id}")
async def websocket_endpoint_voice_agent(websocket: WebSocket, user_id: str):
    # Treat provided user_id as a guest token to allow quick-start integration
    guest_token = f"guest_{user_id}"
    await websocket_endpoint(websocket, token=guest_token)