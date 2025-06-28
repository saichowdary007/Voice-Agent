#!/usr/bin/env python3
"""
FastAPI Web Server for the Voice Agent.
This replaces the CLI interface for Docker/production deployment.
"""
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, validator
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

# Import Voice Agent modules
try:
    from src.config import USE_REALTIME_STT, WHISPER_MODEL
    if USE_REALTIME_STT:
        logger.info(f"🚀 Server-side STT enabled (faster-whisper, model='{WHISPER_MODEL}')")
        try:
            from src.stt import STT
            # Use the same Whisper model name from config (default: 'tiny')
            stt_instance = STT(model_size=WHISPER_MODEL, device="auto")
            logger.info(f"✅ Faster-whisper STT initialized with model '{WHISPER_MODEL}'")
        except Exception as e:
            logger.error(f"❌ Failed to load faster-whisper STT: {e}")
            USE_REALTIME_STT = False
            stt_instance = None
    else:
        logger.info("🚀 Using Web Speech API only (no server-side STT)")
        stt_instance = None
except ImportError:
    logger.warning("⚠️ STT configuration not found, defaulting to browser Web Speech API")
    USE_REALTIME_STT = False
    stt_instance = None

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

# Initialize TTS engine
try:
    from src.tts import TTS
    tts_engine = TTS()
    logger.info("✅ TTS engine initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize TTS engine: {e}")
    tts_engine = None

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

# Import VAD
from src.vad import VAD  # Voice Activity Detection

# Global VAD instance (set during startup)
vad_instance = None

# FastAPI app
app = FastAPI(
    title="Voice Agent API",
    description="Real-time voice interaction API with 3D audio visualization",
    version="1.0.0"
)

# API Router
api_router = APIRouter(prefix="/api")

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

    @validator('text')
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
            "stt": "available" if USE_REALTIME_STT else "unavailable",
            "llm": "available" if llm_interface else "unavailable", 
            "tts": "available" if tts_engine else "unavailable",
            "database": "connected" if USE_SUPABASE else "disabled",
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

from src.websocket_handlers import (
    handle_text_message,
    handle_audio_chunk,
    handle_ping,
    handle_vad_status,
    handle_start_listening,
    handle_stop_listening,
    handle_heartbeat,
    handle_unknown_message,
)

# WebSocket endpoint for real-time voice interaction
@app.websocket("/ws/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str):
    """
    WebSocket endpoint for real-time voice interaction.
    Handles audio streaming and real-time messaging with actual STT/LLM/TTS processing.
    """
    try:
        logger.info(f"🔌 New WebSocket connection attempt with token: {token[:20]}...")
        
        # CRITICAL: Accept the WebSocket connection FIRST before any other operations
        await websocket.accept()
        logger.info("✅ WebSocket connection accepted")
        
        # -----------------------------------
        # Authenticate user (token path param)
        # -----------------------------------
        if token.startswith("guest_"):
            # Guest / unauthenticated connection – use token as pseudo user_id
            user_id = token
            logger.info(f"🎭 Guest connection established: {user_id}")
        else:
            try:
                user = await get_current_user(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))
                user_id = user["id"]
                logger.info(f"🔐 Authenticated connection for user: {user_id}")
            except HTTPException as e:
                # Enhanced error logging for WebSocket authentication failures
                logger.error(f"WebSocket authentication failed for token: {token[:20]}... Error: {e}")
                
                # Try auth manager fallback (includes DEBUG_MODE support)
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
        # Send a lightweight welcome message **before** heavy initialization
        # -----------------------------------
        welcome_message = {
            "type": "connection",
            "status": "connected",
            "message": "Voice Agent WebSocket connected",
            "services": {
                "stt": USE_REALTIME_STT,
                "llm": llm_interface is not None,
                "tts": tts_engine is not None,
            },
        }

        try:
            await websocket.send_json(welcome_message)
            logger.info(f"📨 Welcome message sent to user_id: {user_id}")
        except ConnectionClosedError:
            logger.warning(
                f"Failed to send welcome message to user_id: {user_id} (connection closed)"
            )
            return

        # -----------------------------------
        # Register the connection (may trigger heavier operations like model load)
        # -----------------------------------
        await connection_manager.connect(websocket, user_id)
        logger.info(f"✅ WebSocket connected successfully for user_id: {user_id}")
        
        try:
            conversation_mgr = connection_manager.get_conversation_manager(websocket)
            
            # Ensure ConversationManager exists when Supabase integration is expected
            if USE_SUPABASE and conversation_mgr is None:
                logger.error("ConversationManager could not be initialised – aborting WebSocket session")
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                return
            
            message_handlers = {
                "text_message": handle_text_message,
                "audio_chunk": handle_audio_chunk,
                "ping": handle_ping,
                "vad_status": handle_vad_status,
                "start_listening": handle_start_listening,
                "stop_listening": handle_stop_listening,
                "heartbeat": handle_heartbeat,
            }
            
            while True:
                # Receive message from client
                try:
                    # Check connection state before receiving
                    if websocket.client_state != WebSocketState.CONNECTED:
                        logger.info("WebSocket connection no longer active, breaking message loop")
                        break
                        
                    data = await websocket.receive_text()
                    is_text_message = True
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected normally")
                    break
                except ConnectionClosedError:
                    logger.info("WebSocket connection closed")
                    break
                except Exception as e:
                    # If text receive fails, try binary
                    try:
                        data = await websocket.receive_bytes()
                        is_text_message = False
                    except Exception as binary_error:
                        logger.error(f"Failed to receive WebSocket message: {e}, binary attempt: {binary_error}")
                        # Terminate the loop on unrecoverable receive errors
                        break
                
                try:
                    if is_text_message:
                        message = json.loads(data)
                    else:
                        # Handle binary data (convert to base64 for audio processing)
                        message = {
                            "type": "audio_chunk",
                            "data": base64.b64encode(data).decode("ascii"),
                            "is_binary": True
                        }
                    
                    message_type = message.get("type", "unknown")
                    handler = message_handlers.get(message_type, handle_unknown_message)
                    
                    if message_type == "text_message":
                        await handler(websocket, message, conversation_mgr, llm_interface, tts_engine)
                    elif message_type == "audio_chunk":
                        await handler(websocket, message, conversation_mgr, stt_instance, llm_interface, tts_engine, vad_instance)
                    else:

                        await handler(websocket, message)
                        
                except json.JSONDecodeError:
                    await connection_manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "message": "Invalid JSON format"
                        }),
                        websocket
                    )
                    
        except WebSocketDisconnect:
            connection_manager.disconnect(websocket)
            logger.info(f"WebSocket disconnected for user: {user_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            connection_manager.disconnect(websocket)

    except Exception as e:
        logger.error(f"WebSocket error: {e}")

# Application lifecycle events
@app.on_event("startup")
async def startup_event():
    """Initialize application resources on startup."""
    logger.info("🚀 Voice Agent API starting up...")
    
    # Validate critical services
    if not USE_REALTIME_STT:
        logger.warning("⚠️ STT engine not available")
    if not llm_interface:
        logger.warning("⚠️ LLM engine not available")
    if not tts_engine:
        logger.warning("⚠️ TTS engine not available")
    if not auth_manager:
        logger.warning("⚠️ Auth manager not available")
    
    logger.info("✅ Voice Agent API startup complete")

    # Initialize VAD
    global vad_instance
    vad_instance = VAD(sample_rate=16000, mode=2)
    logger.info("✅ VAD initialized (WebRTC mode 2)")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
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

# Server startup
if __name__ == "__main__":
    import multiprocessing
    # Fix multiprocessing issues on macOS
    multiprocessing.freeze_support()
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting Voice Agent API server on {host}:{port}")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=os.getenv("RELOAD", "false").lower() == "true",
        ws_ping_interval=20,
        ws_ping_timeout=10
    ) 