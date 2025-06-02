import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Set, Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketState
import structlog

from ..services.vad_service import VADService
from ..services.stt_service import STTService
from ..services.tts_service import TTSService
from ..services.llm_service import LLMService
from ..services.audio_service import AudioService

from .websocket_handler import WebSocketHandler
from .utils.logging import setup_logging
from .utils.metrics import MetricsCollector
from .config import settings


# Global state
services: Dict[str, Any] = {}
metrics: Optional[MetricsCollector] = None
logger: Optional[structlog.BoundLogger] = None

class SessionManager:
    """Manage WebSocket sessions with state tracking"""
    
    def __init__(self):
        self.sessions: Dict[WebSocket, Dict[str, Any]] = {}
        self.max_sessions = settings.max_concurrent_sessions
    
    async def create_session(self, websocket: WebSocket) -> bool:
        """Create a new session"""
        if logger is None: # Should not happen after lifespan
            logging.error("Logger not initialized in SessionManager") # fallback logging
            return False

        if len(self.sessions) >= self.max_sessions:
            logger.warning(f"Max sessions ({self.max_sessions}) reached")
            return False
        
        session_id = f"session_{int(time.time() * 1000)}"
        
        # Simplified initial session state, WebSocketHandler will manage more details
        self.sessions[websocket] = {
            "id": session_id,
            "created_at": time.time(),
        }
        
        logger.info(f"Created session {session_id}")
        return True
    
    def get_session(self, websocket: WebSocket) -> Optional[Dict[str, Any]]:
        """Get session data"""
        return self.sessions.get(websocket)
    
    def remove_session(self, websocket: WebSocket):
        """Remove session"""
        session = self.sessions.pop(websocket, None)
        if session and logger: # Check logger for safety
            logger.info(f"Removed session {session['id']}")
    
    def get_active_sessions(self) -> int:
        """Get number of active sessions"""
        return len(self.sessions)

session_manager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global services, metrics, logger
    
    # Setup logging
    setup_logging(log_level=settings.log_level)
    logger = structlog.get_logger("app.main")
    
    # Initialize metrics
    if settings.enable_metrics:
        metrics = MetricsCollector()
    
    # Initialize services
    logger.info("Initializing AI services...", app_name=settings.app_name, app_version=settings.app_version)
    
    try:
        # Create service instances
        services['vad'] = VADService(threshold=settings.vad_threshold)
        services['stt'] = STTService()
        services['tts'] = TTSService()
        services['llm'] = LLMService()
        services['audio'] = AudioService(sample_rate=settings.sample_rate)
        
        # Initialize all services
        init_tasks = [
            services['vad'].initialize(),
            services['stt'].initialize(),
            services['tts'].initialize(),
            services['llm'].initialize(),
            services['audio'].initialize()
        ]
        
        await asyncio.gather(*init_tasks)
        
        # Check service status
        services_status_summary = {name: srv.is_available for name, srv in services.items() if hasattr(srv, 'is_available')}
        available_services = sum(services_status_summary.values())
        total_services = len(services_status_summary)
        
        logger.info(f"Services initialized: {available_services}/{total_services}")
        for service_name, status in services_status_summary.items():
            logger.info(f"  {service_name}: {'✅' if status else '❌'}")
        
        if available_services < total_services:
            logger.warning("Some services failed to initialize - running in degraded mode")

        # Preload models if enabled (assuming services have a preload method)
        if settings.preload_models:
            logger.info("Preloading models...")
            preload_tasks = []
            for srv_name, srv_instance in services.items():
                if hasattr(srv_instance, 'preload'):
                    preload_tasks.append(srv_instance.preload())
            if preload_tasks:
                await asyncio.gather(*preload_tasks)
            logger.info("Models preloaded successfully (if preload methods exist and were called)")
            
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=True)
        # Depending on severity, might want to prevent app startup
        # For now, it will start in a degraded state if some services fail
        raise # Reraise to indicate critical startup failure
    
    yield
    
    # Cleanup
    logger.info("Shutting down services...")
    for service_name, service_instance in services.items():
        if hasattr(service_instance, 'cleanup'):
            try:
                await service_instance.cleanup()
                logger.info(f"Service {service_name} cleaned up.")
            except Exception as e:
                logger.error(f"Error cleaning up service {service_name}: {e}", exc_info=True)


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Service-oriented voice assistant with Gemini Flash, sherpa-ncnn, and Piper TTS",
    version=settings.app_version,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    if logger is None: # Should not happen
        logging.error("Logger not initialized for health_check")
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": "logger not available"})

    try:
        # Check if all services are ready
        service_status_details = {}
        all_ready = True
        for name, service_instance in services.items():
            if hasattr(service_instance, 'get_status'):
                status_detail = service_instance.get_status()
                service_status_details[name] = status_detail
                # Assuming get_status returns a dict with a 'ready' boolean or similar
                if isinstance(status_detail, dict) and not status_detail.get('ready', True): # Default to True if no ready field
                    all_ready = False
            elif hasattr(service_instance, 'is_available'): # Fallback to is_available
                is_avail = service_instance.is_available
                service_status_details[name] = {"available": is_avail}
                if not is_avail:
                    all_ready = False
            else:
                service_status_details[name] = {"status": "unknown"}


        current_memory_usage = metrics.get_memory_usage() if metrics else 0
        
        return JSONResponse(
            status_code=200 if all_ready else 503,
            content={
                "status": "healthy" if all_ready else "degraded",
                "services": service_status_details,
                "active_sessions": session_manager.get_active_sessions(),
                "memory_usage_mb": current_memory_usage,
                "config_summary": {
                    "sample_rate": settings.sample_rate,
                    "max_sessions": settings.max_concurrent_sessions,
                    "metrics_enabled": settings.enable_metrics
                }
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/metrics")
async def get_metrics():
    """Metrics endpoint for monitoring"""
    if not metrics: # metrics can be None if not enabled
        raise HTTPException(status_code=503, detail="Metrics not available (disabled or not initialized)")
    
    return JSONResponse(content=metrics.get_all_metrics())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for voice agent communication"""
    if logger is None: # Should not happen
        logging.error("Logger not initialized for websocket_endpoint")
        # Cannot send WebSocket message if logger failed, so close abnormally
        await websocket.close(code=1011) 
        return

    if not await session_manager.create_session(websocket):
        await websocket.close(code=1013, reason="Server overloaded or session creation failed")
        return
    
    session_data = session_manager.get_session(websocket)
    if not session_data: # Should not happen if create_session succeeded
        logger.error("Session data not found after creation.")
        await websocket.close(code=1011, reason="Internal server error creating session")
        return
        
    session_id = session_data["id"]
    # Create a session-specific logger by re-binding the main logger with session_id
    session_logger = logger.bind(session_id=session_id) 
    session_logger.info(f"WebSocket connected", client=str(websocket.client))

    # Pass the global services, metrics, and a session-specific logger to the handler
    # The WebSocketHandler will manage its own detailed session state.
    handler = WebSocketHandler(
        websocket=websocket,
        services=services,
        metrics=metrics,
        logger=session_logger
    )
    
    try:
        # Send initial status (ready, config) to client
        # This could be moved into WebSocketHandler.handle_connection if preferred
        await handler._send_message({
            "type": "status",
            "session_id": session_id,
            "ready": True,
            "config": {
                "sample_rate": settings.sample_rate,
                "frame_duration_ms": settings.audio_frame_ms
            }
        })

        await handler.handle_connection()
        
    except WebSocketDisconnect:
        session_logger.info(f"WebSocket client disconnected")
    except ConnectionResetError: # Specific error for abrupt client disconnects
        session_logger.warning(f"WebSocket connection reset by client")
    except Exception as e:
        # Log with session_id for better traceability
        session_logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        try:
            # Try to send a clean error message to the client
            if websocket.client_state == WebSocketState.CONNECTED: # Ensure WebSocket is still connected
                 await handler._send_error(f"Internal server error: {str(e)}")
        except Exception as send_exc:
            session_logger.error(f"Failed to send error to client after WebSocket error", send_error_exc_info=send_exc)
            pass 
    finally:
        session_logger.info(f"Cleaning up WebSocket session")
        await handler.cleanup()
        session_manager.remove_session(websocket)
        session_logger.info(f"WebSocket connection closed. Active sessions: {session_manager.get_active_sessions()}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service_name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "active_sessions": session_manager.get_active_sessions(),
        "documentation": "/docs"
    }


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "app.main:app",
        host=settings.ws_server_host,
        port=settings.ws_server_port,
        reload=os.getenv("UVICORN_RELOAD", "true").lower() == 'true',
        log_level=settings.log_level.lower()
    ) 