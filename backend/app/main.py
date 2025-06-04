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

# Ensure correct import paths if services are directly under backend/services
from backend.services.vad_service import VADService
from backend.services.stt_service import STTService
from backend.services.tts_service import TTSService
from backend.services.llm_service import LLMService
from backend.services.audio_service import AudioService # Assuming audio_service is the correct one

from .websocket_handler import WebSocketHandler
from .utils.logging import setup_logging
from .utils.metrics import MetricsCollector
from .config import settings


# Global state
services: Dict[str, Any] = {}
metrics: Optional[MetricsCollector] = None
logger: Optional[structlog.BoundLogger] = None # Define logger type for clarity

class SessionManager:
    """Manage WebSocket sessions with state tracking"""
    
    def __init__(self):
        self.sessions: Dict[WebSocket, Dict[str, Any]] = {}
        self.max_sessions = settings.max_concurrent_sessions
    
    async def create_session(self, websocket: WebSocket) -> bool:
        """Create a new session"""
        if logger is None: 
            logging.error("Logger not initialized in SessionManager") 
            return False

        if len(self.sessions) >= self.max_sessions:
            logger.warning(f"Max sessions ({self.max_sessions}) reached, rejecting new session.")
            return False
        
        # Generate a more unique session ID including part of client info
        session_id = f"session_{int(time.time() * 1000)}_{websocket.client.host}_{websocket.client.port}"
        
        self.sessions[websocket] = {
            "id": session_id,
            "created_at": time.time(),
        }
        
        logger.info(f"Created session {session_id} for {websocket.client.host}:{websocket.client.port}")
        return True
    
    def get_session(self, websocket: WebSocket) -> Optional[Dict[str, Any]]:
        """Get session data"""
        return self.sessions.get(websocket)
    
    def remove_session(self, websocket: WebSocket):
        """Remove session"""
        session = self.sessions.pop(websocket, None)
        if session and logger:
            logger.info(f"Removed session {session['id']}")
    
    def get_active_sessions(self) -> int:
        """Get number of active sessions"""
        return len(self.sessions)

session_manager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global services, metrics, logger
    
    setup_logging(log_level=settings.log_level)
    logger = structlog.get_logger("app.main") # Ensure logger is structlog bound logger
    
    if settings.enable_metrics:
        metrics = MetricsCollector()
        logger.info("Metrics collection enabled.")
    else:
        logger.info("Metrics collection disabled.")
    
    logger.info("Initializing AI services...", app_name=settings.app_name, app_version=settings.app_version)
    
    try:
        services['vad'] = VADService(threshold=settings.vad_threshold)
        services['stt'] = STTService()
        services['tts'] = TTSService()
        services['llm'] = LLMService()
        # Ensure 'audio' key is used if that's what WebSocketHandler expects
        services['audio'] = AudioService(sample_rate=settings.sample_rate) 
        
        init_tasks = [srv.initialize() for srv_name, srv in services.items() if hasattr(srv, 'initialize')]
        await asyncio.gather(*init_tasks)
        
        services_status_summary = {name: (srv.is_available if hasattr(srv, 'is_available') else srv.get_status().get('available', False) if hasattr(srv, 'get_status') else 'unknown') for name, srv in services.items()}
        available_services = sum(1 for status in services_status_summary.values() if status is True) # Count True statuses
        total_services = len(services_status_summary)
        
        logger.info(f"Services initialized: {available_services}/{total_services} available.")
        for service_name, status in services_status_summary.items():
            logger.info(f"  {service_name}: {'✅ Available' if status is True else ('❌ Not Available' if status is False else '❓ Unknown')}")
        
        if available_services < total_services:
            logger.warning("Some services failed to initialize - application running in degraded mode.")

        if settings.preload_models:
            logger.info("Preloading models...")
            preload_tasks = [srv.preload() for srv_name, srv in services.items() if hasattr(srv, 'preload')]
            if preload_tasks:
                await asyncio.gather(*preload_tasks)
            logger.info("Models preloading complete (if applicable).")
            
    except Exception as e:
        logger.error(f"Critical failure during service initialization: {e}", exc_info=True)
        raise # Reraise to indicate critical startup failure, preventing app from starting misconfigured
    
    yield
    
    logger.info("Shutting down services...")
    cleanup_tasks = [srv.cleanup() for srv_name, srv in services.items() if hasattr(srv, 'cleanup')]
    if cleanup_tasks:
        results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        for (srv_name, srv_instance), result in zip(services.items(), results):
            if isinstance(result, Exception):
                logger.error(f"Error cleaning up service {srv_name}: {result}", exc_info=result)
            else:
                logger.info(f"Service {srv_name} cleaned up successfully.")
    logger.info("Application shutdown complete.")


app = FastAPI(
    title=settings.app_name,
    description="Service-oriented voice assistant with Gemini Flash, sherpa-ncnn, and Piper TTS",
    version=settings.app_version,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Consider restricting this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    if logger is None: 
        logging.error("Logger not initialized for health_check") # Basic fallback
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": "logger not available"})

    try:
        service_status_details = {}
        all_ready = True
        for name, service_instance in services.items():
            status = "unknown"
            is_srv_ready = False
            if hasattr(service_instance, 'get_status') and callable(service_instance.get_status):
                status_detail = service_instance.get_status()
                service_status_details[name] = status_detail
                is_srv_ready = status_detail.get('available', False) if isinstance(status_detail, dict) else False
            elif hasattr(service_instance, 'is_available'):
                is_srv_ready = service_instance.is_available
                service_status_details[name] = {"available": is_srv_ready}
            else:
                 service_status_details[name] = {"status": "unknown (no status method)"}
            
            if not is_srv_ready:
                all_ready = False
        
        current_memory_usage = metrics.get_memory_usage() if metrics else 0
        
        return JSONResponse(
            status_code=200 if all_ready else 503,
            content={
                "status": "healthy" if all_ready else "degraded",
                "timestamp": time.time(),
                "app_version": settings.app_version,
                "services": service_status_details,
                "active_sessions": session_manager.get_active_sessions(),
                "memory_usage_mb": round(current_memory_usage, 2),
                "config_summary": {
                    "sample_rate": settings.sample_rate,
                    "max_sessions": settings.max_concurrent_sessions,
                    "metrics_enabled": settings.enable_metrics,
                    "log_level": settings.log_level
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
async def get_metrics_endpoint(): # Renamed to avoid conflict with global 'metrics'
    if not settings.enable_metrics or metrics is None:
        raise HTTPException(status_code=404, detail="Metrics not enabled or not available.")
    
    # Example: Return as Prometheus format or JSON
    # For Prometheus, you'd typically use a dedicated library.
    # Here's a simple JSON representation:
    return JSONResponse(content=metrics.get_all_metrics())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global logger # Ensure we are using the initialized global logger
    if logger is None:
        # This case should ideally not be hit if lifespan initializes logger correctly
        logging.basicConfig(level="ERROR") # Basic fallback logging
        logging.error("Critical: Global logger not initialized at websocket_endpoint entry.")
        await websocket.accept() # Accept to send error, then close
        await websocket.close(code=1011, reason="Server logger not initialized")
        return

    await websocket.accept()
    
    if not await session_manager.create_session(websocket):
        logger.warning("Failed to create session (max sessions or other issue). Closing WebSocket.")
        await websocket.close(code=1013, reason="Server overloaded or session creation failed")
        return
    
    session_data = session_manager.get_session(websocket)
    if not session_data: 
        logger.error("Session data not found immediately after creation. This should not happen.")
        await websocket.close(code=1011, reason="Internal server error: session data unavailable post-creation")
        return
        
    session_id = session_data["id"]
    # Create a session-specific logger
    session_logger = logger.bind(session_id=session_id, client_ip=websocket.client.host) 
    session_logger.info(f"WebSocket connection accepted.")

    handler = WebSocketHandler(
        websocket=websocket,
        services=services,
        metrics=metrics, # Pass the global metrics collector
        logger=session_logger # Pass the session-specific logger
    )
    
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
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
        
    except WebSocketDisconnect: # More specific exception for client disconnects
        session_logger.info(f"WebSocket client disconnected.")
    except ConnectionResetError:
        session_logger.warning(f"WebSocket connection reset by client.")
    except Exception as e:
        session_logger.error(f"Unhandled error in WebSocket connection: {str(e)}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await handler._send_error(f"An unexpected server error occurred: {str(e)}")
            except Exception as send_err_exc:
                session_logger.error(f"Failed to send error to client after WebSocket error: {send_err_exc}", exc_info=True)
    finally:
        session_logger.info(f"Performing cleanup for WebSocket session.")
        await handler.cleanup() # Ensure handler's cleanup is called
        session_manager.remove_session(websocket)
        session_logger.info(f"WebSocket connection closed and session removed. Active sessions: {session_manager.get_active_sessions()}")


@app.get("/")
async def root():
    global logger
    if logger: # Check if logger is initialized
        logger.info("Root endpoint accessed.")
    return {
        "service_name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "active_sessions": session_manager.get_active_sessions(),
        "documentation_url": "/docs", # Point to Swagger/OpenAPI docs
        "health_check_url": "/health"
    }


if __name__ == "__main__":
    # This block is for direct execution (e.g., python -m app.main)
    # Ensure logging is set up even if not run via uvicorn command that sets log level
    if logger is None: # If not already set up by lifespan (e.g. if uvicorn isn't managing lifespan)
        setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO").upper())
        logger = structlog.get_logger("app.main.direct_run")
        logger.info("Running Uvicorn server directly...")

    uvicorn.run(
        "app.main:app", # Make sure this matches your app path
        host=settings.ws_server_host,
        port=settings.ws_server_port,
        reload=os.getenv("UVICORN_RELOAD", "true").lower() == 'true',
        log_level=settings.log_level.lower(), # Uvicorn's own log level
        # Removed 'logger' from uvicorn.run as structlog handles it
    )