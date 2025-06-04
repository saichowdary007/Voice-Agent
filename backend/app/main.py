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

# Assuming services are in backend.services package relative to the project root
from backend.services.vad_service import VADService
from backend.services.stt_service import STTService
from backend.services.tts_service import TTSService
from backend.services.llm_service import LLMService
from backend.services.audio_service import AudioService

from .websocket_handler import WebSocketHandler
from .utils.logging import setup_logging # Assuming this setup structlog
from .utils.metrics import MetricsCollector
from .config import settings


# Global state, initialized in lifespan
services: Dict[str, Any] = {}
metrics: Optional[MetricsCollector] = None
logger: Optional[structlog.BoundLogger] = None # Define type for global logger

class SessionManager:
    """Manage WebSocket sessions with state tracking"""
    
    def __init__(self):
        self.sessions: Dict[WebSocket, Dict[str, Any]] = {}
        self.max_sessions = settings.max_concurrent_sessions
        # Ensure logger is available for SessionManager methods if called before app logger is bound
        self._logger = structlog.get_logger(self.__class__.__name__) 
    
    async def create_session(self, websocket: WebSocket) -> bool:
        """Create a new session"""
        if len(self.sessions) >= self.max_sessions:
            self._logger.warning(f"Max concurrent sessions ({self.max_sessions}) reached. Rejecting new session for {websocket.client.host}.")
            return False
        
        session_id = f"session_{int(time.time() * 1000)}_{websocket.client.host}_{websocket.client.port}"
        
        self.sessions[websocket] = {
            "id": session_id,
            "created_at": time.time(),
            "websocket_obj": websocket # Store websocket for potential direct access if needed
        }
        
        self._logger.info(f"Created session {session_id} for client {websocket.client.host}:{websocket.client.port}. Active sessions: {len(self.sessions)}")
        return True
    
    def get_session_id(self, websocket: WebSocket) -> Optional[str]:
        """Get session ID for a given WebSocket object."""
        session_data = self.sessions.get(websocket)
        return session_data["id"] if session_data else None

    def remove_session(self, websocket: WebSocket):
        """Remove session"""
        session_data = self.sessions.pop(websocket, None)
        if session_data:
            self._logger.info(f"Removed session {session_data['id']}. Active sessions: {len(self.sessions)}")
        else:
            self._logger.warning("Attempted to remove a non-existent session.") # Should ideally not happen
    
    def get_active_sessions_count(self) -> int: # Renamed for clarity
        """Get number of active sessions"""
        return len(self.sessions)

session_manager = SessionManager() # Instantiate the manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global services, metrics, logger # Allow modification of global variables
    
    # Setup logging as the very first step
    # The log_level from settings will be used by setup_logging
    setup_logging(log_level=settings.log_level)
    logger = structlog.get_logger("app.main_lifespan") # Get a logger instance
    
    logger.info("Application startup sequence initiated...", 
                app_name=settings.app_name, 
                app_version=settings.app_version,
                log_level=settings.log_level)
    
    # Initialize metrics collector
    if settings.enable_metrics:
        metrics = MetricsCollector()
        logger.info("Metrics collection enabled.")
    else:
        metrics = None # Explicitly None if disabled
        logger.info("Metrics collection disabled.")
    
    logger.info("Initializing AI and Audio services...")
    try:
        # Create service instances
        services['vad'] = VADService(threshold=settings.vad_threshold)
        services['stt'] = STTService()
        services['tts'] = TTSService()
        services['llm'] = LLMService()
        services['audio'] = AudioService(sample_rate=settings.sample_rate, channels=settings.channels)
        
        # Asynchronously initialize all services
        init_coroutines = []
        for service_name, service_instance in services.items():
            if hasattr(service_instance, 'initialize') and callable(service_instance.initialize):
                init_coroutines.append(service_instance.initialize())
        
        results = await asyncio.gather(*init_coroutines, return_exceptions=True)
        
        # Detailed status logging after initialization attempts
        services_status_summary = {}
        all_services_ready = True
        for (service_name, service_instance), result in zip(services.items(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to initialize service '{service_name}': {result}", exc_info=result)
                services_status_summary[service_name] = {"available": False, "error": str(result)}
                all_services_ready = False
            else:
                # Check is_available or get_status after successful initialize()
                status = False
                if hasattr(service_instance, 'is_available'): status = service_instance.is_available
                elif hasattr(service_instance, 'get_status'): status = service_instance.get_status().get('available', False)
                services_status_summary[service_name] = {"available": status}
                if not status: all_services_ready = False
                logger.info(f"Service '{service_name}' initialized. Available: {status}")

        if not all_services_ready:
            logger.warning("One or more services failed to initialize. Application may run in a degraded state or fail if critical services are down.")
            # Consider raising an exception here if certain services are absolutely critical for startup
            # For example: if 'audio' service fails, it might be unrecoverable.
            # if not services.get('audio') or not services['audio'].is_available:
            #     raise RuntimeError("Critical AudioService failed to initialize. Application cannot start.")

        # Preload models if enabled (and if services support it)
        if settings.preload_models:
            logger.info("Attempting to preload AI models...")
            preload_coroutines = []
            for service_name, service_instance in services.items():
                if hasattr(service_instance, 'preload') and callable(service_instance.preload):
                     if hasattr(service_instance, 'is_available') and service_instance.is_available: # Only preload if service is available
                        preload_coroutines.append(service_instance.preload())
                     elif not hasattr(service_instance, 'is_available'): # If no is_available, assume preload can be tried
                        preload_coroutines.append(service_instance.preload())

            if preload_coroutines:
                await asyncio.gather(*preload_coroutines, return_exceptions=True) # Handle preload errors too
            logger.info("Model preloading process completed (if applicable for services).")
            
    except Exception as e:
        logger.critical(f"A critical error occurred during service initialization: {e}", exc_info=True)
        # This exception will prevent the app from starting if it occurs here.
        raise
    
    logger.info("Application startup complete. Listening for requests...")
    yield # Application runs here
    
    # Shutdown sequence
    logger.info("Application shutdown sequence initiated...")
    cleanup_coroutines = []
    for service_name, service_instance in services.items():
        if hasattr(service_instance, 'cleanup') and callable(service_instance.cleanup):
            cleanup_coroutines.append(service_instance.cleanup())
    
    if cleanup_coroutines:
        results = await asyncio.gather(*cleanup_coroutines, return_exceptions=True)
        for (service_name, _), result in zip(services.items(), results):
            if isinstance(result, Exception):
                logger.error(f"Error during cleanup of service '{service_name}': {result}", exc_info=result)
            else:
                logger.info(f"Service '{service_name}' cleaned up successfully.")
    logger.info("Application shutdown complete.")


app = FastAPI(
    title=settings.app_name,
    description="Ultra-Fast Voice Agent with Gemini Flash, Sherpa-NCNN, and Piper TTS",
    version=settings.app_version,
    lifespan=lifespan # Use the new lifespan context manager
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins if hasattr(settings, 'cors_allowed_origins') else ["*"], # Make configurable
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", summary="Health check for the voice agent service")
async def health_check_endpoint(): # Renamed to avoid conflict
    """Provides the operational status of the application and its services."""
    if logger is None: # Should be initialized by lifespan
        # Fallback basic logging if structlog isn't ready
        logging.error("Health check: Global logger not initialized!")
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": "Server logger not available"})

    try:
        service_status_details = {}
        all_services_operational = True
        for name, service_instance in services.items():
            status_info = {"available": False, "details": "Status method not found"}
            if hasattr(service_instance, 'get_status') and callable(service_instance.get_status):
                try:
                    status_detail_dict = service_instance.get_status()
                    status_info = status_detail_dict # Assumes get_status returns a dict
                    if not status_info.get('available', False): # Check for 'available' key
                        all_services_operational = False
                except Exception as e:
                    logger.warning(f"Error getting status for service '{name}': {e}")
                    status_info = {"available": False, "error": str(e)}
                    all_services_operational = False
            elif hasattr(service_instance, 'is_available'): # Fallback
                is_avail = service_instance.is_available
                status_info = {"available": is_avail}
                if not is_avail: all_services_operational = False
            
            service_status_details[name] = status_info

        current_memory_mb = metrics.get_memory_usage() if metrics else 0.0
        
        response_content = {
            "status": "healthy" if all_services_operational else "degraded",
            "timestamp": time.time(),
            "app_version": settings.app_version,
            "active_sessions": session_manager.get_active_sessions_count(),
            "memory_usage_mb": round(current_memory_mb, 2),
            "services": service_status_details,
            "config_summary": {
                "sample_rate": settings.sample_rate,
                "channels": settings.channels,
                "audio_frame_ms": settings.audio_frame_ms,
                "max_concurrent_sessions": settings.max_concurrent_sessions,
                "metrics_enabled": settings.enable_metrics,
                "log_level": settings.log_level,
                "preload_models": settings.preload_models
            }
        }
        return JSONResponse(status_code=200 if all_services_operational else 503, content=response_content)
    except Exception as e:
        logger.error(f"Health check endpoint failed critically: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": f"Internal server error: {str(e)}"})


@app.get("/metrics", summary="Application performance metrics (if enabled)")
async def metrics_endpoint():
    """Exports collected performance metrics."""
    if not settings.enable_metrics or metrics is None:
        raise HTTPException(status_code=404, detail="Metrics collection is disabled or not available.")
    # Consider Prometheus format output here if that's the target system
    return JSONResponse(content=metrics.get_all_metrics())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time voice agent communication."""
    # Logger should be initialized by lifespan by now.
    # Fallback for safety, though this indicates a deeper startup issue if logger is None.
    app_logger = logger if logger else structlog.get_logger("app.websocket_fallback")

    await websocket.accept() # Accept connection first
    
    if not await session_manager.create_session(websocket):
        app_logger.warning("Session creation failed (e.g., max sessions). Closing WebSocket.")
        await websocket.close(code=1013, reason="Server overloaded or unable to create session.")
        return
    
    session_id = session_manager.get_session_id(websocket)
    if not session_id: # Should not happen if create_session succeeded
        app_logger.error("Critical: Session ID not found after successful session creation. Closing WebSocket.")
        await websocket.close(code=1011, reason="Internal server error: Session ID missing.")
        return
        
    # Create a logger instance bound with session-specific context
    session_logger = app_logger.bind(session_id=session_id, client_host=websocket.client.host)
    session_logger.info(f"WebSocket connection accepted.")

    handler = WebSocketHandler(
        websocket=websocket,
        services=services,
        metrics=metrics, 
        logger=session_logger # Pass the new session-bound logger
    )
    
    try:
        # Send initial status/config to client only if WebSocket is still connected
        if websocket.client_state == WebSocketState.CONNECTED:
            await handler._send_message({
                "type": "status",
                "session_id": session_id, # Send the generated session_id to client
                "ready": True,
                "config": { # Send relevant backend config to client
                    "sample_rate": settings.sample_rate,
                    "frame_duration_ms": settings.audio_frame_ms,
                    "channels": settings.channels
                }
            })
        else: # Should not happen immediately after accept, but good check
            session_logger.warning("WebSocket disconnected before initial status message could be sent.")
            # Cleanup will be handled in finally block
            return

        await handler.handle_connection() # Main message handling loop
        
    except WebSocketDisconnect:
        session_logger.info(f"Client disconnected WebSocket gracefully.")
    except ConnectionResetError:
        session_logger.warning(f"Client connection reset abruptly.")
    except Exception as e: # Catch any other unhandled exceptions from handle_connection
        session_logger.error(f"Unhandled error during WebSocket communication: {str(e)}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await handler._send_error(f"A critical server error occurred. Please try reconnecting.")
            except Exception as send_err:
                session_logger.error(f"Failed to send final error message to client: {send_err}", exc_info=True)
    finally:
        session_logger.info(f"Performing final cleanup for WebSocket session...")
        await handler.cleanup() # Ensure handler's internal cleanup is called
        session_manager.remove_session(websocket) # Remove from global session manager
        
        # Final log to confirm closure and state, checking client_state one last time
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                # If not already closed by handler or client, attempt a graceful close
                await websocket.close(code=1000)
                session_logger.info("WebSocket closed from main finally block.")
            except Exception as close_exc:
                session_logger.warning(f"Exception during final WebSocket close attempt: {close_exc}", exc_info=True)
        
        session_logger.info(f"Session processing finished. Final active sessions: {session_manager.get_active_sessions_count()}")


@app.get("/", summary="Root endpoint providing basic service information")
async def root_endpoint(): # Renamed to avoid conflict
    """Returns basic information about the voice agent service."""
    if logger: logger.debug("Root endpoint '/' accessed.")
    return {
        "service_name": settings.app_name,
        "version": settings.app_version,
        "status": "API is running",
        "active_sessions": session_manager.get_active_sessions_count(),
        "websocket_endpoint": "/ws",
        "health_endpoint": "/health",
        "metrics_endpoint": "/metrics (if enabled)"
    }


if __name__ == "__main__":
    # This block is for running the app directly with `python -m backend.app.main`
    # Uvicorn's `reload=True` might interfere with `lifespan` context on reloads.
    # For development, `uvicorn backend.app.main:app --reload` is often preferred.
    
    # Ensure logging is set up if this script is run directly and not via uvicorn CLI that might handle it.
    # The lifespan manager should handle this, but as a safeguard for direct script execution:
    if logger is None:
        setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO").upper())
        # Assign to global logger so it's available if lifespan didn't run (e.g. import error before lifespan)
        logger = structlog.get_logger("app.main_direct_run_fallback") 
        logger.info("Running Uvicorn server directly from __main__...")

    uvicorn.run(
        "backend.app.main:app", # Path to the FastAPI app instance
        host=settings.ws_server_host,
        port=settings.ws_server_port,
        reload=os.getenv("UVICORN_RELOAD", "true").lower() == 'true',
        log_level=settings.log_level.lower(), # Uvicorn's native log level
        # Use default uvicorn logger or configure uvicorn logging separately if needed
        # It's generally better to let structlog handle app logs and uvicorn handle access logs.
    )