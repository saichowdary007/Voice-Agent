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

# Setup logging first
from app.utils.logging import setup_logging
from app.config import settings

# Initialize logging
setup_logging(settings.log_level)
logger = structlog.get_logger("app.main_lifespan")

# Import services after logging is set up
from services.vad_service import VADService
from services.stt_service import STTService
from services.tts_service import TTSService
from services.llm_service import LLMService
from services.audio_service import AudioService
from app.utils.metrics import MetricsCollector
from app.websocket_handler import WebSocketHandler

# Global state, initialized in lifespan
services: Dict[str, Any] = {}
metrics: Optional[MetricsCollector] = None

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
    """Startup and shutdown hooks for the FastAPI application"""
    # Create application-wide service containers
    services = {}
    metrics = MetricsCollector() if settings.enable_metrics else None
    
    try:
        logger.info(
            "Application startup sequence initiated...",
            app_name=settings.app_name,
            app_version=settings.app_version,
            log_level=settings.log_level
        )
        
        # Set up metrics collection if enabled
        if metrics:
            logger.info("Metrics collection enabled.")
            
        # Initialize services
        logger.info("Initializing AI and Audio services...")
        
        # Initialize VAD service
        try:
            services['vad'] = VADService(threshold=settings.vad_threshold)
            await services['vad'].initialize()
        except Exception as vad_e:
            logger.critical(f"A critical error occurred initializing VAD service: {vad_e}", exc_info=True)
            services['vad'] = None
        
        # Initialize STT service
        try:
            services['stt'] = STTService(api_key=settings.azure_speech_key, region=settings.azure_speech_region)
            await services['stt'].initialize()
        except Exception as stt_e:
            logger.critical(f"A critical error occurred initializing STT service: {stt_e}", exc_info=True)
            services['stt'] = None
        
        # Initialize TTS service - using default constructor
        try:
            services['tts'] = TTSService()
            await services['tts'].initialize()
            logger.info(f"TTS Service initialized")
        except Exception as tts_e:
            logger.critical(f"A critical error occurred initializing TTS service: {tts_e}", exc_info=True)
            services['tts'] = None
        
        # Initialize LLM service - using default constructor
        try:
            services['llm'] = LLMService()
            await services['llm'].initialize()
            print(f"LLM Service initialized")
        except Exception as llm_e:
            logger.critical(f"A critical error occurred initializing LLM service: {llm_e}", exc_info=True)
            services['llm'] = None
        
        # Initialize Audio service
        try:
            services['audio'] = AudioService(
                sample_rate=settings.sample_rate,
                channels=settings.channels
            )
            await services['audio'].initialize()
            print(f"AudioService configured: SR={settings.sample_rate}Hz, Chan={settings.channels}")
        except Exception as audio_e:
            logger.critical(f"A critical error occurred initializing Audio service: {audio_e}", exc_info=True)
            services['audio'] = None
        
        # Check service availability
        for service_name, service in services.items():
            available = service is not None and getattr(service, 'is_available', True)
            logger.info(f"Service '{service_name}' initialized. Available: {available}")
        
        # Try to preload AI models if supported by services
        logger.info("Attempting to preload AI models...")
        
        # Preload VAD model
        if services.get('vad') and hasattr(services['vad'], 'preload'):
            try:
                await services['vad'].preload()
            except Exception as e:
                logger.warning(f"Failed to preload VAD model: {e}")
        
        logger.info("Model preloading process completed (if applicable for services).")
        
        # Pass the initialized services to the application state
        app.state.services = services
        app.state.metrics = metrics
        app.state.session_manager = SessionManager()
        
        logger.info("Application startup complete. Listening for requests...")
        
        yield
        
        # Shutdown and cleanup
        logger.info("Application shutdown initiated...")
        
        # Close and clean up services
        for service_name, service in services.items():
            if service and hasattr(service, 'cleanup'):
                logger.info(f"Cleaning up {service_name} service...")
                try:
                    # Handle both sync and async cleanup methods
                    if asyncio.iscoroutinefunction(service.cleanup):
                        await service.cleanup()
                    else:
                        service.cleanup()
                    logger.info(f"{service_name} service cleaned up successfully")
                except Exception as e:
                    logger.error(f"Error cleaning up {service_name} service: {e}")
        
        if metrics:
            metrics.cleanup()
            
        logger.info("Application shutdown complete.")
        
    except Exception as e:
        logger.critical(f"A critical error occurred during service initialization: {e}", exc_info=True)
        # We still yield to allow the app to start, but it may be in a degraded state
        yield


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
        for name, service_instance in app.state.services.items():
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

        current_memory_mb = app.state.metrics.get_memory_usage() if app.state.metrics else 0.0
        
        # Check all services and determine if they're ready
        essential_services = ['vad', 'stt', 'audio']
        essential_services_ready = True
        
        for name in essential_services:
            if name == 'audio':
                # Special case for audio service which uses 'overall_available'
                if not service_status_details.get(name, {}).get('overall_available', False):
                    essential_services_ready = False
                    break
            else:
                # Standard check for other services
                if not service_status_details.get(name, {}).get('available', False):
                    essential_services_ready = False
                    break
        
        # If all essential services are ready, report healthy
        status = "healthy" if essential_services_ready else "degraded"
        
        response_content = {
            "status": status,
            "timestamp": time.time(),
            "app_version": settings.app_version,
            "active_sessions": app.state.session_manager.get_active_sessions_count(),
            "memory_usage_mb": round(current_memory_mb, 2),
            "services": service_status_details,
            "config_summary": {
                "sample_rate": settings.sample_rate,
                "channels": settings.channels,
                "audio_frame_ms": settings.frame_duration_ms,
                "max_concurrent_sessions": settings.max_concurrent_sessions,
                "metrics_enabled": settings.enable_metrics,
                "log_level": settings.log_level,
                "preload_models": settings.preload_models
            }
        }
        
        # Only return 503 if truly unhealthy
        return JSONResponse(
            status_code=200 if status == "healthy" else 503, 
            content=response_content
        )
    except Exception as e:
        logger.error(f"Health check endpoint failed critically: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": f"Internal server error: {str(e)}"})


@app.get("/metrics", summary="Application performance metrics (if enabled)")
async def metrics_endpoint():
    """Exports collected performance metrics."""
    if not settings.enable_metrics or app.state.metrics is None:
        raise HTTPException(status_code=404, detail="Metrics collection is disabled or not available.")
    # Consider Prometheus format output here if that's the target system
    return JSONResponse(content=app.state.metrics.get_all_metrics())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time voice agent communication."""
    # Logger should be initialized by lifespan by now.
    # Fallback for safety, though this indicates a deeper startup issue if logger is None.
    app_logger = logger if logger else structlog.get_logger("app.websocket_fallback")

    try:
        app_logger.info(f"WebSocket connection request from {websocket.client.host}:{websocket.client.port}")
        await websocket.accept() # Accept connection first
        app_logger.info(f"WebSocket connection accepted from {websocket.client.host}:{websocket.client.port}")
    except Exception as e:
        app_logger.error(f"Failed to accept WebSocket connection from {websocket.client.host}:{websocket.client.port}: {e}", exc_info=True)
        return
    
    try:
        app_logger.info(f"Creating session for {websocket.client.host}:{websocket.client.port}")
        if not await session_manager.create_session(websocket):
            app_logger.warning(f"Session creation failed for {websocket.client.host}:{websocket.client.port} (max sessions limit)")
            await websocket.close(code=1013, reason="Server overloaded or unable to create session.")
            return
        
        session_id = session_manager.get_session_id(websocket)
        if not session_id: # Should not happen if create_session succeeded
            app_logger.error(f"Critical: Session ID not found after successful session creation for {websocket.client.host}:{websocket.client.port}")
            await websocket.close(code=1011, reason="Internal server error: Session ID missing.")
            return
            
        # Create a logger instance bound with session-specific context
        session_logger = app_logger.bind(session_id=session_id, client_host=websocket.client.host)
        session_logger.info(f"WebSocket connection accepted and session created.")

        handler = WebSocketHandler(
            websocket=websocket,
            services=services,
            metrics=metrics, 
            logger=session_logger # Pass the new session-bound logger
        )
        
        # Send initial status/config to client in a try-except block
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                session_logger.info(f"Sending initial status message to client")
                await handler._send_message({
                    "type": "status",
                    "session_id": session_id, # Send the generated session_id to client
                    "ready": True,
                    "config": { # Send relevant backend config to client
                        "sample_rate": settings.sample_rate,
                        "frame_duration_ms": settings.frame_duration_ms,
                        "channels": settings.channels
                    }
                })
                session_logger.info(f"Initial status message sent successfully")
            else: # Should not happen immediately after accept, but good check
                session_logger.warning("WebSocket disconnected before initial status message could be sent.")
                # Cleanup will be handled in finally block
                return
        except Exception as msg_e:
            session_logger.error(f"Failed to send initial status message: {msg_e}", exc_info=True)
            # Continue anyway, since this is not critical

        # Main connection handler - wrapped in a try-except with reconnection logic
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries and websocket.client_state == WebSocketState.CONNECTED:
            try:
                session_logger.info(f"Starting main WebSocket handler loop")
                await handler.handle_connection() # Main message handling loop
                session_logger.info(f"WebSocket handler loop exited normally")
                break  # If handle_connection returns normally, exit loop
            except WebSocketDisconnect:
                session_logger.info(f"Client disconnected WebSocket gracefully.")
                break
            except ConnectionResetError:
                session_logger.warning(f"Client connection reset abruptly.")
                break
            except Exception as e: # Catch any other unhandled exceptions from handle_connection
                session_logger.error(f"Unhandled error during WebSocket communication: {str(e)}", exc_info=True)
                retry_count += 1
                
                if retry_count < max_retries and websocket.client_state == WebSocketState.CONNECTED:
                    session_logger.warning(f"Attempting to recover WebSocket handler (retry {retry_count}/{max_retries})...")
                    await asyncio.sleep(0.5)  # Brief pause before retry
                    try:
                        await handler._send_error("A temporary server error occurred. Reconnecting...")
                    except:
                        pass  # Ignore if we can't send the error
                else:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        try:
                            await handler._send_error(f"A critical server error occurred. Please try reconnecting.")
                        except Exception as send_err:
                            session_logger.error(f"Failed to send final error message to client: {send_err}", exc_info=True)
                    break
    finally:
        session_logger.info(f"Performing final cleanup for WebSocket session...")
        try:
            await handler.cleanup() # Ensure handler's internal cleanup is called
        except Exception as cleanup_e:
            session_logger.error(f"Error during handler cleanup: {cleanup_e}", exc_info=True)
            
        session_manager.remove_session(websocket) # Remove from global session manager
        
        # Final log to confirm closure and state, checking client_state one last time
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                # If not already closed by handler or client, attempt a graceful close
                session_logger.info(f"WebSocket still connected. Attempting graceful close.")
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
    # For development, `uvicorn backend.app.main:app --reload` is often preferred.
    
    # Ensure logging is set up if this script is run directly and not via uvicorn CLI that might handle it.
    # The lifespan manager should handle this, but as a safeguard for direct script execution:
    if logger is None:
        setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO").upper())
        # Assign to global logger so it's available if lifespan didn't run (e.g. import error before lifespan)
        logger = structlog.get_logger("app.main_direct_run_fallback") 
        logger.info("Running Uvicorn server directly from __main__...")

    # Configure reload settings to avoid hot-reload storms from generated files
    reload_config = os.getenv("UVICORN_RELOAD", "true").lower() == 'true'
    reload_dirs = None
    reload_excludes = None
    
    if reload_config:
        # Only watch specific directories and exclude problematic patterns
        reload_dirs = ["backend/app", "backend/services"]
        reload_excludes = [
            "__pycache__",
            "*.pyc", 
            "*.pyo",
            "*__pydantic_*",  # Exclude pydantic generated files
            "*.onnx",         # Exclude model files
            "*.log",          # Exclude log files
            "temp_*",         # Exclude temporary files
            ".DS_Store"       # Exclude macOS files
        ]
        logger.info(f"Reload enabled with dirs: {reload_dirs}, excludes: {reload_excludes}")

    uvicorn.run(
        "backend.app.main:app", # Path to the FastAPI app instance
        host=settings.ws_server_host,
        port=settings.ws_server_port,
        reload=reload_config,
        reload_dirs=reload_dirs,
        reload_excludes=reload_excludes,
        log_level=settings.log_level.lower(), # Uvicorn's native log level
        # Use default uvicorn logger or configure uvicorn logging separately if needed
        # It's generally better to let structlog handle app logs and uvicorn handle access logs.
    )