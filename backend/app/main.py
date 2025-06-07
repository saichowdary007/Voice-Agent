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
from dotenv import load_dotenv

# Load .env file first
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

# Setup logging next
from backend.app.utils.logging import setup_logging
from backend.app.config import settings

# Initialize logging
setup_logging(settings.log_level)
logger = structlog.get_logger("app.main")

# Import services after logging is set up
from backend.services.voice_service import VoiceService
from backend.app.websocket_manager import WebSocketManager
from backend.app.utils.metrics import MetricsCollector

# Global state
voice_service: Optional[VoiceService] = None
websocket_manager: Optional[WebSocketManager] = None
metrics: Optional[MetricsCollector] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    global voice_service, websocket_manager, metrics
    
    try:
        logger.info("Starting Voice Agent application...")
        
        # Initialize metrics
        metrics = MetricsCollector() if settings.enable_metrics else None
        
        # Initialize voice service
        voice_service = VoiceService()
        await voice_service.initialize()
        
        # Get service health status
        if hasattr(voice_service, 'get_health_status'):
            services_status = await voice_service.get_health_status()
            
            # Log status of each service
            for service_name, status in services_status.items():
                if service_name != "voice_service":  # Skip the parent service
                    is_available = status.get("available", False)
                    status_str = "✅ Available" if is_available else "❌ Unavailable"
                    
                    if not is_available and "error" in status:
                        logger.error(f"{service_name.upper()}: {status_str} - Error: {status['error']}")
                    else:
                        logger.info(f"{service_name.upper()}: {status_str}")
        
        # Initialize WebSocket manager
        websocket_manager = WebSocketManager(voice_service, voice_service.audio_service, metrics)
        
        # Store in app state
        app.state.voice_service = voice_service
        app.state.websocket_manager = websocket_manager
        app.state.metrics = metrics
        
        logger.info("Voice Agent application started successfully")
        
        yield
        
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down Voice Agent application...")
        
        # Cleanup WebSocket manager
        if websocket_manager:
            await websocket_manager.cleanup()
        
        # Cleanup voice service
        if voice_service:
            await voice_service.cleanup()
        
        logger.info("Voice Agent application shutdown complete")

app = FastAPI(
    title=settings.app_name,
    description="Ultra-Fast Voice Agent with Modern Architecture",
    version=settings.app_version,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        status = {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {}
        }
        
        if voice_service:
            status["services"] = await voice_service.get_health_status()
        
        return status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/metrics")
async def get_metrics():
    """Get application metrics"""
    if not metrics:
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    
    try:
        result = await metrics.get_metrics()
        return result
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for voice communication"""
    if not websocket_manager:
        await websocket.close(code=1011, reason="Service unavailable")
        return
    
    await websocket_manager.handle_connection(websocket)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "endpoints": {
            "websocket": "/ws",
            "health": "/health",
            "metrics": "/metrics"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host="127.0.0.1",
        port=8090,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )