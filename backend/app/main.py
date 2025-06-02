import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from .websocket_handler import WebSocketHandler
from .utils.logging import setup_logging
from .utils.metrics import MetricsCollector
from .audio.stt import STTEngine
from .audio.tts import TTSEngine
from .audio.vad import VADEngine
from .ai.gemini_client import GeminiClient


# Global state
active_connections: Set[WebSocket] = set()
engines: Dict = {}
metrics: MetricsCollector = None
logger = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global engines, metrics, logger
    
    # Setup logging
    setup_logging()
    logger = structlog.get_logger()
    
    # Initialize metrics
    metrics = MetricsCollector()
    
    # Initialize engines
    logger.info("Initializing AI engines...")
    
    try:
        # Initialize STT engine
        engines['stt'] = STTEngine()
        await engines['stt'].initialize()
        
        # Initialize TTS engine  
        engines['tts'] = TTSEngine()
        await engines['tts'].initialize()
        
        # Initialize VAD engine
        engines['vad'] = VADEngine()
        await engines['vad'].initialize()
        
        # Initialize Gemini client
        engines['gemini'] = GeminiClient()
        await engines['gemini'].initialize()
        
        logger.info("All engines initialized successfully")
        
        # Preload models if enabled
        if os.getenv('PRELOAD_MODELS', 'true').lower() == 'true':
            logger.info("Preloading models...")
            await engines['stt'].preload()
            await engines['tts'].preload()
            await engines['vad'].preload()
            logger.info("Models preloaded successfully")
            
    except Exception as e:
        logger.error(f"Failed to initialize engines: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down engines...")
    for engine in engines.values():
        if hasattr(engine, 'cleanup'):
            await engine.cleanup()


# Create FastAPI app
app = FastAPI(
    title="Voice Agent API",
    description="High-performance voice agent with sub-500ms latency",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    try:
        # Check if all engines are ready
        engine_status = {}
        for name, engine in engines.items():
            engine_status[name] = hasattr(engine, 'is_ready') and await engine.is_ready()
        
        all_ready = all(engine_status.values())
        
        return JSONResponse(
            status_code=200 if all_ready else 503,
            content={
                "status": "healthy" if all_ready else "degraded",
                "engines": engine_status,
                "active_connections": len(active_connections),
                "memory_usage_mb": metrics.get_memory_usage() if metrics else 0
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/metrics")
async def get_metrics():
    """Metrics endpoint for monitoring"""
    if not metrics:
        raise HTTPException(status_code=503, detail="Metrics not available")
    
    return JSONResponse(content=metrics.get_all_metrics())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for voice agent communication"""
    global active_connections, engines, metrics, logger
    
    # Check connection limit
    max_connections = int(os.getenv('MAX_CONCURRENT_SESSIONS', '3'))
    if len(active_connections) >= max_connections:
        await websocket.close(code=1008, reason="Server at capacity")
        return
    
    await websocket.accept()
    active_connections.add(websocket)
    
    # Create WebSocket handler
    handler = WebSocketHandler(
        websocket=websocket,
        engines=engines,
        metrics=metrics,
        logger=logger
    )
    
    try:
        logger.info(f"New WebSocket connection. Total: {len(active_connections)}")
        metrics.increment_counter('websocket_connections_total')
        
        # Start handling messages
        await handler.handle_connection()
        
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        error_msg = str(e) if str(e) else f"Unexpected error: {type(e).__name__}"
        logger.error(f"WebSocket error in session {getattr(handler, 'session_id', 'unknown')}: {error_msg}", exc_info=True)
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass  # Connection might already be closed
    finally:
        active_connections.discard(websocket)
        await handler.cleanup()
        logger.info(f"WebSocket connection closed. Total: {len(active_connections)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Voice Agent API",
        "version": "1.0.0",
        "status": "running",
        "active_connections": len(active_connections)
    }


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 