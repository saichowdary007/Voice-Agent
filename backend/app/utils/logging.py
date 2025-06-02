import os
import sys
import logging
from typing import Any, Dict
import structlog
from structlog.typing import FilteringBoundLogger


def setup_logging(log_level: str = None) -> None:
    """Setup structured logging for the application"""
    
    # Get log level from environment
    log_level = log_level or os.getenv('LOG_LEVEL', 'INFO')
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level.upper() # Use the determined log_level
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if os.getenv('LOG_FORMAT') == 'json' 
            else structlog.dev.ConsoleRenderer(colors=True)
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


class VoiceAgentLogger:
    """Enhanced logger for voice agent with performance tracking"""
    
    def __init__(self, name: str):
        self.logger: FilteringBoundLogger = structlog.get_logger(name)
        self.performance_data: Dict[str, Any] = {}
        
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
        
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
        
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
        
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)
        
    def log_latency(self, operation: str, latency_ms: float, **kwargs):
        """Log latency measurement"""
        self.logger.info(
            "latency_measurement",
            operation=operation,
            latency_ms=latency_ms,
            **kwargs
        )
        
    def log_audio_metrics(self, 
                         audio_duration: float,
                         processing_time: float,
                         sample_rate: int,
                         **kwargs):
        """Log audio processing metrics"""
        rtf = processing_time / audio_duration if audio_duration > 0 else 0
        
        self.logger.info(
            "audio_metrics",
            audio_duration_sec=audio_duration,
            processing_time_sec=processing_time,
            real_time_factor=rtf,
            sample_rate=sample_rate,
            **kwargs
        )
        
    def log_session_event(self, 
                         session_id: str,
                         event_type: str,
                         **kwargs):
        """Log session-related events"""
        self.logger.info(
            "session_event",
            session_id=session_id,
            event_type=event_type,
            **kwargs
        )
        
    def log_websocket_event(self,
                           connection_id: str,
                           event_type: str,
                           message_type: str = None,
                           **kwargs):
        """Log WebSocket events"""
        self.logger.info(
            "websocket_event",
            connection_id=connection_id,
            event_type=event_type,
            message_type=message_type,
            **kwargs
        )
        
    def log_error_with_context(self,
                              error: Exception,
                              context: Dict[str, Any],
                              operation: str = None):
        """Log error with rich context"""
        self.logger.error(
            "error_with_context",
            error_type=type(error).__name__,
            error_message=str(error),
            operation=operation,
            context=context,
            exc_info=True
        )


def get_logger(name: str) -> VoiceAgentLogger:
    """Get enhanced logger instance"""
    return VoiceAgentLogger(name)


class PerformanceLogger:
    """Logger specifically for performance monitoring"""
    
    def __init__(self):
        self.logger = structlog.get_logger("performance")
        
    def log_request_timing(self,
                          endpoint: str,
                          method: str,
                          duration_ms: float,
                          status_code: int = None,
                          **kwargs):
        """Log HTTP request timing"""
        self.logger.info(
            "request_timing",
            endpoint=endpoint,
            method=method,
            duration_ms=duration_ms,
            status_code=status_code,
            **kwargs
        )
        
    def log_pipeline_stage(self,
                          stage: str,
                          duration_ms: float,
                          session_id: str = None,
                          **kwargs):
        """Log pipeline stage timing"""
        self.logger.info(
            "pipeline_stage",
            stage=stage,
            duration_ms=duration_ms,
            session_id=session_id,
            **kwargs
        )
        
    def log_model_performance(self,
                             model_name: str,
                             input_size: int,
                             output_size: int,
                             processing_time_ms: float,
                             **kwargs):
        """Log ML model performance"""
        self.logger.info(
            "model_performance",
            model_name=model_name,
            input_size=input_size,
            output_size=output_size,
            processing_time_ms=processing_time_ms,
            **kwargs
        )
        
    def log_resource_usage(self,
                          cpu_percent: float,
                          memory_mb: float,
                          gpu_memory_mb: float = None,
                          **kwargs):
        """Log system resource usage"""
        self.logger.info(
            "resource_usage",
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            **kwargs
        ) 