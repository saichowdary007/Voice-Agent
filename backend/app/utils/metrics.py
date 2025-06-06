import os
import time
import psutil
import threading
import asyncio
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MetricData:
    """Individual metric data point"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class LatencyMetric:
    """Latency measurement"""
    operation: str
    value: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterMetric:
    """Counter measurement"""
    name: str
    value: int
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Simple metrics collector for performance monitoring"""
    
    def __init__(self):
        self.metrics: Dict[str, List[MetricData]] = {}
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.start_time = time.time()
        
        # Initialize basic metrics
        self.counters.update({
            "websocket_connections": 0,
            "audio_frames_processed": 0,
            "speech_sessions": 0,
            "ai_requests": 0,
            "tts_requests": 0,
            "errors": 0
        })
        
        self.gauges.update({
            "active_sessions": 0,
            "average_processing_time": 0.0,
            "memory_usage_mb": 0.0
        })
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        self.counters[name] = self.counters.get(name, 0) + value
        
        # Store detailed metric
        metric = MetricData(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(metric)
        
        # Keep only last 1000 entries per metric
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric value"""
        self.gauges[name] = value
        
        # Store detailed metric
        metric = MetricData(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(metric)
        
        # Keep only last 100 entries for gauges
        if len(self.metrics[name]) > 100:
            self.metrics[name] = self.metrics[name][-100:]
    
    def record_latency(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a latency measurement"""
        metric_name = f"{name}_latency"
        
        metric = MetricData(
            name=metric_name,
            value=duration,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append(metric)
        
        # Keep only last 500 latency measurements
        if len(self.metrics[metric_name]) > 500:
            self.metrics[metric_name] = self.metrics[metric_name][-500:]
    
    def get_counter(self, name: str) -> int:
        """Get current counter value"""
        return self.counters.get(name, 0)
    
    def get_gauge(self, name: str) -> float:
        """Get current gauge value"""
        return self.gauges.get(name, 0.0)
    
    def get_average_latency(self, name: str, window_seconds: int = 60) -> float:
        """Get average latency for a metric over a time window"""
        metric_name = f"{name}_latency"
        
        if metric_name not in self.metrics:
            return 0.0
        
        cutoff_time = time.time() - window_seconds
        recent_metrics = [
            m for m in self.metrics[metric_name] 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return 0.0
        
        return sum(m.value for m in recent_metrics) / len(recent_metrics)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        uptime = time.time() - self.start_time
        
        # Calculate some derived metrics
        total_requests = (
            self.get_counter("ai_requests") + 
            self.get_counter("tts_requests")
        )
        
        requests_per_second = total_requests / uptime if uptime > 0 else 0
        
        return {
            "timestamp": time.time(),
            "uptime_seconds": uptime,
            "counters": self.counters.copy(),
            "gauges": self.gauges.copy(),
            "derived": {
                "requests_per_second": round(requests_per_second, 2),
                "error_rate": (
                    self.get_counter("errors") / max(total_requests, 1)
                ) * 100,
                "average_latencies": {
                    "audio_processing": self.get_average_latency("audio_processing"),
                    "ai_response": self.get_average_latency("ai_response"),
                    "tts_generation": self.get_average_latency("tts_generation")
                }
            }
        }
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.set_gauge("memory_usage_mb", memory_mb)
            return memory_mb
        except ImportError:
            logger.warning("psutil not available, cannot measure memory usage")
            return 0.0
        except Exception as e:
            logger.error(f"Error measuring memory usage: {e}")
            return 0.0
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.start_time = time.time()
        
        logger.info("Metrics reset")
    
    async def cleanup(self):
        """Cleanup metrics collector"""
        logger.info("Cleaning up metrics collector")
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()


class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alert_thresholds = {
            'latency_p95_ms': 1000,  # 1 second
            'latency_p99_ms': 2000,  # 2 seconds
            'error_rate_percent': 5,  # 5%
            'memory_usage_mb': 400,   # 400MB
            'cpu_usage_percent': 80   # 80%
        }
        
    def check_performance_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance issues and return alerts"""
        alerts = []
        
        # Check latency alerts
        end_to_end_stats = self.metrics.get_latency_stats('end_to_end_latency')
        if end_to_end_stats:
            if end_to_end_stats.get('p95', 0) > self.alert_thresholds['latency_p95_ms']:
                alerts.append({
                    'type': 'latency_high',
                    'metric': 'end_to_end_latency_p95',
                    'value': end_to_end_stats['p95'],
                    'threshold': self.alert_thresholds['latency_p95_ms'],
                    'severity': 'warning'
                })
                
        # Check system resource alerts
        system_metrics = self.metrics.get_system_metrics()
        
        if system_metrics.get('process_memory_mb', 0) > self.alert_thresholds['memory_usage_mb']:
            alerts.append({
                'type': 'memory_high',
                'metric': 'process_memory_mb',
                'value': system_metrics['process_memory_mb'],
                'threshold': self.alert_thresholds['memory_usage_mb'],
                'severity': 'warning'
            })
            
        if system_metrics.get('cpu_percent', 0) > self.alert_thresholds['cpu_usage_percent']:
            alerts.append({
                'type': 'cpu_high',
                'metric': 'cpu_percent',
                'value': system_metrics['cpu_percent'],
                'threshold': self.alert_thresholds['cpu_usage_percent'],
                'severity': 'warning'
            })
            
        return alerts
        
    def get_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        score = 100.0
        
        # Latency impact
        end_to_end_stats = self.metrics.get_latency_stats('end_to_end_latency')
        if end_to_end_stats:
            p95_latency = end_to_end_stats.get('p95', 0)
            if p95_latency > 500:  # Target: <500ms
                score -= min(30, (p95_latency - 500) / 50)  # Reduce score based on excess latency
                
        # System resource impact
        system_metrics = self.metrics.get_system_metrics()
        cpu_usage = system_metrics.get('cpu_percent', 0)
        memory_usage = system_metrics.get('process_memory_mb', 0)
        
        if cpu_usage > 70:
            score -= min(20, (cpu_usage - 70) / 3)
            
        if memory_usage > 300:
            score -= min(20, (memory_usage - 300) / 20)
            
        # Error rate impact
        error_rates = self.metrics.get_all_metrics().get('error_rates', {})
        total_error_rate = sum(error_rates.values())
        if total_error_rate > 1:
            score -= min(30, total_error_rate * 5)
            
        return max(0, score) 