import os
import time
import psutil
import threading
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger()


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
    """Collects and manages performance metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.enabled = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
        
        # Metric storage
        self.latency_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counter_metrics: Dict[str, int] = defaultdict(int)
        self.gauge_metrics: Dict[str, float] = defaultdict(float)
        
        # Performance tracking
        self.request_times: deque = deque(maxlen=max_history)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # System metrics
        self.system_metrics_interval = 30  # seconds
        self.last_system_check = 0
        self.system_stats = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
    def record_latency(self, operation: str, latency: float, metadata: Dict[str, Any] = None):
        """Record latency measurement"""
        if not self.enabled:
            return
            
        with self.lock:
            metric = LatencyMetric(
                operation=operation,
                value=latency,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            
            self.latency_metrics[operation].append(metric)
            
        logger.debug(f"Recorded latency: {operation} = {latency:.3f}ms")
        
    def increment_counter(self, name: str, labels: Dict[str, str] = None):
        """Increment counter metric"""
        if not self.enabled:
            return
            
        with self.lock:
            key = self._make_counter_key(name, labels)
            self.counter_metrics[key] += 1
            
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set gauge metric value"""
        if not self.enabled:
            return
            
        with self.lock:
            key = self._make_counter_key(name, labels)
            self.gauge_metrics[key] = value
            
    def record_request_time(self, duration: float):
        """Record HTTP request duration"""
        if not self.enabled:
            return
            
        with self.lock:
            self.request_times.append({
                'duration': duration,
                'timestamp': time.time()
            })
            
    def record_error(self, error_type: str):
        """Record error occurrence"""
        if not self.enabled:
            return
            
        with self.lock:
            self.error_counts[error_type] += 1
            
    def get_latency_stats(self, operation: str) -> Dict[str, float]:
        """Get latency statistics for operation"""
        if not self.enabled or operation not in self.latency_metrics:
            return {}
            
        with self.lock:
            metrics = list(self.latency_metrics[operation])
            
        if not metrics:
            return {}
            
        values = [m.value for m in metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'p50': self._percentile(values, 50),
            'p95': self._percentile(values, 95),
            'p99': self._percentile(values, 99),
            'latest': values[-1] if values else 0
        }
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        current_time = time.time()
        
        # Update system metrics if needed
        if current_time - self.last_system_check > self.system_metrics_interval:
            self._update_system_metrics()
            self.last_system_check = current_time
            
        return self.system_stats.copy()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
            
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        if not self.enabled:
            return {}
            
        # Get key latency metrics
        key_operations = [
            'end_to_end_latency',
            'stt_processing',
            'tts_generation',
            'ai_response',
            'websocket_round_trip'
        ]
        
        latency_stats = {}
        for op in key_operations:
            stats = self.get_latency_stats(op)
            if stats:
                latency_stats[op] = stats
                
        # Get counter metrics
        with self.lock:
            counters = dict(self.counter_metrics)
            gauges = dict(self.gauge_metrics)
            
        # Get system metrics
        system_metrics = self.get_system_metrics()
        
        # Calculate error rates
        error_rates = {}
        total_requests = sum(counters.get(k, 0) for k in counters if 'request' in k)
        if total_requests > 0:
            for error_type, count in self.error_counts.items():
                error_rates[error_type] = (count / total_requests) * 100
                
        return {
            'latency': latency_stats,
            'counters': counters,
            'gauges': gauges,
            'system': system_metrics,
            'error_rates': error_rates,
            'timestamp': time.time()
        }
        
    def _update_system_metrics(self):
        """Update system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            
            self.system_stats = {
                'cpu_percent': cpu_percent,
                'memory_total_gb': memory.total / (1024**3),
                'memory_used_gb': memory.used / (1024**3),
                'memory_percent': memory.percent,
                'process_memory_mb': process_memory.rss / (1024**2),
                'process_cpu_percent': process.cpu_percent(),
                'disk_used_gb': disk.used / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'disk_percent': (disk.used / disk.total) * 100,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
            
    def _make_counter_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create key for counter with labels"""
        if not labels:
            return name
            
        label_str = ','.join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"
        
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
            
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
        
    def reset_metrics(self):
        """Reset all collected metrics"""
        with self.lock:
            self.latency_metrics.clear()
            self.counter_metrics.clear()
            self.gauge_metrics.clear()
            self.request_times.clear()
            self.error_counts.clear()
            
        logger.info("Metrics reset")
        
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        if not self.enabled:
            return ""
            
        lines = []
        timestamp = int(time.time() * 1000)
        
        # Export latency metrics
        for operation, metrics in self.latency_metrics.items():
            if metrics:
                stats = self.get_latency_stats(operation)
                for stat_name, value in stats.items():
                    metric_name = f"voice_agent_latency_{stat_name}"
                    lines.append(f'{metric_name}{{operation="{operation}"}} {value} {timestamp}')
                    
        # Export counters
        with self.lock:
            for name, value in self.counter_metrics.items():
                metric_name = f"voice_agent_counter_{name.replace('[', '_').replace(']', '').replace('=', '_').replace(',', '_')}"
                lines.append(f'{metric_name} {value} {timestamp}')
                
        # Export gauges
        with self.lock:
            for name, value in self.gauge_metrics.items():
                metric_name = f"voice_agent_gauge_{name.replace('[', '_').replace(']', '').replace('=', '_').replace(',', '_')}"
                lines.append(f'{metric_name} {value} {timestamp}')
                
        # Export system metrics
        system_metrics = self.get_system_metrics()
        for name, value in system_metrics.items():
            if isinstance(value, (int, float)) and name != 'timestamp':
                metric_name = f"voice_agent_system_{name}"
                lines.append(f'{metric_name} {value} {timestamp}')
                
        return '\n'.join(lines)

    def cleanup(self):
        """Clean up resources used by the metrics collector"""
        logger.info("Cleaning up metrics collector...")
        with self.lock:
            # Clear all metrics
            self.latency_metrics.clear()
            self.counter_metrics.clear()
            self.gauge_metrics.clear()
            self.request_times.clear()
            self.error_counts.clear()
            self.system_stats.clear()
        logger.info("Metrics collector cleaned up successfully")


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