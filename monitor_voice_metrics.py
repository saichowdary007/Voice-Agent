#!/usr/bin/env python3
"""
Voice processing metrics monitoring script.
Tracks RMS levels, VAD performance, and connection health as recommended.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from collections import deque, defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceMetricsMonitor:
    """Monitor voice processing metrics and generate reports."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        
        # Metrics storage
        self.rms_history = deque(maxlen=history_size)
        self.vad_decisions = deque(maxlen=history_size)
        self.connection_events = deque(maxlen=history_size)
        self.transcription_results = deque(maxlen=history_size)
        
        # Performance counters
        self.stats = {
            'total_audio_chunks': 0,
            'speech_chunks': 0,
            'silence_chunks': 0,
            'successful_transcriptions': 0,
            'failed_transcriptions': 0,
            'connection_drops': 0,
            'false_positives': 0,  # VAD said speech but no transcript
            'missed_speech': 0,    # Transcript but VAD said silence
        }
        
        # Thresholds for alerts
        self.alert_thresholds = {
            'low_rms_threshold': 0.01,
            'high_rms_threshold': 0.5,
            'vad_accuracy_threshold': 0.85,
            'transcription_success_rate': 0.90,
        }
        
        self.start_time = time.time()
    
    def log_audio_chunk(self, rms_level: float, is_speech: bool, 
                       has_transcript: bool = False, transcript: str = ""):
        """Log metrics for an audio chunk."""
        timestamp = time.time()
        
        # Store metrics
        self.rms_history.append({
            'timestamp': timestamp,
            'rms': rms_level,
            'is_speech': is_speech,
            'has_transcript': has_transcript,
            'transcript': transcript
        })
        
        # Update counters
        self.stats['total_audio_chunks'] += 1
        if is_speech:
            self.stats['speech_chunks'] += 1
        else:
            self.stats['silence_chunks'] += 1
        
        if has_transcript:
            self.stats['successful_transcriptions'] += 1
        elif is_speech and not has_transcript:
            self.stats['false_positives'] += 1
        elif not is_speech and has_transcript:
            self.stats['missed_speech'] += 1
    
    def log_connection_event(self, event_type: str, details: str = ""):
        """Log connection-related events."""
        timestamp = time.time()
        
        self.connection_events.append({
            'timestamp': timestamp,
            'event': event_type,
            'details': details
        })
        
        if event_type == 'disconnect':
            self.stats['connection_drops'] += 1
    
    def get_current_stats(self) -> Dict:
        """Get current performance statistics."""
        runtime = time.time() - self.start_time
        
        # Calculate rates
        total_chunks = self.stats['total_audio_chunks']
        if total_chunks > 0:
            speech_rate = self.stats['speech_chunks'] / total_chunks
            vad_accuracy = 1.0 - (self.stats['false_positives'] + self.stats['missed_speech']) / total_chunks
            transcription_rate = self.stats['successful_transcriptions'] / max(self.stats['speech_chunks'], 1)
        else:
            speech_rate = 0.0
            vad_accuracy = 1.0
            transcription_rate = 0.0
        
        # RMS statistics
        if self.rms_history:
            rms_values = [entry['rms'] for entry in self.rms_history]
            rms_stats = {
                'mean': np.mean(rms_values),
                'std': np.std(rms_values),
                'min': np.min(rms_values),
                'max': np.max(rms_values),
                'percentile_25': np.percentile(rms_values, 25),
                'percentile_75': np.percentile(rms_values, 75),
            }
        else:
            rms_stats = {}
        
        return {
            'runtime_seconds': runtime,
            'total_chunks': total_chunks,
            'speech_rate': speech_rate,
            'vad_accuracy': vad_accuracy,
            'transcription_success_rate': transcription_rate,
            'connection_drops': self.stats['connection_drops'],
            'rms_statistics': rms_stats,
            'raw_stats': self.stats.copy()
        }
    
    def check_alerts(self) -> List[str]:
        """Check for performance alerts."""
        alerts = []
        stats = self.get_current_stats()
        
        # Check RMS levels
        if 'rms_statistics' in stats and stats['rms_statistics']:
            rms_mean = stats['rms_statistics']['mean']
            if rms_mean < self.alert_thresholds['low_rms_threshold']:
                alerts.append(f"⚠️ Low RMS levels detected: {rms_mean:.4f} (users may be too far from mic)")
            elif rms_mean > self.alert_thresholds['high_rms_threshold']:
                alerts.append(f"⚠️ High RMS levels detected: {rms_mean:.4f} (gain may be too high)")
        
        # Check VAD accuracy
        if stats['vad_accuracy'] < self.alert_thresholds['vad_accuracy_threshold']:
            alerts.append(f"⚠️ VAD accuracy low: {stats['vad_accuracy']:.2f}")
        
        # Check transcription success rate
        if stats['transcription_success_rate'] < self.alert_thresholds['transcription_success_rate']:
            alerts.append(f"⚠️ Transcription success rate low: {stats['transcription_success_rate']:.2f}")
        
        # Check connection stability
        if stats['connection_drops'] > 5:
            alerts.append(f"⚠️ Multiple connection drops: {stats['connection_drops']}")
        
        return alerts
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance report."""
        stats = self.get_current_stats()
        alerts = self.check_alerts()
        
        report = []
        report.append("=" * 60)
        report.append("VOICE PROCESSING PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Runtime: {stats['runtime_seconds']:.1f} seconds")
        report.append("")
        
        # Overall metrics
        report.append("OVERALL METRICS:")
        report.append(f"  Total Audio Chunks: {stats['total_chunks']}")
        report.append(f"  Speech Detection Rate: {stats['speech_rate']:.2%}")
        report.append(f"  VAD Accuracy: {stats['vad_accuracy']:.2%}")
        report.append(f"  Transcription Success Rate: {stats['transcription_success_rate']:.2%}")
        report.append(f"  Connection Drops: {stats['connection_drops']}")
        report.append("")
        
        # RMS statistics
        if stats['rms_statistics']:
            rms = stats['rms_statistics']
            report.append("RMS LEVEL ANALYSIS:")
            report.append(f"  Mean RMS: {rms['mean']:.4f}")
            report.append(f"  Std Dev: {rms['std']:.4f}")
            report.append(f"  Range: {rms['min']:.4f} - {rms['max']:.4f}")
            report.append(f"  25th Percentile: {rms['percentile_25']:.4f}")
            report.append(f"  75th Percentile: {rms['percentile_75']:.4f}")
            report.append("")
        
        # Detailed breakdown
        raw = stats['raw_stats']
        report.append("DETAILED BREAKDOWN:")
        report.append(f"  Speech Chunks: {raw['speech_chunks']}")
        report.append(f"  Silence Chunks: {raw['silence_chunks']}")
        report.append(f"  Successful Transcriptions: {raw['successful_transcriptions']}")
        report.append(f"  Failed Transcriptions: {raw['failed_transcriptions']}")
        report.append(f"  False Positives (VAD): {raw['false_positives']}")
        report.append(f"  Missed Speech (VAD): {raw['missed_speech']}")
        report.append("")
        
        # Alerts
        if alerts:
            report.append("ALERTS:")
            for alert in alerts:
                report.append(f"  {alert}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if stats['speech_rate'] < 0.1:
            report.append("  • Speech rate very low - check microphone sensitivity")
        if stats['vad_accuracy'] < 0.8:
            report.append("  • VAD accuracy low - consider tuning thresholds")
        if stats['transcription_success_rate'] < 0.8:
            report.append("  • Transcription rate low - check audio quality and STT service")
        if not alerts:
            report.append("  • System performing within normal parameters")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_metrics_to_file(self, filepath: str):
        """Save metrics data to JSON file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'stats': self.get_current_stats(),
            'rms_history': list(self.rms_history),
            'connection_events': list(self.connection_events)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"📊 Metrics saved to {filepath}")
    
    def plot_rms_histogram(self, save_path: Optional[str] = None):
        """Generate RMS level histogram."""
        if not self.rms_history:
            logger.warning("No RMS data to plot")
            return
        
        rms_values = [entry['rms'] for entry in self.rms_history]
        
        plt.figure(figsize=(10, 6))
        plt.hist(rms_values, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('RMS Level')
        plt.ylabel('Frequency')
        plt.title('RMS Level Distribution')
        plt.grid(True, alpha=0.3)
        
        # Add threshold lines
        plt.axvline(self.alert_thresholds['low_rms_threshold'], 
                   color='red', linestyle='--', label='Low Threshold')
        plt.axvline(self.alert_thresholds['high_rms_threshold'], 
                   color='red', linestyle='--', label='High Threshold')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📈 RMS histogram saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


# Global monitor instance
_monitor = None


def get_voice_monitor() -> VoiceMetricsMonitor:
    """Get global voice metrics monitor."""
    global _monitor
    if _monitor is None:
        _monitor = VoiceMetricsMonitor()
    return _monitor


async def run_monitoring_demo():
    """Run a demonstration of the monitoring system."""
    logger.info("🔍 Starting Voice Metrics Monitoring Demo")
    
    monitor = get_voice_monitor()
    
    # Simulate some audio processing events
    logger.info("Simulating audio processing events...")
    
    # Simulate normal operation
    for i in range(100):
        # Simulate varying RMS levels
        if i < 20:
            # Silence period
            rms = np.random.normal(0.01, 0.005)
            is_speech = False
            has_transcript = False
        elif i < 60:
            # Speech period
            rms = np.random.normal(0.15, 0.05)
            is_speech = True
            has_transcript = np.random.random() > 0.1  # 90% success rate
        else:
            # Mixed period
            rms = np.random.normal(0.08, 0.03)
            is_speech = np.random.random() > 0.6
            has_transcript = is_speech and (np.random.random() > 0.15)
        
        # Ensure RMS is positive
        rms = max(0.001, rms)
        
        transcript = f"Test transcript {i}" if has_transcript else ""
        monitor.log_audio_chunk(rms, is_speech, has_transcript, transcript)
        
        # Simulate occasional connection issues
        if i == 30:
            monitor.log_connection_event('disconnect', 'Network timeout')
        elif i == 31:
            monitor.log_connection_event('reconnect', 'Connection restored')
    
    # Generate and display report
    report = monitor.generate_report()
    print(report)
    
    # Save metrics
    metrics_dir = Path("voice_metrics")
    metrics_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    monitor.save_metrics_to_file(f"voice_metrics/metrics_{timestamp}.json")
    
    # Generate plots
    try:
        monitor.plot_rms_histogram(f"voice_metrics/rms_histogram_{timestamp}.png")
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")
    
    logger.info("✅ Monitoring demo completed")


if __name__ == "__main__":
    asyncio.run(run_monitoring_demo())