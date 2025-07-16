#!/usr/bin/env python3
"""
Performance monitoring script for Voice Agent.
Tracks latency metrics and provides real-time performance insights.
"""

import asyncio
import json
import time
import websockets
import base64
import logging
from datetime import datetime
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor Voice Agent performance metrics."""
    
    def __init__(self):
        self.metrics: List[Dict] = []
        self.session_start = time.time()
    
    def log_interaction(self, interaction_data: Dict):
        """Log a single voice interaction with timing data."""
        interaction_data['timestamp'] = datetime.utcnow().isoformat()
        interaction_data['session_time'] = time.time() - self.session_start
        self.metrics.append(interaction_data)
        
        # Log to console
        total_latency = interaction_data.get('total_latency_ms', 0)
        stt_latency = interaction_data.get('stt_latency_ms', 0)
        llm_latency = interaction_data.get('llm_latency_ms', 0)
        tts_latency = interaction_data.get('tts_latency_ms', 0)
        
        logger.info(f"🎯 Interaction #{len(self.metrics)}")
        logger.info(f"   Total: {total_latency:.0f}ms | STT: {stt_latency:.0f}ms | LLM: {llm_latency:.0f}ms | TTS: {tts_latency:.0f}ms")
        
        # Performance assessment
        if total_latency < 3000:
            logger.info("   ✅ EXCELLENT performance (< 3s)")
        elif total_latency < 5000:
            logger.info("   ⚡ GOOD performance (< 5s)")
        elif total_latency < 10000:
            logger.info("   ⚠️  ACCEPTABLE performance (< 10s)")
        else:
            logger.warning("   🐌 SLOW performance (> 10s)")
    
    def get_summary(self) -> Dict:
        """Get performance summary statistics."""
        if not self.metrics:
            return {"message": "No interactions recorded yet"}
        
        total_latencies = [m.get('total_latency_ms', 0) for m in self.metrics]
        stt_latencies = [m.get('stt_latency_ms', 0) for m in self.metrics if m.get('stt_latency_ms')]
        llm_latencies = [m.get('llm_latency_ms', 0) for m in self.metrics if m.get('llm_latency_ms')]
        tts_latencies = [m.get('tts_latency_ms', 0) for m in self.metrics if m.get('tts_latency_ms')]
        
        def safe_avg(lst): return sum(lst) / len(lst) if lst else 0
        def safe_min(lst): return min(lst) if lst else 0
        def safe_max(lst): return max(lst) if lst else 0
        
        return {
            "total_interactions": len(self.metrics),
            "session_duration_minutes": (time.time() - self.session_start) / 60,
            "average_total_latency_ms": safe_avg(total_latencies),
            "min_total_latency_ms": safe_min(total_latencies),
            "max_total_latency_ms": safe_max(total_latencies),
            "average_stt_latency_ms": safe_avg(stt_latencies),
            "average_llm_latency_ms": safe_avg(llm_latencies),
            "average_tts_latency_ms": safe_avg(tts_latencies),
            "interactions_under_3s": len([l for l in total_latencies if l < 3000]),
            "interactions_under_5s": len([l for l in total_latencies if l < 5000]),
            "performance_grade": self._calculate_grade(total_latencies)
        }
    
    def _calculate_grade(self, latencies: List[float]) -> str:
        """Calculate overall performance grade."""
        if not latencies:
            return "N/A"
        
        avg_latency = sum(latencies) / len(latencies)
        under_3s_pct = len([l for l in latencies if l < 3000]) / len(latencies) * 100
        
        if avg_latency < 2000 and under_3s_pct > 80:
            return "A+ (Excellent)"
        elif avg_latency < 3000 and under_3s_pct > 60:
            return "A (Very Good)"
        elif avg_latency < 5000 and under_3s_pct > 40:
            return "B (Good)"
        elif avg_latency < 8000:
            return "C (Acceptable)"
        else:
            return "D (Needs Improvement)"
    
    def save_metrics(self, filename: str = None):
        """Save metrics to JSON file."""
        if not filename:
            filename = f"voice_metrics/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump({
                "summary": self.get_summary(),
                "interactions": self.metrics
            }, f, indent=2)
        
        logger.info(f"📊 Metrics saved to {filename}")

async def test_voice_performance():
    """Test voice performance by simulating interactions."""
    monitor = PerformanceMonitor()
    
    logger.info("🚀 Starting Voice Agent performance test...")
    logger.info("📝 This will simulate voice interactions and measure latency")
    
    # Test data - simulate different types of interactions
    test_phrases = [
        "Hello, how are you?",
        "What's the weather like today?",
        "Tell me a joke",
        "What can you help me with?",
        "Thank you, goodbye"
    ]
    
    try:
        # Connect to Voice Agent WebSocket
        uri = "ws://localhost:8000/ws/demo_token"  # Use demo token for testing
        
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected to Voice Agent WebSocket")
            
            for i, phrase in enumerate(test_phrases, 1):
                logger.info(f"\n🎤 Test {i}/{len(test_phrases)}: '{phrase}'")
                
                start_time = time.time()
                
                # Send text message (simulating voice input)
                message = {
                    "type": "text",
                    "text": phrase,
                    "language": "en"
                }
                
                await websocket.send(json.dumps(message))
                
                # Wait for response
                stt_time = llm_time = tts_time = 0
                total_start = time.time()
                
                try:
                    # Wait for text response
                    response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    response_data = json.loads(response)
                    
                    total_time = (time.time() - total_start) * 1000
                    
                    # Log the interaction
                    monitor.log_interaction({
                        "test_number": i,
                        "input_text": phrase,
                        "response_type": response_data.get("type"),
                        "total_latency_ms": total_time,
                        "stt_latency_ms": stt_time,  # Would be populated in real voice test
                        "llm_latency_ms": total_time * 0.6,  # Estimate
                        "tts_latency_ms": total_time * 0.3,  # Estimate
                        "success": True
                    })
                    
                    # Small delay between tests
                    await asyncio.sleep(1)
                    
                except asyncio.TimeoutError:
                    logger.error(f"❌ Test {i} timed out after 30 seconds")
                    monitor.log_interaction({
                        "test_number": i,
                        "input_text": phrase,
                        "total_latency_ms": 30000,
                        "success": False,
                        "error": "timeout"
                    })
                
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("📊 PERFORMANCE TEST SUMMARY")
    logger.info("="*60)
    
    summary = monitor.get_summary()
    logger.info(f"Total Interactions: {summary['total_interactions']}")
    logger.info(f"Average Latency: {summary['average_total_latency_ms']:.0f}ms")
    logger.info(f"Best Response: {summary['min_total_latency_ms']:.0f}ms")
    logger.info(f"Worst Response: {summary['max_total_latency_ms']:.0f}ms")
    logger.info(f"Under 3s: {summary['interactions_under_3s']}/{summary['total_interactions']}")
    logger.info(f"Performance Grade: {summary['performance_grade']}")
    
    # Save results
    monitor.save_metrics()
    
    logger.info("\n🎯 Performance test complete!")

if __name__ == "__main__":
    asyncio.run(test_voice_performance())