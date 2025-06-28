#!/usr/bin/env python3
"""
Real-time WebSocket monitoring dashboard for voice pipeline diagnostics.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional
import websockets
import aiohttp
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeMonitor:
    """Real-time monitoring of WebSocket connections and voice pipeline."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")
        self.is_monitoring = False
        
        # Metrics storage (last 100 entries)
        self.connection_times = deque(maxlen=100)
        self.response_times = deque(maxlen=100)
        self.error_counts = defaultdict(int)
        self.active_connections = 0
        self.total_messages = 0
        self.failed_messages = 0
        
        # Pipeline stage timings
        self.stt_times = deque(maxlen=50)
        self.llm_times = deque(maxlen=50)
        self.tts_times = deque(maxlen=50)
        self.end_to_end_times = deque(maxlen=50)
        
    async def start_monitoring(self, duration: int = 300):
        """Start real-time monitoring for specified duration (seconds)."""
        logger.info(f"🔍 Starting real-time monitoring for {duration} seconds...")
        self.is_monitoring = True
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitor_connections()),
            asyncio.create_task(self._monitor_pipeline_performance()),
            asyncio.create_task(self._display_dashboard()),
            asyncio.create_task(self._auto_stop(duration))
        ]
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        finally:
            self.is_monitoring = False
    
    async def _auto_stop(self, duration: int):
        """Automatically stop monitoring after duration."""
        await asyncio.sleep(duration)
        self.is_monitoring = False
        logger.info(f"⏰ Monitoring completed after {duration} seconds")
    
    async def _monitor_connections(self):
        """Monitor WebSocket connection health."""
        while self.is_monitoring:
            try:
                # Test connection
                start_time = time.perf_counter()
                
                # Get auth token
                async with aiohttp.ClientSession() as session:
                    auth_payload = {
                        "email": f"monitor_{int(time.time())}@example.com",
                        "password": "test123"
                    }
                    
                    async with session.post(
                        f"{self.base_url}/api/auth/debug-signin",
                        json=auth_payload,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            auth_data = await resp.json()
                            token = auth_data["access_token"]
                        else:
                            self.error_counts["auth_failed"] += 1
                            await asyncio.sleep(5)
                            continue
                
                # Test WebSocket connection
                ws_uri = f"{self.ws_url}/ws/{token}"
                
                async with websockets.connect(
                    ws_uri, 
                    timeout=5,
                    ping_interval=20
                ) as websocket:
                    connect_time = (time.perf_counter() - start_time) * 1000
                    self.connection_times.append(connect_time)
                    self.active_connections += 1
                    
                    # Send test message
                    test_message = {
                        "type": "ping",
                        "timestamp": time.time()
                    }
                    
                    message_start = time.perf_counter()
                    await websocket.send(json.dumps(test_message))
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        response_time = (time.perf_counter() - message_start) * 1000
                        self.response_times.append(response_time)
                        self.total_messages += 1
                    except asyncio.TimeoutError:
                        self.error_counts["response_timeout"] += 1
                        self.failed_messages += 1
                    
                    self.active_connections -= 1
                    
            except websockets.exceptions.ConnectionClosed:
                self.error_counts["connection_closed"] += 1
            except websockets.exceptions.InvalidURI:
                self.error_counts["invalid_uri"] += 1
            except Exception as e:
                self.error_counts["connection_error"] += 1
                logger.debug(f"Connection monitoring error: {e}")
            
            await asyncio.sleep(2)  # Test every 2 seconds
    
    async def _monitor_pipeline_performance(self):
        """Monitor voice pipeline performance."""
        while self.is_monitoring:
            try:
                # Get auth token
                async with aiohttp.ClientSession() as session:
                    auth_payload = {
                        "email": f"pipeline_test_{int(time.time())}@example.com",
                        "password": "test123"
                    }
                    
                    async with session.post(
                        f"{self.base_url}/api/auth/debug-signin",
                        json=auth_payload,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status != 200:
                            await asyncio.sleep(10)
                            continue
                        
                        auth_data = await resp.json()
                        token = auth_data["access_token"]
                
                # Test pipeline via WebSocket
                ws_uri = f"{self.ws_url}/ws/{token}"
                
                async with websockets.connect(ws_uri, timeout=10) as websocket:
                    # Send text message to test pipeline
                    pipeline_start = time.perf_counter()
                    
                    test_message = {
                        "type": "text_message",
                        "text": "Hello, how are you today?",
                        "language": "en"
                    }
                    
                    await websocket.send(json.dumps(test_message))
                    
                    # Track pipeline stages
                    stt_time = None
                    llm_time = None
                    tts_time = None
                    
                    timeout_time = time.time() + 15  # 15 second timeout
                    
                    while time.time() < timeout_time:
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                            response_data = json.loads(response)
                            current_time = time.perf_counter()
                            
                            response_type = response_data.get("type")
                            
                            if response_type == "stt_result" and stt_time is None:
                                stt_time = (current_time - pipeline_start) * 1000
                                self.stt_times.append(stt_time)
                                
                            elif response_type == "text_response" and llm_time is None:
                                llm_time = (current_time - pipeline_start) * 1000
                                self.llm_times.append(llm_time)
                                
                            elif response_type in ["audio_stream", "tts_audio"] and tts_time is None:
                                tts_time = (current_time - pipeline_start) * 1000
                                self.tts_times.append(tts_time)
                                
                                # Calculate end-to-end time
                                end_to_end = (current_time - pipeline_start) * 1000
                                self.end_to_end_times.append(end_to_end)
                                break
                                
                        except asyncio.TimeoutError:
                            break
                    
            except Exception as e:
                logger.debug(f"Pipeline monitoring error: {e}")
            
            await asyncio.sleep(10)  # Test pipeline every 10 seconds
    
    async def _display_dashboard(self):
        """Display real-time monitoring dashboard."""
        while self.is_monitoring:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H")
            
            # Header
            print("=" * 80)
            print("🔍 VOICE AGENT WEBSOCKET MONITORING DASHBOARD")
            print("=" * 80)
            print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Connection Statistics
            print("🔌 CONNECTION STATISTICS")
            print("-" * 40)
            if self.connection_times:
                avg_connect = sum(self.connection_times) / len(self.connection_times)
                max_connect = max(self.connection_times)
                min_connect = min(self.connection_times)
                print(f"Average Connection Time: {avg_connect:.1f}ms")
                print(f"Min/Max Connection Time: {min_connect:.1f}ms / {max_connect:.1f}ms")
            else:
                print("No connection data available")
            
            print(f"Active Connections: {self.active_connections}")
            print(f"Total Messages: {self.total_messages}")
            print(f"Failed Messages: {self.failed_messages}")
            
            if self.total_messages > 0:
                success_rate = ((self.total_messages - self.failed_messages) / self.total_messages) * 100
                print(f"Success Rate: {success_rate:.1f}%")
            print()
            
            # Response Time Statistics
            print("⚡ RESPONSE TIME STATISTICS")
            print("-" * 40)
            if self.response_times:
                avg_response = sum(self.response_times) / len(self.response_times)
                max_response = max(self.response_times)
                min_response = min(self.response_times)
                print(f"Average Response Time: {avg_response:.1f}ms")
                print(f"Min/Max Response Time: {min_response:.1f}ms / {max_response:.1f}ms")
            else:
                print("No response time data available")
            print()
            
            # Pipeline Performance
            print("🎤 VOICE PIPELINE PERFORMANCE")
            print("-" * 40)
            
            if self.end_to_end_times:
                avg_e2e = sum(self.end_to_end_times) / len(self.end_to_end_times)
                print(f"Average End-to-End: {avg_e2e:.1f}ms")
                
                if self.stt_times:
                    avg_stt = sum(self.stt_times) / len(self.stt_times)
                    print(f"Average STT Time: {avg_stt:.1f}ms")
                
                if self.llm_times:
                    avg_llm = sum(self.llm_times) / len(self.llm_times)
                    print(f"Average LLM Time: {avg_llm:.1f}ms")
                
                if self.tts_times:
                    avg_tts = sum(self.tts_times) / len(self.tts_times)
                    print(f"Average TTS Time: {avg_tts:.1f}ms")
            else:
                print("No pipeline performance data available")
            print()
            
            # Error Statistics
            print("❌ ERROR STATISTICS")
            print("-" * 40)
            if self.error_counts:
                for error_type, count in self.error_counts.items():
                    print(f"{error_type}: {count}")
            else:
                print("No errors detected")
            print()
            
            # Performance Indicators
            print("🎯 PERFORMANCE INDICATORS")
            print("-" * 40)
            
            # Connection health
            if self.connection_times and len(self.connection_times) >= 5:
                recent_connects = list(self.connection_times)[-5:]
                avg_recent = sum(recent_connects) / len(recent_connects)
                if avg_recent < 1000:
                    print("✅ Connection Health: EXCELLENT")
                elif avg_recent < 2000:
                    print("🟡 Connection Health: GOOD")
                else:
                    print("❌ Connection Health: POOR")
            
            # Pipeline health
            if self.end_to_end_times and len(self.end_to_end_times) >= 3:
                recent_e2e = list(self.end_to_end_times)[-3:]
                avg_recent_e2e = sum(recent_e2e) / len(recent_e2e)
                if avg_recent_e2e < 3000:
                    print("✅ Pipeline Health: EXCELLENT")
                elif avg_recent_e2e < 5000:
                    print("🟡 Pipeline Health: GOOD")
                else:
                    print("❌ Pipeline Health: POOR")
            
            # Error rate
            total_attempts = self.total_messages + self.failed_messages
            if total_attempts > 0:
                error_rate = (self.failed_messages / total_attempts) * 100
                if error_rate < 5:
                    print("✅ Error Rate: LOW")
                elif error_rate < 15:
                    print("🟡 Error Rate: MODERATE")
                else:
                    print("❌ Error Rate: HIGH")
            
            print()
            print("Press Ctrl+C to stop monitoring")
            print("=" * 80)
            
            await asyncio.sleep(2)  # Update dashboard every 2 seconds

    def generate_report(self) -> Dict:
        """Generate a summary report of monitoring data."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_duration": "N/A",
            "connection_stats": {
                "total_connections": len(self.connection_times),
                "average_connect_time_ms": sum(self.connection_times) / len(self.connection_times) if self.connection_times else 0,
                "max_connect_time_ms": max(self.connection_times) if self.connection_times else 0,
                "min_connect_time_ms": min(self.connection_times) if self.connection_times else 0
            },
            "response_stats": {
                "total_messages": self.total_messages,
                "failed_messages": self.failed_messages,
                "success_rate": ((self.total_messages - self.failed_messages) / self.total_messages * 100) if self.total_messages > 0 else 0,
                "average_response_time_ms": sum(self.response_times) / len(self.response_times) if self.response_times else 0
            },
            "pipeline_stats": {
                "average_end_to_end_ms": sum(self.end_to_end_times) / len(self.end_to_end_times) if self.end_to_end_times else 0,
                "average_stt_ms": sum(self.stt_times) / len(self.stt_times) if self.stt_times else 0,
                "average_llm_ms": sum(self.llm_times) / len(self.llm_times) if self.llm_times else 0,
                "average_tts_ms": sum(self.tts_times) / len(self.tts_times) if self.tts_times else 0
            },
            "error_counts": dict(self.error_counts)
        }
        
        return report

async def main():
    """Main function for real-time monitoring."""
    monitor = RealTimeMonitor()
    
    try:
        # Start monitoring for 5 minutes by default
        await monitor.start_monitoring(duration=300)
        
        # Generate and save report
        report = monitor.generate_report()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"websocket_monitor_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Monitoring report saved to: {filename}")
        
    except KeyboardInterrupt:
        logger.info("🛑 Monitoring stopped by user")
        
        # Generate final report
        report = monitor.generate_report()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"websocket_monitor_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Final monitoring report saved to: {filename}")

if __name__ == "__main__":
    asyncio.run(main())