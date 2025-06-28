#!/usr/bin/env python3
"""
WebSocket Voice Pipeline Diagnostics Tool
Comprehensive testing and monitoring for the voice interaction pipeline.
"""

import asyncio
import json
import time
import logging
import websockets
import aiohttp
import base64
import statistics
from typing import Dict, List, Optional, Any
from datetime import datetime
import psutil
import threading
from dataclasses import dataclass, field
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LatencyMetrics:
    """Stores latency measurements for different pipeline stages."""
    websocket_connect: List[float] = field(default_factory=list)
    auth_handshake: List[float] = field(default_factory=list)
    audio_upload: List[float] = field(default_factory=list)
    stt_processing: List[float] = field(default_factory=list)
    llm_processing: List[float] = field(default_factory=list)
    tts_processing: List[float] = field(default_factory=list)
    audio_download: List[float] = field(default_factory=list)
    end_to_end: List[float] = field(default_factory=list)

@dataclass
class ConnectionStats:
    """WebSocket connection statistics."""
    total_connections: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    connection_errors: List[str] = field(default_factory=list)
    average_connect_time: float = 0.0
    max_connect_time: float = 0.0
    min_connect_time: float = float('inf')

@dataclass
class PipelineStats:
    """Voice pipeline processing statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_breakdown: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    processing_times: LatencyMetrics = field(default_factory=LatencyMetrics)

class WebSocketDiagnostics:
    """Comprehensive WebSocket and voice pipeline diagnostics."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")
        self.connection_stats = ConnectionStats()
        self.pipeline_stats = PipelineStats()
        self.active_connections: List[websockets.WebSocketServerProtocol] = []
        self.system_metrics = deque(maxlen=100)
        self.is_monitoring = False
        
    async def test_websocket_connectivity(self) -> Dict[str, Any]:
        """Test basic WebSocket connectivity and handshake."""
        logger.info("🔌 Testing WebSocket connectivity...")
        
        results = {
            "endpoint_accessible": False,
            "handshake_successful": False,
            "connection_time_ms": 0,
            "error_message": None,
            "server_response": None
        }
        
        try:
            # Test basic HTTP endpoint first
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as resp:
                    if resp.status == 200:
                        results["endpoint_accessible"] = True
                        logger.info("✅ HTTP endpoint accessible")
                    else:
                        results["error_message"] = f"HTTP endpoint returned {resp.status}"
                        return results
            
            # Test WebSocket connection
            start_time = time.perf_counter()
            
            # Try guest connection first
            ws_uri = f"{self.ws_url}/ws/guest_test_{int(time.time())}"
            
            async with websockets.connect(
                ws_uri,
                timeout=10,
                ping_interval=20,
                ping_timeout=10
            ) as websocket:
                connect_time = (time.perf_counter() - start_time) * 1000
                results["connection_time_ms"] = connect_time
                results["handshake_successful"] = True
                
                # Send ping and wait for response
                ping_message = {
                    "type": "ping",
                    "timestamp": time.time()
                }
                
                await websocket.send(json.dumps(ping_message))
                
                # Wait for response with timeout
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    results["server_response"] = json.loads(response)
                    logger.info(f"✅ WebSocket handshake successful ({connect_time:.1f}ms)")
                except asyncio.TimeoutError:
                    results["error_message"] = "Server did not respond to ping within 5 seconds"
                    
        except websockets.exceptions.ConnectionClosed as e:
            results["error_message"] = f"WebSocket connection closed: {e}"
            logger.error(f"❌ WebSocket connection closed: {e}")
        except websockets.exceptions.InvalidURI as e:
            results["error_message"] = f"Invalid WebSocket URI: {e}"
            logger.error(f"❌ Invalid WebSocket URI: {e}")
        except Exception as e:
            results["error_message"] = f"WebSocket connection failed: {e}"
            logger.error(f"❌ WebSocket connection failed: {e}")
            
        return results

    async def test_authenticated_connection(self) -> Dict[str, Any]:
        """Test WebSocket connection with authentication."""
        logger.info("🔐 Testing authenticated WebSocket connection...")
        
        results = {
            "auth_successful": False,
            "token_obtained": False,
            "websocket_auth_successful": False,
            "auth_time_ms": 0,
            "error_message": None
        }
        
        try:
            # Get authentication token
            auth_start = time.perf_counter()
            
            async with aiohttp.ClientSession() as session:
                auth_payload = {
                    "email": f"test_{int(time.time())}@example.com",
                    "password": "test123"
                }
                
                async with session.post(
                    f"{self.base_url}/api/auth/debug-signin",
                    json=auth_payload
                ) as resp:
                    if resp.status == 200:
                        auth_data = await resp.json()
                        token = auth_data.get("access_token")
                        if token:
                            results["token_obtained"] = True
                            results["auth_successful"] = True
                            auth_time = (time.perf_counter() - auth_start) * 1000
                            results["auth_time_ms"] = auth_time
                            
                            # Test WebSocket with token
                            ws_uri = f"{self.ws_url}/ws/{token}"
                            
                            async with websockets.connect(ws_uri, timeout=10) as websocket:
                                results["websocket_auth_successful"] = True
                                logger.info(f"✅ Authenticated WebSocket connection successful ({auth_time:.1f}ms)")
                                
                                # Test message exchange
                                test_message = {
                                    "type": "text_message",
                                    "text": "Hello, this is a test message",
                                    "language": "en"
                                }
                                
                                await websocket.send(json.dumps(test_message))
                                
                                # Wait for response
                                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                                response_data = json.loads(response)
                                logger.info(f"📨 Received response: {response_data.get('type', 'unknown')}")
                                
                    else:
                        results["error_message"] = f"Authentication failed: {resp.status}"
                        
        except Exception as e:
            results["error_message"] = f"Authentication test failed: {e}"
            logger.error(f"❌ Authentication test failed: {e}")
            
        return results

    async def measure_latency_pipeline(self, num_tests: int = 5) -> Dict[str, Any]:
        """Measure end-to-end latency for the voice pipeline."""
        logger.info(f"⏱️ Measuring pipeline latency ({num_tests} tests)...")
        
        latencies = []
        errors = []
        
        for i in range(num_tests):
            try:
                latency = await self._single_latency_test()
                latencies.append(latency)
                logger.info(f"Test {i+1}/{num_tests}: {latency['end_to_end']:.1f}ms")
            except Exception as e:
                errors.append(str(e))
                logger.error(f"Test {i+1}/{num_tests} failed: {e}")
                
        if latencies:
            end_to_end_times = [l['end_to_end'] for l in latencies]
            results = {
                "total_tests": num_tests,
                "successful_tests": len(latencies),
                "failed_tests": len(errors),
                "average_latency_ms": statistics.mean(end_to_end_times),
                "median_latency_ms": statistics.median(end_to_end_times),
                "min_latency_ms": min(end_to_end_times),
                "max_latency_ms": max(end_to_end_times),
                "std_dev_ms": statistics.stdev(end_to_end_times) if len(end_to_end_times) > 1 else 0,
                "detailed_latencies": latencies,
                "errors": errors
            }
        else:
            results = {
                "total_tests": num_tests,
                "successful_tests": 0,
                "failed_tests": len(errors),
                "errors": errors,
                "error_message": "All latency tests failed"
            }
            
        return results

    async def _single_latency_test(self) -> Dict[str, float]:
        """Perform a single end-to-end latency test."""
        timings = {}
        
        # Get auth token
        auth_start = time.perf_counter()
        async with aiohttp.ClientSession() as session:
            auth_payload = {
                "email": f"latency_test_{int(time.time())}@example.com",
                "password": "test123"
            }
            
            async with session.post(
                f"{self.base_url}/api/auth/debug-signin",
                json=auth_payload
            ) as resp:
                auth_data = await resp.json()
                token = auth_data["access_token"]
                
        timings['auth'] = (time.perf_counter() - auth_start) * 1000
        
        # Connect WebSocket
        ws_start = time.perf_counter()
        ws_uri = f"{self.ws_url}/ws/{token}"
        
        async with websockets.connect(ws_uri, timeout=10) as websocket:
            timings['websocket_connect'] = (time.perf_counter() - ws_start) * 1000
            
            # Send test message
            message_start = time.perf_counter()
            test_message = {
                "type": "text_message",
                "text": "What is the capital of France?",
                "language": "en"
            }
            
            await websocket.send(json.dumps(test_message))
            timings['message_sent'] = (time.perf_counter() - message_start) * 1000
            
            # Wait for response
            response_start = time.perf_counter()
            response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            timings['response_received'] = (time.perf_counter() - response_start) * 1000
            
            # Calculate end-to-end
            timings['end_to_end'] = (time.perf_counter() - message_start) * 1000
            
        return timings

    async def test_voice_pipeline(self) -> Dict[str, Any]:
        """Test the complete voice interaction pipeline."""
        logger.info("🎤 Testing voice pipeline...")
        
        results = {
            "audio_capture_test": False,
            "stt_test": False,
            "llm_test": False,
            "tts_test": False,
            "audio_playback_test": False,
            "pipeline_errors": [],
            "processing_times": {}
        }
        
        try:
            # Generate test audio data (simulate 1 second of audio)
            sample_rate = 16000
            duration = 1.0
            test_audio = self._generate_test_audio(sample_rate, duration)
            
            # Get auth token
            async with aiohttp.ClientSession() as session:
                auth_payload = {
                    "email": f"voice_test_{int(time.time())}@example.com",
                    "password": "test123"
                }
                
                async with session.post(
                    f"{self.base_url}/api/auth/debug-signin",
                    json=auth_payload
                ) as resp:
                    auth_data = await resp.json()
                    token = auth_data["access_token"]
            
            # Test voice pipeline via WebSocket
            ws_uri = f"{self.ws_url}/ws/{token}"
            
            async with websockets.connect(ws_uri, timeout=10) as websocket:
                # Send audio chunk
                audio_start = time.perf_counter()
                
                audio_message = {
                    "type": "audio_chunk",
                    "data": base64.b64encode(test_audio).decode('ascii'),
                    "is_final": True,
                    "sample_rate": sample_rate
                }
                
                await websocket.send(json.dumps(audio_message))
                results["audio_capture_test"] = True
                
                # Wait for processing responses
                timeout_time = time.time() + 30  # 30 second timeout
                
                while time.time() < timeout_time:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        response_data = json.loads(response)
                        
                        response_type = response_data.get("type")
                        
                        if response_type == "stt_result":
                            results["stt_test"] = True
                            results["processing_times"]["stt"] = (time.perf_counter() - audio_start) * 1000
                            logger.info(f"✅ STT processing successful")
                            
                        elif response_type == "text_response":
                            results["llm_test"] = True
                            results["processing_times"]["llm"] = (time.perf_counter() - audio_start) * 1000
                            logger.info(f"✅ LLM processing successful")
                            
                        elif response_type == "audio_stream" or response_type == "tts_audio":
                            results["tts_test"] = True
                            results["audio_playback_test"] = True
                            results["processing_times"]["tts"] = (time.perf_counter() - audio_start) * 1000
                            logger.info(f"✅ TTS processing successful")
                            break
                            
                        elif response_type == "error":
                            results["pipeline_errors"].append(response_data.get("message", "Unknown error"))
                            
                    except asyncio.TimeoutError:
                        break
                        
                results["processing_times"]["end_to_end"] = (time.perf_counter() - audio_start) * 1000
                
        except Exception as e:
            results["pipeline_errors"].append(f"Voice pipeline test failed: {e}")
            logger.error(f"❌ Voice pipeline test failed: {e}")
            
        return results

    def _generate_test_audio(self, sample_rate: int, duration: float) -> bytes:
        """Generate test audio data (sine wave)."""
        import numpy as np
        
        # Generate a 440Hz sine wave (A note)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        frequency = 440.0
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        return audio_data.tobytes()

    async def monitor_system_resources(self, duration: int = 60) -> Dict[str, Any]:
        """Monitor system resources during testing."""
        logger.info(f"📊 Monitoring system resources for {duration} seconds...")
        
        metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "network_io": [],
            "disk_io": [],
            "duration_seconds": duration
        }
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics["cpu_usage"].append(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics["memory_usage"].append(memory.percent)
            
            # Network I/O
            network = psutil.net_io_counters()
            metrics["network_io"].append({
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv
            })
            
            # Disk I/O
            disk = psutil.disk_io_counters()
            if disk:
                metrics["disk_io"].append({
                    "read_bytes": disk.read_bytes,
                    "write_bytes": disk.write_bytes
                })
            
            await asyncio.sleep(1)
        
        # Calculate averages
        results = {
            "average_cpu_percent": statistics.mean(metrics["cpu_usage"]),
            "max_cpu_percent": max(metrics["cpu_usage"]),
            "average_memory_percent": statistics.mean(metrics["memory_usage"]),
            "max_memory_percent": max(metrics["memory_usage"]),
            "duration_seconds": duration,
            "raw_metrics": metrics
        }
        
        return results

    async def run_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """Run all diagnostic tests and return comprehensive results."""
        logger.info("🔍 Running comprehensive WebSocket and voice pipeline diagnostics...")
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0
            }
        }
        
        # Test 1: Basic WebSocket connectivity
        logger.info("\n" + "="*50)
        logger.info("TEST 1: WebSocket Connectivity")
        logger.info("="*50)
        
        connectivity_results = await self.test_websocket_connectivity()
        results["websocket_connectivity"] = connectivity_results
        results["test_summary"]["total_tests"] += 1
        
        if connectivity_results["handshake_successful"]:
            results["test_summary"]["passed_tests"] += 1
            logger.info("✅ WebSocket connectivity test PASSED")
        else:
            results["test_summary"]["failed_tests"] += 1
            logger.error("❌ WebSocket connectivity test FAILED")
        
        # Test 2: Authentication
        logger.info("\n" + "="*50)
        logger.info("TEST 2: Authentication")
        logger.info("="*50)
        
        auth_results = await self.test_authenticated_connection()
        results["authentication"] = auth_results
        results["test_summary"]["total_tests"] += 1
        
        if auth_results["websocket_auth_successful"]:
            results["test_summary"]["passed_tests"] += 1
            logger.info("✅ Authentication test PASSED")
        else:
            results["test_summary"]["failed_tests"] += 1
            logger.error("❌ Authentication test FAILED")
        
        # Test 3: Latency measurement
        logger.info("\n" + "="*50)
        logger.info("TEST 3: Latency Measurement")
        logger.info("="*50)
        
        latency_results = await self.measure_latency_pipeline(num_tests=3)
        results["latency_measurement"] = latency_results
        results["test_summary"]["total_tests"] += 1
        
        if latency_results.get("successful_tests", 0) > 0:
            results["test_summary"]["passed_tests"] += 1
            logger.info("✅ Latency measurement test PASSED")
        else:
            results["test_summary"]["failed_tests"] += 1
            logger.error("❌ Latency measurement test FAILED")
        
        # Test 4: Voice pipeline
        logger.info("\n" + "="*50)
        logger.info("TEST 4: Voice Pipeline")
        logger.info("="*50)
        
        voice_results = await self.test_voice_pipeline()
        results["voice_pipeline"] = voice_results
        results["test_summary"]["total_tests"] += 1
        
        if voice_results["stt_test"] and voice_results["llm_test"] and voice_results["tts_test"]:
            results["test_summary"]["passed_tests"] += 1
            logger.info("✅ Voice pipeline test PASSED")
        else:
            results["test_summary"]["failed_tests"] += 1
            logger.error("❌ Voice pipeline test FAILED")
        
        # Test 5: System resources (run in background)
        logger.info("\n" + "="*50)
        logger.info("TEST 5: System Resource Monitoring")
        logger.info("="*50)
        
        resource_results = await self.monitor_system_resources(duration=10)
        results["system_resources"] = resource_results
        results["test_summary"]["total_tests"] += 1
        results["test_summary"]["passed_tests"] += 1  # Always passes
        
        # Generate summary report
        logger.info("\n" + "="*50)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("="*50)
        
        summary = results["test_summary"]
        success_rate = (summary["passed_tests"] / summary["total_tests"]) * 100
        
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed_tests']}")
        logger.info(f"Failed: {summary['failed_tests']}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            logger.info("🎉 Overall system health: GOOD")
        elif success_rate >= 60:
            logger.warning("⚠️ Overall system health: FAIR")
        else:
            logger.error("❌ Overall system health: POOR")
        
        return results

async def main():
    """Main function to run diagnostics."""
    diagnostics = WebSocketDiagnostics()
    
    try:
        results = await diagnostics.run_comprehensive_diagnostics()
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"websocket_diagnostics_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📄 Diagnostic results saved to: {filename}")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("🛑 Diagnostics interrupted by user")
    except Exception as e:
        logger.error(f"❌ Diagnostics failed: {e}")
        raise

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import numpy
        import psutil
        import websockets
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install with: pip install numpy psutil websockets")
        exit(1)
    
    asyncio.run(main())