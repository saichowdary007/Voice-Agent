#!/usr/bin/env python3
"""
WebSocket Audio Fixes Validation Script
Validates that all fixes work together to achieve sub-3-second response times
and eliminate 1011 errors while maintaining connection stability.
"""

import asyncio
import json
import time
import base64
import statistics
import websockets
import numpy as np
from typing import List, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import psutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketFixesValidator:
    """Validates WebSocket audio fixes implementation."""
    
    def __init__(self, server_url: str = "ws://localhost:8000"):
        self.server_url = server_url
        self.test_results = {
            'protocol_negotiation': [],
            'connection_stability': [],
            'audio_processing': [],
            'latency_measurements': [],
            'error_rates': [],
            'concurrent_connections': []
        }
    
    def generate_test_audio(self, duration_ms: int = 250, sample_rate: int = 16000) -> bytes:
        """Generate test audio data matching the required format."""
        samples = int(sample_rate * duration_ms / 1000)
        # Generate sine wave with some noise (simulating speech)
        t = np.linspace(0, duration_ms / 1000, samples, False)
        frequency = 440  # A4 note
        audio_signal = 0.3 * np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.normal(0, 1, samples)
        audio_int16 = (audio_signal * 32767).astype(np.int16)
        return audio_int16.tobytes()
    
    async def test_protocol_negotiation(self) -> Dict[str, Any]:
        """Test WebSocket protocol negotiation for both binary and stream-audio."""
        logger.info("🔍 Testing WebSocket protocol negotiation...")
        
        results = {
            'binary_protocol': False,
            'stream_audio_protocol': False,
            'no_protocol': False,
            'errors': []
        }
        
        # Test binary protocol
        try:
            async with websockets.connect(
                f"{self.server_url}/ws/test_token",
                subprotocols=["binary"]
            ) as websocket:
                logger.info(f"✅ Binary protocol negotiated: {websocket.subprotocol}")
                results['binary_protocol'] = websocket.subprotocol == "binary"
        except Exception as e:
            results['errors'].append(f"Binary protocol test failed: {e}")
        
        # Test stream-audio protocol
        try:
            async with websockets.connect(
                f"{self.server_url}/ws/test_token",
                subprotocols=["stream-audio"]
            ) as websocket:
                logger.info(f"✅ Stream-audio protocol negotiated: {websocket.subprotocol}")
                results['stream_audio_protocol'] = websocket.subprotocol == "stream-audio"
        except Exception as e:
            results['errors'].append(f"Stream-audio protocol test failed: {e}")
        
        # Test no protocol (backward compatibility)
        try:
            async with websockets.connect(f"{self.server_url}/ws/test_token") as websocket:
                logger.info("✅ No protocol connection successful")
                results['no_protocol'] = True
        except Exception as e:
            results['errors'].append(f"No protocol test failed: {e}")
        
        self.test_results['protocol_negotiation'].append(results)
        return results
    
    async def test_connection_stability(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """Test connection stability over time to ensure 1011 errors are eliminated."""
        logger.info(f"🔍 Testing connection stability for {duration_minutes} minutes...")
        
        results = {
            'duration_seconds': duration_minutes * 60,
            'connection_drops': 0,
            'error_1011_count': 0,
            'heartbeat_responses': 0,
            'total_messages': 0,
            'errors': []
        }
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        try:
            async with websockets.connect(
                f"{self.server_url}/ws/test_token",
                subprotocols=["binary"],
                ping_interval=20,
                ping_timeout=10
            ) as websocket:
                
                while time.time() < end_time:
                    try:
                        # Send heartbeat
                        heartbeat_msg = {
                            "type": "heartbeat",
                            "timestamp": int(time.time() * 1000)
                        }
                        await websocket.send(json.dumps(heartbeat_msg))
                        results['total_messages'] += 1
                        
                        # Wait for response
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                            response_data = json.loads(response)
                            if response_data.get('type') == 'heartbeat_ack':
                                results['heartbeat_responses'] += 1
                        except asyncio.TimeoutError:
                            logger.warning("⚠️ Heartbeat timeout")
                        
                        # Wait before next heartbeat
                        await asyncio.sleep(10)
                        
                    except websockets.exceptions.ConnectionClosed as e:
                        results['connection_drops'] += 1
                        if e.code == 1011:
                            results['error_1011_count'] += 1
                        logger.error(f"❌ Connection closed: {e.code} - {e.reason}")
                        break
                    except Exception as e:
                        results['errors'].append(str(e))
                        logger.error(f"❌ Connection error: {e}")
                        break
        
        except Exception as e:
            results['errors'].append(f"Connection test failed: {e}")
        
        # Calculate success rate
        if results['total_messages'] > 0:
            results['heartbeat_success_rate'] = results['heartbeat_responses'] / results['total_messages']
        else:
            results['heartbeat_success_rate'] = 0.0
        
        self.test_results['connection_stability'].append(results)
        return results
    
    async def test_audio_processing_performance(self, num_chunks: int = 50) -> Dict[str, Any]:
        """Test audio processing performance with correct format."""
        logger.info(f"🔍 Testing audio processing performance with {num_chunks} chunks...")
        
        results = {
            'total_chunks': num_chunks,
            'successful_chunks': 0,
            'processing_times': [],
            'chunk_sizes': [],
            'format_validation_passes': 0,
            'errors': []
        }
        
        try:
            async with websockets.connect(
                f"{self.server_url}/ws/test_token",
                subprotocols=["binary"]
            ) as websocket:
                
                for i in range(num_chunks):
                    try:
                        # Generate test audio (250ms, mono, 16kHz as per requirements)
                        test_audio = self.generate_test_audio(duration_ms=250)
                        encoded_audio = base64.b64encode(test_audio).decode('ascii')
                        
                        # Create audio message
                        audio_msg = {
                            "type": "audio_chunk",
                            "data": encoded_audio,
                            "is_final": i == num_chunks - 1,
                            "format": "pcm_s16le",
                            "sample_rate": 16000,
                            "channels": 1,
                            "timestamp": int(time.time() * 1000)
                        }
                        
                        # Measure processing time
                        start_time = time.perf_counter()
                        await websocket.send(json.dumps(audio_msg))
                        
                        # Wait for acknowledgment
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                            processing_time = (time.perf_counter() - start_time) * 1000
                            
                            results['processing_times'].append(processing_time)
                            results['chunk_sizes'].append(len(test_audio))
                            results['successful_chunks'] += 1
                            
                            # Check if format validation passed
                            response_data = json.loads(response)
                            if response_data.get('type') != 'error':
                                results['format_validation_passes'] += 1
                            
                        except asyncio.TimeoutError:
                            logger.warning(f"⚠️ Timeout on chunk {i}")
                        
                        # Small delay between chunks
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        results['errors'].append(f"Chunk {i} failed: {e}")
        
        except Exception as e:
            results['errors'].append(f"Audio processing test failed: {e}")
        
        # Calculate statistics
        if results['processing_times']:
            results['avg_processing_time'] = statistics.mean(results['processing_times'])
            results['max_processing_time'] = max(results['processing_times'])
            results['min_processing_time'] = min(results['processing_times'])
            results['processing_time_std'] = statistics.stdev(results['processing_times']) if len(results['processing_times']) > 1 else 0
        
        results['success_rate'] = results['successful_chunks'] / results['total_chunks']
        results['format_validation_rate'] = results['format_validation_passes'] / results['total_chunks']
        
        self.test_results['audio_processing'].append(results)
        return results
    
    async def test_end_to_end_latency(self, num_tests: int = 10) -> Dict[str, Any]:
        """Test end-to-end latency to ensure sub-3-second response times."""
        logger.info(f"🔍 Testing end-to-end latency with {num_tests} voice interactions...")
        
        results = {
            'total_tests': num_tests,
            'successful_tests': 0,
            'latencies': [],
            'sub_3_second_count': 0,
            'errors': []
        }
        
        try:
            async with websockets.connect(
                f"{self.server_url}/ws/test_token",
                subprotocols=["binary"]
            ) as websocket:
                
                for i in range(num_tests):
                    try:
                        # Generate longer audio for realistic test (1 second)
                        test_audio = self.generate_test_audio(duration_ms=1000)
                        encoded_audio = base64.b64encode(test_audio).decode('ascii')
                        
                        # Create final audio message
                        audio_msg = {
                            "type": "audio_chunk",
                            "data": encoded_audio,
                            "is_final": True,
                            "language": "en",
                            "timestamp": int(time.time() * 1000)
                        }
                        
                        # Measure end-to-end latency
                        start_time = time.perf_counter()
                        await websocket.send(json.dumps(audio_msg))
                        
                        # Wait for final response (text or audio)
                        response_received = False
                        timeout_start = time.perf_counter()
                        
                        while not response_received and (time.perf_counter() - timeout_start) < 5.0:
                            try:
                                response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                                response_data = json.loads(response)
                                
                                # Check for final response types
                                if response_data.get('type') in ['text_response', 'tts_audio', 'audio_response']:
                                    end_time = time.perf_counter()
                                    latency = (end_time - start_time) * 1000  # Convert to ms
                                    
                                    results['latencies'].append(latency)
                                    results['successful_tests'] += 1
                                    
                                    if latency < 3000:  # Sub-3-second requirement
                                        results['sub_3_second_count'] += 1
                                    
                                    response_received = True
                                    logger.info(f"✅ Test {i+1}: {latency:.1f}ms latency")
                                
                            except asyncio.TimeoutError:
                                continue
                        
                        if not response_received:
                            results['errors'].append(f"Test {i+1}: No response received within 5 seconds")
                        
                        # Wait between tests
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        results['errors'].append(f"Test {i+1} failed: {e}")
        
        except Exception as e:
            results['errors'].append(f"Latency test failed: {e}")
        
        # Calculate statistics
        if results['latencies']:
            results['avg_latency'] = statistics.mean(results['latencies'])
            results['max_latency'] = max(results['latencies'])
            results['min_latency'] = min(results['latencies'])
            results['latency_std'] = statistics.stdev(results['latencies']) if len(results['latencies']) > 1 else 0
        
        results['success_rate'] = results['successful_tests'] / results['total_tests']
        results['sub_3_second_rate'] = results['sub_3_second_count'] / results['total_tests']
        
        self.test_results['latency_measurements'].append(results)
        return results
    
    async def test_concurrent_connections(self, num_connections: int = 10) -> Dict[str, Any]:
        """Test system stability under concurrent connections."""
        logger.info(f"🔍 Testing {num_connections} concurrent connections...")
        
        results = {
            'total_connections': num_connections,
            'successful_connections': 0,
            'connection_times': [],
            'errors': []
        }
        
        async def create_connection(connection_id: int):
            """Create a single WebSocket connection."""
            try:
                start_time = time.perf_counter()
                async with websockets.connect(
                    f"{self.server_url}/ws/test_token_{connection_id}",
                    subprotocols=["binary"]
                ) as websocket:
                    connection_time = (time.perf_counter() - start_time) * 1000
                    results['connection_times'].append(connection_time)
                    
                    # Send a test message
                    test_msg = {
                        "type": "connection",
                        "message": f"Test connection {connection_id}",
                        "timestamp": int(time.time() * 1000)
                    }
                    await websocket.send(json.dumps(test_msg))
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        results['successful_connections'] += 1
                        logger.info(f"✅ Connection {connection_id}: {connection_time:.1f}ms")
                    except asyncio.TimeoutError:
                        logger.warning(f"⚠️ Connection {connection_id}: No response")
                    
                    # Keep connection alive briefly
                    await asyncio.sleep(2)
                    
            except Exception as e:
                results['errors'].append(f"Connection {connection_id} failed: {e}")
        
        # Create all connections concurrently
        tasks = [create_connection(i) for i in range(num_connections)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate statistics
        if results['connection_times']:
            results['avg_connection_time'] = statistics.mean(results['connection_times'])
            results['max_connection_time'] = max(results['connection_times'])
        
        results['success_rate'] = results['successful_connections'] / results['total_connections']
        
        self.test_results['concurrent_connections'].append(results)
        return results
    
    def measure_system_performance(self) -> Dict[str, Any]:
        """Measure system performance metrics."""
        logger.info("🔍 Measuring system performance...")
        
        # Get CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Get process-specific metrics if possible
        try:
            current_process = psutil.Process()
            process_memory = current_process.memory_info()
            process_cpu = current_process.cpu_percent()
        except:
            process_memory = None
            process_cpu = None
        
        return {
            'system_cpu_percent': cpu_percent,
            'system_memory_percent': memory.percent,
            'system_memory_available': memory.available,
            'process_memory_rss': process_memory.rss if process_memory else None,
            'process_memory_vms': process_memory.vms if process_memory else None,
            'process_cpu_percent': process_cpu,
            'timestamp': time.time()
        }
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of all WebSocket fixes."""
        logger.info("🚀 Starting comprehensive WebSocket fixes validation...")
        
        validation_results = {
            'start_time': time.time(),
            'system_performance_before': self.measure_system_performance(),
            'tests': {}
        }
        
        try:
            # Test 1: Protocol Negotiation
            logger.info("\n" + "="*50)
            logger.info("TEST 1: Protocol Negotiation")
            logger.info("="*50)
            validation_results['tests']['protocol_negotiation'] = await self.test_protocol_negotiation()
            
            # Test 2: Connection Stability (shorter duration for testing)
            logger.info("\n" + "="*50)
            logger.info("TEST 2: Connection Stability")
            logger.info("="*50)
            validation_results['tests']['connection_stability'] = await self.test_connection_stability(duration_minutes=1)
            
            # Test 3: Audio Processing Performance
            logger.info("\n" + "="*50)
            logger.info("TEST 3: Audio Processing Performance")
            logger.info("="*50)
            validation_results['tests']['audio_processing'] = await self.test_audio_processing_performance(num_chunks=20)
            
            # Test 4: End-to-End Latency
            logger.info("\n" + "="*50)
            logger.info("TEST 4: End-to-End Latency")
            logger.info("="*50)
            validation_results['tests']['latency'] = await self.test_end_to_end_latency(num_tests=5)
            
            # Test 5: Concurrent Connections
            logger.info("\n" + "="*50)
            logger.info("TEST 5: Concurrent Connections")
            logger.info("="*50)
            validation_results['tests']['concurrent_connections'] = await self.test_concurrent_connections(num_connections=5)
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            validation_results['error'] = str(e)
        
        validation_results['end_time'] = time.time()
        validation_results['total_duration'] = validation_results['end_time'] - validation_results['start_time']
        validation_results['system_performance_after'] = self.measure_system_performance()
        
        return validation_results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive validation report."""
        report = []
        report.append("🎯 WEBSOCKET AUDIO FIXES VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Total Duration: {results['total_duration']:.1f} seconds")
        report.append(f"Start Time: {time.ctime(results['start_time'])}")
        report.append(f"End Time: {time.ctime(results['end_time'])}")
        report.append("")
        
        # Protocol Negotiation Results
        if 'protocol_negotiation' in results['tests']:
            pn = results['tests']['protocol_negotiation']
            report.append("📡 PROTOCOL NEGOTIATION")
            report.append("-" * 30)
            report.append(f"Binary Protocol: {'✅ PASS' if pn['binary_protocol'] else '❌ FAIL'}")
            report.append(f"Stream-Audio Protocol: {'✅ PASS' if pn['stream_audio_protocol'] else '❌ FAIL'}")
            report.append(f"No Protocol (Compatibility): {'✅ PASS' if pn['no_protocol'] else '❌ FAIL'}")
            if pn['errors']:
                report.append(f"Errors: {len(pn['errors'])}")
            report.append("")
        
        # Connection Stability Results
        if 'connection_stability' in results['tests']:
            cs = results['tests']['connection_stability']
            report.append("🔗 CONNECTION STABILITY")
            report.append("-" * 30)
            report.append(f"Duration: {cs['duration_seconds']} seconds")
            report.append(f"Connection Drops: {cs['connection_drops']}")
            report.append(f"1011 Errors: {cs['error_1011_count']} {'✅ ELIMINATED' if cs['error_1011_count'] == 0 else '❌ STILL PRESENT'}")
            report.append(f"Heartbeat Success Rate: {cs['heartbeat_success_rate']:.1%}")
            report.append("")
        
        # Audio Processing Results
        if 'audio_processing' in results['tests']:
            ap = results['tests']['audio_processing']
            report.append("🎵 AUDIO PROCESSING")
            report.append("-" * 30)
            report.append(f"Success Rate: {ap['success_rate']:.1%}")
            report.append(f"Format Validation Rate: {ap['format_validation_rate']:.1%}")
            if ap.get('avg_processing_time'):
                report.append(f"Avg Processing Time: {ap['avg_processing_time']:.1f}ms")
                report.append(f"Max Processing Time: {ap['max_processing_time']:.1f}ms")
            report.append("")
        
        # Latency Results
        if 'latency' in results['tests']:
            lat = results['tests']['latency']
            report.append("⚡ END-TO-END LATENCY")
            report.append("-" * 30)
            report.append(f"Success Rate: {lat['success_rate']:.1%}")
            report.append(f"Sub-3-Second Rate: {lat['sub_3_second_rate']:.1%} {'✅ TARGET MET' if lat['sub_3_second_rate'] >= 0.8 else '❌ TARGET MISSED'}")
            if lat.get('avg_latency'):
                report.append(f"Average Latency: {lat['avg_latency']:.1f}ms")
                report.append(f"Max Latency: {lat['max_latency']:.1f}ms")
                report.append(f"Min Latency: {lat['min_latency']:.1f}ms")
            report.append("")
        
        # Concurrent Connections Results
        if 'concurrent_connections' in results['tests']:
            cc = results['tests']['concurrent_connections']
            report.append("🔀 CONCURRENT CONNECTIONS")
            report.append("-" * 30)
            report.append(f"Success Rate: {cc['success_rate']:.1%}")
            if cc.get('avg_connection_time'):
                report.append(f"Avg Connection Time: {cc['avg_connection_time']:.1f}ms")
                report.append(f"Max Connection Time: {cc['max_connection_time']:.1f}ms")
            report.append("")
        
        # Overall Assessment
        report.append("🎯 OVERALL ASSESSMENT")
        report.append("-" * 30)
        
        # Check if key requirements are met
        protocol_ok = results['tests'].get('protocol_negotiation', {}).get('binary_protocol', False)
        no_1011_errors = results['tests'].get('connection_stability', {}).get('error_1011_count', 1) == 0
        latency_ok = results['tests'].get('latency', {}).get('sub_3_second_rate', 0) >= 0.8
        
        if protocol_ok and no_1011_errors and latency_ok:
            report.append("✅ ALL CRITICAL REQUIREMENTS MET")
            report.append("   - Protocol negotiation working")
            report.append("   - 1011 errors eliminated")
            report.append("   - Sub-3-second response times achieved")
        else:
            report.append("❌ SOME REQUIREMENTS NOT MET")
            if not protocol_ok:
                report.append("   - Protocol negotiation issues")
            if not no_1011_errors:
                report.append("   - 1011 errors still present")
            if not latency_ok:
                report.append("   - Latency targets not met")
        
        return "\n".join(report)


async def main():
    """Main validation function."""
    # Check if server is running
    server_url = os.getenv("WEBSOCKET_SERVER_URL", "ws://localhost:8000")
    
    validator = WebSocketFixesValidator(server_url)
    
    try:
        # Run comprehensive validation
        results = await validator.run_comprehensive_validation()
        
        # Generate and display report
        report = validator.generate_report(results)
        print("\n" + report)
        
        # Save results to file
        with open("websocket_fixes_validation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("\n✅ Validation complete! Results saved to websocket_fixes_validation_results.json")
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())