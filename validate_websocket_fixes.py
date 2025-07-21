#!/usr/bin/env python3
"""
Validation script for the 5 critical WebSocket fixes identified in the forensic analysis.
Tests each fix to ensure proper implementation and functionality.
"""

import asyncio
import websockets
import json
import base64
import time
import logging
import subprocess
import sys
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebSocketFixValidator:
    """Validates the 5 critical WebSocket fixes."""
    
    def __init__(self, server_url: str = "ws://localhost:8080"):
        self.server_url = server_url
        self.test_results: Dict[str, Dict[str, Any]] = {}
        
    async def validate_all_fixes(self) -> Dict[str, Dict[str, Any]]:
        """Run all validation tests."""
        logger.info("🧪 Starting WebSocket fixes validation...")
        
        # Test each fix
        await self.test_fix_1_protocol_handshake()
        await self.test_fix_2_ping_timeout()
        await self.test_fix_3_audio_format()
        await self.test_fix_4_ffmpeg_streaming()
        await self.test_fix_5_vad_reset_loop()
        
        # Generate summary
        self.generate_summary()
        
        return self.test_results
    
    async def test_fix_1_protocol_handshake(self):
        """FIX #1: Test WebSocket protocol handshake accepts any protocol."""
        logger.info("🔧 Testing FIX #1: Protocol handshake compatibility...")
        
        test_protocols = [
            "binary",
            "stream-audio", 
            "unknown-protocol",
            None  # No protocol
        ]
        
        results = []
        
        for protocol in test_protocols:
            try:
                # Create WebSocket connection with specific protocol
                extra_headers = {}
                if protocol:
                    extra_headers["Sec-WebSocket-Protocol"] = protocol
                
                # Use demo token for testing
                ws_url = f"{self.server_url}/ws/guest_test_user"
                
                async with websockets.connect(
                    ws_url, 
                    extra_headers=extra_headers,
                    timeout=5
                ) as websocket:
                    # Send a test message
                    test_message = {
                        "type": "connection",
                        "message": f"Testing protocol: {protocol or 'none'}"
                    }
                    await websocket.send(json.dumps(test_message))
                    
                    # Wait for response
                    response = await asyncio.wait_for(websocket.recv(), timeout=3)
                    response_data = json.loads(response)
                    
                    results.append({
                        "protocol": protocol or "none",
                        "status": "success",
                        "response": response_data.get("type", "unknown")
                    })
                    
                    logger.info(f"✅ Protocol '{protocol or 'none'}' accepted successfully")
                    
            except Exception as e:
                results.append({
                    "protocol": protocol or "none", 
                    "status": "failed",
                    "error": str(e)
                })
                logger.error(f"❌ Protocol '{protocol or 'none'}' failed: {e}")
        
        self.test_results["fix_1_protocol_handshake"] = {
            "description": "WebSocket accepts any protocol to avoid 1011 handshake failures",
            "results": results,
            "passed": all(r["status"] == "success" for r in results)
        }
    
    async def test_fix_2_ping_timeout(self):
        """FIX #2: Test ping timeout is disabled."""
        logger.info("🔧 Testing FIX #2: Ping timeout disabled...")
        
        try:
            ws_url = f"{self.server_url}/ws/guest_test_user"
            
            async with websockets.connect(ws_url, timeout=5) as websocket:
                # Send connection message
                await websocket.send(json.dumps({
                    "type": "connection",
                    "message": "Testing ping timeout"
                }))
                
                # Wait for initial response
                await websocket.recv()
                
                # Wait longer than the old ping interval (25 seconds)
                logger.info("⏳ Waiting 30 seconds to test ping timeout...")
                start_time = time.time()
                
                try:
                    # Send periodic heartbeats to keep connection alive
                    for i in range(6):  # 6 * 5 = 30 seconds
                        await asyncio.sleep(5)
                        
                        # Send heartbeat
                        await websocket.send(json.dumps({
                            "type": "heartbeat",
                            "timestamp": time.time()
                        }))
                        
                        # Try to receive response (optional)
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=1)
                            logger.debug(f"Heartbeat response: {response}")
                        except asyncio.TimeoutError:
                            pass  # No response is fine
                    
                    elapsed = time.time() - start_time
                    
                    # If we get here, ping timeout is disabled
                    self.test_results["fix_2_ping_timeout"] = {
                        "description": "Ping timeout disabled to prevent 1011 errors during processing",
                        "elapsed_seconds": elapsed,
                        "passed": True,
                        "status": "success"
                    }
                    logger.info(f"✅ Connection survived {elapsed:.1f}s without ping timeout")
                    
                except websockets.exceptions.ConnectionClosed as e:
                    # Connection closed - ping timeout might still be active
                    elapsed = time.time() - start_time
                    self.test_results["fix_2_ping_timeout"] = {
                        "description": "Ping timeout disabled to prevent 1011 errors during processing",
                        "elapsed_seconds": elapsed,
                        "passed": False,
                        "status": "failed",
                        "error": f"Connection closed after {elapsed:.1f}s: {e}"
                    }
                    logger.error(f"❌ Connection closed after {elapsed:.1f}s - ping timeout may still be active")
                    
        except Exception as e:
            self.test_results["fix_2_ping_timeout"] = {
                "description": "Ping timeout disabled to prevent 1011 errors during processing",
                "passed": False,
                "status": "failed", 
                "error": str(e)
            }
            logger.error(f"❌ Ping timeout test failed: {e}")
    
    async def test_fix_3_audio_format(self):
        """FIX #3: Test audio format is 16kHz mono."""
        logger.info("🔧 Testing FIX #3: Audio format validation...")
        
        # This test validates the configuration, not runtime behavior
        # since we can't easily test browser MediaRecorder from Python
        
        try:
            # Check frontend configuration
            frontend_config_checks = []
            
            # Read the AudioVisualizer component
            try:
                with open("react-frontend/src/components/AudioVisualizer.tsx", "r") as f:
                    content = f.read()
                    
                # Check for 16kHz sample rate
                if "sampleRate: 16000" in content:
                    frontend_config_checks.append({
                        "check": "16kHz sample rate configured",
                        "status": "passed"
                    })
                else:
                    frontend_config_checks.append({
                        "check": "16kHz sample rate configured", 
                        "status": "failed",
                        "error": "sampleRate: 16000 not found"
                    })
                
                # Check for mono channel
                if "channelCount: 1" in content:
                    frontend_config_checks.append({
                        "check": "Mono channel configured",
                        "status": "passed"
                    })
                else:
                    frontend_config_checks.append({
                        "check": "Mono channel configured",
                        "status": "failed", 
                        "error": "channelCount: 1 not found"
                    })
                
                # Check for WebM/Opus format
                if "audio/webm;codecs=opus" in content:
                    frontend_config_checks.append({
                        "check": "WebM/Opus format configured",
                        "status": "passed"
                    })
                else:
                    frontend_config_checks.append({
                        "check": "WebM/Opus format configured",
                        "status": "failed",
                        "error": "audio/webm;codecs=opus not found"
                    })
                
                # Check for 250ms chunks
                if "start(250)" in content:
                    frontend_config_checks.append({
                        "check": "250ms chunks configured",
                        "status": "passed"
                    })
                else:
                    frontend_config_checks.append({
                        "check": "250ms chunks configured",
                        "status": "failed",
                        "error": "start(250) not found"
                    })
                    
            except FileNotFoundError:
                frontend_config_checks.append({
                    "check": "AudioVisualizer.tsx exists",
                    "status": "failed",
                    "error": "File not found"
                })
            
            all_passed = all(check["status"] == "passed" for check in frontend_config_checks)
            
            self.test_results["fix_3_audio_format"] = {
                "description": "Audio format configured as 16kHz mono WebM/Opus with 250ms chunks",
                "checks": frontend_config_checks,
                "passed": all_passed,
                "status": "success" if all_passed else "failed"
            }
            
            if all_passed:
                logger.info("✅ Audio format configuration validated")
            else:
                logger.error("❌ Audio format configuration issues found")
                
        except Exception as e:
            self.test_results["fix_3_audio_format"] = {
                "description": "Audio format configured as 16kHz mono WebM/Opus with 250ms chunks",
                "passed": False,
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Audio format test failed: {e}")
    
    async def test_fix_4_ffmpeg_streaming(self):
        """FIX #4: Test persistent ffmpeg streaming processor."""
        logger.info("🔧 Testing FIX #4: Persistent ffmpeg streaming...")
        
        try:
            # Check if StreamingAudioProcessor is properly implemented
            implementation_checks = []
            
            # Read the audio preprocessor
            try:
                with open("src/audio_preprocessor.py", "r") as f:
                    content = f.read()
                
                # Check for persistent ffmpeg process
                if "self.ffmpeg_process: Optional[subprocess.Popen] = None" in content:
                    implementation_checks.append({
                        "check": "Persistent ffmpeg process attribute",
                        "status": "passed"
                    })
                else:
                    implementation_checks.append({
                        "check": "Persistent ffmpeg process attribute",
                        "status": "failed",
                        "error": "ffmpeg_process attribute not found"
                    })
                
                # Check for streaming initialization
                if "initialize_streaming_ffmpeg" in content:
                    implementation_checks.append({
                        "check": "Streaming ffmpeg initialization method",
                        "status": "passed"
                    })
                else:
                    implementation_checks.append({
                        "check": "Streaming ffmpeg initialization method",
                        "status": "failed",
                        "error": "initialize_streaming_ffmpeg method not found"
                    })
                
                # Check for low delay flags
                if "'-fflags', '+nobuffer'" in content and "'-flags', 'low_delay'" in content:
                    implementation_checks.append({
                        "check": "Low delay ffmpeg flags",
                        "status": "passed"
                    })
                else:
                    implementation_checks.append({
                        "check": "Low delay ffmpeg flags",
                        "status": "failed",
                        "error": "Low delay flags not found"
                    })
                
                # Check for process reuse logic
                if "if not self.is_initialized" in content:
                    implementation_checks.append({
                        "check": "Process reuse logic",
                        "status": "passed"
                    })
                else:
                    implementation_checks.append({
                        "check": "Process reuse logic",
                        "status": "failed",
                        "error": "Process reuse logic not found"
                    })
                    
            except FileNotFoundError:
                implementation_checks.append({
                    "check": "audio_preprocessor.py exists",
                    "status": "failed",
                    "error": "File not found"
                })
            
            # Test actual ffmpeg availability
            try:
                result = subprocess.run(["ffmpeg", "-version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    implementation_checks.append({
                        "check": "ffmpeg binary available",
                        "status": "passed"
                    })
                else:
                    implementation_checks.append({
                        "check": "ffmpeg binary available",
                        "status": "failed",
                        "error": "ffmpeg command failed"
                    })
            except (subprocess.TimeoutExpired, FileNotFoundError):
                implementation_checks.append({
                    "check": "ffmpeg binary available",
                    "status": "failed",
                    "error": "ffmpeg not found or timeout"
                })
            
            all_passed = all(check["status"] == "passed" for check in implementation_checks)
            
            self.test_results["fix_4_ffmpeg_streaming"] = {
                "description": "Persistent ffmpeg process eliminates CPU spike from spawning new processes",
                "checks": implementation_checks,
                "passed": all_passed,
                "status": "success" if all_passed else "failed"
            }
            
            if all_passed:
                logger.info("✅ Persistent ffmpeg streaming implementation validated")
            else:
                logger.error("❌ Persistent ffmpeg streaming issues found")
                
        except Exception as e:
            self.test_results["fix_4_ffmpeg_streaming"] = {
                "description": "Persistent ffmpeg process eliminates CPU spike from spawning new processes",
                "passed": False,
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ ffmpeg streaming test failed: {e}")
    
    async def test_fix_5_vad_reset_loop(self):
        """FIX #5: Test VAD reset loop is eliminated."""
        logger.info("🔧 Testing FIX #5: VAD reset loop elimination...")
        
        try:
            implementation_checks = []
            
            # Check websocket handlers
            try:
                with open("src/websocket_handlers.py", "r") as f:
                    content = f.read()
                
                # Check that per-chunk VAD reset is removed
                if "# FIX #5: Remove per-chunk VAD/STT resets" in content:
                    implementation_checks.append({
                        "check": "Per-chunk VAD reset removed",
                        "status": "passed"
                    })
                else:
                    implementation_checks.append({
                        "check": "Per-chunk VAD reset removed",
                        "status": "failed",
                        "error": "FIX #5 comment not found in websocket_handlers.py"
                    })
                
                # Check that reset calls are commented out or removed
                vad_reset_count = content.count("vad_instance.reset_state()")
                stt_reset_count = content.count("await stt_instance._reset_state()")
                
                if vad_reset_count == 0 and stt_reset_count == 0:
                    implementation_checks.append({
                        "check": "VAD/STT reset calls removed from audio processing",
                        "status": "passed"
                    })
                else:
                    implementation_checks.append({
                        "check": "VAD/STT reset calls removed from audio processing",
                        "status": "failed",
                        "error": f"Found {vad_reset_count} VAD resets and {stt_reset_count} STT resets"
                    })
                    
            except FileNotFoundError:
                implementation_checks.append({
                    "check": "websocket_handlers.py exists",
                    "status": "failed",
                    "error": "File not found"
                })
            
            # Check server.py for disconnect cleanup
            try:
                with open("server.py", "r") as f:
                    content = f.read()
                
                # Check that VAD/STT reset is moved to disconnect
                if "# FIX #5: Reset VAD and STT state when WebSocket closes" in content:
                    implementation_checks.append({
                        "check": "VAD/STT reset moved to disconnect handler",
                        "status": "passed"
                    })
                else:
                    implementation_checks.append({
                        "check": "VAD/STT reset moved to disconnect handler",
                        "status": "failed",
                        "error": "FIX #5 disconnect cleanup not found"
                    })
                    
            except FileNotFoundError:
                implementation_checks.append({
                    "check": "server.py exists",
                    "status": "failed",
                    "error": "File not found"
                })
            
            all_passed = all(check["status"] == "passed" for check in implementation_checks)
            
            self.test_results["fix_5_vad_reset_loop"] = {
                "description": "VAD reset loop eliminated - only reset on WebSocket close",
                "checks": implementation_checks,
                "passed": all_passed,
                "status": "success" if all_passed else "failed"
            }
            
            if all_passed:
                logger.info("✅ VAD reset loop elimination validated")
            else:
                logger.error("❌ VAD reset loop elimination issues found")
                
        except Exception as e:
            self.test_results["fix_5_vad_reset_loop"] = {
                "description": "VAD reset loop eliminated - only reset on WebSocket close",
                "passed": False,
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ VAD reset loop test failed: {e}")
    
    def generate_summary(self):
        """Generate a summary of all test results."""
        logger.info("\n" + "="*60)
        logger.info("🧪 WEBSOCKET FIXES VALIDATION SUMMARY")
        logger.info("="*60)
        
        total_fixes = len(self.test_results)
        passed_fixes = sum(1 for result in self.test_results.values() if result.get("passed", False))
        
        logger.info(f"Total fixes tested: {total_fixes}")
        logger.info(f"Fixes passed: {passed_fixes}")
        logger.info(f"Fixes failed: {total_fixes - passed_fixes}")
        logger.info(f"Success rate: {(passed_fixes/total_fixes)*100:.1f}%")
        
        logger.info("\nDetailed Results:")
        logger.info("-" * 40)
        
        for fix_name, result in self.test_results.items():
            status_icon = "✅" if result.get("passed", False) else "❌"
            logger.info(f"{status_icon} {fix_name.upper().replace('_', ' ')}")
            logger.info(f"   {result.get('description', 'No description')}")
            
            if not result.get("passed", False) and "error" in result:
                logger.info(f"   Error: {result['error']}")
            
            if "checks" in result:
                for check in result["checks"]:
                    check_icon = "✅" if check["status"] == "passed" else "❌"
                    logger.info(f"   {check_icon} {check['check']}")
                    if check["status"] == "failed" and "error" in check:
                        logger.info(f"      Error: {check['error']}")
        
        logger.info("\n" + "="*60)
        
        if passed_fixes == total_fixes:
            logger.info("🎉 ALL FIXES VALIDATED SUCCESSFULLY!")
            logger.info("The WebSocket implementation should now be stable and performant.")
        else:
            logger.warning("⚠️  Some fixes need attention before deployment.")
            logger.info("Please review the failed tests and make necessary corrections.")
        
        logger.info("="*60)


async def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate WebSocket fixes")
    parser.add_argument("--server", default="ws://localhost:8080", 
                       help="WebSocket server URL (default: ws://localhost:8080)")
    parser.add_argument("--output", help="Output results to JSON file")
    
    args = parser.parse_args()
    
    validator = WebSocketFixValidator(args.server)
    
    try:
        results = await validator.validate_all_fixes()
        
        if args.output:
            import json
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        
        # Exit with appropriate code
        all_passed = all(result.get("passed", False) for result in results.values())
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())