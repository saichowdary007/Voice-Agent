#!/usr/bin/env python3
import asyncio
import json
import websockets
import logging
import time
import argparse
import datetime
import sys
import os
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("service_monitor.log")
    ]
)
logger = logging.getLogger("service_monitor")

# Configuration
DEFAULT_WS_URL = "ws://localhost:8003/ws"
DEFAULT_CHECK_INTERVAL = 300  # seconds (5 minutes)
DEFAULT_TEST_AUDIO_DIR = "test_audio"
SERVICE_NAMES = ["vad", "stt", "llm", "tts", "audio"]

class ServiceMonitor:
    def __init__(self, ws_url: str = DEFAULT_WS_URL, test_audio_dir: str = DEFAULT_TEST_AUDIO_DIR):
        self.ws_url = ws_url
        self.test_audio_dir = test_audio_dir
        self.websocket = None
        self.session_id = None
        self.services_status = {
            service: {"status": "unknown", "last_check": None, "details": None}
            for service in SERVICE_NAMES
        }
        self.received_messages = []
        self.health_history = []
        
    async def connect(self) -> bool:
        """Establish WebSocket connection to the backend"""
        try:
            logger.info(f"Connecting to {self.ws_url}...")
            self.websocket = await websockets.connect(self.ws_url)
            
            # Wait for initial status message
            status_message = await asyncio.wait_for(self.websocket.recv(), 10)
            status_data = json.loads(status_message)
            
            if status_data.get("type") == "status" and status_data.get("ready") is True:
                self.session_id = status_data.get("session_id")
                logger.info(f"Connected successfully. Session ID: {self.session_id}")
                return True
            else:
                logger.error(f"Unexpected initial message: {status_data}")
                return False
                
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            return False
    
    async def check_services(self) -> Dict[str, Dict[str, Any]]:
        """Check all services and return status"""
        if not self.websocket:
            logger.error("Not connected. Call connect() first.")
            return self.services_status
            
        try:
            # Start message listener
            listener_task = asyncio.create_task(self.message_listener())
            
            # Clear previous messages
            self.received_messages = []
            
            # Check each service
            await self.check_audio_service()
            await asyncio.sleep(1)
            
            await self.check_vad_service()
            await asyncio.sleep(1)
            
            await self.check_stt_service() 
            await asyncio.sleep(1)
            
            await self.check_llm_service()
            await asyncio.sleep(1)
            
            await self.check_tts_service()
            
            # Allow time for final responses
            await asyncio.sleep(3)
            
            # Cancel listener
            listener_task.cancel()
            try:
                await listener_task
            except asyncio.CancelledError:
                pass
                
            # Update timestamps
            now = datetime.datetime.now()
            for service in self.services_status:
                self.services_status[service]["last_check"] = now
                
            # Save history
            self.health_history.append({
                "timestamp": now.isoformat(),
                "status": {s: self.services_status[s]["status"] for s in self.services_status}
            })
                
            return self.services_status
            
        except Exception as e:
            logger.error(f"Error during service monitoring: {str(e)}")
            return self.services_status
        finally:
            if self.websocket:
                await self.websocket.close()
                logger.info("WebSocket connection closed")
    
    async def message_listener(self):
        """Listen for and process incoming messages"""
        try:
            while True:
                message = await self.websocket.recv()
                try:
                    data = json.loads(message)
                    logger.debug(f"Received: {data.get('type', 'unknown')}")
                    self.received_messages.append(data)
                    
                    # Update service status based on message type
                    if data.get("type") == "transcript":
                        self.services_status["stt"]["status"] = "working"
                        self.services_status["stt"]["details"] = data
                        
                    elif data.get("type") == "ai_response":
                        self.services_status["llm"]["status"] = "working"
                        self.services_status["llm"]["details"] = data
                        
                    elif data.get("type") == "tts_start" or data.get("type") == "tts_complete":
                        self.services_status["tts"]["status"] = "working"
                        self.services_status["tts"]["details"] = data
                        
                    elif data.get("type") == "audio_chunk":
                        # Audio service is working if we get audio chunks back
                        self.services_status["audio"]["status"] = "working"
                        self.services_status["audio"]["details"] = {"received_audio": True}
                        
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in message listener: {str(e)}")
    
    async def check_audio_service(self):
        """Check audio service by sending test audio"""
        logger.info("Checking Audio Service...")
        try:
            # Try to use a real audio file first
            test_file = os.path.join(self.test_audio_dir, "tone_440hz.webm")
            if os.path.exists(test_file):
                with open(test_file, "rb") as f:
                    audio_data = f.read()
                
                await self.websocket.send(audio_data)
                logger.info(f"Sent test audio data from {test_file}")
            else:
                # Fall back to dummy WebM header with proper EBML structure
                webm_header = bytes([
                    # EBML Header (proper format)
                    0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x00, 0x00, 0x00, 
                    0x00, 0x00, 0x00, 0x1F, 0x42, 0x86, 0x81, 0x01,
                    0x42, 0xF7, 0x81, 0x01, 0x42, 0xF2, 0x81, 0x04, 
                    0x42, 0xF3, 0x81, 0x08, 0x42, 0x82, 0x84, 0x77,
                    0x65, 0x62, 0x6D, 0x42, 0x87, 0x81, 0x02, 0x42, 
                    0x85, 0x81, 0x02
                ])
                
                await self.websocket.send(webm_header)
                logger.info("Sent dummy audio data (no test file found)")
            
            # Audio service status will be updated by message listener when receiving response
            self.services_status["audio"]["status"] = "pending"
            
            # Wait a bit for processing
            await asyncio.sleep(5)
            
            # If no response received, mark as failing
            if self.services_status["audio"]["status"] == "pending":
                self.services_status["audio"]["status"] = "failing"
                self.services_status["audio"]["details"] = "No response to test audio"
            
        except Exception as e:
            logger.error(f"Error checking audio service: {str(e)}")
            self.services_status["audio"]["status"] = "error"
            self.services_status["audio"]["details"] = str(e)
    
    async def check_vad_service(self):
        """Check VAD service by sending EOS message"""
        logger.info("Checking VAD Service...")
        try:
            # Send EOS message which will trigger VAD processing
            await self.websocket.send(json.dumps({"type": "eos"}))
            logger.info("Sent EOS message to test VAD")
            
            # Mark as pending
            self.services_status["vad"]["status"] = "pending"
            
            # Wait for response with retries
            start_time = time.time()
            max_retries = 2
            retry_count = 0
            
            while retry_count < max_retries and time.time() - start_time < 10:
                # Check if we got any VAD-related messages
                for msg in self.received_messages:
                    # Look for direct VAD messages or indirect evidence of VAD working
                    if (msg.get("type") == "transcript" or 
                        msg.get("type") == "vad_status" or 
                        "speech_segment" in str(msg).lower()):
                        # If we get any related message after EOS, VAD is working
                        self.services_status["vad"]["status"] = "working"
                        self.services_status["vad"]["details"] = "VAD processed EOS signal"
                        return
                
                # If we didn't get a response yet, wait a bit and retry
                if retry_count > 0:
                    # Send another EOS with force flag
                    await self.websocket.send(json.dumps({
                        "type": "eos", 
                        "force_finalize": True
                    }))
                
                await asyncio.sleep(1.0)
                retry_count += 1
            
            # If timeout with no response
            self.services_status["vad"]["status"] = "failing"
            self.services_status["vad"]["details"] = "No VAD response to EOS message"
            
        except Exception as e:
            logger.error(f"Error checking VAD service: {str(e)}")
            self.services_status["vad"]["status"] = "error"
            self.services_status["vad"]["details"] = str(e)
    
    async def check_stt_service(self):
        """Check STT service by sending a text command"""
        logger.info("Checking STT Service...")
        try:
            # Send a text command to test STT
            await self.websocket.send(json.dumps({
                "type": "text_command", 
                "text": "MONITOR_STT_SERVICE_CHECK"
            }))
            logger.info("Sent test command for STT")
            
            # Mark as pending
            self.services_status["stt"]["status"] = "pending"
            
            # Wait for response
            await asyncio.sleep(5)
            
            # STT status will be updated by message listener
            if self.services_status["stt"]["status"] == "pending":
                self.services_status["stt"]["status"] = "failing"
                self.services_status["stt"]["details"] = "No response from STT service"
            
        except Exception as e:
            logger.error(f"Error checking STT service: {str(e)}")
            self.services_status["stt"]["status"] = "error"
            self.services_status["stt"]["details"] = str(e)
    
    async def check_llm_service(self):
        """Check LLM service by sending a text command"""
        logger.info("Checking LLM Service...")
        try:
            # Send a text command to test LLM
            await self.websocket.send(json.dumps({
                "type": "text_command", 
                "text": "What is 2+2?"
            }))
            logger.info("Sent test message to LLM")
            
            # Mark as pending
            self.services_status["llm"]["status"] = "pending"
            
            # Wait for response
            await asyncio.sleep(5)
            
            # LLM status will be updated by message listener
            if self.services_status["llm"]["status"] == "pending":
                self.services_status["llm"]["status"] = "failing"
                self.services_status["llm"]["details"] = "No response from LLM service"
            
        except Exception as e:
            logger.error(f"Error checking LLM service: {str(e)}")
            self.services_status["llm"]["status"] = "error"
            self.services_status["llm"]["details"] = str(e)
    
    async def check_tts_service(self):
        """Check TTS service by checking for audio responses"""
        logger.info("Checking TTS Service...")
        try:
            # Send a text command that should trigger TTS
            await self.websocket.send(json.dumps({
                "type": "text_command", 
                "text": "This is a TTS monitor check"
            }))
            logger.info("Sent test message to trigger TTS")
            
            # Mark as pending
            self.services_status["tts"]["status"] = "pending"
            
            # Wait for response
            await asyncio.sleep(5)
            
            # TTS status will be updated by message listener
            if self.services_status["tts"]["status"] == "pending":
                self.services_status["tts"]["status"] = "failing"
                self.services_status["tts"]["details"] = "No TTS audio generated"
            
        except Exception as e:
            logger.error(f"Error checking TTS service: {str(e)}")
            self.services_status["tts"]["status"] = "error"
            self.services_status["tts"]["details"] = str(e)

    def print_status(self):
        """Print current service status"""
        print("\n" + "="*50)
        print(f"SERVICE STATUS - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        
        all_working = True
        
        for service, status in self.services_status.items():
            status_str = status["status"].upper()
            last_check = status.get("last_check")
            last_check_str = last_check.strftime("%H:%M:%S") if last_check else "Never"
            
            if status["status"] == "working":
                status_emoji = "✅"
            elif status["status"] == "failing":
                status_emoji = "❌"
                all_working = False
            elif status["status"] == "error":
                status_emoji = "⚠️"
                all_working = False
            else:
                status_emoji = "❓"
                all_working = False
                
            print(f"{service.upper():5} SERVICE: {status_emoji} {status_str} (Last check: {last_check_str})")
            
            if status["status"] != "working" and status["details"]:
                print(f"  → Details: {status['details']}")
                
        print("="*50)
        print("OVERALL HEALTH:", "✅ HEALTHY" if all_working else "❌ ISSUES DETECTED")
        print("="*50)
        
        return all_working

    def save_status_report(self, filename: str = "health_report.json"):
        """Save status history to a JSON file"""
        try:
            with open(filename, "w") as f:
                json.dump({
                    "current_status": {
                        s: {
                            "status": self.services_status[s]["status"],
                            "last_check": self.services_status[s]["last_check"].isoformat() 
                                if self.services_status[s]["last_check"] else None
                        } for s in self.services_status
                    },
                    "history": self.health_history
                }, f, indent=2)
            
            logger.info(f"Saved health report to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving health report: {str(e)}")
            return False

async def continuous_monitoring(monitor: ServiceMonitor, interval_seconds: int, max_checks: int = None):
    """Run continuous monitoring with specified interval"""
    check_count = 0
    
    while max_checks is None or check_count < max_checks:
        logger.info(f"Starting monitoring cycle {check_count + 1}")
        
        # Connect and check services
        if await monitor.connect():
            await monitor.check_services()
            all_working = monitor.print_status()
            
            # Save report every time
            monitor.save_status_report()
            
            # Log issues to alert
            if not all_working:
                logger.warning("⚠️ Service issues detected! Check health_report.json for details.")
            
            check_count += 1
            
            # Wait for next check
            if max_checks is None or check_count < max_checks:
                logger.info(f"Next check in {interval_seconds} seconds...")
                await asyncio.sleep(interval_seconds)
        else:
            logger.error("Failed to connect to backend. Retrying in 30 seconds...")
            await asyncio.sleep(30)

async def main():
    parser = argparse.ArgumentParser(description="Monitor Voice Agent Backend Services")
    parser.add_argument("--url", default=DEFAULT_WS_URL, help=f"WebSocket URL (default: {DEFAULT_WS_URL})")
    parser.add_argument("--interval", type=int, default=DEFAULT_CHECK_INTERVAL, help=f"Check interval in seconds (default: {DEFAULT_CHECK_INTERVAL})")
    parser.add_argument("--test-audio-dir", default=DEFAULT_TEST_AUDIO_DIR, help=f"Directory with test audio files (default: {DEFAULT_TEST_AUDIO_DIR})")
    parser.add_argument("--single", action="store_true", help="Run a single check and exit")
    args = parser.parse_args()
    
    monitor = ServiceMonitor(ws_url=args.url, test_audio_dir=args.test_audio_dir)
    
    if args.single:
        # Single check mode
        if await monitor.connect():
            await monitor.check_services()
            monitor.print_status()
            monitor.save_status_report()
        else:
            logger.error("Failed to connect to backend. Check aborted.")
            sys.exit(1)
    else:
        # Continuous monitoring mode
        logger.info(f"Starting continuous monitoring (interval: {args.interval}s)")
        await continuous_monitoring(monitor, args.interval)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        sys.exit(0) 