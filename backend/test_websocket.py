#!/usr/bin/env python3

import asyncio
import websockets
import json
import sys
import os
import time
import logging
import argparse
import base64
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("websocket_test")

class WebSocketTester:
    """Test WebSocket communication with the backend voice service"""
    
    def __init__(self, ws_url="ws://localhost:8003/ws"):
        self.ws_url = ws_url
        self.websocket = None
        self.test_audio_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_audio")
        self.checkpoints = {}
        self.response_times = {}
    
    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            logger.info(f"Connecting to {self.ws_url}...")
            self.websocket = await websockets.connect(self.ws_url)
            
            # Wait for initial status message
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") == "status" and data.get("ready") is True:
                logger.info(f"✅ Connected successfully. Session ID: {data.get('session_id')}")
                return True
            else:
                logger.error(f"❌ Unexpected initial message: {data}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    async def send_audio_file(self, filename):
        """Send a test audio file to the server"""
        file_path = os.path.join(self.test_audio_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"❌ Test file not found: {file_path}")
            return False
        
        try:
            # Record checkpoint for sending audio
            self.record_checkpoint("audio_send_start")
            
            # Send the audio file
            with open(file_path, "rb") as f:
                audio_data = f.read()
            
            logger.info(f"Sending audio file: {filename} ({len(audio_data)} bytes)")
            await self.websocket.send(audio_data)
            
            # Record checkpoint after sending
            self.record_checkpoint("audio_send_complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error sending audio: {e}")
            return False
    
    async def send_end_of_stream(self):
        """Send end-of-stream marker"""
        try:
            # Record checkpoint
            self.record_checkpoint("eos_send")
            
            await self.websocket.send(json.dumps({
                "type": "eos",
                "force_finalize": True,
                "timestamp": time.time()
            }))
            
            logger.info("Sent end-of-stream marker")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error sending end-of-stream: {e}")
            return False
    
    async def send_text_command(self, text):
        """Send a text command directly to test LLM processing"""
        try:
            # Record checkpoint
            self.record_checkpoint("text_command_send")
            
            await self.websocket.send(json.dumps({
                "type": "text_command",
                "text": text,
                "timestamp": time.time()
            }))
            
            logger.info(f"Sent text command: '{text}'")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error sending text command: {e}")
            return False
    
    async def listen_for_responses(self, timeout=30):
        """Listen for responses with timeout"""
        try:
            # Keep track of what we've received
            received_types = set()
            responses = []
            start_time = time.time()
            
            logger.info(f"Listening for responses (timeout: {timeout}s)...")
            
            while time.time() - start_time < timeout:
                try:
                    # Wait for response with 5 second timeout
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                    
                    # Process response
                    try:
                        data = json.loads(response)
                        msg_type = data.get("type", "unknown")
                        
                        # Record when we received different message types
                        if msg_type not in received_types:
                            self.record_checkpoint(f"received_{msg_type}")
                            received_types.add(msg_type)
                        
                        # Process specific message types
                        if msg_type == "transcript":
                            if data.get("partial"):
                                logger.info(f"👂 Partial transcript: '{data.get('text', '')}'")
                            elif data.get("final"):
                                logger.info(f"👂 Final transcript: '{data.get('text', '')}'")
                                
                        elif msg_type == "ai_response":
                            logger.info(f"🤖 AI response: '{data.get('text', '')}'")
                            
                        elif msg_type == "tts_start":
                            logger.info("🔊 TTS generation started")
                            
                        elif msg_type == "tts_complete":
                            logger.info("🔊 TTS generation completed")
                            
                        elif msg_type == "audio_chunk":
                            audio_size = len(data.get("audio_data", ""))
                            logger.info(f"🔊 Received audio chunk: {audio_size} bytes")
                            
                        elif msg_type == "error":
                            logger.error(f"❌ Server error: {data.get('message', 'Unknown error')}")
                            
                        responses.append(data)
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Received non-JSON response: {response[:100]}...")
                        
                except asyncio.TimeoutError:
                    logger.info("No response received in 5 seconds, continuing to listen...")
                    continue
                
                # Check if we've received all expected message types
                expected_types = {"transcript", "ai_response", "tts_start", "audio_chunk", "tts_complete"}
                if expected_types.issubset(received_types):
                    logger.info("✅ Received all expected message types")
                    break
            
            # Record metrics
            if "received_transcript" in self.checkpoints and "audio_send_start" in self.checkpoints:
                stt_time = self.checkpoints["received_transcript"] - self.checkpoints["audio_send_start"]
                self.response_times["stt"] = round(stt_time * 1000)
                
            if "received_ai_response" in self.checkpoints and "received_transcript" in self.checkpoints:
                llm_time = self.checkpoints["received_ai_response"] - self.checkpoints["received_transcript"]
                self.response_times["llm"] = round(llm_time * 1000)
                
            if "received_audio_chunk" in self.checkpoints and "received_ai_response" in self.checkpoints:
                tts_time = self.checkpoints["received_audio_chunk"] - self.checkpoints["received_ai_response"]
                self.response_times["tts"] = round(tts_time * 1000)
            
            return responses
            
        except Exception as e:
            logger.error(f"❌ Error listening for responses: {e}")
            return []
    
    def record_checkpoint(self, name):
        """Record a timing checkpoint"""
        self.checkpoints[name] = time.time()
        logger.debug(f"Checkpoint: {name}")
    
    async def close(self):
        """Close the WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket connection closed")
    
    def print_performance_report(self):
        """Print performance report based on checkpoints"""
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        # Print timing for each stage
        print("Response Times:")
        for stage, ms in self.response_times.items():
            print(f"  {stage.upper():10}: {ms} ms")
            
        # Print latency metrics if we have complete flow
        if "stt" in self.response_times and "llm" in self.response_times and "tts" in self.response_times:
            total_latency = sum(self.response_times.values())
            print(f"\nTotal pipeline latency: {total_latency} ms")
            target = 500  # 500ms target
            status = "✅ TARGET MET" if total_latency <= target else "❌ ABOVE TARGET"
            print(f"Target (500ms): {status}")
        
        print("="*60)

async def main():
    parser = argparse.ArgumentParser(description="Test Voice Agent WebSocket and Audio Processing")
    parser.add_argument("--url", default="ws://localhost:8003/ws", help="WebSocket URL")
    parser.add_argument("--audio", default="tone_440hz.webm", help="Test audio file in test_audio directory")
    parser.add_argument("--text", default="What is the current time?", help="Test text query")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    args = parser.parse_args()
    
    tester = WebSocketTester(ws_url=args.url)
    
    try:
        # Connect to WebSocket
        if not await tester.connect():
            return
        
        # Run audio test if specified
        if args.audio:
            print("\n" + "="*60)
            print(f"TESTING AUDIO PROCESSING FLOW")
            print("="*60)
            
            # Send audio
            if await tester.send_audio_file(args.audio):
                # Send EOS
                await tester.send_end_of_stream()
                # Listen for responses
                await tester.listen_for_responses(timeout=args.timeout)
            
        # Run text test if specified
        if args.text:
            print("\n" + "="*60)
            print(f"TESTING TEXT PROCESSING FLOW")
            print("="*60)
            
            # Send text command
            if await tester.send_text_command(args.text):
                # Listen for responses
                await tester.listen_for_responses(timeout=args.timeout)
        
        # Print performance report
        tester.print_performance_report()
        
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main()) 