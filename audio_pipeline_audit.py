#!/usr/bin/env python3
"""
Audio Pipeline Audit Script
Traces the complete audio flow from WebSocket to Deepgram Agent to identify where it breaks.
"""

import asyncio
import json
import logging
import os
import sys
import time
import websockets
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioPipelineAuditor:
    def __init__(self, ws_url, auth_token):
        self.ws_url = ws_url
        self.auth_token = auth_token
        self.websocket = None
        self.audit_results = {
            'connection': False,
            'authentication': False,
            'settings_handshake': False,
            'audio_streaming': False,
            'user_speech_events': False,
            'agent_text_response': False,
            'agent_audio_response': False,
            'errors': []
        }
        
    async def connect(self):
        """Connect to WebSocket and authenticate"""
        try:
            logger.info(f"🔗 Connecting to WebSocket: {self.ws_url}")
            self.websocket = await websockets.connect(
                self.ws_url,
                subprotocols=["binary"]
            )
            self.audit_results['connection'] = True
            logger.info("✅ WebSocket connection established")
            return True
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            self.audit_results['errors'].append(f"Connection failed: {e}")
            return False
    
    async def send_settings(self):
        """Send settings handshake to initialize Deepgram Agent"""
        try:
            settings_message = {
                "type": "settings",
                "audio": {
                    "input": {
                        "encoding": "linear16",
                        "sample_rate": 16000,
                        "channels": 1
                    },
                    "output": {
                        "encoding": "linear16", 
                        "sample_rate": 24000,
                        "channels": 1
                    }
                }
            }
            
            logger.info("📤 Sending settings handshake...")
            await self.websocket.send(json.dumps(settings_message))
            
            # Wait for settings_applied response
            response = await asyncio.wait_for(self.websocket.recv(), timeout=10)
            response_data = json.loads(response)
            
            if response_data.get('type') == 'settings_applied':
                self.audit_results['settings_handshake'] = True
                logger.info("✅ Settings handshake successful")
                return True
            else:
                logger.error(f"❌ Unexpected settings response: {response_data}")
                self.audit_results['errors'].append(f"Settings handshake failed: {response_data}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Settings handshake failed: {e}")
            self.audit_results['errors'].append(f"Settings handshake error: {e}")
            return False
    
    async def send_test_audio(self):
        """Send test audio data to simulate speech"""
        try:
            # Generate test PCM16 audio (sine wave at 440Hz for 2 seconds)
            import math
            sample_rate = 16000
            duration = 2.0
            frequency = 440
            
            samples = []
            for i in range(int(sample_rate * duration)):
                t = i / sample_rate
                sample = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * t))
                samples.extend([sample & 0xFF, (sample >> 8) & 0xFF])
            
            test_audio = bytes(samples)
            
            logger.info(f"📤 Sending test audio: {len(test_audio)} bytes")
            
            # Send audio in chunks
            chunk_size = 1600  # 50ms at 16kHz
            for i in range(0, len(test_audio), chunk_size):
                chunk = test_audio[i:i+chunk_size]
                is_final = (i + chunk_size >= len(test_audio))
                
                audio_message = {
                    "type": "audio_chunk",
                    "data": chunk.hex(),  # Send as hex string for testing
                    "is_final": is_final,
                    "format": "pcm16",
                    "sample_rate": 16000
                }
                
                await self.websocket.send(json.dumps(audio_message))
                await asyncio.sleep(0.05)  # 50ms delay between chunks
            
            self.audit_results['audio_streaming'] = True
            logger.info("✅ Test audio streaming completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Audio streaming failed: {e}")
            self.audit_results['errors'].append(f"Audio streaming error: {e}")
            return False
    
    async def listen_for_responses(self, timeout=30):
        """Listen for responses from the server"""
        try:
            logger.info(f"👂 Listening for responses (timeout: {timeout}s)...")
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=5)
                    response_data = json.loads(response)
                    
                    msg_type = response_data.get('type')
                    logger.info(f"📨 Received: {msg_type}")
                    
                    # Track different response types
                    if msg_type == 'UserStartedSpeaking':
                        self.audit_results['user_speech_events'] = True
                        logger.info("✅ User speech detection working")
                    
                    elif msg_type == 'agent_text':
                        self.audit_results['agent_text_response'] = True
                        content = response_data.get('content', '')
                        logger.info(f"✅ Agent text response: {content[:100]}...")
                    
                    elif msg_type == 'tts_audio':
                        self.audit_results['agent_audio_response'] = True
                        audio_size = len(response_data.get('data', ''))
                        logger.info(f"✅ Agent audio response: {audio_size} bytes")
                    
                    elif msg_type == 'error':
                        error_msg = response_data.get('message', '')
                        logger.error(f"❌ Server error: {error_msg}")
                        self.audit_results['errors'].append(f"Server error: {error_msg}")
                    
                    else:
                        logger.info(f"📋 Other response: {response_data}")
                
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"❌ Error receiving response: {e}")
                    break
            
            logger.info("👂 Response listening completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Response listening failed: {e}")
            self.audit_results['errors'].append(f"Response listening error: {e}")
            return False
    
    async def run_audit(self):
        """Run complete audio pipeline audit"""
        logger.info("🔍 Starting Audio Pipeline Audit...")
        
        try:
            # Step 1: Connect
            if not await self.connect():
                return self.audit_results
            
            # Step 2: Settings handshake
            if not await self.send_settings():
                return self.audit_results
            
            # Step 3: Send test audio and listen for responses simultaneously
            audio_task = asyncio.create_task(self.send_test_audio())
            listen_task = asyncio.create_task(self.listen_for_responses(30))
            
            await asyncio.gather(audio_task, listen_task)
            
        except Exception as e:
            logger.error(f"❌ Audit failed: {e}")
            self.audit_results['errors'].append(f"Audit error: {e}")
        
        finally:
            if self.websocket:
                await self.websocket.close()
        
        return self.audit_results
    
    def print_audit_report(self):
        """Print detailed audit report"""
        print("\n" + "="*60)
        print("🔍 AUDIO PIPELINE AUDIT REPORT")
        print("="*60)
        
        # Connection status
        print(f"🔗 WebSocket Connection: {'✅ PASS' if self.audit_results['connection'] else '❌ FAIL'}")
        print(f"🔐 Authentication: {'✅ PASS' if self.audit_results['authentication'] else '❌ FAIL'}")
        print(f"⚙️  Settings Handshake: {'✅ PASS' if self.audit_results['settings_handshake'] else '❌ FAIL'}")
        
        # Audio pipeline
        print(f"🎵 Audio Streaming: {'✅ PASS' if self.audit_results['audio_streaming'] else '❌ FAIL'}")
        print(f"🎤 Speech Detection: {'✅ PASS' if self.audit_results['user_speech_events'] else '❌ FAIL'}")
        
        # Agent responses
        print(f"💬 Agent Text Response: {'✅ PASS' if self.audit_results['agent_text_response'] else '❌ FAIL'}")
        print(f"🔊 Agent Audio Response: {'✅ PASS' if self.audit_results['agent_audio_response'] else '❌ FAIL'}")
        
        # Overall status
        all_critical_pass = (
            self.audit_results['connection'] and
            self.audit_results['settings_handshake'] and
            self.audit_results['audio_streaming']
        )
        
        agent_working = (
            self.audit_results['agent_text_response'] and
            self.audit_results['agent_audio_response']
        )
        
        print("\n" + "-"*60)
        print("📊 DIAGNOSIS:")
        
        if not all_critical_pass:
            print("❌ CRITICAL: Basic audio pipeline is broken")
        elif not self.audit_results['user_speech_events']:
            print("❌ ISSUE: Speech detection not working (STT problem)")
        elif not agent_working:
            print("❌ ISSUE: Agent not responding (LLM/TTS problem)")
        else:
            print("✅ SUCCESS: Audio pipeline working correctly")
        
        # Errors
        if self.audit_results['errors']:
            print("\n🚨 ERRORS DETECTED:")
            for i, error in enumerate(self.audit_results['errors'], 1):
                print(f"   {i}. {error}")
        
        print("="*60)

async def main():
    # Configuration
    WS_URL = "ws://127.0.0.1:8000/ws/test-token"  # We'll use a test token
    AUTH_TOKEN = "test-token"
    
    # Check if server is running
    try:
        import requests
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        logger.info("✅ Server is running")
    except:
        logger.error("❌ Server is not running. Please start the server first.")
        return
    
    # Run audit
    auditor = AudioPipelineAuditor(WS_URL, AUTH_TOKEN)
    results = await auditor.run_audit()
    auditor.print_audit_report()

if __name__ == "__main__":
    asyncio.run(main())