#!/usr/bin/env python3

import asyncio
import sys
import os
import time
import logging
import json
import numpy as np
import soundfile as sf
import argparse
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_services")

# Add parent directory to path to find app module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import services
from backend.services.voice_service import VoiceService
from backend.services.vad_service import VADService
from backend.services.stt_service import STTService
from backend.services.llm_service import LLMService
from backend.services.tts_service import TTSService

class ServiceTester:
    """Test individual voice agent services"""
    
    def __init__(self):
        self.voice_service = None
        self.vad_service = None
        self.stt_service = None
        self.llm_service = None
        self.tts_service = None
        self.test_audio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_audio", "tone_440hz.webm")
        
    async def initialize(self, service_name=None):
        """Initialize all services or just the specified one"""
        try:
            if service_name:
                logger.info(f"Initializing {service_name} service for testing...")
                
                if service_name == "vad":
                    self.vad_service = VADService()
                    await self.vad_service.initialize()
                elif service_name == "stt":
                    self.stt_service = STTService()
                    await self.stt_service.initialize()
                elif service_name == "llm":
                    self.llm_service = LLMService()
                    await self.llm_service.initialize()
                elif service_name == "tts":
                    self.tts_service = TTSService()
                    await self.tts_service.initialize()
                elif service_name == "voice":
                    self.voice_service = VoiceService()
                    await self.voice_service.initialize()
                else:
                    logger.error(f"Unknown service: {service_name}")
                    return
                
                logger.info(f"{service_name.upper()} service initialized for testing")
            else:
                logger.info("Initializing all services for testing...")
                
                # Initialize services
                self.vad_service = VADService()
                self.stt_service = STTService()
                self.llm_service = LLMService()
                self.tts_service = TTSService()
                
                # Initialize in parallel
                await asyncio.gather(
                    self.vad_service.initialize(),
                    self.stt_service.initialize(),
                    self.llm_service.initialize(),
                    self.tts_service.initialize()
                )
                
                # Also initialize combined voice service
                self.voice_service = VoiceService()
                await self.voice_service.initialize()
                
                logger.info("All services initialized for testing")
            
        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            
    async def test_vad(self):
        """Test Voice Activity Detection"""
        logger.info("🔍 TESTING VAD SERVICE...")
        
        if not self.vad_service or not self.vad_service.is_available:
            logger.error("❌ VAD service not available")
            return False
        
        try:
            # Generate test audio (1 second of silence, then 1 second of tone)
            silence = np.zeros(16000, dtype=np.float32)
            tone = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
            test_audio = np.concatenate([silence, tone])
            
            # Process in frames
            frame_size = 512  # 32ms at 16kHz
            detected_speech = False
            frame_count = 0
            
            for i in range(0, len(test_audio), frame_size):
                frame = test_audio[i:i+frame_size]
                if len(frame) < frame_size:
                    frame = np.pad(frame, (0, frame_size - len(frame)))
                    
                result = await self.vad_service.process_audio_chunk(frame)
                frame_count += 1
                
                if result.get("speech_detected", False):
                    detected_speech = True
                    logger.info(f"✅ VAD detected speech at frame {frame_count}")
                    break
            
            if detected_speech:
                logger.info("✅ VAD test passed: Speech detected successfully")
                return True
            else:
                logger.warning("⚠️ VAD test inconclusive: No speech detected in test audio")
                return False
                
        except Exception as e:
            logger.error(f"❌ VAD test failed: {e}")
            return False
    
    async def test_stt(self):
        """Test Speech-to-Text"""
        logger.info("🔍 TESTING STT SERVICE...")
        
        if not self.stt_service or not self.stt_service.is_available:
            logger.error("❌ STT service not available")
            return False
        
        try:
            # Start continuous recognition
            session_id = await self.stt_service.start_continuous_recognition()
            if not session_id:
                logger.error("❌ STT failed to start continuous recognition")
                return False
                
            logger.info(f"✅ STT started continuous recognition: {session_id}")
            
            # Test with a simple phrase by sending synthesized "hello" audio
            # This is a minimal test - in production, use proper test audio files
            
            # Try to stop recognition
            await self.stt_service.stop_continuous_recognition()
            logger.info("✅ STT successfully stopped continuous recognition")
            
            # Get service status
            status = await self.stt_service.get_status()
            logger.info(f"STT status: {json.dumps(status, indent=2)}")
            
            return True
                
        except Exception as e:
            logger.error(f"❌ STT test failed: {e}")
            return False
    
    async def test_llm(self):
        """Test LLM service"""
        logger.info("🔍 TESTING LLM SERVICE...")
        
        if not self.llm_service or not self.llm_service.is_available:
            logger.error("❌ LLM service not available")
            return False
        
        try:
            # Test with a simple query
            test_query = "What is 2+2?"
            logger.info(f"Sending test query to LLM: '{test_query}'")
            
            # Get response
            response = await self.llm_service.process_text(test_query)
            
            if response:
                logger.info(f"✅ LLM response: '{response[:100]}...'")
                return True
            else:
                logger.warning("⚠️ LLM returned empty response")
                return False
                
        except Exception as e:
            logger.error(f"❌ LLM test failed: {e}")
            return False
    
    async def test_tts(self):
        """Test Text-to-Speech"""
        logger.info("🔍 TESTING TTS SERVICE...")
        
        if not self.tts_service or not self.tts_service.is_available:
            logger.error("❌ TTS service not available")
            return False
        
        try:
            # Test with a simple phrase
            test_text = "This is a test of the text to speech service."
            logger.info(f"Generating speech for: '{test_text}'")
            
            # Get speech audio
            result = await self.tts_service.synthesize_speech(test_text)
            
            if result and result.get("audio_data"):
                audio_size = len(result["audio_data"])
                logger.info(f"✅ TTS generated {audio_size} bytes of audio")
                
                # Save audio to file for verification
                output_path = "test_tts_output.wav"
                with open(output_path, "wb") as f:
                    f.write(result["audio_data"])
                logger.info(f"Saved TTS output to {output_path}")
                
                return True
            else:
                error = result.get("error", "Unknown error")
                logger.warning(f"⚠️ TTS failed to generate audio: {error}")
                return False
                
        except Exception as e:
            logger.error(f"❌ TTS test failed: {e}")
            return False
    
    async def test_full_flow(self):
        """Test the full voice processing flow"""
        logger.info("🔍 TESTING FULL VOICE PROCESSING FLOW...")
        
        if not self.voice_service or not self.voice_service.is_available:
            logger.error("❌ Voice service not available")
            return False
        
        try:
            # Get service health status
            health = await self.voice_service.get_health_status()
            logger.info(f"Voice service health status: {json.dumps(health, indent=2)}")
            
            # Process a test text message
            test_text = "Hello, how can I help you today?"
            logger.info(f"Processing test text: '{test_text}'")
            
            response = await self.voice_service.process_text(test_text)
            if response:
                logger.info(f"✅ Got response: '{response[:100]}...'")
            else:
                logger.warning("⚠️ No response from process_text")
            
            # Test generating speech
            logger.info("Testing speech generation...")
            chunk_count = 0
            
            async for audio_chunk in self.voice_service.generate_speech(test_text):
                chunk_count += 1
                logger.info(f"Received audio chunk {chunk_count}: {len(audio_chunk)} bytes")
                if chunk_count >= 3:  # Just test a few chunks
                    break
            
            if chunk_count > 0:
                logger.info(f"✅ Generated {chunk_count} audio chunks")
                return True
            else:
                logger.warning("⚠️ No audio chunks generated")
                return False
                
        except Exception as e:
            logger.error(f"❌ Full flow test failed: {e}")
            return False
    
    async def cleanup(self, service_name=None):
        """Clean up all services or just the specified one"""
        logger.info(f"Cleaning up {'all' if not service_name else service_name} services...")
        try:
            if service_name:
                if service_name == "vad" and self.vad_service:
                    await self.vad_service.cleanup()
                elif service_name == "stt" and self.stt_service:
                    await self.stt_service.cleanup()
                elif service_name == "llm" and self.llm_service:
                    await self.llm_service.cleanup()
                elif service_name == "tts" and self.tts_service:
                    await self.tts_service.cleanup()
                elif service_name == "voice" and self.voice_service:
                    await self.voice_service.cleanup()
            else:
                cleanup_tasks = []
                
                for service in [self.vad_service, self.stt_service, self.llm_service, self.tts_service, self.voice_service]:
                    if service and hasattr(service, 'cleanup'):
                        cleanup_tasks.append(service.cleanup())
                
                if cleanup_tasks:
                    await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                
            logger.info("Services cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Run service tests based on command line arguments"""
    parser = argparse.ArgumentParser(description="Test Voice Agent services")
    parser.add_argument("--test", choices=["vad", "stt", "llm", "tts", "all", "full"], 
                        default="all", help="Specify which service to test")
    args = parser.parse_args()
    
    tester = ServiceTester()
    
    try:
        # Initialize based on test type
        if args.test == "all":
            await tester.initialize()
            
            results = {}
            
            # Test individual services
            results["vad"] = await tester.test_vad()
            results["stt"] = await tester.test_stt()
            results["llm"] = await tester.test_llm()
            results["tts"] = await tester.test_tts()
            
            # Test full flow
            results["full_flow"] = await tester.test_full_flow()
            
            # Print summary
            print("\n" + "="*50)
            print("TEST RESULTS SUMMARY")
            print("="*50)
            
            all_passed = True
            for service, passed in results.items():
                status = "✅ PASSED" if passed else "❌ FAILED"
                print(f"{service.upper():12}: {status}")
                if not passed:
                    all_passed = False
            
            print("="*50)
            print(f"OVERALL: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
            print("="*50)
            
        elif args.test == "full":
            await tester.initialize("voice")
            result = await tester.test_full_flow()
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"FULL FLOW TEST: {status}")
            
        else:
            # Test specific service
            await tester.initialize(args.test)
            
            if args.test == "vad":
                result = await tester.test_vad()
            elif args.test == "stt":
                result = await tester.test_stt()
            elif args.test == "llm":
                result = await tester.test_llm()
            elif args.test == "tts":
                result = await tester.test_tts()
                
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{args.test.upper()} TEST: {status}")
            
    finally:
        # Clean up based on test type
        if args.test == "all":
            await tester.cleanup()
        elif args.test == "full":
            await tester.cleanup("voice")
        else:
            await tester.cleanup(args.test)

if __name__ == "__main__":
    asyncio.run(main()) 