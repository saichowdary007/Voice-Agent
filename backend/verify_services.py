#!/usr/bin/env python3
"""
Service Verification Script

This script tests each service component independently to ensure they are working correctly.
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("verify_services")

# Load environment variables
load_dotenv()

async def verify_vad_service():
    """Verify Voice Activity Detection service"""
    logger.info("Testing VAD service...")
    
    try:
        from backend.services.vad_service import VADService
        
        # Initialize VAD service
        vad = VADService()
        await vad.initialize()
        
        if vad.is_available:
            logger.info("✅ VAD service initialized successfully")
            
            # Clean up
            await vad.cleanup()
            return True
        else:
            logger.error("❌ VAD service failed to initialize")
            return False
            
    except Exception as e:
        logger.error(f"❌ VAD service verification error: {e}")
        return False

async def verify_stt_service():
    """Verify Speech-to-Text service"""
    logger.info("Testing STT service...")
    
    try:
        from backend.services.stt_service import STTService
        
        # Initialize STT service
        stt = STTService()
        await stt.initialize()
        
        if stt.is_available:
            logger.info("✅ STT service initialized successfully")
            
            # Test session creation
            session_id = await stt.start_continuous_recognition()
            if session_id:
                logger.info(f"✅ STT session created: {session_id}")
                
                # Clean up
                await stt.stop_continuous_recognition()
                await stt.cleanup()
                return True
            else:
                logger.error("❌ STT session creation failed")
                await stt.cleanup()
                return False
        else:
            logger.error("❌ STT service failed to initialize")
            return False
            
    except Exception as e:
        logger.error(f"❌ STT service verification error: {e}")
        return False

async def verify_llm_service():
    """Verify Language Model service"""
    logger.info("Testing LLM service...")
    
    try:
        from backend.services.llm_service import LLMService
        
        # Initialize LLM service
        llm = LLMService()
        await llm.initialize()
        
        if llm.is_available:
            logger.info("✅ LLM service initialized successfully")
            
            # Test response generation
            response = await llm.generate_response("Hello, how are you?")
            if response:
                logger.info(f"✅ LLM response generated: '{response[:50]}...'")
                
                # Clean up
                await llm.cleanup()
                return True
            else:
                logger.error("❌ LLM response generation failed")
                await llm.cleanup()
                return False
        else:
            logger.error("❌ LLM service failed to initialize")
            return False
            
    except Exception as e:
        logger.error(f"❌ LLM service verification error: {e}")
        return False

async def verify_tts_service():
    """Verify Text-to-Speech service"""
    logger.info("Testing TTS service...")
    
    try:
        from backend.services.tts_service import TTSService
        
        # Initialize TTS service
        tts = TTSService()
        await tts.initialize()
        
        if tts.is_available:
            logger.info("✅ TTS service initialized successfully")
            
            # Test speech generation
            test_text = "This is a test of the text-to-speech service."
            
            # Mock service should be used if Piper is not available
            if hasattr(tts, 'use_mock') and tts.use_mock:
                logger.info("Using mock TTS service")
                
            chunks_received = 0
            async for audio_chunk in tts.generate_speech_stream(test_text):
                chunks_received += 1
                if chunks_received == 1:
                    logger.info(f"✅ Received first audio chunk: {len(audio_chunk)} bytes")
                    
            if chunks_received > 0:
                logger.info(f"✅ TTS generated {chunks_received} audio chunks")
                
                # Clean up
                await tts.cleanup()
                return True
            else:
                logger.error("❌ TTS speech generation failed")
                await tts.cleanup()
                return False
        else:
            logger.error("❌ TTS service failed to initialize")
            return False
            
    except Exception as e:
        logger.error(f"❌ TTS service verification error: {e}")
        return False

async def verify_voice_service():
    """Verify complete Voice service integration"""
    logger.info("Testing Voice service integration...")
    
    try:
        from backend.services.voice_service import VoiceService
        
        # Initialize Voice service
        voice = VoiceService()
        await voice.initialize()
        
        if voice.is_available:
            logger.info("✅ Voice service initialized successfully")
            
            # Get health status
            health = await voice.get_health_status()
            
            # Log service status
            for service_name, status in health.items():
                if service_name != "voice_service" and isinstance(status, dict):
                    is_available = status.get("available", False)
                    status_str = "✅ Available" if is_available else "❌ Unavailable"
                    logger.info(f"{service_name.upper()}: {status_str}")
            
            # Clean up
            await voice.cleanup()
            return True
        else:
            logger.error("❌ Voice service failed to initialize")
            return False
            
    except Exception as e:
        logger.error(f"❌ Voice service verification error: {e}")
        return False

async def main():
    """Run all verification tests"""
    logger.info("Starting service verification...")
    
    # Run verification tests
    vad_result = await verify_vad_service()
    stt_result = await verify_stt_service()
    llm_result = await verify_llm_service()
    tts_result = await verify_tts_service()
    voice_result = await verify_voice_service()
    
    # Print summary
    logger.info("\n=== Verification Summary ===")
    logger.info(f"VAD Service: {'✅ PASSED' if vad_result else '❌ FAILED'}")
    logger.info(f"STT Service: {'✅ PASSED' if stt_result else '❌ FAILED'}")
    logger.info(f"LLM Service: {'✅ PASSED' if llm_result else '❌ FAILED'}")
    logger.info(f"TTS Service: {'✅ PASSED' if tts_result else '❌ FAILED'}")
    logger.info(f"Voice Service Integration: {'✅ PASSED' if voice_result else '❌ FAILED'}")
    
    # Overall result
    if all([vad_result, stt_result, llm_result, tts_result, voice_result]):
        logger.info("\n✅ ALL SERVICES VERIFIED SUCCESSFULLY")
    else:
        logger.error("\n❌ SOME SERVICES FAILED VERIFICATION")

if __name__ == "__main__":
    asyncio.run(main()) 