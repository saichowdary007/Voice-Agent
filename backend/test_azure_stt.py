import asyncio
import os
import sys
import time
import numpy as np
import wave
import structlog
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the STT service directly
from services.stt_service import STTService, STTResult

# Set up logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

async def test_with_audio_file(audio_file_path: str):
    """Test the Azure STT service with an audio file"""
    logger.info(f"Testing Azure STT with audio file: {audio_file_path}")
    
    # Check if file exists
    if not os.path.exists(audio_file_path):
        logger.error(f"Audio file not found: {audio_file_path}")
        return
    
    # Initialize STT service
    stt_service = STTService()
    await stt_service.initialize()
    
    if not stt_service.is_available:
        logger.error("Azure STT service not available. Check API key and region.")
        return
    
    logger.info("Azure STT service initialized successfully")
    
    # Load audio file
    try:
        with wave.open(audio_file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            logger.info(f"Audio file properties: {sample_rate}Hz, {channels} channels, {sample_width} bytes per sample")
            
            # Azure STT expects 16kHz mono PCM audio
            if sample_rate != 16000 or channels != 1:
                logger.warning(f"Audio format should be 16kHz mono for optimal results (got {sample_rate}Hz, {channels} channels)")
            
            # Read all audio data
            audio_data = wav_file.readframes(wav_file.getnframes())
    except Exception as e:
        logger.error(f"Error reading audio file: {e}")
        return
    
    logger.info(f"Read {len(audio_data)} bytes of audio data")
    
    # Process audio in chunks to simulate streaming
    chunk_size = 1600  # 100ms at 16kHz
    chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
    
    try:
        # Process each chunk
        for i, chunk in enumerate(chunks):
            stt_result = await stt_service.process_frame(chunk)
            if stt_result.partial_text:
                logger.info(f"Partial result ({i+1}/{len(chunks)}): {stt_result.partial_text}")
            
            # Brief pause to simulate real-time streaming
            await asyncio.sleep(0.05)
        
        # Finalize STT processing
        final_result = await stt_service.finalize()
        if final_result.final_text:
            logger.info(f"Final result: {final_result.final_text}")
        else:
            logger.warning("No final text received from STT service")
    except Exception as e:
        logger.error(f"Error during STT processing: {e}")
    finally:
        # Clean up resources
        await stt_service.cleanup()
        logger.info("STT service cleaned up")

async def main():
    # Check if audio file path is provided as command line argument
    if len(sys.argv) > 1:
        audio_file_path = sys.argv[1]
    else:
        # Default test audio file (you should place a test WAV file here)
        audio_file_path = "test_audio.wav"
    
    await test_with_audio_file(audio_file_path)

if __name__ == "__main__":
    asyncio.run(main()) 