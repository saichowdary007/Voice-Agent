"""
Speech-to-Text (STT) module with Deepgram integration.
Uses Deepgram's nova-3 model for high-accuracy, low-latency transcription.
"""
import logging

# Import Deepgram STT
try:
    from src.stt_deepgram import STT as DeepgramSTT
    logger = logging.getLogger(__name__)
    logger.info("🚀 Using Deepgram STT for high-accuracy transcription")
    STT = DeepgramSTT
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Failed to import Deepgram STT: {e}")
    logger.error("Please ensure DEEPGRAM_API_KEY is set and deepgram-sdk is installed")
    raise ImportError("Deepgram STT is required but not available")
