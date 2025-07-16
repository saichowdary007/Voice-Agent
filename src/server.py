# Initialize VAD
from src.config import VAD_AGGRESSIVENESS
global vad_instance
vad_instance = VAD(sample_rate=16000, mode=VAD_AGGRESSIVENESS)
logger.info("✅ VAD initialized (WebRTC mode %s)", VAD_AGGRESSIVENESS) 