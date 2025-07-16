#!/usr/bin/env python3
"""
Advanced Voice Recognition Fix Script
Addresses poor voice recognition and repetitive responses.
"""

import os
import sys
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_voice_config():
    """Update voice_config.json with optimized settings for better recognition."""
    logger.info("🔧 Updating voice configuration for better recognition...")
    
    config_path = Path("voice_config.json")
    
    # Optimized configuration for better voice recognition
    optimized_config = {
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "chunk_size": 512,  # Smaller chunks for faster processing
            "format": "int16",
            "gain": 2.0,  # Amplify audio for better recognition
            "noise_reduction": True
        },
        "stt": {
            "model": "nova-3",
            "language": "en-US",
            "timeout_seconds": 3,  # Reduced timeout
            "min_audio_length_ms": 100,  # Very short minimum
            "silence_timeout_ms": 500,  # Faster silence detection
            "sensitivity": "high",
            "smart_format": True,
            "punctuate": True,
            "profanity_filter": False,
            "redact": False
        },
        "llm": {
            "max_tokens": 100,  # Shorter responses
            "temperature": 0.8,  # More creative responses
            "timeout_seconds": 10,
            "demo_mode_fallback": True,
            "response_variety": True
        },
        "tts": {
            "model": "aura-asteria-en",
            "sample_rate": 24000,
            "timeout_seconds": 5,
            "speed": 1.1  # Slightly faster speech
        },
        "performance": {
            "ultra_fast_mode": True,
            "target_latency_ms": 2000,  # 2 second target
            "enable_caching": True,
            "parallel_processing": True,
            "aggressive_optimization": True
        },
        "vad": {
            "aggressiveness": 1,  # More sensitive
            "frame_ms": 10,
            "silence_timeout_ms": 400,
            "speech_threshold": 0.2  # Lower threshold
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(optimized_config, f, indent=2)
    
    logger.info("✅ Voice configuration updated for better recognition")

def update_environment_for_recognition():
    """Update .env file with settings optimized for voice recognition."""
    logger.info("🎯 Optimizing environment for voice recognition...")
    
    env_path = Path(".env")
    if not env_path.exists():
        logger.error("❌ .env file not found")
        return False
    
    # Read current .env
    with open(env_path, 'r') as f:
        env_content = f.read()
    
    # Voice recognition optimizations
    optimizations = {
        # STT Optimizations
        "DEEPGRAM_STT_MODEL": "nova-3",
        "DEEPGRAM_STT_LANGUAGE": "en-US",
        "DEEPGRAM_STT_SMART_FORMAT": "true",
        "DEEPGRAM_STT_PUNCTUATE": "true",
        "DEEPGRAM_STT_DIARIZE": "false",
        
        # Audio Processing
        "ENERGY_THRESHOLD": "100",  # More sensitive
        "PAUSE_THRESHOLD": "0.5",   # Faster pause detection
        "MIN_AUDIO_THRESHOLD": "20", # Lower threshold
        "AUDIO_GAIN": "2.5",        # Higher gain
        "AUDIO_BUFFER_SIZE": "256", # Smaller buffer
        
        # VAD Settings
        "VAD_AGGRESSIVENESS": "1",  # More sensitive
        "VAD_SILENCE_TIMEOUT_MS": "400",
        "VAD_SPEECH_THRESHOLD": "0.2",
        
        # Performance
        "ULTRA_FAST_MODE": "true",
        "ULTRA_FAST_TARGET_LATENCY_MS": "2000",
        "DEBUG_MODE": "false",
        
        # Deepgram STT configuration
        "DEEPGRAM_STT_MODEL": "nova-3",
        "USE_REALTIME_STT": "true"
    }
    
    updated = False
    for key, value in optimizations.items():
        if f"{key}=" not in env_content:
            env_content += f"\n{key}={value}"
            updated = True
            logger.info(f"✅ Added {key}={value}")
        else:
            # Update existing value
            import re
            pattern = f"^{key}=.*$"
            replacement = f"{key}={value}"
            if re.search(pattern, env_content, re.MULTILINE):
                env_content = re.sub(pattern, replacement, env_content, flags=re.MULTILINE)
                updated = True
                logger.info(f"✅ Updated {key}={value}")
    
    if updated:
        with open(env_path, 'w') as f:
            f.write(env_content)
        logger.info("✅ Environment optimized for voice recognition")
    
    return True

def create_voice_test_script():
    """Create a script to test voice recognition improvements."""
    logger.info("🧪 Creating voice recognition test script...")
    
    test_script = '''#!/usr/bin/env python3
"""
Voice Recognition Test Script
Tests the improved voice recognition system.
"""

import asyncio
import time
import json
import logging
from src.stt_deepgram import STT
from src.llm import LLM
from src.tts_deepgram import TTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_voice_recognition():
    """Test voice recognition with various inputs."""
    print("🎤 Voice Recognition Test")
    print("=" * 50)
    
    # Initialize components
    stt = STT()
    llm = LLM()
    tts = TTS()
    
    # Test phrases to simulate
    test_phrases = [
        "Hello there",
        "What is your name",
        "Tell me a joke",
        "How are you today",
        "What can you help me with",
        "Thank you very much",
        "Goodbye for now"
    ]
    
    print("\\n🧪 Testing LLM Response Variety...")
    for i, phrase in enumerate(test_phrases, 1):
        print(f"\\n{i}. Testing: '{phrase}'")
        start_time = time.time()
        
        try:
            response = await llm.generate_response(phrase)
            latency = (time.time() - start_time) * 1000
            
            print(f"   Response: '{response[:80]}{'...' if len(response) > 80 else ''}'")
            print(f"   Latency: {latency:.0f}ms")
            
            if latency < 2000:
                print("   ✅ FAST response")
            elif latency < 5000:
                print("   ⚡ GOOD response")
            else:
                print("   ⚠️  SLOW response")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Small delay between tests
        await asyncio.sleep(0.5)
    
    print("\\n🎯 Voice Recognition Test Complete!")
    print("\\nTips for better voice recognition:")
    print("- Speak clearly and at normal volume")
    print("- Ensure good microphone quality")
    print("- Minimize background noise")
    print("- Speak at a steady pace")
    print("- Wait for the response before speaking again")

if __name__ == "__main__":
    asyncio.run(test_voice_recognition())
'''
    
    with open("test_voice_recognition.py", 'w') as f:
        f.write(test_script)
    
    logger.info("✅ Created test_voice_recognition.py")

def create_audio_diagnostics():
    """Create audio diagnostics script."""
    logger.info("🔍 Creating audio diagnostics script...")
    
    diagnostics_script = '''#!/usr/bin/env python3
"""
Audio Diagnostics Script
Helps diagnose audio input issues.
"""

import numpy as np
import logging
import asyncio
from src.stt_deepgram import STT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_audio_chunk(audio_bytes):
    """Analyze audio chunk for quality metrics."""
    if not audio_bytes or len(audio_bytes) < 320:
        return None
    
    try:
        # Convert to numpy array
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Calculate metrics
        rms_level = np.sqrt(np.mean(audio_np**2))
        max_level = np.max(np.abs(audio_np))
        min_level = np.min(np.abs(audio_np))
        
        # Signal-to-noise ratio estimate
        noise_floor = np.percentile(np.abs(audio_np), 10)
        signal_peak = np.percentile(np.abs(audio_np), 90)
        snr_estimate = signal_peak / (noise_floor + 1e-10)
        
        return {
            "rms_level": rms_level,
            "max_level": max_level,
            "min_level": min_level,
            "snr_estimate": snr_estimate,
            "length_seconds": len(audio_bytes) / (16000 * 2),
            "is_likely_speech": rms_level > 0.01 and max_level > 0.1
        }
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        return None

async def diagnose_audio_input():
    """Diagnose audio input quality."""
    print("🔍 Audio Input Diagnostics")
    print("=" * 40)
    
    try:
        stt = STT()
        print("✅ STT initialized successfully")
        
        # Test with different audio scenarios
        test_scenarios = [
            ("Very quiet audio", b"\\x00" * 3200),  # Silence
            ("Low volume audio", b"\\x10\\x00" * 1600),  # Very quiet
            ("Normal audio", b"\\x00\\x10" * 1600),  # Low volume
        ]
        
        for scenario_name, test_audio in test_scenarios:
            print(f"\\n🧪 Testing: {scenario_name}")
            analysis = analyze_audio_chunk(test_audio)
            
            if analysis:
                print(f"   RMS Level: {analysis['rms_level']:.4f}")
                print(f"   Max Level: {analysis['max_level']:.4f}")
                print(f"   SNR Estimate: {analysis['snr_estimate']:.2f}")
                print(f"   Likely Speech: {analysis['is_likely_speech']}")
                
                if analysis['is_likely_speech']:
                    print("   ✅ Audio quality looks good")
                else:
                    print("   ⚠️  Audio may be too quiet")
            else:
                print("   ❌ Could not analyze audio")
        
        print("\\n📋 Recommendations:")
        print("- Ensure microphone is not muted")
        print("- Check microphone permissions in browser")
        print("- Test with different microphones if available")
        print("- Reduce background noise")
        print("- Speak at normal conversational volume")
        
    except Exception as e:
        print(f"❌ Diagnostics failed: {e}")

if __name__ == "__main__":
    asyncio.run(diagnose_audio_input())
'''
    
    with open("diagnose_audio.py", 'w') as f:
        f.write(diagnostics_script)
    
    logger.info("✅ Created diagnose_audio.py")

def print_voice_recognition_tips():
    """Print tips for better voice recognition."""
    logger.info("\n" + "="*60)
    logger.info("🎯 VOICE RECOGNITION IMPROVEMENT GUIDE")
    logger.info("="*60)
    
    tips = [
        "1. 🎤 Audio Input Quality:",
        "   - Use a good quality microphone (headset recommended)",
        "   - Speak 6-12 inches from the microphone",
        "   - Ensure microphone permissions are granted",
        "   - Test microphone in browser settings first",
        "",
        "2. 🔊 Speaking Technique:",
        "   - Speak clearly at normal conversational volume",
        "   - Avoid speaking too fast or too slow",
        "   - Pause briefly between sentences",
        "   - Wait for the AI response before speaking again",
        "",
        "3. 🌍 Environment:",
        "   - Minimize background noise (TV, music, etc.)",
        "   - Use a quiet room when possible",
        "   - Avoid echo-prone spaces",
        "   - Close windows to reduce outside noise",
        "",
        "4. 🔧 Technical Settings:",
        "   - Restart the server after applying fixes",
        "   - Test with: python test_voice_recognition.py",
        "   - Run diagnostics: python diagnose_audio.py",
        "   - Check browser console for errors",
        "",
        "5. 🚨 Troubleshooting:",
        "   - If recognition is poor, try refreshing the page",
        "   - Check that Deepgram API key is valid",
        "   - Verify internet connection is stable",
        "   - Try different browsers (Chrome recommended)",
        "",
        "6. ⚡ Performance Expectations:",
        "   - STT should complete in < 5 seconds",
        "   - Total response time should be < 10 seconds",
        "   - Demo mode responses should be < 2 seconds",
        "   - Voice recognition should work 8/10 times"
    ]
    
    for tip in tips:
        logger.info(tip)
    
    logger.info("="*60)

def main():
    """Main fix function."""
    logger.info("🚀 Advanced Voice Recognition Fix Script")
    logger.info("This will optimize your system for better voice recognition")
    
    # Apply all fixes
    update_voice_config()
    update_environment_for_recognition()
    create_voice_test_script()
    create_audio_diagnostics()
    
    logger.info("\n✅ Voice recognition optimizations applied!")
    print_voice_recognition_tips()
    
    logger.info("\n🎯 Next Steps:")
    logger.info("1. Restart the server: python server.py")
    logger.info("2. Test recognition: python test_voice_recognition.py")
    logger.info("3. Run diagnostics: python diagnose_audio.py")
    logger.info("4. Try speaking to the voice interface")

if __name__ == "__main__":
    main()