#!/usr/bin/env python3
"""
Voice Interface Fix Script
This script helps diagnose and fix common voice interface issues.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_microphone_audio():
    """Test if microphone is capturing audio properly"""
    try:
        import pyaudio
        import numpy as np
        import wave
        
        logger.info("🎤 Testing microphone audio capture...")
        
        # Audio settings
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        RECORD_SECONDS = 3
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # List available audio devices
        logger.info("Available audio devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                logger.info(f"  {i}: {info['name']} (inputs: {info['maxInputChannels']})")
        
        # Record audio
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
        
        logger.info(f"🔴 Recording for {RECORD_SECONDS} seconds... Please speak!")
        
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        logger.info("🔴 Recording finished")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save to file
        test_file = "test_microphone.wav"
        wf = wave.open(test_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        # Analyze audio
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
        max_val = np.max(np.abs(audio_data))
        
        logger.info(f"📊 Audio analysis:")
        logger.info(f"  - RMS level: {rms:.2f}")
        logger.info(f"  - Max level: {max_val}")
        logger.info(f"  - Duration: {len(audio_data) / RATE:.1f}s")
        logger.info(f"  - Saved to: {test_file}")
        
        if rms < 100:
            logger.warning("⚠️ Audio level is very low - check microphone volume")
        elif rms > 10000:
            logger.warning("⚠️ Audio level is very high - may be clipping")
        else:
            logger.info("✅ Audio levels look good")
            
        return test_file
        
    except ImportError:
        logger.error("❌ PyAudio not installed. Install with: pip install pyaudio")
        return None
    except Exception as e:
        logger.error(f"❌ Microphone test failed: {e}")
        return None

async def test_stt_with_file(audio_file):
    """Test STT with a recorded audio file"""
    try:
        from src.stt_deepgram import STT
        
        logger.info(f"🎯 Testing STT with {audio_file}")
        
        stt = STT()
        
        # Read audio file
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()
        
        # Test transcription
        transcript = await stt.transcribe_bytes(audio_bytes)
        
        if transcript and transcript.strip():
            logger.info(f"✅ STT Success: '{transcript}'")
            return True
        else:
            logger.warning("⚠️ No transcript returned")
            return False
            
    except Exception as e:
        logger.error(f"❌ STT test failed: {e}")
        return False

async def fix_frontend_vad():
    """Fix frontend Voice Activity Detection settings"""
    try:
        logger.info("🔧 Fixing frontend VAD settings...")
        
        # Read the AudioVisualizer component
        frontend_file = "react-frontend/src/components/AudioVisualizer.tsx"
        
        if not os.path.exists(frontend_file):
            logger.error(f"❌ Frontend file not found: {frontend_file}")
            return False
        
        with open(frontend_file, 'r') as f:
            content = f.read()
        
        # Look for VAD thresholds and suggest improvements
        if "SPEECH_THRESHOLD" in content:
            logger.info("📝 Current VAD settings found in frontend")
            
            # Extract current thresholds
            import re
            speech_match = re.search(r'SPEECH_THRESHOLD\s*=\s*(\d+)', content)
            silence_match = re.search(r'SILENCE_THRESHOLD\s*=\s*(\d+)', content)
            
            if speech_match and silence_match:
                speech_thresh = int(speech_match.group(1))
                silence_thresh = int(silence_match.group(1))
                
                logger.info(f"  - Speech threshold: {speech_thresh}")
                logger.info(f"  - Silence threshold: {silence_thresh}")
                
                # Suggest better values
                if speech_thresh > 5:
                    logger.warning("⚠️ Speech threshold might be too high")
                    logger.info("💡 Try lowering SPEECH_THRESHOLD to 3-5")
                
                if silence_thresh < 10:
                    logger.warning("⚠️ Silence threshold might be too low")
                    logger.info("💡 Try increasing SILENCE_THRESHOLD to 15-20")
        
        logger.info("✅ Frontend VAD analysis complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Frontend VAD fix failed: {e}")
        return False

async def create_test_audio():
    """Create a test audio file with speech"""
    try:
        logger.info("🎵 Creating test audio with speech...")
        
        # Try to use text-to-speech to create test audio
        try:
            import pyttsx3
            
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            
            test_text = "Hello, this is a test of the voice recognition system. Can you hear me clearly?"
            
            # Save to file
            test_file = "test_speech.wav"
            engine.save_to_file(test_text, test_file)
            engine.runAndWait()
            
            logger.info(f"✅ Created test speech file: {test_file}")
            return test_file
            
        except ImportError:
            logger.warning("⚠️ pyttsx3 not available, creating sine wave test")
            
            # Create a simple sine wave as fallback
            import numpy as np
            import wave
            
            sample_rate = 16000
            duration = 3.0
            frequency = 440  # A note
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
            
            # Add some variation to make it more speech-like
            modulation = np.sin(2 * np.pi * 5 * t) * 0.1
            audio_data = audio_data * (1 + modulation)
            
            # Convert to 16-bit PCM
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            test_file = "test_tone.wav"
            with wave.open(test_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            logger.info(f"✅ Created test tone file: {test_file}")
            return test_file
            
    except Exception as e:
        logger.error(f"❌ Test audio creation failed: {e}")
        return None

async def main():
    """Main diagnostic and fix routine"""
    logger.info("🚀 Voice Interface Diagnostic Tool")
    logger.info("=" * 50)
    
    # Step 1: Test microphone
    logger.info("\n1. Testing microphone...")
    mic_file = await test_microphone_audio()
    
    if mic_file:
        # Step 2: Test STT with recorded audio
        logger.info("\n2. Testing STT with recorded audio...")
        stt_success = await test_stt_with_file(mic_file)
        
        if not stt_success:
            logger.info("\n3. Creating test speech audio...")
            test_file = await create_test_audio()
            
            if test_file:
                logger.info("4. Testing STT with generated speech...")
                await test_stt_with_file(test_file)
    
    # Step 3: Analyze frontend settings
    logger.info("\n5. Analyzing frontend VAD settings...")
    await fix_frontend_vad()
    
    # Step 4: Provide recommendations
    logger.info("\n" + "=" * 50)
    logger.info("🎯 RECOMMENDATIONS:")
    logger.info("=" * 50)
    
    logger.info("1. 🎤 Microphone Setup:")
    logger.info("   - Ensure microphone permissions are granted")
    logger.info("   - Check system audio input levels")
    logger.info("   - Use a good quality microphone if possible")
    
    logger.info("\n2. 🔊 Audio Levels:")
    logger.info("   - Speak clearly and at normal volume")
    logger.info("   - Avoid background noise")
    logger.info("   - Test in a quiet environment")
    
    logger.info("\n3. ⚙️ Frontend Settings:")
    logger.info("   - Lower SPEECH_THRESHOLD to 3-5 for better sensitivity")
    logger.info("   - Increase SILENCE_THRESHOLD to 15-20 for better detection")
    logger.info("   - Adjust audio gain if levels are too low")
    
    logger.info("\n4. 🔧 Backend Settings:")
    logger.info("   - Deepgram STT is now optimized")
    logger.info("   - Audio buffering has been improved")
    logger.info("   - Silence detection prevents empty transcriptions")
    
    logger.info("\n✅ Diagnostic complete!")

if __name__ == "__main__":
    asyncio.run(main())