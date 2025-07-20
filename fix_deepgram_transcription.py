#!/usr/bin/env python3
"""
Deepgram Transcription Fix
Diagnoses and fixes the empty transcript issue with comprehensive testing.
"""

import asyncio
import logging
import os
import sys
import numpy as np
import wave
import io
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import DEEPGRAM_API_KEY, DEEPGRAM_STT_MODEL
from src.stt_deepgram import DeepgramSTT
from src.audio_preprocessor import get_audio_preprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_audio(duration_seconds=2, sample_rate=16000, frequency=440):
    """Generate a test tone for audio testing."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    # Generate a simple sine wave
    audio = np.sin(2 * np.pi * frequency * t) * 0.3  # 30% amplitude
    
    # Add some speech-like modulation
    modulation = np.sin(2 * np.pi * 5 * t) * 0.1  # 5Hz modulation
    audio = audio * (1 + modulation)
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


def create_wav_file(pcm_data, sample_rate=16000):
    """Convert PCM data to WAV format."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    
    wav_buffer.seek(0)
    return wav_buffer.read()


async def test_deepgram_api_key():
    """Test if the Deepgram API key is valid."""
    logger.info("🔑 Testing Deepgram API key...")
    
    if not DEEPGRAM_API_KEY:
        logger.error("❌ DEEPGRAM_API_KEY is not set!")
        return False
    
    logger.info(f"API Key: {DEEPGRAM_API_KEY[:10]}...{DEEPGRAM_API_KEY[-4:]}")
    
    try:
        from deepgram import DeepgramClient
        client = DeepgramClient(DEEPGRAM_API_KEY)
        
        # Test with a simple audio file
        test_audio = generate_test_audio(1, 16000, 440)  # 1 second tone
        wav_audio = create_wav_file(test_audio)
        
        payload = {"buffer": wav_audio}
        
        from deepgram import PrerecordedOptions
        options = PrerecordedOptions(
            model="nova-2",  # Use nova-2 as fallback
            language="en-US",
            smart_format=True,
            punctuate=True,
        )
        
        response = await asyncio.to_thread(
            client.listen.prerecorded.v("1").transcribe_file,
            payload,
            options
        )
        
        logger.info("✅ Deepgram API key is valid")
        logger.info(f"Response: {response}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Deepgram API key test failed: {e}")
        return False


async def test_audio_preprocessing():
    """Test the audio preprocessing pipeline."""
    logger.info("🎵 Testing audio preprocessing...")
    
    try:
        preprocessor = get_audio_preprocessor()
        
        # Generate test audio with speech-like characteristics
        test_audio = generate_test_audio(2, 16000, 200)  # Lower frequency, more speech-like
        
        logger.info(f"Original audio: {len(test_audio)} bytes")
        
        # Test preprocessing
        processed_audio, metadata = preprocessor.preprocess_audio_chunk(test_audio, inject_preroll=True)
        
        logger.info(f"Processed audio: {len(processed_audio)} bytes")
        logger.info(f"Metadata: {metadata}")
        
        # Analyze processed audio
        audio_np = np.frombuffer(processed_audio, dtype=np.int16).astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_np**2))
        peak = np.max(np.abs(audio_np))
        
        logger.info(f"Processed audio - RMS: {rms:.4f}, Peak: {peak:.4f}")
        
        if rms < 0.001:
            logger.warning("⚠️ Processed audio appears to be silence!")
            return False
        
        logger.info("✅ Audio preprocessing working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Audio preprocessing test failed: {e}")
        return False


async def test_deepgram_transcription():
    """Test Deepgram transcription with various audio samples."""
    logger.info("🎤 Testing Deepgram transcription...")
    
    try:
        stt = DeepgramSTT()
        
        # Test 1: Simple tone (should return empty)
        logger.info("Test 1: Simple tone")
        tone_audio = generate_test_audio(1, 16000, 440)
        wav_tone = create_wav_file(tone_audio)
        
        result1 = await stt.transcribe_bytes(wav_tone)
        logger.info(f"Tone result: '{result1}'")
        
        # Test 2: More complex audio (multiple frequencies)
        logger.info("Test 2: Complex audio")
        complex_audio = generate_test_audio(2, 16000, 200)  # Lower frequency
        # Add some noise to make it more speech-like
        noise = np.random.normal(0, 0.05, len(complex_audio)//2)
        complex_audio_np = np.frombuffer(complex_audio, dtype=np.int16).astype(np.float32) / 32768.0
        complex_audio_np += noise
        complex_audio_with_noise = (complex_audio_np * 32767).astype(np.int16).tobytes()
        wav_complex = create_wav_file(complex_audio_with_noise)
        
        result2 = await stt.transcribe_bytes(wav_complex)
        logger.info(f"Complex audio result: '{result2}'")
        
        # Test 3: Try with different models
        logger.info("Test 3: Testing different models")
        
        # Override config temporarily
        import src.config as config
        original_model = config.DEEPGRAM_STT_MODEL
        
        for model in ["nova-2", "enhanced", "base"]:
            try:
                config.DEEPGRAM_STT_MODEL = model
                logger.info(f"Testing with model: {model}")
                
                result = await stt.transcribe_bytes(wav_complex)
                logger.info(f"Model {model} result: '{result}'")
                
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
        
        # Restore original model
        config.DEEPGRAM_STT_MODEL = original_model
        
        logger.info("✅ Deepgram transcription tests completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Deepgram transcription test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


async def test_real_microphone_audio():
    """Test with real microphone audio if available."""
    logger.info("🎙️ Testing with real microphone audio...")
    
    try:
        import speech_recognition as sr
        
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        logger.info("Adjusting for ambient noise...")
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        
        logger.info("Please speak for 3 seconds...")
        with microphone as source:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
        
        # Get raw audio data
        audio_bytes = audio.get_wav_data()
        logger.info(f"Captured {len(audio_bytes)} bytes of audio")
        
        # Test with Deepgram
        stt = DeepgramSTT()
        result = await stt.transcribe_bytes(audio_bytes)
        
        logger.info(f"Microphone transcription result: '{result}'")
        
        if result and result.strip():
            logger.info("✅ Real microphone audio transcription successful!")
            return True
        else:
            logger.warning("⚠️ No transcription from real microphone audio")
            return False
        
    except ImportError:
        logger.warning("speech_recognition not available, skipping microphone test")
        return True
    except Exception as e:
        logger.error(f"❌ Microphone test failed: {e}")
        return False


async def fix_configuration():
    """Fix common configuration issues."""
    logger.info("🔧 Fixing configuration issues...")
    
    fixes_applied = []
    
    # Fix 1: Update .env file with correct model
    env_path = Path(".env")
    if env_path.exists():
        env_content = env_path.read_text()
        
        # Fix model mismatch
        if "DEEPGRAM_STT_MODEL=nova-2" in env_content:
            env_content = env_content.replace("DEEPGRAM_STT_MODEL=nova-2", "DEEPGRAM_STT_MODEL=nova-3")
            fixes_applied.append("Updated DEEPGRAM_STT_MODEL to nova-3")
        
        # Add missing endpointing setting
        if "DEEPGRAM_STT_ENDPOINTING" not in env_content:
            env_content += "\nDEEPGRAM_STT_ENDPOINTING=300\n"
            fixes_applied.append("Added DEEPGRAM_STT_ENDPOINTING=300")
        
        # Add filler words setting
        if "DEEPGRAM_STT_FILLER_WORDS" not in env_content:
            env_content += "DEEPGRAM_STT_FILLER_WORDS=false\n"
            fixes_applied.append("Added DEEPGRAM_STT_FILLER_WORDS=false")
        
        # Write back if changes were made
        if fixes_applied:
            env_path.write_text(env_content)
            logger.info(f"✅ Applied {len(fixes_applied)} configuration fixes")
            for fix in fixes_applied:
                logger.info(f"  - {fix}")
        else:
            logger.info("No configuration fixes needed")
    
    return fixes_applied


async def create_audio_test_samples():
    """Create test audio samples for debugging."""
    logger.info("🎵 Creating test audio samples...")
    
    debug_dir = Path("debug_audio")
    debug_dir.mkdir(exist_ok=True)
    
    # Sample 1: Pure tone
    tone_audio = generate_test_audio(2, 16000, 440)
    tone_wav = create_wav_file(tone_audio)
    (debug_dir / "test_tone.wav").write_bytes(tone_wav)
    
    # Sample 2: Speech-like audio
    speech_audio = generate_test_audio(2, 16000, 200)
    # Add formant-like structure
    speech_np = np.frombuffer(speech_audio, dtype=np.int16).astype(np.float32) / 32768.0
    # Add second harmonic
    t = np.linspace(0, 2, len(speech_np), False)
    speech_np += 0.2 * np.sin(2 * np.pi * 400 * t)  # Second harmonic
    speech_np += 0.1 * np.sin(2 * np.pi * 800 * t)  # Third harmonic
    speech_audio_enhanced = (speech_np * 32767).astype(np.int16).tobytes()
    speech_wav = create_wav_file(speech_audio_enhanced)
    (debug_dir / "test_speech_like.wav").write_bytes(speech_wav)
    
    # Sample 3: Noisy audio
    noise = np.random.normal(0, 0.1, len(speech_np))
    noisy_audio = speech_np + noise
    noisy_audio = np.clip(noisy_audio, -1, 1)  # Prevent clipping
    noisy_wav = create_wav_file((noisy_audio * 32767).astype(np.int16).tobytes())
    (debug_dir / "test_noisy.wav").write_bytes(noisy_wav)
    
    logger.info(f"✅ Created test samples in {debug_dir}/")
    return True


async def main():
    """Run comprehensive Deepgram transcription diagnostics and fixes."""
    logger.info("🚀 Starting Deepgram Transcription Fix")
    logger.info("=" * 60)
    
    # Step 1: Fix configuration
    await fix_configuration()
    
    # Step 2: Test API key
    api_key_ok = await test_deepgram_api_key()
    if not api_key_ok:
        logger.error("❌ Cannot proceed without valid Deepgram API key")
        return
    
    # Step 3: Test audio preprocessing
    preprocessing_ok = await test_audio_preprocessing()
    
    # Step 4: Test Deepgram transcription
    transcription_ok = await test_deepgram_transcription()
    
    # Step 5: Create test samples
    await create_audio_test_samples()
    
    # Step 6: Test with real microphone (optional)
    try:
        await test_real_microphone_audio()
    except KeyboardInterrupt:
        logger.info("Microphone test skipped by user")
    
    # Summary
    logger.info("=" * 60)
    logger.info("🏁 Diagnostic Summary:")
    logger.info(f"  API Key: {'✅' if api_key_ok else '❌'}")
    logger.info(f"  Audio Preprocessing: {'✅' if preprocessing_ok else '❌'}")
    logger.info(f"  Deepgram Transcription: {'✅' if transcription_ok else '❌'}")
    
    if api_key_ok and preprocessing_ok and transcription_ok:
        logger.info("✅ All tests passed! The issue should be resolved.")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above for details.")
    
    # Recommendations
    logger.info("\n📋 Recommendations:")
    logger.info("1. Restart your server after configuration changes")
    logger.info("2. Test with the created audio samples in debug_audio/")
    logger.info("3. Check your microphone permissions and levels")
    logger.info("4. Monitor the server logs for any remaining issues")


if __name__ == "__main__":
    asyncio.run(main())