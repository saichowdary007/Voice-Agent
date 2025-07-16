#!/usr/bin/env python3
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
            ("Very quiet audio", b"\x00" * 3200),  # Silence
            ("Low volume audio", b"\x10\x00" * 1600),  # Very quiet
            ("Normal audio", b"\x00\x10" * 1600),  # Low volume
        ]
        
        for scenario_name, test_audio in test_scenarios:
            print(f"\n🧪 Testing: {scenario_name}")
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
        
        print("\n📋 Recommendations:")
        print("- Ensure microphone is not muted")
        print("- Check microphone permissions in browser")
        print("- Test with different microphones if available")
        print("- Reduce background noise")
        print("- Speak at normal conversational volume")
        
    except Exception as e:
        print(f"❌ Diagnostics failed: {e}")

if __name__ == "__main__":
    asyncio.run(diagnose_audio_input())
