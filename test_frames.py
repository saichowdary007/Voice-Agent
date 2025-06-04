#!/usr/bin/env python3
import asyncio
import websockets
import json
import numpy as np

async def test_frame_config():
    uri = 'ws://localhost:8003/ws'
    try:
        async with websockets.connect(uri) as websocket:
            # Get initial configuration
            message = await websocket.recv()
            data = json.loads(message)
            config = data.get('config', {})
            
            print("📐 Frame Configuration:")
            print(f"  Sample Rate: {config.get('sample_rate')} Hz")
            print(f"  Frame Duration: {config.get('frame_duration_ms')} ms") 
            print(f"  Channels: {config.get('channels')}")
            
            # Calculate expected frame size
            sample_rate = config.get('sample_rate', 16000)
            frame_ms = config.get('frame_duration_ms', 32)
            channels = config.get('channels', 1)
            
            samples_per_frame = sample_rate * frame_ms // 1000
            bytes_per_frame = samples_per_frame * 2 * channels  # 16-bit audio
            
            print(f"  Expected samples per frame: {samples_per_frame}")
            print(f"  Expected bytes per frame: {bytes_per_frame}")
            
            # Test with correct frame size
            print(f"\n🧪 Testing with {samples_per_frame} samples...")
            
            # Create test audio frame (512 samples of sine wave)
            frequency = 440  # A4 note
            duration_ms = frame_ms
            t = np.linspace(0, duration_ms/1000, samples_per_frame, False)
            audio_samples = np.sin(2 * np.pi * frequency * t) * 0.1  # Low amplitude
            
            # Convert to 16-bit PCM bytes
            audio_int16 = (audio_samples * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            print(f"  Generated audio: {len(audio_bytes)} bytes = {len(audio_int16)} samples")
            
            # Send test audio frame
            await websocket.send(audio_bytes)
            print("  ✅ Sent test audio frame")
            
            # Wait for potential VAD response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                response_data = json.loads(response)
                print(f"  📨 Response: {response_data}")
            except asyncio.TimeoutError:
                print("  ⏰ No immediate response (normal for VAD)")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_frame_config()) 