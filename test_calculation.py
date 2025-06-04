#!/usr/bin/env python3

# Backend configuration (from config.py)
sample_rate = 16000  # Hz
audio_frame_ms = 32  # ms
channels = 1
bytes_per_sample = 2  # 16-bit audio

# Calculate backend frame size
backend_frame_size = sample_rate * audio_frame_ms // 1000 * bytes_per_sample * channels
backend_samples_per_frame = sample_rate * audio_frame_ms // 1000

print("🔧 Backend Configuration:")
print(f"  Sample Rate: {sample_rate} Hz")
print(f"  Frame Duration: {audio_frame_ms} ms")
print(f"  Expected frame size: {backend_frame_size} bytes")
print(f"  Expected samples per frame: {backend_samples_per_frame}")

# Frontend configuration (from VoiceAgent.tsx)
frontend_buffer_timeout = 120  # ms

# Calculate frontend chunk size
frontend_chunk_samples = sample_rate * frontend_buffer_timeout // 1000
frontend_chunk_bytes = frontend_chunk_samples * bytes_per_sample * channels

print("\n📱 Frontend Configuration:")
print(f"  Buffer Timeout: {frontend_buffer_timeout} ms")
print(f"  Chunk size: {frontend_chunk_bytes} bytes")
print(f"  Chunk samples: {frontend_chunk_samples}")

# Calculate how many backend frames per frontend chunk
frames_per_chunk = frontend_chunk_samples / backend_samples_per_frame
bytes_per_chunk_exact = frames_per_chunk * backend_frame_size

print("\n🧮 Conversion Analysis:")
print(f"  Frontend chunks per backend frame: {frames_per_chunk:.2f}")
print(f"  Should process {frames_per_chunk:.2f} backend frames per frontend chunk")

# VAD expectation
vad_expected_samples = 512  # From VAD error message
print(f"\n🎯 VAD Requirements:")
print(f"  VAD expects: {vad_expected_samples} samples")
print(f"  Backend provides: {backend_samples_per_frame} samples")
print(f"  Frontend sends: {frontend_chunk_samples} samples")

if frontend_chunk_samples == 1920:
    print("\n❌ ISSUE FOUND:")
    print(f"  Frontend is sending {frontend_chunk_samples} samples to VAD directly!")
    print(f"  This should be sliced into {frames_per_chunk:.1f} frames of {backend_samples_per_frame} samples each")
else:
    print("\n✅ No direct issue found with calculations") 