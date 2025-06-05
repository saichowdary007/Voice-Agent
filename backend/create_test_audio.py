import numpy as np
import wave

# Create a sine wave at 440 Hz for 3 seconds
sample_rate = 16000
duration = 3  # seconds
frequency = 440  # Hz

# Generate the sine wave
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
data = np.sin(2 * np.pi * frequency * t) * 0.5  # 50% amplitude

# Convert to 16-bit PCM
audio_data = (data * 32767).astype(np.int16)

# Save as WAV file
with wave.open('test_audio.wav', 'wb') as wav_file:
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 16-bit
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(audio_data.tobytes())

print("Created test_audio.wav with a 440 Hz tone for 3 seconds at 16 kHz sample rate.") 