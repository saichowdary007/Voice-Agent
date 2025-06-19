"""
Refactored Speech-to-Text (STT) module to use the simple and effective `speech_recognition` library.
"""
import speech_recognition as sr
from src.config import ENERGY_THRESHOLD, PAUSE_THRESHOLD
import asyncio

class STT:
    """
    Handles Speech-to-Text conversion using Google's free web API via the
    `speech_recognition` library. It automatically handles silence detection.
    """
    def __init__(self, calibrate_mic: bool = True, microphone_index: int = None):
        """Create an STT engine.

        Args:
            calibrate_mic: When *True* the local microphone will be opened and
                calibrated for ambient noise. This is useful for CLI demos but
                will fail on headless servers (no audio hardware). Set to *False* 
                only in server environments.
            microphone_index: Specific microphone index to use. If None, uses default.
        """
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = ENERGY_THRESHOLD
        self.recognizer.pause_threshold = PAUSE_THRESHOLD
        self.recognizer.non_speaking_duration = PAUSE_THRESHOLD

        # Always initialize microphone for voice recognition to work
        self.microphone = None
        try:
            # List available microphones for debugging
            mics = sr.Microphone.list_microphone_names()
            print(f"🎤 Available microphones: {mics}")
            
            if microphone_index is not None:
                self.microphone = sr.Microphone(device_index=microphone_index)
                print(f"🎤 Using microphone {microphone_index}: {mics[microphone_index] if microphone_index < len(mics) else 'Unknown'}")
            else:
                self.microphone = sr.Microphone()
                print("🎤 Using default microphone")
                
            if calibrate_mic:
                print("🎙️  Calibrating microphone for ambient noise... Please be quiet for a moment.")
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print(f"✅ Microphone calibrated. Energy threshold: {self.recognizer.energy_threshold}")
            else:
                print(f"🎤 Microphone initialized without calibration. Energy threshold: {self.recognizer.energy_threshold}")
                
        except Exception as e:
            print(f"❌ Error initializing microphone: {e}")
            print("💡 Try running with different microphone_index or check audio permissions")
            # Still create a microphone object to prevent None errors
            try:
                self.microphone = sr.Microphone()
            except:
                pass

    def listen_and_transcribe(self, timeout: int = None, phrase_time_limit: int = None) -> str:
        """
        Listens for a single phrase from the microphone and transcribes it.
        This function will block until a phrase is detected.

        Args:
            timeout: Maximum time to wait for phrase to start (seconds)
            phrase_time_limit: Maximum time to record for (seconds)

        Returns:
            The transcribed text as a string, or None if speech could not be recognized.
        """
        if self.microphone is None:
            print("❌ Microphone not initialized. Cannot listen for audio.")
            return None
            
        try:
            with self.microphone as source:
                print("\n👂 Listening for your command...")
                # Add timeout and phrase_time_limit for better control
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
            
            print("🧠 Transcribing...")
            # Use Google's free web recognizer
            transcript = self.recognizer.recognize_google(audio)
            print(f"🎤 You said: {transcript}")
            return transcript

        except sr.WaitTimeoutError:
            print("⌛ Listening timed out while waiting for phrase to start.")
            return None
        except sr.UnknownValueError:
            print("🤔 Sorry, I didn't catch that. Could you please repeat?")
            return None
        except sr.RequestError as e:
            print(f"📡 Could not request results from Google Speech Recognition service; {e}")
            return None
        except Exception as e:
            print(f"❌ An unexpected error occurred in STT: {e}")
            return None

    async def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw PCM/FLAC/WAV bytes sent by the client.

        The client is expected to send **16-bit little-endian PCM** at the
        specified *sample_rate* (default 16 kHz). This keeps the server-side
        dependency surface minimal and still works well with
        ``speech_recognition``.
        """
        if not audio_bytes:
            return ""

        try:
            # speech_recognition expects AudioData – build one from raw bytes
            audio_data = sr.AudioData(audio_bytes, sample_rate, 2)  # 2 bytes per sample (16-bit)
            text = await asyncio.to_thread(self.recognizer.recognize_google, audio_data)
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            print(f"📡 Google STT request failed: {e}")
            return ""
        except Exception as e:
            print(f"❌ Unexpected STT error: {e}")
            return ""

    def test_microphone(self):
        """Test microphone input and show audio levels"""
        if self.microphone is None:
            print("❌ No microphone available for testing")
            return
            
        try:
            with self.microphone as source:
                print("🎤 Testing microphone... Speak now!")
                print("Current energy threshold:", self.recognizer.energy_threshold)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
                print(f"Audio captured! Length: {len(audio.frame_data)} bytes")
                print("🔊 Testing transcription...")
                text = self.recognizer.recognize_google(audio)
                print(f"✅ Transcribed: '{text}'")
                return text
        except Exception as e:
            print(f"❌ Microphone test failed: {e}")
            return None

# --- Example Usage ---
if __name__ == '__main__':
    print("\n--- STT Module Example ---")
    
    # Try with calibration first
    stt = STT(calibrate_mic=True)
    
    if stt.microphone is None:
        print("❌ Could not initialize microphone. Exiting.")
        exit(1)
    
    # First, run a quick test
    print("\n🧪 Running microphone test...")
    test_result = stt.test_microphone()
    
    if test_result:
        print(f"\n✅ Test successful! Detected: '{test_result}'")
    else:
        print("\n⚠️  Test failed, but let's try the full example...")
    
    print("\n🎙️  Now speak a sentence for the main example...")
    text = stt.listen_and_transcribe(timeout=10, phrase_time_limit=10)
    if text:
        print(f"\n✅ Successfully transcribed: '{text}'")
    else:
        print("\n❌ Could not transcribe speech.")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check microphone permissions in System Preferences")
        print("   2. Try speaking louder or closer to the microphone")
        print("   3. Ensure a stable internet connection for Google STT")
        print("   4. Try a different microphone if available")
    print("--- STT Example Complete ---")
