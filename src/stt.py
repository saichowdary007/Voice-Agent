"""
Refactored Speech-to-Text (STT) module to use the simple and effective `speech_recognition` library.
"""
import speech_recognition as sr
from src.config import ENERGY_THRESHOLD, PAUSE_THRESHOLD

class STT:
    """
    Handles Speech-to-Text conversion using Google's free web API via the
    `speech_recognition` library. It automatically handles silence detection.
    """
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = ENERGY_THRESHOLD
        self.recognizer.pause_threshold = PAUSE_THRESHOLD
        self.recognizer.non_speaking_duration = PAUSE_THRESHOLD
        self.microphone = sr.Microphone()

        # Calibrate for ambient noise upon initialization
        print("🎙️  Calibrating microphone for ambient noise... Please be quiet for a moment.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        print("✅ Microphone calibrated.")

    def listen_and_transcribe(self) -> str:
        """
        Listens for a single phrase from the microphone and transcribes it.
        This function will block until a phrase is detected.

        Returns:
            The transcribed text as a string, or None if speech could not be recognized.
        """
        try:
            with self.microphone as source:
                print("\n👂 Listening for your command...")
                audio = self.recognizer.listen(source)
            
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
            print(f"An unexpected error occurred in STT: {e}")
            return None

# --- Example Usage ---
if __name__ == '__main__':
    stt = STT()
    print("\n--- STT Module Example ---")
    print("Speak a sentence into the microphone.")
    text = stt.listen_and_transcribe()
    if text:
        print(f"\nSuccessfully transcribed: '{text}'")
    else:
        print("\nCould not transcribe speech.")
    print("--- STT Example Complete ---")
