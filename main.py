import pyaudio
import time
import numpy as np
import collections
import traceback

from src.vad import VAD
from src.stt import STT
from src.llm import LLM
from src.tts import TTS
from src.conversation import ConversationManager
from src.audio_utils import AudioPlayer
from src.language_detection import LanguageDetector
from src.config import (
    INPUT_SAMPLE_RATE, INPUT_CHANNELS, INPUT_FORMAT,
    VAD_FRAME_MS, VAD_SILENCE_TIMEOUT_MS
)

# --- Constants ---
WAKE_WORD = "hey gemini"
CHUNK_SIZE = int(INPUT_SAMPLE_RATE * (VAD_FRAME_MS / 1000.0))
SILENCE_FRAMES = int(VAD_SILENCE_TIMEOUT_MS / VAD_FRAME_MS)

class VoiceAgent:
    def __init__(self):
        print("Initializing Voice Agent...")
        try:
            self.p = pyaudio.PyAudio()
            self.vad = VAD()
            self.stt = STT()
            self.llm = LLM()
            self.tts = TTS()
            self.language_detector = LanguageDetector()
            self.conversation = ConversationManager()
            self.player = AudioPlayer()
            self.state = "IDLE" # IDLE, LISTENING, PROCESSING, SPEAKING
            print("Voice Agent Initialized.")
        except Exception as e:
            print("Failed to initialize Voice Agent components. See error below.")
            traceback.print_exc()
            # Exit or handle gracefully
            raise e

    def run(self):
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=INPUT_CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        print("\n--- Voice Agent is running ---")
        print("Say 'Hey Gemini' to wake me up.")
        print("---------------------------------")
        
        voiced_frames = collections.deque(maxlen=SILENCE_FRAMES)
        audio_buffer = []
        is_speaking = False

        try:
            while True:
                try:
                    frame = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    is_speech = self.vad.is_speech(frame)

                    if self.state == "SPEAKING" and is_speech:
                        print("Barge-in detected! Stopping playback.")
                        self.player.stop_playback()
                        self.state = "LISTENING"
                        is_speaking = False
                        audio_buffer = [frame] # Start new buffer with the frame that triggered barge-in
                        voiced_frames.clear()
                        voiced_frames.append(is_speech)
                        continue

                    if self.state == "IDLE":
                        if is_speech:
                            audio_buffer.append(frame)
                            if len(audio_buffer) > 50:
                                full_audio = b''.join(audio_buffer)
                                audio_np = STT.audio_bytes_to_numpy(full_audio)
                                text = self.stt.transcribe(audio_np).lower()
                                
                                if WAKE_WORD in text:
                                    print(f"Wake word detected! '{text.strip()}'")
                                    self.state = "LISTENING"
                                    audio_buffer = []
                                    voiced_frames.clear()
                                    # Play confirmation sound
                                    self.state = "SPEAKING"
                                    self.player.start_playback()
                                    for chunk in self.tts.synthesize_stream("I'm listening", lang='en'):
                                        self.player.write_chunk(chunk)
                                    self.player.stop_playback()
                                    self.state = "LISTENING" # Ready to listen for command

                                else:
                                    audio_buffer.pop(0)

                    elif self.state == "LISTENING":
                        audio_buffer.append(frame)
                        voiced_frames.append(is_speech)
                        
                        if len(voiced_frames) == SILENCE_FRAMES and all(not s for s in voiced_frames):
                            self.state = "PROCESSING"
                            print("End of speech detected. Processing...")
                            
                            full_audio = b''.join(audio_buffer)
                            audio_np = STT.audio_bytes_to_numpy(full_audio)
                            
                            audio_buffer = []
                            voiced_frames.clear()

                            user_text = self.stt.transcribe(audio_np)
                            print(f"You said: {user_text}")

                            if not user_text.strip():
                                self.state = "IDLE"
                                continue
                            
                            detected_lang = self.language_detector.detect_language(user_text)
                            llm_context = self.conversation.get_context_for_llm(user_text)
                            self.conversation.add_message("user", user_text)
                            
                            assistant_response = self.llm.generate_response(user_text, llm_context, language=detected_lang)
                            self.conversation.add_message("model", assistant_response)
                            print(f"Assistant ({detected_lang}): {assistant_response}")
                            
                            self.state = "SPEAKING"
                            is_speaking = True
                            
                            self.player.start_playback()
                            for audio_chunk in self.tts.synthesize_stream(assistant_response, lang=detected_lang):
                                if self.player.is_playing():
                                    self.player.write_chunk(audio_chunk)
                                else:
                                    print("Playback stopped, likely due to barge-in.")
                                    is_speaking = False
                                    break
                            
                            self.player.stop_playback()
                            
                            if is_speaking: # If it finished without interruption
                                self.state = "IDLE"
                            is_speaking = False
                except Exception as e:
                    print("\nAn error occurred in the main loop:")
                    traceback.print_exc()
                    print("Restarting listening loop...")
                    # Reset state to recover
                    self.state = "IDLE"
                    audio_buffer.clear()
                    voiced_frames.clear()
                    self.player.stop_playback()
                    time.sleep(1) # Prevent rapid-fire error loops

        except KeyboardInterrupt:
            print("Stopping Voice Agent.")
        finally:
            stream.stop_stream()
            stream.close()
            self.player.stop_playback()
            self.p.terminate()

if __name__ == "__main__":
    agent = VoiceAgent()
    agent.run()
