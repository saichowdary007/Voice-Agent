#!/usr/bin/env python3
"""
Minimal Deepgram Voice Agent runner using Python SDK with Gemini (Google) as the think provider.

- Requires environment variables:
  - DEEPGRAM_API_KEY
  - GEMINI_API_KEY (used to call Google's Generative Language API v1beta endpoint)

This script:
- Connects to the Deepgram Agent WebSocket via the official SDK
- Configures audio in/out and agent providers (listen/speak/think)
- Streams a sample WAV (spacewalk.wav) to the agent
- Saves responses to output-*.wav and logs events to chatlog.txt
"""

import os
import time
import json
import threading
import requests

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    AgentWebSocketEvents,
    AgentKeepAlive,
)
from deepgram.clients.agent.v1.websocket.options import SettingsOptions


def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)

    header = bytearray(44)
    header[0:4] = b"RIFF"
    header[4:8] = b"\x00\x00\x00\x00"  # file size (unused here)
    header[8:12] = b"WAVE"
    header[12:16] = b"fmt "
    header[16:20] = b"\x10\x00\x00\x00"  # PCM fmt chunk size
    header[20:22] = b"\x01\x00"  # PCM
    header[22:24] = channels.to_bytes(2, "little")
    header[24:28] = sample_rate.to_bytes(4, "little")
    header[28:32] = byte_rate.to_bytes(4, "little")
    header[32:34] = block_align.to_bytes(2, "little")
    header[34:36] = bits_per_sample.to_bytes(2, "little")
    header[36:40] = b"data"
    header[40:44] = b"\x00\x00\x00\x00"  # data size (unused here)
    return header


def main():
    try:
        dg_api_key = os.getenv("DEEPGRAM_API_KEY")
        if not dg_api_key:
            raise ValueError("DEEPGRAM_API_KEY environment variable is not set")

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        # We will attach Google's endpoint to the provider for Gemini usage.
        # If GEMINI_API_KEY is not set, the request may fail unless your Deepgram project
        # has Gemini configured server-side.

        # Initialize Deepgram client
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram = DeepgramClient(dg_api_key, config)
        connection = deepgram.agent.websocket.v("1")

        # Configure the Voice Agent
        options = SettingsOptions()
        # Audio input configuration (PCM16 24kHz)
        options.audio.input.encoding = "linear16"
        options.audio.input.sample_rate = 24000
        # Audio output configuration
        options.audio.output.encoding = "linear16"
        options.audio.output.sample_rate = 24000
        options.audio.output.container = "wav"

        # Agent configuration
        options.agent.language = "en"
        options.agent.listen.provider.type = "deepgram"
        options.agent.listen.provider.model = "nova-3"

        # Use Google (Gemini) for thinking
        options.agent.think.provider.type = "google"
        options.agent.think.provider.model = os.getenv("DG_THINK_MODEL", "gemini-2.0-flash")
        options.agent.think.prompt = "You are a friendly AI assistant."

        # Attach Google endpoint when GEMINI_API_KEY is provided
        if gemini_api_key:
            model = options.agent.think.provider.model or "gemini-2.0-flash"
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_api_key}"
            # The SDK supports setting additional provider fields dynamically via dict access
            # when attributes are not explicitly modeled.
            options.agent.think.provider.endpoint = endpoint

        options.agent.speak.provider.type = "deepgram"
        options.agent.speak.provider.model = "aura-2-thalia-en"
        options.agent.greeting = "Hello! How can I help you today?"

        # Keep Alive loop
        def send_keep_alive():
            while True:
                time.sleep(5)
                try:
                    connection.send(str(AgentKeepAlive()))
                except Exception:
                    break

        threading.Thread(target=send_keep_alive, daemon=True).start()

        # Runtime state
        audio_buffer = bytearray()
        file_counter = 0
        processing_complete = False

        # Event handlers
        def on_audio_data(self, data, **kwargs):
            nonlocal audio_buffer
            audio_buffer.extend(data)

        def on_agent_audio_done(self, agent_audio_done, **kwargs):
            nonlocal audio_buffer, file_counter, processing_complete
            if len(audio_buffer) > 0:
                with open(f"output-{file_counter}.wav", "wb") as f:
                    f.write(create_wav_header())
                    f.write(audio_buffer)
            audio_buffer = bytearray()
            file_counter += 1
            processing_complete = True

        def on_conversation_text(self, conversation_text, **kwargs):
            with open("chatlog.txt", "a") as chatlog:
                chatlog.write(f"{json.dumps(conversation_text.__dict__)}\n")

        def on_welcome(self, welcome, **kwargs):
            with open("chatlog.txt", "a") as chatlog:
                chatlog.write(f"Welcome message: {welcome}\n")

        def on_settings_applied(self, settings_applied, **kwargs):
            with open("chatlog.txt", "a") as chatlog:
                chatlog.write(f"Settings applied: {settings_applied}\n")

        def on_user_started_speaking(self, user_started_speaking, **kwargs):
            with open("chatlog.txt", "a") as chatlog:
                chatlog.write(f"User Started Speaking: {user_started_speaking}\n")

        def on_agent_thinking(self, agent_thinking, **kwargs):
            with open("chatlog.txt", "a") as chatlog:
                chatlog.write(f"Agent Thinking: {agent_thinking}\n")

        def on_agent_started_speaking(self, agent_started_speaking, **kwargs):
            nonlocal audio_buffer
            audio_buffer = bytearray()
            with open("chatlog.txt", "a") as chatlog:
                chatlog.write(f"Agent Started Speaking: {agent_started_speaking}\n")

        def on_close(self, close, **kwargs):
            with open("chatlog.txt", "a") as chatlog:
                chatlog.write(f"Connection closed: {close}\n")

        def on_error(self, error, **kwargs):
            with open("chatlog.txt", "a") as chatlog:
                chatlog.write(f"Error: {error}\n")

        def on_unhandled(self, unhandled, **kwargs):
            with open("chatlog.txt", "a") as chatlog:
                chatlog.write(f"Unhandled event: {unhandled}\n")

        # Register handlers
        connection.on(AgentWebSocketEvents.AudioData, on_audio_data)
        connection.on(AgentWebSocketEvents.AgentAudioDone, on_agent_audio_done)
        connection.on(AgentWebSocketEvents.ConversationText, on_conversation_text)
        connection.on(AgentWebSocketEvents.Welcome, on_welcome)
        connection.on(AgentWebSocketEvents.SettingsApplied, on_settings_applied)
        connection.on(AgentWebSocketEvents.UserStartedSpeaking, on_user_started_speaking)
        connection.on(AgentWebSocketEvents.AgentThinking, on_agent_thinking)
        connection.on(AgentWebSocketEvents.AgentStartedSpeaking, on_agent_started_speaking)
        connection.on(AgentWebSocketEvents.Close, on_close)
        connection.on(AgentWebSocketEvents.Error, on_error)
        connection.on(AgentWebSocketEvents.Unhandled, on_unhandled)

        # Start connection
        if not connection.start(options):
            print("Failed to start connection")
            return

        # Stream demo audio to agent (skip 44-byte WAV header)
        response = requests.get("https://dpgr.am/spacewalk.wav", stream=True)
        header = response.raw.read(44)
        if header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
            print("Invalid WAV header from demo audio")
            return

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                connection.send(chunk)
                time.sleep(0.05)

        # Wait up to 30s for agent response to complete
        start = time.time()
        timeout = 30
        while not processing_complete and (time.time() - start) < timeout:
            time.sleep(0.25)

        connection.finish()
        print("Finished. Check output-*.wav and chatlog.txt")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()


