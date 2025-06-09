#!/usr/bin/env python3
"""
Main entry point for the refactored, ultra-fast voice agent.
Now with user authentication.
"""
# Set this environment variable before importing any other modules.
# This prevents a deadlock issue with the tokenizers library when used in a forked process.
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import getpass
from src.stt import STT
from src.llm import LLM
from src.tts import TTS
from src.conversation import ConversationManager
from src.auth import AuthManager
from src.config import USE_SUPABASE

class VoiceAgent:
    """
    A simple, fast, and conversational voice agent.
    """
    def __init__(self, user_id: str, auth_manager: AuthManager):
        print("🚀 Initializing Ultra-Fast Voice Agent...")
        self.user_id = user_id
        self.auth_manager = auth_manager
        self.stt = STT()
        self.llm = LLM()
        self.tts = TTS()
        if USE_SUPABASE:
            self.conversation = ConversationManager(user_id=self.user_id)
        else:
            self.conversation = None
        print("✅ Agent Initialized. Ready to chat!")

    async def run(self):
        """
        The main async loop for the voice agent.
        """
        while True:
            # 1. Listen and Transcribe
            user_text = self.stt.listen_and_transcribe()

            if not user_text:
                continue

            # Check for sign-out command
            if "sign out" in user_text.lower().strip() or "log out" in user_text.lower().strip():
                self.auth_manager.sign_out()
                break

            # 2. Get Conversation History & User Profile (if enabled)
            history = []
            profile_facts = []
            if self.conversation:
                history, profile_facts = await asyncio.gather(
                    self.conversation.get_context_for_llm(user_text),
                    self.conversation.get_user_profile()
                )

            # 3. Generate AI Response
            print("🤖 Thinking...")
            ai_response = await self.llm.generate_response(user_text, history, profile_facts)
            print(f"💬 AI: {ai_response}")

            # 4. Speak the Response
            await self.tts.speak(ai_response)

            # 5. Update history and learn new facts
            if self.conversation:
                await self.conversation.add_message("user", user_text)
                await self.conversation.add_message("model", ai_response)

                new_facts = await self.llm.extract_facts(f"User: {user_text}\nAI: {ai_response}")
                if new_facts:
                    await self.conversation.update_user_profile(new_facts)

async def main():
    auth_manager = AuthManager()
    session = None

    while not session:
        print("\n--- Voice Agent Login ---")
        action = input("Choose an action: [1] Login, [2] Sign Up, [3] Exit: ").strip()
        
        if action == "3":
            print("👋 Goodbye!")
            return

        if action not in ["1", "2"]:
            print("Invalid choice. Please try again.")
            continue

        email = input("Enter your email: ").strip()
        password = getpass.getpass("Enter your password: ").strip()

        if action == "1":
            session = auth_manager.sign_in(email, password)
        elif action == "2":
            auth_manager.sign_up(email, password)
            print("\nPlease log in with your new credentials.")

    # If login is successful, start the agent
    agent = VoiceAgent(user_id=session.user.id, auth_manager=auth_manager)
    await agent.run()

if __name__ == "__main__":
    try:
        if USE_SUPABASE:
            asyncio.run(main())
        else:
            print("Authentication is disabled because Supabase is not configured in your .env file.")
            print("Running in offline mode without user profiles or history.")
            # Simplified offline agent if needed in the future
            # For now, we'll just exit.
    except KeyboardInterrupt:
        print("\n gracefully shutting down")
