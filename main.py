#!/usr/bin/env python3
"""
Main entry point for the refactored, ultra-fast voice agent.
Now with user authentication, proper error handling, and resource management.
"""
# Set this environment variable before importing any other modules.
# This prevents a deadlock issue with the tokenizers library when used in a forked process.
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import getpass
import logging
import signal
import sys
from typing import Optional

from src.stt import STT
from src.llm import LLM
from src.tts import TTS
from src.conversation import ConversationManager
from src.auth import AuthManager
from src.config import USE_SUPABASE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceAgent:
    """
    A simple, fast, and conversational voice agent with proper resource management.
    """
    def __init__(self, user_id: str, auth_manager: AuthManager):
        logger.info("🚀 Initializing Voice Agent...")
        self.user_id = user_id
        self.auth_manager = auth_manager
        self.running = False
        
        # Initialize components with error handling
        try:
            self.stt = STT()
            logger.info("✅ STT initialized")
        except Exception as e:
            logger.error(f"❌ STT initialization failed: {e}")
            raise
            
        try:
            self.llm = LLM()
            logger.info("✅ LLM initialized")
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")
            raise
            
        try:
            self.tts = TTS()
            logger.info("✅ TTS initialized")
        except Exception as e:
            logger.error(f"❌ TTS initialization failed: {e}")
            raise
        
        # Initialize conversation manager if Supabase is available
        if USE_SUPABASE:
            try:
                self.conversation = ConversationManager(user_id=self.user_id)
                logger.info("✅ Conversation manager initialized")
            except Exception as e:
                logger.warning(f"⚠️ Conversation manager failed to initialize: {e}")
                self.conversation = None
        else:
            self.conversation = None
            logger.info("ℹ️ Running without conversation history")
        
        logger.info("✅ Voice Agent initialized successfully!")

    async def cleanup(self):
        """Clean up resources."""
        logger.info("🧹 Cleaning up resources...")
        try:
            if hasattr(self.llm, 'cleanup_session'):
                await self.llm.cleanup_session()
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

    async def run(self):
        """
        The main async loop for the voice agent with proper error handling.
        """
        self.running = True
        logger.info("🎙️ Voice Agent is now listening...")
        
        try:
            while self.running:
                try:
                    # 1. Listen and Transcribe
                    user_text = self.stt.listen_and_transcribe()

                    if not user_text:
                        continue

                    # Check for exit commands
                    if any(cmd in user_text.lower().strip() for cmd in 
                          ["sign out", "log out", "exit", "quit", "goodbye"]):
                        logger.info("User requested to exit")
                        self.auth_manager.sign_out()
                        break

                    # 2. Get Conversation History & User Profile (if enabled)
                    history = []
                    profile_facts = []
                    if self.conversation:
                        try:
                            history, profile_facts = await asyncio.gather(
                                self.conversation.get_context_for_llm(user_text),
                                self.conversation.get_user_profile(),
                                return_exceptions=True
                            )
                            
                            # Handle exceptions from gather
                            if isinstance(history, Exception):
                                logger.warning(f"Failed to get history: {history}")
                                history = []
                            if isinstance(profile_facts, Exception):
                                logger.warning(f"Failed to get profile: {profile_facts}")
                                profile_facts = []
                                
                        except Exception as e:
                            logger.warning(f"Failed to get conversation context: {e}")
                            history, profile_facts = [], []

                    # 3. Generate AI Response
                    logger.info("🤖 Thinking...")
                    try:
                        ai_response = await self.llm.generate_response(
                            user_text, history, profile_facts
                        )
                        logger.info(f"💬 AI: {ai_response}")
                    except Exception as e:
                        logger.error(f"LLM generation failed: {e}")
                        ai_response = "I'm having trouble processing that right now. Please try again."

                    # 4. Speak the Response
                    try:
                        await self.tts.speak(ai_response)
                    except Exception as e:
                        logger.error(f"TTS failed: {e}")
                        print(f"AI: {ai_response}")  # Fallback to text output

                    # 5. Update history and learn new facts
                    if self.conversation:
                        try:
                            # Save conversation
                            await asyncio.gather(
                                self.conversation.add_message("user", user_text),
                                self.conversation.add_message("model", ai_response),
                                return_exceptions=True
                            )

                            # Extract and update user profile facts
                            new_facts = await self.llm.extract_facts(
                                f"User: {user_text}\nAI: {ai_response}"
                            )
                            if new_facts:
                                await self.conversation.update_user_profile(new_facts)
                        except Exception as e:
                            logger.warning(f"Failed to save conversation: {e}")

                except KeyboardInterrupt:
                    logger.info("Received interrupt signal")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    print("Something went wrong. Continuing...")
                    
        finally:
            self.running = False
            await self.cleanup()

def setup_signal_handlers(agent: VoiceAgent):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        agent.running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main application entry point with proper error handling."""
    auth_manager = AuthManager()
    session = None

    try:
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
            if not email:
                print("Email cannot be empty.")
                continue
                
            password = getpass.getpass("Enter your password: ").strip()
            if not password:
                print("Password cannot be empty.")
                continue

            try:
                if action == "1":
                    session = auth_manager.sign_in(email, password)
                elif action == "2":
                    session = auth_manager.sign_up(email, password)
                    if session:
                        print("\nSign-up successful! You are now logged in.")
                    else:
                        print("\nSign-up failed. Please try logging in with existing credentials.")
            except Exception as e:
                logger.error(f"Authentication error: {e}")
                print(f"Authentication failed: {e}")

        if session:
            # Create and run the voice agent
            agent = VoiceAgent(user_id=session.user.id, auth_manager=auth_manager)
            setup_signal_handlers(agent)
            await agent.run()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"An error occurred: {e}")
    finally:
        logger.info("Application shutting down")

if __name__ == "__main__":
    try:
        if USE_SUPABASE:
            asyncio.run(main())
        else:
            print("Authentication is disabled because Supabase is not configured in your .env file.")
            print("Running in offline mode without user profiles or history.")
            print("Please configure SUPABASE_URL and SUPABASE_KEY in your .env file to enable full functionality.")

            # --- Offline fallback --------------------------------------------------
            async def _offline_run():
                """Run the VoiceAgent in a local–only demo mode."""
                # Force demo mode so AuthManager does not attempt any network calls
                auth_manager_offline = AuthManager(demo_mode=True)
                # Use a static user id to satisfy VoiceAgent constructor
                offline_user_id = "offline_user"
                agent = VoiceAgent(user_id=offline_user_id, auth_manager=auth_manager_offline)
                setup_signal_handlers(agent)
                await agent.run()

            # Execute the offline agent loop
            asyncio.run(_offline_run())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
