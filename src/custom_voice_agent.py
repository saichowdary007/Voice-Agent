"""
Custom Voice Agent that integrates Supabase conversation history with LLM responses
This bypasses Deepgram's built-in LLM to use our custom logic
"""
import asyncio
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from src.llm import LLM
from src.conversation import ConversationManager
from src.config import USE_SUPABASE

logger = logging.getLogger(__name__)


class CustomVoiceAgent:
    """
    Custom voice agent that:
    1. Receives speech-to-text from Deepgram
    2. Retrieves conversation history from Supabase
    3. Generates responses using custom LLM logic
    4. Sends text back to Deepgram for TTS
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.llm = LLM()
        self.conversation_mgr = None
        
        if USE_SUPABASE and user_id:
            try:
                self.conversation_mgr = ConversationManager(user_id)
                logger.info(f"✅ Custom voice agent initialized for user: {user_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize conversation manager: {e}")
        else:
            logger.info("Custom voice agent running without conversation history")
    
    async def process_user_speech(self, user_text: str) -> Optional[str]:
        """
        Process user speech and generate response with Supabase context
        
        Args:
            user_text: The transcribed user speech
            
        Returns:
            AI response text or None if processing fails
        """
        if not user_text.strip():
            return None
            
        try:
            logger.info(f"🎤 Processing user speech: {user_text}")
            
            # Get conversation history and user profile from Supabase
            history = []
            profile_facts = []
            
            if self.conversation_mgr:
                try:
                    history, profile_facts = await asyncio.gather(
                        self.conversation_mgr.get_context_for_llm(user_text),
                        self.conversation_mgr.get_user_profile(),
                        return_exceptions=True
                    )
                    
                    # Handle exceptions from gather
                    if isinstance(history, Exception):
                        logger.warning(f"Failed to get history: {history}")
                        history = []
                    if isinstance(profile_facts, Exception):
                        logger.warning(f"Failed to get profile: {profile_facts}")
                        profile_facts = []
                        
                    logger.info(f"📚 Retrieved {len(history)} history messages and {len(profile_facts)} profile facts")
                    
                except Exception as e:
                    logger.warning(f"Failed to get conversation context: {e}")
                    history, profile_facts = [], []
            
            # Generate AI response using LLM with Supabase context
            ai_response = await self.llm.generate_response(
                user_text=user_text,
                conversation_history=history,
                user_profile=profile_facts
            )
            
            logger.info(f"🤖 Generated response: {ai_response[:100]}...")
            
            # Save conversation to Supabase
            if self.conversation_mgr:
                try:
                    await asyncio.gather(
                        self.conversation_mgr.add_message("user", user_text),
                        self.conversation_mgr.add_message("assistant", ai_response),
                        return_exceptions=True
                    )
                    
                    # Extract and update user profile facts
                    try:
                        new_facts = await self.llm.extract_facts(
                            f"User: {user_text}\nAI: {ai_response}"
                        )
                        if new_facts:
                            await self.conversation_mgr.update_user_profile(new_facts)
                            logger.info(f"📝 Updated profile with {len(new_facts)} new facts")
                    except Exception as e:
                        logger.warning(f"Failed to extract/save facts: {e}")
                        
                except Exception as e:
                    logger.warning(f"Failed to save conversation: {e}")
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error processing user speech: {e}")
            return "I'm having trouble processing that right now. Please try again."
    
    async def get_conversation_summary(self) -> str:
        """Get a summary of recent conversation for context"""
        if not self.conversation_mgr:
            return "No conversation history available."
            
        try:
            recent_history = await self.conversation_mgr.get_recent_context(max_results=5)
            if not recent_history:
                return "No recent conversation history."
                
            summary_parts = []
            for msg in reversed(recent_history):  # Show oldest first
                role = "You" if msg['role'] == 'user' else "AI"
                text = msg['text'][:100] + "..." if len(msg['text']) > 100 else msg['text']
                summary_parts.append(f"{role}: {text}")
                
            return "Recent conversation:\n" + "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Failed to get conversation summary: {e}")
            return "Error retrieving conversation history."
    
    async def clear_conversation_history(self) -> bool:
        """Clear conversation history for this user"""
        if not self.conversation_mgr:
            return False
            
        try:
            success = await self.conversation_mgr.clear_history()
            if success:
                logger.info(f"🗑️ Cleared conversation history for user: {self.user_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to clear conversation history: {e}")
            return False