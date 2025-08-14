"""
Simplified ConversationManager for Deepgram Voice Agent with Supabase.
Handles basic conversation history without ML embeddings or sentence transformers.
"""
import asyncio
import logging
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from src.config import USE_SUPABASE, SUPABASE_URL, SUPABASE_KEY
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)

# For type checkers, import the type
if TYPE_CHECKING:
    from supabase import Client

# Conditional imports to avoid dependency issues
if USE_SUPABASE:
    try:
        from supabase import create_client
        SUPABASE_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"⚠️ Supabase dependencies not available: {e}")
        SUPABASE_AVAILABLE = False
        create_client = None
else:
    SUPABASE_AVAILABLE = False
    create_client = None

class ConversationManager:
    """
    Simplified conversation manager for Deepgram Voice Agent with Supabase.
    Handles basic conversation history without ML embeddings.
    """

    def __init__(self, user_id: str):
        if not user_id:
            raise ValueError("A user ID must be provided to initialize the ConversationManager.")
        self.user_id = user_id
        self.use_supabase = USE_SUPABASE and SUPABASE_AVAILABLE
        self.supabase: Optional['Client'] = self._connect_supabase()
        
        if self.use_supabase and self.supabase and SUPABASE_AVAILABLE:
            logger.info("✅ Conversation history is enabled (Supabase).")
        else:
            logger.warning("⚠️ Conversation history is disabled. Supabase not configured in .env file.")

    def _connect_supabase(self) -> Optional['Client']:
        """Establishes a connection to Supabase if configured."""
        if self.use_supabase and SUPABASE_AVAILABLE and create_client:
            try:
                return create_client(SUPABASE_URL, SUPABASE_KEY)
            except Exception as e:
                logger.error(f"❌ Failed to connect to Supabase: {e}")
                return None
        return None

    async def add_message(self, role: str, text: str) -> bool:
        """
        Adds a message to the conversation history in Supabase.
        
        Returns:
            True if successful, False otherwise.
        """
        if not self.use_supabase or not self.supabase:
            return False

        try:
            # Use asyncio.to_thread for database operation
            assert self.supabase is not None
            await asyncio.to_thread(
                lambda: self.supabase.table('conversation_history').insert({
                    'session_id': self.user_id,
                    'role': role,
                    'text': text
                }).execute()
            )
            return True
        except Exception as e:
            logger.error(f"❌ Error adding message to Supabase: {e}")
            return False

    async def get_recent_context(self, max_results: int = 10) -> List[Dict]:
        """Retrieves the most recent messages."""
        if not self.use_supabase or not self.supabase:
            return []
            
        try:
            assert self.supabase is not None
            response = await asyncio.to_thread(
                lambda: self.supabase.table('conversation_history')
                .select('role, text, created_at')
                .eq('session_id', self.user_id)
                .order('created_at', desc=True)
                .limit(max_results)
                .execute()
            )
            return response.data or []
        except Exception as e:
            logger.error(f"❌ Error fetching recent history from Supabase: {e}")
            return []

    async def get_context_for_llm(self, current_text: str = "") -> List[Dict[str, Any]]:
        """
        Retrieves recent conversation history formatted for LLM consumption.
        
        Returns:
            Formatted message history for LLM consumption.
        """
        if not self.use_supabase or not self.supabase:
            return []
        
        try:
            # Get recent history
            recent_history = await self.get_recent_context(max_results=8)
            
            # Sort by timestamp (oldest first for proper conversation flow)
            sorted_history = sorted(
                recent_history, 
                key=lambda x: datetime.fromisoformat(x['created_at'])
            )
            
            # Format for the Gemini API
            return [
                {"role": row['role'], "parts": [{"text": row['text']}]} 
                for row in sorted_history
            ]
        except Exception as e:
            logger.error(f"❌ Error getting context for LLM: {e}")
            return []

    async def get_user_profile(self) -> List[Dict[str, str]]:
        """Retrieves basic user profile facts."""
        if not self.use_supabase or not self.supabase:
            return []
            
        try:
            assert self.supabase is not None
            response = await asyncio.to_thread(
                lambda: self.supabase.table('user_profile')
                .select('key, value')
                .eq('user_id', self.user_id)
                .execute()
            )
            return response.data or []
        except Exception as e:
            logger.error(f"❌ Error fetching user profile from Supabase: {e}")
            return []

    async def update_user_profile(self, facts: List[Dict[str, str]]) -> bool:
        """
        Updates (upserts) a list of facts for the user.
        
        Returns:
            True if successful, False otherwise.
        """
        if not self.use_supabase or not self.supabase or not facts:
            return False

        try:
            # Process facts in parallel with error handling
            tasks = []
            assert self.supabase is not None
            for fact in facts:
                if 'key' in fact and 'value' in fact:
                    task = asyncio.to_thread(
                        lambda f=fact: self.supabase.rpc(
                            'upsert_user_profile',
                            {
                                'p_user_id': self.user_id, 
                                'p_key': f['key'], 
                                'p_value': f['value']
                            }
                        ).execute()
                    )
                    tasks.append(task)
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for any failures
                failures = [r for r in results if isinstance(r, Exception)]
                if failures:
                    logger.warning(f"Some profile updates failed: {len(failures)}/{len(tasks)}")
                else:
                    logger.info(f"✅ Updated user profile with {len(facts)} facts")
                
                return len(failures) == 0
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating user profile in Supabase: {e}")
            return False

    async def clear_history(self) -> bool:
        """
        Clears the history for the current session in Supabase.
        
        Returns:
            True if successful, False otherwise.
        """
        if not self.use_supabase or not self.supabase:
            return False
            
        try:
            assert self.supabase is not None
            await asyncio.to_thread(
                lambda: self.supabase.table('conversation_history')
                .delete()
                .eq('session_id', self.user_id)
                .execute()
            )
            logger.info(f"✅ History cleared for session: {self.user_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing history in Supabase: {e}")
            return False

    async def handle_user_turn(self, user_msg: str, assistant_msg: str, llm_interface=None) -> None:
        """
        Simple handler for user turns - just stores the conversation.
        
        Args:
            user_msg: The user's message
            assistant_msg: The assistant's response
            llm_interface: LLM interface (optional, for future use)
        """
        if not user_msg.strip():
            return
            
        try:
            # Store the conversation messages
            await self.add_message("user", user_msg)
            await self.add_message("model", assistant_msg)
            
            # Extract basic facts if LLM interface is available
            if llm_interface:
                try:
                    facts = await llm_interface.extract_facts(f"User: {user_msg}\nAI: {assistant_msg}")
                    if facts:
                        await self.update_user_profile(facts)
                except Exception as e:
                    logger.warning(f"Failed to extract facts: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error in handle_user_turn: {e}")

# --- Example Usage ---
async def main():
    """Example of how to use the simplified ConversationManager."""
    print("--- Simplified ConversationManager Example ---")
    manager = ConversationManager("user_123")
    
    if not manager.use_supabase:
        print("Please configure Supabase in your .env file to run this example.")
        return

    print("\nClearing history for a fresh start...")
    await manager.clear_history()

    print("\nAdding new messages...")
    await manager.add_message("user", "What is the capital of Italy?")
    await manager.add_message("model", "The capital of Italy is Rome.")
    await manager.add_message("user", "And what is its most famous landmark?")
    await manager.add_message("model", "That would likely be the Colosseum.")

    print("\nGetting context for LLM...")
    context = await manager.get_context_for_llm()

    print("\nRetrieved Context:")
    for item in context:
        print(f"- {item['role']}: {item['parts'][0]['text']}")
    
    print("\n--- ConversationManager Example Complete ---")

if __name__ == '__main__':
    print("To test this module, run server.py or use an asyncio REPL.")