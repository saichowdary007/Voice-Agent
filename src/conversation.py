"""
Refactored ConversationManager to be fully asynchronous and use the Supabase Python client.
Enhanced with proper error handling, type safety, and optimized async patterns.
"""
import asyncio
import logging
from typing import List, Dict, Optional
from src.config import USE_SUPABASE, SUPABASE_URL, SUPABASE_KEY
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)

# Conditional imports to avoid dependency issues
if USE_SUPABASE:
    try:
        from sentence_transformers import SentenceTransformer
        from supabase import create_client, Client
        SUPABASE_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"⚠️ Supabase dependencies not available: {e}")
        SUPABASE_AVAILABLE = False
        SentenceTransformer = None
        create_client = None
        Client = None
else:
    SUPABASE_AVAILABLE = False
    SentenceTransformer = None
    create_client = None
    Client = None

class ConversationManager:
    """
    Manages conversation state and history using Supabase.
    This class is designed to be used in an async environment with proper error handling.
    """
    def __init__(self, user_id: str):
        if not user_id:
            raise ValueError("A user ID must be provided to initialize the ConversationManager.")
        self.user_id = user_id
        self.use_supabase = USE_SUPABASE and SUPABASE_AVAILABLE
        self.supabase: Optional[Client] = self._connect_supabase()
        
        if self.use_supabase and self.supabase and SUPABASE_AVAILABLE:
            logger.info("✅ Conversation history is enabled (Supabase).")
            # The embedding model is loaded only if Supabase is in use.
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model: {e}")
                self.embedding_model = None
        else:
            logger.warning("⚠️ Conversation history is disabled. Supabase not configured in .env file.")
            self.embedding_model = None

    def _connect_supabase(self) -> Optional[Client]:
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
        if not self.use_supabase or not self.supabase or not self.embedding_model:
            return False

        try:
            embedding = self.embedding_model.encode(text).tolist()
            
            # Use asyncio.to_thread for CPU-bound operation
            await asyncio.to_thread(
                lambda: self.supabase.table('conversation_history').insert({
                    'session_id': self.user_id,
                    'role': role,
                    'text': text,
                    'embedding': embedding
                }).execute()
            )
            return True
        except Exception as e:
            logger.error(f"❌ Error adding message to Supabase: {e}")
            return False

    async def _get_semantic_context(self, current_text: str, max_results: int = 3) -> List[Dict]:
        """Retrieves semantically similar messages from the past."""
        if not self.embedding_model:
            return []
            
        try:
            current_embedding = self.embedding_model.encode(current_text).tolist()
            response = await asyncio.to_thread(
                lambda: self.supabase.rpc(
                    'match_conversations', 
                    {
                        'query_embedding': current_embedding, 
                        'match_threshold': 0.7, 
                        'match_count': max_results,
                        'session_id': self.user_id  # Add session filter
                    }
                ).execute()
            )
            return response.data or []
        except Exception as e:
            logger.error(f"❌ Vector search error: {e}")
            return []

    async def _get_recent_context(self, max_results: int = 4) -> List[Dict]:
        """Retrieves the most recent messages."""
        try:
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

    async def get_context_for_llm(self, current_text: str) -> List[Dict[str, str]]:
        """
        Retrieves a combined context of recent and semantically relevant messages.
        
        Returns:
            Formatted message history for LLM consumption.
        """
        if not self.use_supabase or not self.supabase or not self.embedding_model:
            return []
        
        try:
            # Concurrently fetch recent and semantic history
            tasks = [
                self._get_recent_context(),
                self._get_semantic_context(current_text)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            recent_history = results[0] if not isinstance(results[0], Exception) else []
            semantic_history = results[1] if not isinstance(results[1], Exception) else []

            # Combine and deduplicate the histories
            combined_history = {}
            
            # Process semantic history
            for item in semantic_history:
                item_text = item.get('content')
                if item_text:
                    combined_history[item_text] = item

            # Process recent history
            for item in recent_history:
                item_text = item.get('text')
                if item_text:
                    # Unify the key to 'content' for consistent formatting
                    item['content'] = item_text
                    combined_history[item_text] = item
            
            # Sort the unique history items by timestamp
            sorted_history = sorted(
                list(combined_history.values()), 
                key=lambda x: datetime.fromisoformat(x['created_at'])
            )
            
            # Format for the Gemini API
            return [
                {"role": row['role'], "parts": [{"text": row['content']}]} 
                for row in sorted_history
            ]
        except Exception as e:
            logger.error(f"❌ Error getting context for LLM: {e}")
            return []

    async def get_user_profile(self) -> List[Dict[str, str]]:
        """Retrieves all facts for the current user."""
        if not self.use_supabase or not self.supabase:
            return []
            
        try:
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

# --- Example Usage ---
async def main():
    """Example of how to use the async ConversationManager."""
    print("--- ConversationManager Example ---")
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

    print("\nSearching for context related to 'famous places there'...")
    context = await manager.get_context_for_llm("famous places there")

    print("\nRetrieved Context:")
    for item in context:
        print(f"- {item['role']}: {item['parts'][0]['text']}")
    
    print("\n--- ConversationManager Example Complete ---")

if __name__ == '__main__':
    # This requires a running event loop to work.
    # To run this file directly: python -m asyncio
    # And then in the REPL: from src.conversation import main; await main()
    # Or simply run the main_refactored.py which uses this class.
    print("To test this module, run main_refactored.py or use an asyncio REPL.")
