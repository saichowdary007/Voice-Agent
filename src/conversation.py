"""
Refactored ConversationManager to be fully asynchronous and use the Supabase Python client.
Enhanced with proper error handling, type safety, and optimized async patterns.
"""
import asyncio
import logging
import os
import re
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
        self.embedding_model = self._load_embedding_model() if self.use_supabase else None
        self.vector_search_available = True  # Track if vector search is available
        
        # Personal fact storage settings
        self.max_snippets_per_user = 200
        self.similarity_threshold = 0.88  # For deduplication
        
        if self.use_supabase and self.supabase and SUPABASE_AVAILABLE:
            logger.info("✅ Conversation history is enabled (Supabase).")
        else:
            logger.warning("⚠️ Conversation history is disabled. Supabase not configured in .env file.")

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
        if not self.embedding_model or not self.vector_search_available:
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
            # Check for HTTP 404 (function not found) or general function errors
            error_str = str(e).lower()
            if ('404' in error_str or 'function' in error_str and 'match_conversations' in error_str or 
                'does not exist' in error_str):
                logger.warning("⚠️ Vector search function not available - disabling semantic context")
                self.vector_search_available = False  # Disable future attempts
            else:
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

    def _load_embedding_model(self):
        if self.use_supabase and SUPABASE_AVAILABLE and SentenceTransformer:
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Embedding model loaded successfully")
                return model
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model: {e}")
                return None
        return None

    # === PERSONAL FACT STORAGE PIPELINE ===

    async def _classify_personal_fact(self, sentence: str, llm_interface) -> bool:
        """
        Fast classification to determine if sentence contains personal information.
        Returns True if it contains personal facts worth saving.
        """
        if not llm_interface:
            return False
            
        fact_prompt = f"""
Decide whether the USER sentence contains PERSONAL FACTS worth saving:

USER: "{sentence}"

Respond with exactly one token:
- YES  -> contains new personal info
- NO   -> does NOT contain personal info
"""
        
        try:
            # Use a simplified call for classification
            response = await llm_interface.generate_response(
                user_text=fact_prompt,
                conversation_history=[],
                user_profile=[]
            )
            
            # Clean and check response
            decision = response.strip().upper()
            return decision == "YES"
            
        except Exception as e:
            logger.error(f"❌ Error in personal fact classification: {e}")
            return False

    async def _summarize_to_bullet(self, text: str, llm_interface) -> str:
        """
        Summarizes user text into a single bullet point.
        """
        if not llm_interface:
            return text
            
        prompt = f"""
Rewrite the USER's sentence as one bullet starting with '• '. 
Keep concrete nouns & numbers; drop filler words.

USER: "{text}"
"""
        
        try:
            response = await llm_interface.generate_response(
                user_text=prompt,
                conversation_history=[],
                user_profile=[]
            )
            
            bullet = response.strip()
            if not bullet.startswith('•'):
                bullet = f"• {bullet}"
            return bullet
            
        except Exception as e:
            logger.error(f"❌ Error in summarization: {e}")
            return f"• {text}"

    def _redact_pii(self, text: str) -> str:
        """
        Basic PII redaction using regex patterns.
        """
        # Phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # Email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # SSN patterns
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        
        # Credit card patterns
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)
        
        return text

    async def _check_duplicate(self, fact: str) -> bool:
        """
        Check if a similar fact already exists using cosine similarity.
        Returns True if duplicate exists (similarity > threshold).
        """
        if not self.embedding_model or not self.use_supabase or not self.supabase:
            return False
            
        try:
            fact_embedding = self.embedding_model.encode(fact).tolist()
            
            # Get existing snippets for this user
            response = await asyncio.to_thread(
                lambda: self.supabase.table('memory_snippets')
                .select('content, embedding')
                .eq('user_id', self.user_id)
                .execute()
            )
            
            existing_snippets = response.data or []
            
            # Check similarity with existing snippets
            for snippet in existing_snippets:
                if 'embedding' in snippet and snippet['embedding']:
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(fact_embedding, snippet['embedding'])
                    if similarity > self.similarity_threshold:
                        return True
                        
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking for duplicates: {e}")
            return False

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import numpy as np
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            
            if norm_vec1 == 0 or norm_vec2 == 0:
                return 0
                
            return dot_product / (norm_vec1 * norm_vec2)
        except Exception as e:
            logger.error(f"❌ Error calculating cosine similarity: {e}")
            return 0

    async def _store_fact(self, fact: str) -> bool:
        """
        Store a personal fact in the memory_snippets table.
        """
        if not self.use_supabase or not self.supabase or not self.embedding_model:
            return False
            
        try:
            # Redact PII
            clean_fact = self._redact_pii(fact)
            
            # Check for duplicates
            if await self._check_duplicate(clean_fact):
                logger.debug(f"Skipping duplicate fact: {clean_fact}")
                return True  # Not an error, just a duplicate
            
            # Generate embedding
            embedding = self.embedding_model.encode(clean_fact).tolist()
            
            # Store the fact
            await asyncio.to_thread(
                lambda: self.supabase.table('memory_snippets').insert({
                    'user_id': self.user_id,
                    'content': clean_fact,
                    'importance': 7,  # Default importance
                    'embedding': embedding
                }).execute()
            )
            
            # Prune old facts if over limit
            await self._prune_old_facts()
            
            logger.info(f"✅ Stored personal fact: {clean_fact}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing fact: {e}")
            return False

    async def _prune_old_facts(self) -> None:
        """
        Keep only the most recent facts, prune by lowest importance.
        """
        if not self.use_supabase or not self.supabase:
            return
            
        try:
            # Count current snippets
            count_response = await asyncio.to_thread(
                lambda: self.supabase.table('memory_snippets')
                .select('id', count='exact')
                .eq('user_id', self.user_id)
                .execute()
            )
            
            current_count = count_response.count
            
            if current_count <= self.max_snippets_per_user:
                return
                
            # Get snippets to delete (oldest and lowest importance)
            snippets_to_delete = current_count - self.max_snippets_per_user
            
            response = await asyncio.to_thread(
                lambda: self.supabase.table('memory_snippets')
                .select('id')
                .eq('user_id', self.user_id)
                .order('importance', desc=False)
                .order('created_at', desc=False)
                .limit(snippets_to_delete)
                .execute()
            )
            
            if response.data:
                ids_to_delete = [item['id'] for item in response.data]
                await asyncio.to_thread(
                    lambda: self.supabase.table('memory_snippets')
                    .delete()
                    .in_('id', ids_to_delete)
                    .execute()
                )
                logger.info(f"✅ Pruned {len(ids_to_delete)} old facts")
                
        except Exception as e:
            logger.error(f"❌ Error pruning old facts: {e}")

    async def handle_user_turn(self, user_msg: str, assistant_msg: str, llm_interface) -> None:
        """
        Main pipeline for handling user turns and storing personal facts.
        
        Args:
            user_msg: The user's message
            assistant_msg: The assistant's response
            llm_interface: LLM interface for classification and summarization
        """
        if not user_msg.strip():
            return
            
        try:
            # 1. Check for explicit verbs
            user_lower = user_msg.lower()
            explicit_keywords = ("remember", "save this", "note this")
            
            if any(keyword in user_lower for keyword in explicit_keywords):
                # Store raw or summarized version
                bullet = await self._summarize_to_bullet(user_msg, llm_interface)
                await self._store_fact(bullet)
                return
            
            # 2. Run the classification
            if await self._classify_personal_fact(user_msg, llm_interface):
                # 3. Summarize into one bullet
                bullet = await self._summarize_to_bullet(user_msg, llm_interface)
                await self._store_fact(bullet)
            
        except Exception as e:
            logger.error(f"❌ Error in handle_user_turn: {e}")

    async def get_memory_snippets(self, query_text: str, max_results: int = 5) -> List[Dict]:
        """
        Retrieve memory snippets semantically similar to the query.
        """
        if not self.embedding_model or not self.use_supabase or not self.supabase:
            return []
            
        try:
            query_embedding = self.embedding_model.encode(query_text).tolist()
            
            response = await asyncio.to_thread(
                lambda: self.supabase.rpc(
                    'match_memory_snippets',
                    {
                        'query_embedding': query_embedding,
                        'match_threshold': 0.35,
                        'match_count': max_results,
                        'user_id': self.user_id
                    }
                ).execute()
            )
            
            return response.data or []
            
        except Exception as e:
            error_str = str(e).lower()
            if '404' in error_str or 'function' in error_str:
                logger.warning("⚠️ Memory snippets search function not available")
            else:
                logger.error(f"❌ Error retrieving memory snippets: {e}")
            return []

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
