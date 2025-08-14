"""
AgentConversationBridge
Bridges Deepgram events to Supabase conversation storage via ConversationManager.
"""
import logging
from typing import Optional

from src.conversation import ConversationManager

logger = logging.getLogger(__name__)


class AgentConversationBridge:
    """Convenience wrapper over ConversationManager for storing conversation text."""

    def __init__(self, user_id: Optional[str]):
        self.user_id = user_id
        self._manager: Optional[ConversationManager] = None
        if user_id:
            try:
                self._manager = ConversationManager(user_id)
            except Exception as e:
                logger.warning(f"Conversation bridge init failed: {e}")

    async def record_conversation_text(self, role: str, content: str) -> None:
        if not self._manager or not content:
            return
        try:
            await self._manager.add_message(role, content)
        except Exception as e:
            logger.debug(f"Conversation store failed: {e}")
