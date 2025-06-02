import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import structlog

from .gemini_client import Message

logger = structlog.get_logger()


@dataclass
class ConversationSession:
    """Represents a conversation session"""
    session_id: str
    start_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    messages: List[Message] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextManager:
    """Manages conversation context and session state"""
    
    def __init__(self, max_messages: int = 20, session_timeout: float = 1800.0):
        self.max_messages = max_messages  # Maximum messages to keep in context
        self.session_timeout = session_timeout  # 30 minutes session timeout
        
        # Current session
        self.current_session: Optional[ConversationSession] = None
        
        # Context summarization
        self.summary_threshold = 15  # Summarize when context gets too long
        self.current_summary = ""
        
    def start_session(self, session_id: str) -> ConversationSession:
        """Start a new conversation session"""
        self.current_session = ConversationSession(session_id=session_id)
        logger.info(f"Started new conversation session: {session_id}")
        return self.current_session
        
    def get_current_session(self) -> Optional[ConversationSession]:
        """Get current session, create if none exists"""
        if not self.current_session:
            import uuid
            session_id = str(uuid.uuid4())[:8]
            self.start_session(session_id)
            
        return self.current_session
        
    def add_user_message(self, content: str, metadata: Dict[str, Any] = None):
        """Add user message to conversation context"""
        session = self.get_current_session()
        
        message = Message(
            role='user',
            content=content,
            timestamp=time.time()
        )
        
        session.messages.append(message)
        session.last_activity = time.time()
        
        if metadata:
            session.metadata.update(metadata)
            
        self._manage_context_size()
        
        logger.debug(f"Added user message: '{content[:50]}...'")
        
    def add_ai_message(self, content: str, metadata: Dict[str, Any] = None):
        """Add AI response to conversation context"""
        session = self.get_current_session()
        
        message = Message(
            role='assistant',
            content=content,
            timestamp=time.time()
        )
        
        session.messages.append(message)
        session.last_activity = time.time()
        
        if metadata:
            session.metadata.update(metadata)
            
        self._manage_context_size()
        
        logger.debug(f"Added AI message: '{content[:50]}...'")
        
    def get_messages(self) -> List[Message]:
        """Get conversation messages for AI processing"""
        session = self.get_current_session()
        
        if not session:
            return []
            
        # Include summary if available
        messages = []
        
        if self.current_summary and len(session.messages) > self.summary_threshold:
            # Add summary as context
            summary_message = Message(
                role='assistant',
                content=f"[Previous conversation summary: {self.current_summary}]",
                timestamp=time.time() - 3600  # Mark as older
            )
            messages.append(summary_message)
            
            # Keep only recent messages
            messages.extend(session.messages[-10:])
        else:
            messages = session.messages
            
        return messages
        
    def _manage_context_size(self):
        """Manage context size and create summaries when needed"""
        session = self.get_current_session()
        
        if not session or len(session.messages) <= self.max_messages:
            return
            
        # Need to summarize old messages
        if len(session.messages) > self.summary_threshold:
            self._create_context_summary()
            
        # Keep only recent messages
        if len(session.messages) > self.max_messages:
            removed_count = len(session.messages) - self.max_messages
            session.messages = session.messages[-self.max_messages:]
            logger.debug(f"Trimmed {removed_count} old messages from context")
            
    def _create_context_summary(self):
        """Create summary of older conversation context"""
        session = self.get_current_session()
        
        if not session or len(session.messages) <= self.summary_threshold:
            return
            
        # Get messages to summarize (older half)
        split_point = len(session.messages) // 2
        messages_to_summarize = session.messages[:split_point]
        
        # Create simple summary
        user_topics = []
        ai_responses = []
        
        for msg in messages_to_summarize:
            if msg.role == 'user':
                user_topics.append(msg.content[:100])
            elif msg.role == 'assistant':
                ai_responses.append(msg.content[:100])
                
        # Generate summary
        if user_topics:
            topics_summary = f"User discussed: {', '.join(user_topics[:3])}"
            if len(user_topics) > 3:
                topics_summary += f" and {len(user_topics) - 3} other topics"
                
            self.current_summary = topics_summary
            
        # Remove summarized messages
        session.messages = session.messages[split_point:]
        
        logger.debug(f"Created context summary: '{self.current_summary[:100]}...'")
        
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        session = self.get_current_session()
        
        if not session:
            return {}
            
        user_messages = [m for m in session.messages if m.role == 'user']
        ai_messages = [m for m in session.messages if m.role == 'assistant']
        
        duration = time.time() - session.start_time
        
        return {
            'session_id': session.session_id,
            'duration_seconds': duration,
            'total_messages': len(session.messages),
            'user_messages': len(user_messages),
            'ai_messages': len(ai_messages),
            'last_activity': session.last_activity,
            'has_summary': bool(self.current_summary)
        }
        
    def clear_context(self):
        """Clear conversation context"""
        if self.current_session:
            session_id = self.current_session.session_id
            self.current_session = None
            self.current_summary = ""
            logger.info(f"Cleared context for session: {session_id}")
            
    def is_duplicate_user_message(self, content: str) -> bool:
        """Check if the user message is a duplicate of the last user message"""
        session = self.get_current_session()
        
        if not session or not session.messages:
            return False
            
        # Look for the last user message
        for message in reversed(session.messages):
            if message.role == 'user':
                # Compare content with some tolerance for slight variations
                return message.content.strip().lower() == content.strip().lower()
                
        return False 