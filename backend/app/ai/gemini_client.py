import os
import asyncio
from typing import List, Dict, Any, AsyncGenerator, Optional
import google.generativeai as genai
import structlog
from dataclasses import dataclass

logger = structlog.get_logger()


@dataclass
class Message:
    """Chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float


class GeminiClient:
    """Google Gemini 2.0 Flash client for voice interactions"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        self.demo_mode = False
        
        # Demo responses for fallback/testing
        self.demo_responses = [
            "Hello! I'm Nova, your voice assistant. How can I help you today?",
            "That's interesting! Tell me more about that.",
            "I understand what you're saying. How can I assist you further?",
            "Great question! Let me think about that for a moment.",
            "I appreciate you sharing that with me. What else would you like to know?",
            "That sounds fascinating. Can you tell me more details?",
            "I'm here to help with whatever you need. What's on your mind?",
            "Thank you for that information. How can I help you next?",
            "I find that quite intriguing. Would you like to explore that further?",
            "Excellent point! Is there anything specific you'd like to discuss?"
        ]
        self.demo_response_index = 0
        
        # If no API key is provided, enable demo mode
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            self.demo_mode = True
            logger.warning("No valid Gemini API key found. Running in DEMO MODE with mock responses.")
        else:
            logger.info("Gemini API key found, enabling full AI responses.")
            
        self.model_name = "gemini-2.0-flash-exp"
        self.model: Optional[genai.GenerativeModel] = None
        self.chat_session: Optional[Any] = None
        
        # Voice-optimized generation config
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            max_output_tokens=150,  # Keep responses concise for voice
            candidate_count=1,
            stop_sequences=None,
        )
        
        # Safety settings for voice interactions
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        # System prompt for voice interactions
        self.system_prompt = """You are Nova, a friendly and helpful voice assistant. 

Key guidelines for voice interactions:
- Keep responses concise and conversational (1-2 sentences max)
- Speak naturally as if having a real conversation
- Be warm, engaging, and personable
- Avoid long explanations unless specifically asked
- Use simple language that sounds good when spoken
- If asked to do something you can't do, politely explain briefly
- Stay focused on the conversation context
- Express emotions and personality in your voice

Remember: Users are speaking to you, so respond as you would in a natural conversation."""
        
    async def initialize(self):
        """Initialize Gemini client"""
        try:
            logger.info("Initializing Gemini client...")
            
            if self.demo_mode:
                logger.info("Running in demo mode - no API calls will be made")
                return
            
            # Configure API
            genai.configure(api_key=self.api_key)
            
            # Create model
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                system_instruction=self.system_prompt
            )
            
            # Test connection
            test_response = await asyncio.to_thread(
                self.model.generate_content,
                "Hello"
            )
            
            if not test_response or not test_response.text:
                raise RuntimeError("Failed to get test response from Gemini")
                
            logger.info(f"Gemini client initialized with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
            
    async def is_ready(self) -> bool:
        """Check if Gemini client is ready"""
        if self.demo_mode:
            return True
        return self.model is not None
        
    async def generate_response(self, messages: List[Message]) -> AsyncGenerator[str, None]:
        """Generate streaming response from conversation history"""
        try:
            if self.demo_mode:
                # Return demo response
                response_text = self.demo_responses[self.demo_response_index % len(self.demo_responses)]
                self.demo_response_index += 1
                
                # Simulate streaming by yielding word by word with small delay
                words = response_text.split()
                for word in words:
                    yield word + " "
                    await asyncio.sleep(0.05)  # Small delay to simulate streaming
                return
                
            if not self.model:
                raise RuntimeError("Gemini client not initialized")
                
            # Convert messages to Gemini format
            history = self._convert_messages_to_history(messages[:-1])  # All except last
            current_message = messages[-1].content if messages else ""
            
            # Start chat session with history
            if history:
                chat = self.model.start_chat(history=history)
            else:
                chat = self.model.start_chat()
                
            logger.debug(f"Generating response for: '{current_message[:50]}...'")
            
            # Generate streaming response
            response = await asyncio.to_thread(
                chat.send_message,
                current_message,
                stream=True
            )
            
            full_response = ""
            async for chunk in self._async_response_iterator(response):
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text
                    
            logger.debug(f"Generated response: '{full_response[:100]}...'")
            
        except Exception as e:
            logger.error(f"Gemini response generation error: {e}")
            # Yield error response
            yield "I'm sorry, I'm having trouble processing that right now. Could you try again?"
            
    async def _async_response_iterator(self, response):
        """Convert sync response iterator to async"""
        def sync_iterator():
            try:
                for chunk in response:
                    yield chunk
            except Exception as e:
                logger.error(f"Response iteration error: {e}")
                
        # Run in thread to avoid blocking
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()
        
        def producer():
            try:
                for chunk in sync_iterator():
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)
                
        # Start producer in thread
        await asyncio.to_thread(producer)
        
        # Consume from queue
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
            
    def _convert_messages_to_history(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert messages to Gemini chat history format"""
        history = []
        
        for msg in messages:
            # Map roles
            if msg.role == 'user':
                role = 'user'
            elif msg.role == 'assistant':
                role = 'model'
            else:
                continue  # Skip unknown roles
                
            history.append({
                'role': role,
                'parts': [msg.content]
            })
            
        return history
        
    async def generate_simple_response(self, prompt: str) -> str:
        """Generate simple non-streaming response"""
        try:
            if self.demo_mode:
                # Return demo response
                response_text = self.demo_responses[self.demo_response_index % len(self.demo_responses)]
                self.demo_response_index += 1
                return response_text
                
            if not self.model:
                raise RuntimeError("Gemini client not initialized")
                
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            return response.text if response and response.text else ""
            
        except Exception as e:
            logger.error(f"Simple response generation error: {e}")
            return "I'm sorry, I couldn't process that request."
            
    async def cleanup(self):
        """Clean up Gemini client resources"""
        try:
            self.model = None
            self.chat_session = None
            logger.info("Gemini client cleaned up")
            
        except Exception as e:
            logger.error(f"Gemini cleanup error: {e}")


class ConversationOptimizer:
    """Optimizes conversations for voice interactions"""
    
    @staticmethod
    def optimize_for_voice(text: str) -> str:
        """Optimize text response for voice synthesis"""
        if not text:
            return text
            
        # Remove markdown formatting
        import re
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Links
        
        # Replace symbols with words
        replacements = {
            '&': ' and ',
            '@': ' at ',
            '%': ' percent ',
            '#': ' number ',
            '$': ' dollars ',
            '+': ' plus ',
            '=': ' equals ',
            '<': ' less than ',
            '>': ' greater than ',
            '|': ' or ',
            '/': ' slash ',
            '\\': ' backslash ',
        }
        
        for symbol, word in replacements.items():
            text = text.replace(symbol, word)
            
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Ensure proper sentence ending
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
            
        return text
        
    @staticmethod
    def split_long_response(text: str, max_length: int = 200) -> List[str]:
        """Split long responses into voice-friendly chunks"""
        if len(text) <= max_length:
            return [text]
            
        chunks = []
        sentences = text.split('.')
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence += '.'
            
            if len(current_chunk + sentence) <= max_length:
                current_chunk += sentence + ' '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ' '
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks if chunks else [text]


class ContextAwareGemini(GeminiClient):
    """Context-aware Gemini client with conversation memory"""
    
    def __init__(self):
        super().__init__()
        self.conversation_context = []
        self.max_context_length = 10  # Keep last 10 exchanges
        
    async def generate_contextual_response(self, user_input: str, context: Dict[str, Any] = None) -> AsyncGenerator[str, None]:
        """Generate response with conversation context"""
        try:
            # Add user input to context
            import time
            user_message = Message(
                role='user',
                content=user_input,
                timestamp=time.time()
            )
            
            # Build context-aware messages
            messages = self.conversation_context[-self.max_context_length:] + [user_message]
            
            # Generate response
            full_response = ""
            async for token in self.generate_response(messages):
                full_response += token
                yield token
                
            # Add assistant response to context
            assistant_message = Message(
                role='assistant',
                content=full_response,
                timestamp=time.time()
            )
            
            self.conversation_context.extend([user_message, assistant_message])
            
            # Trim context if too long
            if len(self.conversation_context) > self.max_context_length * 2:
                self.conversation_context = self.conversation_context[-self.max_context_length * 2:]
                
        except Exception as e:
            logger.error(f"Contextual response error: {e}")
            yield "I'm having trouble with that. Could you rephrase your question?"
            
    def clear_context(self):
        """Clear conversation context"""
        self.conversation_context = []
        logger.debug("Conversation context cleared") 