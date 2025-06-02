import os
import asyncio
import logging
import time
from typing import Dict, Any, AsyncGenerator, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

class LLMService:
    """LLM service using Gemini 2.0 Flash for ultra-fast responses"""
    
    def __init__(self):
        self.model = None
        self.is_available = False
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = "gemini-2.0-flash"
        
        # Conversation history
        self.conversation_history = []
        self.max_history_length = 10  # Keep last 10 exchanges
        
    async def initialize(self):
        """Initialize the Gemini model"""
        try:
            if not self.api_key or self.api_key == "test_key_for_demo":
                logger.warning("No valid Gemini API key provided, using mock responses")
                self.is_available = False
                return
                
            logger.info("Initializing Gemini 2.0 Flash model...")
            
            # Configure Gemini
            genai.configure(api_key=self.api_key)
            
            # Create model with optimized settings for speed
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=40,
                max_output_tokens=1024,
                candidate_count=1,
            )
            
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            # Try to create model with system_instruction (newer versions)
            try:
                self.model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    system_instruction="You are a helpful voice assistant. Respond naturally and conversationally. Keep responses concise and engaging for voice interaction. Avoid using markdown formatting or special characters that don't work well in speech."
                )
            except TypeError:
                # Fallback for older versions without system_instruction
                logger.info("Using older google-generativeai version, creating model without system_instruction")
                self.model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
            
            # Test the model
            test_response = self.model.generate_content("Hello")
            if test_response and test_response.text:
                self.is_available = True
                logger.info("✅ Gemini 2.0 Flash initialized successfully")
            else:
                logger.error("Gemini test failed - no response text")
                self.is_available = False
                
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.is_available = False
    
    async def generate_response(self, user_input: str) -> str:
        """
        Generate a complete response from user input
        
        Args:
            user_input: User's text input
            
        Returns:
            Generated response text
        """
        if not self.is_available:
            return self._get_mock_response()
        
        try:
            # Add user input to history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Create prompt with conversation context
            prompt = self._build_prompt(user_input)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                response_text = response.text.strip()
                
                # Add response to history
                self.conversation_history.append({"role": "assistant", "content": response_text})
                
                # Trim history if too long
                self._trim_history()
                
                logger.info(f"LLM generated response: '{response_text[:100]}...'")
                return response_text
            else:
                logger.error("No response text from Gemini")
                return self._get_mock_response()
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._get_mock_response()
    
    async def generate_streaming(self, user_input: str) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from user input
        
        Args:
            user_input: User's text input
            
        Yields:
            Response tokens as they are generated
        """
        if not self.is_available:
            # Mock streaming response
            mock_response = self._get_mock_response()
            words = mock_response.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.05)  # Simulate streaming delay
            return
        
        try:
            # Add user input to history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Create prompt with conversation context
            prompt = self._build_prompt(user_input)
            
            # Generate streaming response
            response_text = ""
            response = self.model.generate_content(prompt, stream=True)
            
            for chunk in response:
                if chunk.text:
                    chunk_text = chunk.text
                    response_text += chunk_text
                    yield chunk_text
                    
            # Add complete response to history
            if response_text:
                self.conversation_history.append({"role": "assistant", "content": response_text.strip()})
                self._trim_history()
                logger.info(f"LLM streamed response: '{response_text[:100]}...'")
                
        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            # Fallback to mock response
            mock_response = self._get_mock_response()
            words = mock_response.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.05)
    
    def _build_prompt(self, user_input: str) -> str:
        """Build prompt with conversation context"""
        if not self.conversation_history:
            return user_input
        
        # Build context from recent history
        context_parts = []
        for exchange in self.conversation_history[-6:]:  # Last 3 exchanges
            if exchange["role"] == "user":
                context_parts.append(f"User: {exchange['content']}")
            else:
                context_parts.append(f"Assistant: {exchange['content']}")
        
        context = "\n".join(context_parts)
        prompt = f"Conversation context:\n{context}\n\nUser: {user_input}\nAssistant:"
        
        return prompt
    
    def _trim_history(self):
        """Trim conversation history to maintain performance"""
        if len(self.conversation_history) > self.max_history_length * 2:
            # Keep last max_history_length exchanges (user + assistant pairs)
            self.conversation_history = self.conversation_history[-self.max_history_length * 2:]
    
    def _get_mock_response(self) -> str:
        """Get a mock response for demo/fallback"""
        mock_responses = [
            "I'm here to help! What would you like to know?",
            "That's interesting. Tell me more about that.",
            "I understand. How can I assist you further?",
            "Great question! Let me think about that.",
            "I appreciate you sharing that with me.",
            "That sounds fascinating. What else would you like to explore?",
            "I'm listening. Please continue.",
            "Thank you for that information. What's next?",
            "I find that quite intriguing. Can you elaborate?",
            "Excellent point. How can I help you with that?"
        ]
        
        import random
        return random.choice(mock_responses)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation"""
        return {
            "total_exchanges": len(self.conversation_history) // 2,
            "last_user_input": self.conversation_history[-2]["content"] if len(self.conversation_history) >= 2 else None,
            "last_response": self.conversation_history[-1]["content"] if self.conversation_history else None,
            "history_length": len(self.conversation_history)
        }
    
    async def test_connection(self) -> bool:
        """Test LLM service connection"""
        if not self.is_available:
            return False
        
        try:
            test_response = await self.generate_response("Hello, this is a test.")
            return len(test_response) > 0
        except Exception as e:
            logger.error(f"LLM connection test failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get LLM service status"""
        return {
            "available": self.is_available,
            "service": f"Gemini {self.model_name}" if self.is_available else "Mock LLM",
            "api_key_configured": bool(self.api_key and self.api_key != "test_key_for_demo"),
            "conversation_exchanges": len(self.conversation_history) // 2,
            "max_history_length": self.max_history_length
        } 