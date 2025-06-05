import os
import asyncio
import logging
import time
from typing import Dict, Any, AsyncGenerator, Optional, List, Union
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

class LLMService:
    """LLM service using Gemini 2.0 Flash for ultra-fast responses"""
    
    def __init__(self):
        self.model = None
        self.is_available = False
        # Use GOOGLE_API_KEY to match the config
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model_name = "gemini-2.0-flash-exp"
        
        # Conversation history
        self.conversation_history = []
        self.max_history_length = 10  # Keep last 10 exchanges
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # Initial retry delay in seconds
        
        logger.info("LLM Service initialized with model: %s", self.model_name)
        
    async def initialize(self):
        """Initialize the Gemini model"""
        try:
            # Check for valid API key (exclude common placeholders)
            invalid_keys = ["test_key_for_demo", "your-google-api-key-here", "", None]
            if not self.api_key or self.api_key in invalid_keys:
                logger.warning("No valid Gemini API key provided, using mock responses")
                logger.info("To enable Gemini LLM: Get API key from https://aistudio.google.com/apikey")
                logger.info("Then update GOOGLE_API_KEY in your .env file")
                self.is_available = False
                return
                
            logger.info("Initializing Gemini 2.0 Flash model...")
            
            # Configure Gemini
            def configure_genai():
                genai.configure(api_key=self.api_key)
            
            await asyncio.to_thread(configure_genai)
            
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
            def test_model():
                test_response = self.model.generate_content("Hello")
                return test_response
            
            test_response = await asyncio.to_thread(test_model)
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
            
            # Generate response with retry logic
            retry_count = 0
            while retry_count <= self.max_retries:
                try:
                    def generate():
                        return self.model.generate_content(prompt)
                    
                    response = await asyncio.wait_for(
                        asyncio.to_thread(generate),
                        timeout=60.0  # 60 seconds timeout
                    )
                    
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
                        retry_count += 1
                        if retry_count <= self.max_retries:
                            await asyncio.sleep(self.retry_delay * retry_count)  # Exponential backoff
                        
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning(f"LLM generation attempt {retry_count+1} failed: {e}")
                    retry_count += 1
                    if retry_count <= self.max_retries:
                        await asyncio.sleep(self.retry_delay * retry_count)  # Exponential backoff
                    else:
                        logger.error(f"All {self.max_retries} LLM generation attempts failed")
                        break
            
            # If we've exhausted all retries, return a mock response
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
            
            # Generate streaming response with retry logic
            retry_count = 0
            while retry_count <= self.max_retries:
                try:
                    # Generate streaming response
                    response_text = ""
                    
                    def generate_stream():
                        return self.model.generate_content(prompt, stream=True)
                    
                    response = await asyncio.wait_for(
                        asyncio.to_thread(generate_stream),
                        timeout=30.0  # 30 seconds timeout for stream initialization
                    )
                    
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
                        return
                    else:
                        logger.warning("No text generated in streaming response")
                        retry_count += 1
                        if retry_count <= self.max_retries:
                            await asyncio.sleep(self.retry_delay * retry_count)  # Exponential backoff
                        
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning(f"LLM streaming attempt {retry_count+1} failed: {e}")
                    retry_count += 1
                    if retry_count <= self.max_retries:
                        await asyncio.sleep(self.retry_delay * retry_count)  # Exponential backoff
                    else:
                        logger.error(f"All {self.max_retries} LLM streaming attempts failed")
                        break
            
            # If we've exhausted all retries, return a mock response
            mock_response = self._get_mock_response()
            words = mock_response.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.05)
                
        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            # Fallback to mock response
            mock_response = self._get_mock_response()
            words = mock_response.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.05)
    
    async def generate_response_stream(self, messages: list) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from a list of messages
        
        Args:
            messages: List of message objects with role and content
            
        Yields:
            Response tokens as they are generated
        """
        logger.info("LLM generate_response_stream called with %d messages", len(messages))
        
        if not self.is_available:
            logger.warning("LLM not available, using mock streaming response")
            # Mock streaming response
            mock_response = self._get_mock_response()
            words = mock_response.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.05)  # Simulate streaming delay
            return
        
        try:
            # Process incoming messages and update conversation history
            processed_messages = []
            
            for message in messages:
                # Check if message is a dict or has attributes
                if isinstance(message, dict):
                    role = message.get("role")
                    content = message.get("content")
                else:
                    # Assume it's an object with attributes
                    role = getattr(message, "role", None)
                    content = getattr(message, "content", None)
                
                if role and content:
                    processed_messages.append({"role": role, "content": content})
                    logger.debug("Processed message - Role: %s, Content: %s", role, content[:30] + "..." if len(content) > 30 else content)
            
            # Update conversation history with all valid messages
            self.conversation_history.extend(processed_messages)
            self._trim_history()
            
            # Extract the latest user message
            user_message = None
            for message in reversed(processed_messages):
                if message["role"] == "user" and message["content"]:
                    user_message = message["content"]
                    break
            
            if not user_message:
                logger.warning("No valid user message found in messages list")
                mock_response = self._get_mock_response()
                for word in mock_response.split():
                    yield word + " "
                    await asyncio.sleep(0.05)
                return
                
            # Generate response with retry logic
            retry_count = 0
            while retry_count <= self.max_retries:
                try:
                    # Generate response using Gemini model
                    prompt = self._build_prompt(user_message)
                    logger.debug("Built prompt for LLM: %s", prompt[:100] + "..." if len(prompt) > 100 else prompt)
                    
                    def generate_stream():
                        logger.info("Starting generate_content with stream=True")
                        return self.model.generate_content(prompt, stream=True)
                    
                    logger.info("Waiting for stream initialization (timeout: 30s)")
                    response = await asyncio.wait_for(
                        asyncio.to_thread(generate_stream),
                        timeout=30.0  # 30 seconds timeout for stream initialization
                    )
                    
                    # Stream the response
                    logger.info("Stream initialized successfully, beginning token streaming")
                    response_text = ""
                    token_count = 0
                    for chunk in response:
                        if chunk.text:
                            chunk_text = chunk.text
                            response_text += chunk_text
                            token_count += 1
                            if token_count % 10 == 0:
                                logger.debug("Streamed %d tokens so far", token_count)
                            yield chunk_text
                            
                    # Add complete response to history
                    if response_text:
                        self.conversation_history.append({"role": "assistant", "content": response_text.strip()})
                        self._trim_history()
                        logger.info(f"LLM streamed complete response: {token_count} tokens, '{response_text[:100]}...'")
                        return
                    else:
                        logger.warning("No text generated in streaming response")
                        retry_count += 1
                        if retry_count <= self.max_retries:
                            logger.info("Retrying stream generation (attempt %d/%d) after delay: %ds", 
                                      retry_count, self.max_retries, self.retry_delay * retry_count)
                            await asyncio.sleep(self.retry_delay * retry_count)  # Exponential backoff
                    
                except asyncio.TimeoutError as te:
                    logger.warning(f"LLM streaming timeout (attempt {retry_count+1}/{self.max_retries+1}): {te}")
                    retry_count += 1
                    if retry_count <= self.max_retries:
                        logger.info("Retrying after timeout (attempt %d/%d) with delay: %ds", 
                                  retry_count, self.max_retries, self.retry_delay * retry_count)
                        await asyncio.sleep(self.retry_delay * retry_count)  # Exponential backoff
                    else:
                        logger.error(f"All {self.max_retries} LLM streaming attempts timed out")
                        break
                except Exception as e:
                    logger.warning(f"LLM streaming attempt {retry_count+1} failed: {e}")
                    retry_count += 1
                    if retry_count <= self.max_retries:
                        logger.info("Retrying after error (attempt %d/%d) with delay: %ds", 
                                  retry_count, self.max_retries, self.retry_delay * retry_count)
                        await asyncio.sleep(self.retry_delay * retry_count)  # Exponential backoff
                    else:
                        logger.error(f"All {self.max_retries} LLM streaming attempts failed")
                        break
                
            # If we've exhausted all retries, return a mock response
            logger.warning("Falling back to mock response after failed retries")
            mock_response = self._get_mock_response()
            words = mock_response.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.05)
                
        except Exception as e:
            logger.error(f"LLM response stream failed with critical error: {e}")
            # Fallback to mock response
            logger.info("Using mock response after critical error")
            mock_response = self._get_mock_response()
            words = mock_response.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.05)
                
        finally:
            logger.info("LLM generate_response_stream completed")
    
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
            "max_history_length": self.max_history_length,
            "max_retries": self.max_retries
        }
    
    async def cleanup(self):
        """Clean up LLM resources"""
        try:
            self.model = None
            self.conversation_history = []
            self.is_available = False
            logger.info("LLM service cleaned up")
            
        except Exception as e:
            logger.error(f"LLM cleanup error: {e}") 