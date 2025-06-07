import os
import asyncio
import logging
import time
from typing import Dict, Any, AsyncGenerator, Optional, List, Union
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError

logger = logging.getLogger(__name__)

class LLMService:
    """LLM service using Gemini 2.0 Flash for ultra-fast responses"""
    
    def __init__(self):
        self.model = None
        self.is_available = False
        # Standardize API key usage
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY")
            if self.api_key:
                logger.warning("Using GEMINI_API_KEY instead of GOOGLE_API_KEY. Please update your environment variables for consistency.")
        
        self.model_name = "gemini-2.0-flash-exp"
        
        # Conversation history
        self.conversation_history = []
        self.max_history_length = 10  # Keep last 10 exchanges
        
        # Retry and timeout configuration - from config where available
        self.max_retries = 3
        self.initial_retry_delay = 1.0  # Initial retry delay in seconds
        self.max_retry_delay = 8.0  # Maximum retry delay in seconds
        self.backoff_factor = 2.0  # Exponential backoff multiplier
        self.retry_delay = 1.0  # Base delay for retries
        
        # Timeouts in seconds - use app settings if available
        try:
            from backend.app.config import settings
            self.connect_timeout = settings.ai_response_timeout / 2  # Half the response timeout for connection
            self.response_timeout = settings.ai_response_timeout
        except (ImportError, AttributeError):
            logger.warning("Could not load timeout settings from app config, using defaults")
            self.connect_timeout = 30.0  # 30 seconds for initial connection
            self.response_timeout = 60.0  # 60 seconds for full response
        
        logger.info("LLM Service initialized with model: %s", self.model_name)
        logger.info(f"Timeout settings - Connect: {self.connect_timeout}s, Response: {self.response_timeout}s")
        logger.info(f"Retry settings - Max retries: {self.max_retries}, Initial delay: {self.initial_retry_delay}s")
        
    async def initialize(self):
        """Initialize the Gemini model with robust error handling"""
        try:
            # Check for valid API key (exclude common placeholders)
            invalid_keys = ["test_key_for_demo", "your-google-api-key-here", "", None]
            
            # Check if we should use mock service
            if (not self.api_key or self.api_key in invalid_keys) and os.getenv("ENABLE_MOCK_SERVICES", "false").lower() == "true":
                logger.info("Using mock LLM responses (GOOGLE_API_KEY not set or invalid, but ENABLE_MOCK_SERVICES=true)")
                self.is_available = True
                return
                
            if not self.api_key or self.api_key in invalid_keys:
                logger.warning("No valid Gemini API key provided, using mock responses")
                logger.info("To enable Gemini LLM: Get API key from https://aistudio.google.com/apikey")
                logger.info("Then update GOOGLE_API_KEY in your .env file")
                self.is_available = False
                return
                
            logger.info("Initializing Gemini 2.0 Flash model...")
            
            # Configure Gemini with timeout
            try:
                # Use asyncio.wait_for to add timeout for API initialization
                await asyncio.wait_for(
                    self._configure_genai(),
                    timeout=self.connect_timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"Gemini API initialization timed out after {self.connect_timeout}s")
                self.is_available = False
                return
            
            # Create model with optimized settings for ultra-low latency
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.9,  # Slightly higher for more diverse but still fast responses
                top_k=20,   # Reduced for faster token selection
                max_output_tokens=512,  # Reduced for faster generation
                candidate_count=1,
                # Enable streaming optimizations if available
                stop_sequences=None,
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
                    system_instruction="You are a helpful voice assistant. Respond naturally and conversationally. Keep responses concise and engaging for voice interaction. Avoid using markdown formatting or special characters that don't work well in speech. Prioritize speed and clarity."
                )
            except TypeError:
                # Fallback for older versions without system_instruction
                logger.info("Using older google-generativeai version, creating model without system_instruction")
                self.model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
            
            # Test the model with timeout
            try:
                test_response = await asyncio.wait_for(
                    self._test_model(),
                    timeout=self.connect_timeout
                )
                
                if test_response and hasattr(test_response, 'text') and test_response.text:
                    self.is_available = True
                    logger.info("✅ Gemini 2.0 Flash initialized successfully")
                else:
                    logger.error("Gemini test failed - no response text")
                    self.is_available = False
            except asyncio.TimeoutError:
                logger.error(f"Gemini test request timed out after {self.connect_timeout}s")
                self.is_available = False
                
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.is_available = False
    
    async def _configure_genai(self):
        """Configure Gemini API with error handling"""
        def configure_genai():
            genai.configure(api_key=self.api_key)
        
        await asyncio.to_thread(configure_genai)
    
    async def _test_model(self):
        """Test the model with a simple prompt"""
        def test_model():
            return self.model.generate_content("Hello")
        
        return await asyncio.to_thread(test_model)
    
    # Enhanced retry decorator for generate_content with explicit backoff settings
    @retry(
        stop=stop_after_attempt(3),  # Maximum number of retry attempts
        wait=wait_exponential(multiplier=1, min=1, max=8),  # Exponential backoff: 1s, 2s, 4s, 8s
        retry=retry_if_exception_type((asyncio.TimeoutError, ConnectionError, IOError, Exception)),  # Expanded retry exceptions
        reraise=True  # Re-raise the last exception
    )
    async def _generate_with_retry(self, prompt, stream=False):
        """Generate content with enhanced retry logic and explicit timeout"""
        start_time = time.time()
        logger.info(f"Starting LLM generation request (timeout: {self.response_timeout}s)")
        
        def generate():
            return self.model.generate_content(prompt, stream=stream)
        
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(generate),
                timeout=self.response_timeout
            )
        except asyncio.TimeoutError as e:
            elapsed = time.time() - start_time
            logger.warning(f"LLM request timed out after {elapsed:.1f}s (timeout: {self.response_timeout}s)")
            raise  # Re-raise for retry handling
        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(f"LLM request failed after {elapsed:.1f}s: {str(e)}")
            raise  # Re-raise for retry handling
    
    async def generate_response(self, user_input: str) -> str:
        """
        Generate a complete response from user input with improved error handling
        
        Args:
            user_input: User's text input
            
        Returns:
            Generated response text
        """
        if not self.is_available:
            return self._get_mock_response()
        
        # Add user input to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Create prompt with conversation context
        prompt = self._build_prompt(user_input)
        
        try:
            # Use the enhanced retry-wrapped function
            start_time = time.time()
            logger.info(f"Generating LLM response for input: '{user_input[:50]}...'")
            
            try:
                response = await self._generate_with_retry(prompt)
                
                if response and hasattr(response, 'text'):
                    response_text = response.text.strip()
                    
                    # Add response to history
                    self.conversation_history.append({"role": "assistant", "content": response_text})
                    
                    # Trim history if too long
                    self._trim_history()
                    
                    elapsed = time.time() - start_time
                    logger.info(f"LLM generated response in {elapsed:.2f}s: '{response_text[:100]}...'")
                    return response_text
                else:
                    logger.error("No response text from Gemini")
                    return self._get_mock_response()
            except RetryError as re:
                logger.error(f"All retry attempts failed: {str(re)}")
                return self._get_mock_response()
                
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"LLM generation timed out after {elapsed:.2f}s (max: {self.response_timeout}s)")
            return self._get_mock_response()
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"LLM generation failed after {elapsed:.2f}s: {e}")
            return self._get_mock_response()
    
    async def generate_streaming(self, user_input: str) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from user input with improved resilience
        
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
        
        # Add user input to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Create prompt with conversation context
        prompt = self._build_prompt(user_input)
        
        # Track timings for detailed logging
        stream_start = time.time()
        first_token_received = False
        token_start_time = None
        
        # Outer try block for overall streaming operation
        try:
            # Attempt to get streaming response with retry and timeout for connection
            logger.info(f"Starting streaming request with timeout {self.response_timeout}s")
            
            try:
                response = await self._generate_with_retry(prompt, stream=True)
                
                response_text = ""
                stream_timeout = self.response_timeout + 30  # Extra time for streaming
                token_count = 0
                
                # Process each chunk with timeout protection
                for chunk in response:
                    # Track first token latency
                    if not first_token_received:
                        first_token_received = True
                        time_to_first_token = time.time() - stream_start
                        token_start_time = time.time()
                        logger.info(f"First token received after {time_to_first_token:.2f}s")
                    
                    # Check for stream timeout
                    current_time = time.time()
                    if current_time - stream_start > stream_timeout:
                        logger.warning(f"Stream processing exceeded timeout of {stream_timeout}s")
                        break
                    
                    # Check for token stall (more than 10 seconds without a token)
                    if first_token_received and (current_time - token_start_time > 10):
                        logger.warning(f"Token streaming stalled for >10s, breaking stream")
                        break
                    
                    if hasattr(chunk, 'text') and chunk.text:
                        chunk_text = chunk.text
                        response_text += chunk_text
                        token_count += 1
                        token_start_time = current_time  # Reset token timer
                        
                        # Log progress periodically
                        if token_count % 20 == 0:
                            elapsed = time.time() - stream_start
                            logger.debug(f"Streamed {token_count} tokens in {elapsed:.2f}s")
                            
                        yield chunk_text
                
                # Add complete response to history if we got something
                total_elapsed = time.time() - stream_start
                if response_text:
                    self.conversation_history.append({"role": "assistant", "content": response_text.strip()})
                    self._trim_history()
                    logger.info(f"LLM streamed response completed in {total_elapsed:.2f}s: {token_count} tokens, '{response_text[:100]}...'")
                else:
                    logger.warning(f"No text generated in streaming response after {total_elapsed:.2f}s")
                    # Yield mock response if nothing was generated
                    mock_response = self._get_mock_response()
                    words = mock_response.split()
                    for word in words:
                        yield word + " "
                        await asyncio.sleep(0.05)
            
            except RetryError as re:
                logger.error(f"All streaming retry attempts failed: {str(re)}")
                # Fall back to mock response
                mock_response = self._get_mock_response()
                words = mock_response.split()
                for word in words:
                    yield word + " "
                    await asyncio.sleep(0.05)
                
        except asyncio.TimeoutError:
            elapsed = time.time() - stream_start
            logger.error(f"LLM streaming timed out after {elapsed:.2f}s (timeout: {self.response_timeout}s)")
            # Fall back to mock response
            mock_response = self._get_mock_response()
            words = mock_response.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.05)
        except Exception as e:
            elapsed = time.time() - stream_start
            logger.error(f"LLM streaming failed after {elapsed:.2f}s: {e}")
            # Fall back to mock response
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
        
        stream_start = time.time()
        
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
                
            # Build prompt with full context
            prompt = self._build_prompt(user_message)
            logger.debug("Built prompt for LLM: %s", prompt[:100] + "..." if len(prompt) > 100 else prompt)
            
            # Attempt to generate with enhanced retry logic
            try:
                # Generate stream with explicit timeout handling
                logger.info(f"Starting stream generation with timeout {self.connect_timeout}s for initial connection")
                
                try:
                    def generate_stream():
                        return self.model.generate_content(prompt, stream=True)
                    
                    # Use a shorter timeout for the initial connection, longer for the streaming
                    response = await asyncio.wait_for(
                        asyncio.to_thread(generate_stream),
                        timeout=self.connect_timeout
                    )
                    
                    # Stream the response with more detailed metrics
                    logger.info("Stream initialized successfully, beginning token streaming")
                    response_text = ""
                    token_count = 0
                    first_token_time = None
                    token_start_time = time.time()
                    stream_timeout = self.response_timeout + 30  # Additional time for streaming
                    
                    for chunk in response:
                        current_time = time.time()
                        
                        # Record first token timing
                        if token_count == 0:
                            first_token_time = current_time
                            first_token_latency = first_token_time - stream_start
                            logger.info(f"First token received after {first_token_latency:.2f}s")
                        
                        # Check for overall stream timeout
                        if current_time - stream_start > stream_timeout:
                            logger.warning(f"Stream exceeded timeout of {stream_timeout}s, breaking")
                            break
                        
                        # Check for stalled token stream (10s without tokens)
                        if token_count > 0 and (current_time - token_start_time) > 10:
                            logger.warning(f"Token stream stalled for >10s, breaking stream")
                            break
                        
                        if hasattr(chunk, 'text') and chunk.text:
                            chunk_text = chunk.text
                            response_text += chunk_text
                            token_count += 1
                            token_start_time = current_time  # Reset stall timer
                            
                            # Log progress periodically
                            if token_count % 20 == 0:
                                elapsed = current_time - stream_start
                                tokens_per_second = token_count / elapsed if elapsed > 0 else 0
                                logger.debug(f"Streamed {token_count} tokens in {elapsed:.2f}s ({tokens_per_second:.1f} t/s)")
                            
                            yield chunk_text
                    
                    # Log final metrics
                    total_time = time.time() - stream_start
                    tokens_per_second = token_count / total_time if total_time > 0 else 0
                    
                    # Add complete response to history
                    if response_text:
                        self.conversation_history.append({"role": "assistant", "content": response_text.strip()})
                        self._trim_history()
                        logger.info(f"LLM stream completed in {total_time:.2f}s: {token_count} tokens at {tokens_per_second:.1f} t/s")
                    else:
                        logger.warning(f"No text generated in stream after {total_time:.2f}s")
                        # Fall back to mock
                        mock_response = self._get_mock_response()
                        for word in mock_response.split():
                            yield word + " "
                            await asyncio.sleep(0.05)
                
                except asyncio.TimeoutError:
                    elapsed = time.time() - stream_start
                    logger.error(f"Stream initialization timed out after {elapsed:.2f}s")
                    # Fall back to mock response
                    mock_response = self._get_mock_response()
                    for word in mock_response.split():
                        yield word + " "
                        await asyncio.sleep(0.05)
                        
                except Exception as e:
                    elapsed = time.time() - stream_start
                    logger.error(f"Stream generation failed after {elapsed:.2f}s: {str(e)}")
                    # Fall back to mock response
                    mock_response = self._get_mock_response()
                    for word in mock_response.split():
                        yield word + " "
                        await asyncio.sleep(0.05)
                        
            except RetryError as re:
                logger.error(f"All retry attempts for stream failed: {str(re)}")
                # Fall back to mock response
                mock_response = self._get_mock_response()
                words = mock_response.split()
                for word in words:
                    yield word + " "
                    await asyncio.sleep(0.05)
                
        except Exception as e:
            elapsed = time.time() - stream_start
            logger.error(f"LLM response stream failed with critical error after {elapsed:.2f}s: {e}")
            # Fallback to mock response
            logger.info("Using mock response after critical error")
            mock_response = self._get_mock_response()
            words = mock_response.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.05)
                
        finally:
            total_elapsed = time.time() - stream_start
            logger.info(f"LLM generate_response_stream completed in {total_elapsed:.2f}s")
    
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
            start_time = time.time()
            test_response = await self.generate_response("Hello, this is a test.")
            elapsed = time.time() - start_time
            logger.info(f"LLM connection test completed in {elapsed:.2f}s")
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
            "max_retries": self.max_retries,
            "connect_timeout": self.connect_timeout,
            "response_timeout": self.response_timeout
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
    
    async def process_text(self, text: str) -> Optional[str]:
        """
        Process text input and return AI response
        
        Args:
            text: Input text to process
            
        Returns:
            AI response text or None if failed
        """
        try:
            start_time = time.time()
            response = await self.generate_response(text)
            elapsed = time.time() - start_time
            logger.info(f"Text processing completed in {elapsed:.2f}s")
            return response if response else None
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return None 