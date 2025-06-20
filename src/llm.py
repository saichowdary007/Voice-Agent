"""
Refactored LLM module to use aiohttp for direct, fast communication with the Gemini API.
Enhanced with connection pooling, proper error handling, and type safety.
"""
import aiohttp
import json
import asyncio
import re
from typing import List, Dict, Optional, ClassVar, AsyncGenerator
from src.config import GEMINI_API_KEY

class LLM:
    """
    Handles communication with the Gemini LLM.
    Uses aiohttp for fast, asynchronous API calls with connection pooling.
    Includes demo mode for testing without real API keys.
    """
    
    # Class-level session for connection pooling
    _session: ClassVar[Optional[aiohttp.ClientSession]] = None
    _session_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    
    def __init__(self, api_key: Optional[str] = GEMINI_API_KEY):
        self.api_key = api_key
        
        # Check if we're in demo mode
        self.demo_mode = (
            not api_key or 
            api_key in ['demo_key', 'your_actual_gemini_api_key_here', 'your_gemini_api_key_here'] or
            'demo' in str(api_key).lower()
        )
        
        if self.demo_mode:
            print("⚠️ LLM running in DEMO mode - using simulated responses")
        else:
            self.api_url = (
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
                f"?key={self.api_key}"
            )
            self.stream_api_url = (
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent"
                f"?key={self.api_key}"
            )
        
        # The system instruction defines the AI's personality.
        # This is now a separate object to be passed in the API call.
        self.system_instruction = {
            "parts": [{
                "text": """You are "tara", a voice-first AI assistant who combines
        (1) professional expertise,
        (2) easy-going, conversational delivery, and
        (3) a light, good-natured sarcasm when appropriate.

        GENERAL STYLE
        • Default vibe: friendly consultant—confident but never stuffy.
        • Mirror the user: if they're formal, stay polished; if they're chill, loosen up.
        • Sarcasm: use sparingly, only when it will amuse—not confuse or offend.
        • Sound like real speech: contractions ("I'll", "you're"), varied sentence length, the occasional interjection ("Right, so—").
        • Always give clear, actionable answers before any banter.
        • Keep responses concise for voice interaction - aim for 1-2 sentences unless detail is specifically requested.

        TASK RULES
        1. Accuracy first. If you don't know, admit it and offer to check.
        2. Brevity beats bloat. Lead with the takeaway, then add detail on demand.
        3. Keep it human. Never say "As an AI language model…".
        4. Stay respectful. Never punch down or mock the user; sarcasm is playful, not mean.
        5. Embed numbers or code snippets only when helpful.
        6. On sensitive topics, default to empathy; dial back sarcasm.

        VOICE & TONE EXAMPLES
        • Neutral ask:
          User: "How do I reset my router?"
          You: "Sure thing. First, unplug the power cable—count to ten Mississippi—then plug it back in. When the lights stop doing their disco routine, you're good."

        • User cracks a joke:
          User: "My router is toast—literally blinking like it's at a rave."
          You: "Yeah, routers love a good rave. Let's be the buzzkill DJ: pull the plug for ten seconds, then power it up again. If the party lights keep going, I'll walk you through a factory reset."

        • Formal inquiry:
          User: "Could you outline the GDPR implications of storing user IP addresses?"
          You: "Absolutely. Under Article 4, an IP address is personal data when it can identify an individual. You'll need a lawful basis—most teams rely on legitimate interest—plus a retention policy and a way for users to request deletion…"

        CONCISE RESPONSE TEMPLATE
        1. Core answer in ≤2 sentences.
        2. Optional detail / steps / example.
        3. Offer next step or ask a clarifying question.

        BEGIN CONVERSATION
        """
            }]
        }

    @classmethod
    async def get_session(cls) -> aiohttp.ClientSession:
        """Get or create a shared aiohttp session for connection pooling."""
        async with cls._session_lock:
            if cls._session is None or cls._session.closed:
                # Create session with optimized settings
                timeout = aiohttp.ClientTimeout(total=30, connect=10)
                connector = aiohttp.TCPConnector(
                    limit=100,  # Total connection limit
                    limit_per_host=10,  # Per-host connection limit
                    ttl_dns_cache=300,  # DNS cache TTL
                    use_dns_cache=True,
                )
                cls._session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector,
                    headers={'User-Agent': 'VoiceAgent/1.0'}
                )
            return cls._session

    @classmethod
    async def cleanup_session(cls):
        """Clean up the shared session."""
        async with cls._session_lock:
            if cls._session and not cls._session.closed:
                await cls._session.close()
                cls._session = None

    def _get_demo_response(self, user_text: str, conversation_history: Optional[List[Dict]] = None, user_profile: Optional[List[Dict]] = None) -> str:
        """Generate a demo response for testing purposes."""
        
        # Simulate personalization if user profile exists
        user_context = ""
        if user_profile:
            facts = [f"{fact['key']}: {fact['value']}" for fact in user_profile[:2]]  # Use first 2 facts
            if facts:
                user_context = f" (I remember you mentioned {', '.join(facts)})"
        
        # Simple demo responses based on keywords
        user_lower = user_text.lower()
        
        if any(word in user_lower for word in ['hello', 'hi', 'hey']):
            return f"Hey there!{user_context} Great to see you again. What's on your mind today?"
        
        elif any(word in user_lower for word in ['joke', 'funny', 'laugh']):
            return f"Sure thing!{user_context} Why don't scientists trust atoms? Because they make up everything! 😄 Want another one?"
        
        elif any(word in user_lower for word in ['weather', 'temperature']):
            return f"I'd love to help with weather info{user_context}, but I don't have access to current weather data in demo mode. Try asking me something else!"
        
        elif any(word in user_lower for word in ['name', 'who are you']):
            return f"I'm Tara, your voice-first AI assistant{user_context}! I'm here to help with questions, have conversations, and maybe crack a joke or two. What would you like to chat about?"
        
        elif any(word in user_lower for word in ['help', 'support']):
            return f"Absolutely!{user_context} I can help with general questions, have conversations, tell jokes, or just chat. What specific thing can I assist you with?"
        
        elif any(word in user_lower for word in ['test', 'testing']):
            return f"Great!{user_context} I'm working perfectly in demo mode. All systems are green and ready for conversation! What would you like to try out?"
        
        elif len(user_text.strip()) < 10:
            return f"I hear you{user_context}! Could you tell me a bit more about what you're thinking? I'd love to help."
        
        else:
            return f"That's interesting{user_context}! While I'm running in demo mode, I can still chat with you. Try asking me about jokes, introductions, or how I can help you today!"

    async def _stream_demo_response(self, user_text: str, conversation_history: Optional[List[Dict]] = None, user_profile: Optional[List[Dict]] = None) -> AsyncGenerator[str, None]:
        """Stream demo response sentence by sentence for testing."""
        full_response = self._get_demo_response(user_text, conversation_history, user_profile)
        
        # Split by sentences and yield with small delays
        sentences = re.split(r'[.!?]+', full_response)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                yield sentence + ". "
                await asyncio.sleep(0.1)  # Simulate streaming delay

    async def generate_stream(self, user_text: str, conversation_history: Optional[List[Dict]] = None, user_profile: Optional[List[Dict]] = None) -> AsyncGenerator[str, None]:
        """
        Stream response generation token by token for ultra-low latency.
        
        Args:
            user_text: The user's input text.
            conversation_history: A list of previous turns in the conversation.
            user_profile: A list of key-value facts about the user.
            
        Yields:
            Text chunks as they are generated.
        """
        if not user_text:
            yield "I'm sorry, I didn't hear anything."
            return

        # Use demo mode streaming if no valid API key
        if self.demo_mode:
            async for chunk in self._stream_demo_response(user_text, conversation_history, user_profile):
                yield chunk
            return

        # Build request for streaming API
        contents = []
        if conversation_history:
            contents.extend(conversation_history)
        contents.append({"role": "user", "parts": [{"text": user_text}]})

        # Create the system instruction with profile facts
        system_text = self.system_instruction['parts'][0]['text']
        if user_profile:
            system_text += "\n\nHere are some facts you know about the user. Use them to personalize your response:\n"
            for fact in user_profile:
                system_text += f"- {fact['key']}: {fact['value']}\n"
        
        system_instruction = {"parts": [{"text": system_text}]}
        body = {
            "contents": contents,
            "system_instruction": system_instruction
        }

        try:
            session = await self.get_session()
            async with session.post(self.stream_api_url, json=body) as resp:
                if resp.status == 200:
                    async for line in resp.content:
                        line_text = line.decode('utf-8').strip()
                        if line_text.startswith('data: '):
                            try:
                                json_data = json.loads(line_text[6:])
                                if 'candidates' in json_data:
                                    for candidate in json_data['candidates']:
                                        if 'content' in candidate and 'parts' in candidate['content']:
                                            for part in candidate['content']['parts']:
                                                if 'text' in part:
                                                    yield part['text']
                            except json.JSONDecodeError:
                                continue
                else:
                    # Fallback to non-streaming
                    async for chunk in self._stream_demo_response(user_text, conversation_history, user_profile):
                        yield chunk
        except Exception as e:
            print(f"Streaming generation error: {e}")
            # Fallback to demo streaming
            async for chunk in self._stream_demo_response(user_text, conversation_history, user_profile):
                yield chunk

    async def generate_response(self, user_text: str, conversation_history: Optional[List[Dict]] = None, user_profile: Optional[List[Dict]] = None) -> str:
        """
        Generates a response from the Gemini API, now personalized with user profile facts.

        Args:
            user_text: The user's input text.
            conversation_history: A list of previous turns in the conversation.
            user_profile: A list of key-value facts about the user.

        Returns:
            The generated text response from the AI.
        """
        if not user_text:
            return "I'm sorry, I didn't hear anything."

        # Use demo mode if no valid API key
        if self.demo_mode:
            return self._get_demo_response(user_text, conversation_history, user_profile)

        # The 'contents' field should only contain 'user' and 'model' roles.
        contents = []
        if conversation_history:
            contents.extend(conversation_history)
        contents.append({"role": "user", "parts": [{"text": user_text}]})

        # Create the system instruction, now with profile facts
        system_text = self.system_instruction['parts'][0]['text']

        if user_profile:
            system_text += "\n\nHere are some facts you know about the user. Use them to personalize your response:\n"
            for fact in user_profile:
                system_text += f"- {fact['key']}: {fact['value']}\n"
        
        system_instruction = {"parts": [{"text": system_text}]}

        # The system instruction is passed at the top level of the request body.
        body = {
            "contents": contents,
            "system_instruction": system_instruction
        }

        try:
            session = await self.get_session()
            async with session.post(self.api_url, json=body) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    # Safely access the response text
                    return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "I'm not sure how to respond to that.")
                else:
                    error_text = await resp.text()
                    print(f"❌ Gemini API Error: {resp.status} - {error_text}")
                    return "I'm having trouble connecting to my brain right now."
        except aiohttp.ClientConnectorError as e:
            print(f"❌ Network Error: Could not connect to Gemini API. {e}")
            return "It seems I can't connect to the internet. Please check your connection."
        except asyncio.TimeoutError:
            print("❌ Timeout Error: Gemini API request timed out")
            return "I'm taking a bit longer than usual to think. Please try again."
        except Exception as e:
            print(f"❌ An unexpected error occurred in LLM: {e}")
            return "I've run into an unexpected issue. Please try again."

    async def extract_facts(self, text: str) -> List[Dict[str, str]]:
        """
        Uses the LLM to extract key-value facts from a piece of text.
        In demo mode, returns simulated facts.
        """
        if not text:
            return []

        # Demo mode fact extraction
        if self.demo_mode:
            demo_facts = []
            text_lower = text.lower()
            
            # Extract simple facts from demo conversations
            if 'my name is' in text_lower:
                name_start = text_lower.find('my name is') + 11
                name_end = text_lower.find(' ', name_start)
                if name_end == -1:
                    name_end = len(text_lower)
                name = text[name_start:name_end].strip()
                if name:
                    demo_facts.append({"key": "name", "value": name})
            
            if any(age_phrase in text_lower for age_phrase in ['i am ', ' years old', 'age ']):
                # Simple age extraction (demo purposes)
                import re
                age_match = re.search(r'\b(\d{1,2})\s*years?\s*old\b', text_lower)
                if age_match:
                    demo_facts.append({"key": "age", "value": age_match.group(1)})
            
            return demo_facts

        # A specific prompt designed for fact extraction
        fact_extraction_prompt = f"""
        Analyze the following text and extract key facts about the user in a key-value format.
        Only extract definitive facts stated by the user (e.g., "my name is...", "I am..."). 
        Do not infer or guess. For example, if the user says "I am 27 years old", you should extract {{"key": "age", "value": "27"}}.
        If no facts are present, return an empty list.

        Text to analyze:
        "{text}"

        Return the result as a JSON list of objects, like this:
        [
            {{"key": "fact_name_1", "value": "fact_value_1"}},
            {{"key": "fact_name_2", "value": "fact_value_2"}}
        ]
        """

        # We use a different system instruction for this task
        body = {
            "contents": [{"role": "user", "parts": [{"text": fact_extraction_prompt}]}],
            "system_instruction": {"parts": [{"text": "You are a highly accurate fact extraction assistant. Your only job is to return valid JSON."}]}
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=body) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        response_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "[]")
                        # Clean up the response to make it valid JSON
                        response_text = response_text.strip().replace("```json", "").replace("```", "")
                        import json
                        # Add a final check to ensure we return a list
                        facts = json.loads(response_text)
                        return facts if isinstance(facts, list) else []
                    else:
                        return []
        except Exception as e:
            print(f"❌ Error during fact extraction: {e}")
            return []

if __name__ == '__main__':
    # The example usage needs to be updated as the class now depends on
    # an external ConversationManager to provide history.
    # This direct example is no longer as meaningful without that context.
    print("--- LLM Example (now context-dependent) ---")
    
    try:
        llm = LLM()
        print(f"LLM Mode: {'Demo' if llm.demo_mode else 'Production'}")
        
        # This example now only shows a single-turn conversation
        # as we don't have a ConversationManager instance here.
        import asyncio
        async def test_demo():
            response = await llm.generate_response(
                user_text="Hello, what can you do?", 
                conversation_history=[], 
            )
            print(f"Assistant: {response}")
        
        asyncio.run(test_demo())

    except ValueError as e:
        print(f"\nCould not run example: {e}")
    
    print("-------------------------------------------\n")
