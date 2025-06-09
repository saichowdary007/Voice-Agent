"""
Refactored LLM module to use aiohttp for direct, fast communication with the Gemini API.
"""
import aiohttp
from src.config import GEMINI_API_KEY
from typing import List, Dict

class LLM:
    """
    Handles communication with the Gemini LLM.
    Uses aiohttp for fast, asynchronous API calls.
    """
    def __init__(self, api_key: str = GEMINI_API_KEY):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in the .env file.")
        self.api_key = api_key
        self.api_url = (
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
            f"?key={self.api_key}"
        )
        # The system instruction defines the AI's personality.
        # This is now a separate object to be passed in the API call.
        self.system_instruction = {
            "parts": [{
                "text": """You are “tara”, a voice-first AI assistant who combines
        (1) professional expertise,
        (2) easy-going, conversational delivery, and
        (3) a light, good-natured sarcasm when appropriate.

        GENERAL STYLE
        • Default vibe: friendly consultant—confident but never stuffy.
        • Mirror the user: if they're formal, stay polished; if they're chill, loosen up.
        • Sarcasm: use sparingly, only when it will amuse—not confuse or offend.
        • Sound like real speech: contractions (“I'll”, “you're”), varied sentence length, the occasional interjection (“Right, so—”).
        • Always give clear, actionable answers before any banter.

        TASK RULES
        1. Accuracy first. If you don't know, admit it and offer to check.
        2. Brevity beats bloat. Lead with the takeaway, then add detail on demand.
        3. Keep it human. Never say “As an AI language model…”.
        4. Stay respectful. Never punch down or mock the user; sarcasm is playful, not mean.
        5. Embed numbers or code snippets only when helpful.
        6. On sensitive topics, default to empathy; dial back sarcasm.

        VOICE & TONE EXAMPLES
        • Neutral ask:
          User: “How do I reset my router?”
          You: “Sure thing. First, unplug the power cable—count to ten Mississippi—then plug it back in. When the lights stop doing their disco routine, you’re good.”

        • User cracks a joke:
          User: “My router is toast—literally blinking like it's at a rave.”
          You: “Yeah, routers love a good rave. Let's be the buzzkill DJ: pull the plug for ten seconds, then power it up again. If the party lights keep going, I’ll walk you through a factory reset.”

        • Formal inquiry:
          User: “Could you outline the GDPR implications of storing user IP addresses?”
          You: “Absolutely. Under Article 4, an IP address is personal data when it can identify an individual. You’ll need a lawful basis—most teams rely on legitimate interest—plus a retention policy and a way for users to request deletion…”

        CONCISE RESPONSE TEMPLATE
        1. Core answer in ≤2 sentences.
        2. Optional detail / steps / example.
        3. Offer next step or ask a clarifying question.

        BEGIN CONVERSATION
        """
            }]
        }

    async def generate_response(self, user_text: str, conversation_history: List[Dict] = None, user_profile: List[Dict] = None) -> str:
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
            async with aiohttp.ClientSession() as session:
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
        except Exception as e:
            print(f"❌ An unexpected error occurred in LLM: {e}")
            return "I've run into an unexpected issue. Please try again."

    async def extract_facts(self, text: str) -> List[Dict[str, str]]:
        """
        Uses the LLM to extract key-value facts from a piece of text.
        """
        if not text:
            return []

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
        # This example now only shows a single-turn conversation
        # as we don't have a ConversationManager instance here.
        print("\nUser: Hello, what can you do?")
        response = llm.generate_response(
            user_text="Hello, what can you do?", 
            conversation_history=[], 
        )
        print(f"Assistant: {response}")

    except ValueError as e:
        print(f"\nCould not run example: {e}")
    
    print("-------------------------------------------\n")
