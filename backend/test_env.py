import os
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Print all environment variables to verify loading
print("=== Environment Variables ===")
for key in ["AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "GOOGLE_API_KEY", "ENABLE_MOCK_SERVICES"]:
    value = os.getenv(key)
    if value:
        # Show first few characters for security
        display_value = value[:5] + "..." if key.endswith("KEY") else value
        print(f"{key}: {display_value}")
    else:
        print(f"{key}: NOT SET")

# Test config loading
print("\n=== Config Settings ===")
try:
    from backend.app.config import settings
    print(f"Azure Speech Key: {settings.azure_speech_key[:5]}...")
    print(f"Azure Speech Region: {settings.azure_speech_region}")
    print(f"Google API Key: {settings.google_api_key[:5]}...")
    print(f"Mock Services: {os.getenv('ENABLE_MOCK_SERVICES')}")
except Exception as e:
    print(f"Error loading config: {e}")

# Run test if requested
if len(sys.argv) > 1 and sys.argv[1] == "--test-stt":
    print("\n=== Testing STT Service ===")
    try:
        from backend.services.stt_service import STTService
        import asyncio
        
        async def test_stt():
            stt = STTService()
            await stt.initialize()
            print(f"STT Service available: {stt.is_available}")
            print(f"STT Service using mock: {stt.use_mock}")
            
            if stt.is_available:
                session_id = await stt.start_continuous_recognition()
                print(f"Started session: {session_id}")
                await stt.stop_continuous_recognition()
            
            await stt.cleanup()
        
        asyncio.run(test_stt())
    except Exception as e:
        print(f"Error testing STT: {e}")

# Test LLM if requested
elif len(sys.argv) > 1 and sys.argv[1] == "--test-llm":
    print("\n=== Testing LLM Service ===")
    try:
        from backend.services.llm_service import LLMService
        import asyncio
        
        async def test_llm():
            llm = LLMService()
            await llm.initialize()
            print(f"LLM Service available: {llm.is_available}")
            
            if llm.is_available:
                response = await llm.generate_response("Hello, how are you today?")
                print(f"LLM Response: {response}")
            
            await llm.cleanup()
        
        asyncio.run(test_llm())
    except Exception as e:
        print(f"Error testing LLM: {e}") 