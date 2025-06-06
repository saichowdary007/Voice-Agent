# Backend Issues and Fix Plan

Based on our testing, we've identified several issues with the backend services. This document outlines the problems and provides a plan to fix them.

## Issues Detected

### 1. Voice Activity Detection (VAD)

- **Status**: ⚠️ Partially Working
- **Issue**: VAD model loads successfully but doesn't detect speech in test audio.
- **Root Cause**: Either the test audio doesn't contain speech patterns that the VAD model recognizes, or there's an issue with the audio processing or VAD threshold settings.
- **Fix Plan**:
  - Adjust VAD threshold to be more sensitive (lower value)
  - Use a more clearly spoken test audio file
  - Check frame processing in test_services.py

### 2. Speech-to-Text (STT)

- **Status**: ❌ Not Working
- **Issue**: STT service is not available.
- **Root Cause**: Azure Speech API key is not set or is invalid.
- **Fix Plan**:
  - Set a valid Azure Speech API key in `.env` file
  - If no key is available, implement a mock STT service for testing
  - Add better error handling for missing API keys

### 3. Language Model (LLM)

- **Status**: ❌ Not Working
- **Issue**: LLM service is not available.
- **Root Cause**: Google API key for Gemini is not set or is invalid.
- **Fix Plan**:
  - Set a valid Google API key in `.env` file
  - Ensure Google API key has access to Gemini models
  - Add better fallback mechanism for testing without API keys

### 4. Text-to-Speech (TTS)

- **Status**: ✅ Working
- **Issue**: None, TTS is working properly.
- **Fix Plan**: None needed.

## Comprehensive Fix Plan

### 1. Update Environment Configuration

Create a proper `.env` file with the following:

```
# Core settings
DEBUG=false
LOG_LEVEL=INFO

# Azure Speech-to-Text
AZURE_SPEECH_KEY=your-azure-speech-key
AZURE_SPEECH_REGION=eastus
AZURE_SPEECH_LANGUAGE=en-US

# Google Gemini LLM
GOOGLE_API_KEY=your-google-api-key
LLM_MODEL=gemini-1.0-pro  # Fall back to an older model if flash not available

# Optional: Use mock services for testing
ENABLE_MOCK_SERVICES=true  # Set to true to use mock services when APIs aren't available
```

### 2. Enhance Service Fallbacks

Modify each service to provide a proper fallback when APIs are not available:

#### STT Service:
```python
def initialize(self):
    # Check if ENABLE_MOCK_SERVICES is set
    if not self.speech_key and os.getenv("ENABLE_MOCK_SERVICES", "false").lower() == "true":
        logger.info("Using mock STT service for testing")
        self.use_mock = True
        self.is_available = True
        return
    
    # Rest of initialization...
```

#### LLM Service:
```python
def initialize(self):
    # Check if ENABLE_MOCK_SERVICES is set
    if not self.api_key and os.getenv("ENABLE_MOCK_SERVICES", "false").lower() == "true":
        logger.info("Using mock LLM service for testing")
        self.use_mock = True
        self.is_available = True
        return
    
    # Rest of initialization...
```

### 3. Adjust VAD Settings

Modify VAD service to use more sensitive defaults:

```python
def __init__(self, 
             speech_threshold: float = 0.5,  # Lowered from 0.75 for higher sensitivity
             silence_threshold: float = 0.2,  # Lowered from 0.3
             min_speech_duration: float = 0.25,  # Shorter to detect brief speech
             min_silence_duration: float = 0.5,  # Kept the same
             sample_rate: int = 16000):
    # Rest of the initialization...
```

### 4. Implement Better Error Handling

Add more robust error handling in the WebSocket manager:

```python
async def handle_connection(self, websocket: WebSocket):
    # Add more specific error handling
    try:
        # Check if services are available
        if not self.voice_service.is_available:
            # Check which specific services are unavailable
            health = await self.voice_service.get_health_status()
            unavailable_services = [s for s, status in health.items() 
                                   if status.get("available") is False]
            
            error_msg = f"Voice services unavailable: {', '.join(unavailable_services)}"
            await websocket.close(code=1013, reason=error_msg)
            return
        
        # Rest of the connection handling...
```

### 5. Enhance Testing Scripts

Update test scripts to provide more detailed diagnostics:

- Add more verbose output about which specific part of a service is failing
- Implement tests with real audio samples from the test_audio directory
- Add test coverage for edge cases and error conditions

## Next Steps

1. Implement the fixes outlined above
2. Run tests for each service individually
3. Test the full processing pipeline
4. Verify WebSocket connection and real-time audio processing
5. Test with the frontend application

Once these fixes are implemented, the backend should be fully functional and ready for integration with the frontend. 