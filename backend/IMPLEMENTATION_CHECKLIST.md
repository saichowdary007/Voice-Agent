# Voice Agent Implementation Checklist

This document provides a structured approach to verify that all components of the Voice Agent backend are working correctly. Use this checklist to systematically test the system and identify any issues.

## Prerequisites

Before testing, ensure you have the following set up:

- Python 3.8+ with pip installed
- Required Python packages installed (`pip install -r requirements.txt`)
- Properly configured `.env` file with API keys
- FFmpeg installed for audio processing
- Test audio files available in `test_audio/` directory

## Environment Configuration

- [ ] Check that all required environment variables are set in `.env`
- [ ] Verify Azure Speech API key is valid
- [ ] Verify Google API key is valid
- [ ] Confirm TTS service dependencies are installed

## Component Testing Checklist

### 1. Voice Activity Detection (VAD)

#### Test with:
```bash
python test_services.py --test vad
```

#### Expected Result:
- VAD model loads successfully
- VAD correctly identifies speech in test audio
- Processing time is < 20ms per frame

#### If Failing:
- Check if PyTorch is installed
- Verify if Silero VAD model can be downloaded
- Check CPU/GPU compatibility
- Ensure audio format matches expected input (16kHz mono)

### 2. Speech-to-Text (STT) 

#### Test with:
```bash
python test_services.py --test stt
```

#### Expected Result:
- Azure Speech service connects successfully
- Continuous recognition starts and stops without errors
- Test audio transcription is accurate

#### If Failing:
- Verify Azure Speech key and region in `.env`
- Check network connectivity to Azure services
- Verify audio format is compatible with Azure Speech (16kHz, 16-bit PCM)
- Check Azure service quotas and limits

### 3. Language Model (LLM)

#### Test with:
```bash
python test_services.py --test llm
```

#### Expected Result:
- Gemini API connects successfully
- Test prompt receives a coherent response
- Response generation time is < 1000ms

#### If Failing:
- Verify Google API key in `.env`
- Check network connectivity to Google services
- Verify the requested model is available in your region
- Check usage quotas for the API

### 4. Text-to-Speech (TTS)

#### Test with:
```bash
python test_services.py --test tts
```

#### Expected Result:
- TTS engine initializes successfully
- Test text is converted to speech
- Audio output is clear and correctly formatted

#### If Failing:
- Check if piper TTS is installed and available in PATH
- Verify voice models are downloaded
- Check for disk space issues
- Test if manual TTS generation works outside the app

### 5. Complete Voice Pipeline

#### Test with:
```bash
python test_websocket.py --audio tone_440hz.webm
```

#### Expected Result:
- Audio is processed through the entire pipeline
- VAD identifies speech segments
- STT transcribes the audio
- LLM responds to the transcription
- TTS converts the response to speech
- Total latency is < 500ms

#### If Failing:
- Check service logs for errors
- Verify WebSocket connections are being established
- Ensure audio format is being correctly detected and processed
- Confirm all services are available and responding

## WebSocket Connection Testing

### Test with:
```bash
python test_websocket.py
```

### Expected Result:
- WebSocket connection established
- Server sends initial status message
- Client can send audio data
- Server processes audio and responds appropriately
- Connection remains stable

### If Failing:
- Check if WebSocket server is running on the expected port
- Verify CORS settings if testing from a browser
- Check for firewall or network issues
- Inspect server logs for connection errors

## Audio Processing Testing

### Test with:
```bash
python test_audio_conversion.py
```

### Expected Result:
- WebM/Opus audio is correctly converted to PCM
- Audio format detection works correctly
- Converted audio has correct sample rate and format

### If Failing:
- Verify FFmpeg is installed and available
- Check WebM headers for corruption
- Ensure audio file format is supported
- Check temp directory permissions

## Performance Testing

### Test with:
```bash
python test_websocket.py --audio tone_440hz.webm
```

### Expected Performance Targets:
- VAD processing: < 20ms
- STT processing: < 150ms
- LLM response: < 200ms
- TTS generation: < 100ms
- End-to-end latency: < 500ms

### If Performance is Poor:
- Check CPU/memory usage during processing
- Verify network latency to external services
- Consider model optimizations or caching
- Review audio buffer sizes and processing chunks

## Common Issues and Solutions

### Audio Format Issues
- **Symptom**: "Invalid WebM header" or "FFmpeg conversion errors"
- **Solution**: Use the `fix_webm_header.py` script to repair WebM headers

### Connection Timeouts
- **Symptom**: "Connection timeout" or "No response from server"
- **Solution**: Check network connectivity, increase timeout values in config

### API Rate Limiting
- **Symptom**: "Rate limit exceeded" or "Quota exceeded"
- **Solution**: Implement request throttling, check API quotas, or use caching

### Memory Issues
- **Symptom**: "Out of memory" or application crashes
- **Solution**: Optimize buffer sizes, process audio in smaller chunks

## Final Verification

When all tests pass, verify the complete system by:

1. Starting the backend server: `python -m backend.app.main`
2. Connecting from the frontend application
3. Testing a complete conversation flow
4. Verifying response times meet performance targets

## Troubleshooting Resources

- [Azure Speech Services Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/)
- [Google AI Studio Documentation](https://ai.google.dev/docs)
- [FFmpeg WebM/Opus Documentation](https://trac.ffmpeg.org/wiki/Encode/WebM)
- [WebSocket Protocol RFC](https://tools.ietf.org/html/rfc6455) 