# Voice Agent Testing and Monitoring Tools

This repository contains tools for testing, debugging, and monitoring the Voice Agent application. These tools are designed to help diagnose issues with WebSocket connections, WebM audio handling, and the various AI services.

## Overview of Tools

### Testing Tools

1. **test_services.py**: Tests basic functionality of all services.
   - Checks VAD, STT, LLM, TTS, and Audio services
   - Provides summary of which services are working

2. **test_individual_services.py**: Detailed test of each individual service.
   - Tests each service with more specific diagnostics
   - Provides detailed output about service behavior

3. **test_audio_processing.py**: Specific tests for audio processing.
   - Tests WebM/Opus encoding/decoding
   - Tests audio conversion functions

### Debug Tools

1. **debug_ffmpeg_conversion.py**: Debug FFmpeg conversion issues.
   - Generate valid WebM/Opus files
   - Extract and analyze WebM headers
   - Test different conversion methods

2. **fix_webm_header.py**: Utility to fix WebM header issues.
   - Scan backend code for invalid WebM headers
   - Suggest fixes with proper header formats
   - Generate valid WebM headers in different code formats

3. **generate_test_files.py**: Create test audio files.
   - Generate various test files (tone, noise, etc.)
   - Create corrupted files for testing error handling

### Monitoring Tools

1. **monitor_services.py**: Continuous monitoring of service health.
   - Periodically checks all services
   - Logs issues and maintains health history
   - Provides service status dashboard

## Setup and Usage

### Environment Setup

```bash
# Set up a virtual environment
python3 -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install dependencies
pip install websockets asyncio
```

### Basic Usage

```bash
# Run basic service tests
python test_services.py

# Run detailed service tests
python test_individual_services.py --service all

# Generate test audio files
./generate_test_files.py

# Run the monitoring service
./monitor_services.py --interval 60  # Check every 60 seconds
```

## Issues Identified and Fixes

### WebM Header Issues

The application was experiencing issues with WebM header processing:

1. **Problem**: Invalid EBML numbers in WebM header causing FFmpeg conversion errors
   - **Fix**: Updated the WebM header generation code with valid header format extracted from a properly generated WebM file

2. **Problem**: Raw audio data length not being a multiple of the sample size
   - **Fix**: Ensured proper padding of audio data and correct header structures

### VAD Service Issues

Issues with Voice Activity Detection and EOS (End of Speech) handling:

1. **Problem**: Inconsistent behavior with EOS signals
   - **Fix**: Improved the VAD test with retries and fallback to audio testing
   - **Fix**: Added force_finalize flag to EOS messages

### STT Service Issues

Issues with Speech-to-Text service:

1. **Problem**: Partial transcript support not confirmed
   - **Fix**: Enhanced testing with longer audio sequences
   - **Fix**: Added specific checks for partial transcript detection

## Monitoring

The monitoring system periodically checks all services and maintains a health history in `health_report.json`. Services are classified as:

- ✅ **Working**: Service is responding correctly
- ❌ **Failing**: Service is not responding or giving errors
- ⚠️ **Error**: Error occurred during testing

## Future Improvements

1. Add email/Slack notifications for service issues
2. Implement automatic recovery attempts for failing services
3. Add more detailed performance metrics
4. Create a web dashboard for service status visualization 