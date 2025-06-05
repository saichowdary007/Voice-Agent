# Voice Agent WebSocket Improvements

This document provides instructions on how to run the Voice Agent application with the WebSocket connection improvements.

## Overview

We've implemented several improvements to fix the WebSocket abnormal disconnection issues (code 1006) that were causing voice interactions to fail:

1. Enhanced LLM service with better message processing, timeouts, and retry logic
2. Improved configuration settings for WebSocket connections
3. Better error handling and reconnection logic
4. Testing tools for WebSocket reliability

## Running with Docker

The easiest way to run the application with all improvements is using Docker:

```bash
# Run the application
./run_docker.sh
```

This script will:
1. Build the Docker images with all WebSocket improvements
2. Start the containers for both frontend and backend
3. Make the WebSocket test page available

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8003
- WebSocket URL: ws://localhost:8003/ws
- WebSocket Test Page: http://localhost:3000/test_websocket_improved.html

## Configuration

The following environment variables can be adjusted in the `docker-compose.yml` file:

### Backend Settings
- `APP_WATCHDOG_INACTIVITY_TIMEOUT`: 120 seconds (timeout for inactive sessions)
- `APP_WATCHDOG_CHECK_INTERVAL`: 5 seconds (interval for checking watchdog)
- `APP_AUDIO_CONVERSION_TIMEOUT`: 20 seconds (timeout for audio conversion)
- `APP_AUDIO_PROCESSING_TIMEOUT`: 15 seconds (timeout for audio processing)
- `APP_AI_RESPONSE_TIMEOUT`: 60 seconds (timeout for AI response generation)
- `APP_MESSAGE_SEND_TIMEOUT`: 10 seconds (timeout for sending WebSocket messages)

### Frontend Settings
- `NEXT_PUBLIC_WS_RECONNECT_MAX_ATTEMPTS`: 5 (maximum reconnection attempts)
- `NEXT_PUBLIC_WS_RECONNECT_BASE_DELAY`: 1000 (base delay in ms for reconnection)
- `NEXT_PUBLIC_WS_PING_INTERVAL`: 15000 (interval in ms for sending ping messages)
- `NEXT_PUBLIC_WS_SERVER_TIMEOUT`: 60000 (timeout in ms for server response)

## Testing WebSocket Improvements

1. Open the WebSocket test page at http://localhost:3000/test_websocket_improved.html
2. Click "Connect" to establish a WebSocket connection
3. Enable "Auto-Reconnect" to test the reconnection logic
4. Use the various test buttons to simulate different scenarios:
   - "Test Reconnection" - Simulates a disconnection and tests reconnection
   - "Test Timeout Recovery" - Tests timeout handling with a complex query
   - "Stress Test" - Sends multiple messages rapidly to test stability

## Verifying Improvements

To verify that the WebSocket improvements are working:

1. Check that the frontend successfully reconnects after disconnections
2. Verify that long-running AI responses complete without timeouts
3. Confirm that connection errors are properly handled with meaningful messages
4. Test barge-in (interrupting the AI while it's speaking) to ensure it works reliably

## Troubleshooting

If you encounter issues:

1. Check the Docker logs: `docker-compose logs -f`
2. Look for error messages related to WebSocket connections
3. Verify that the backend service is running and healthy
4. Test the WebSocket connection directly using the test page

For more detailed information about the WebSocket improvements, see the [WEBSOCKET_IMPROVED_FIX.md](WEBSOCKET_IMPROVED_FIX.md) file. 