# Voice Agent with WebSocket Improvements - Docker Setup

This document provides instructions on how to run the Voice Agent application with Docker, including all the WebSocket connection improvements.

## Overview

We've implemented several improvements to fix the WebSocket abnormal disconnection issues (code 1006) that were causing voice interactions to fail:

1. **Enhanced LLM Service**:
   - Robust message processing for different object formats
   - Proper timeout handling for all async operations
   - Retry logic with exponential backoff
   - Improved error handling and graceful fallbacks

2. **Improved Configuration**:
   - Extended timeout settings for all critical operations
   - Better watchdog configuration for session management
   - Optimized ping/pong frequency for connection health

3. **Resilient Connection Handling**:
   - Automatic reconnection with exponential backoff
   - Better error reporting and user feedback
   - Improved connection state management

## Quick Start with Docker

The easiest way to run the application with all WebSocket improvements is using Docker:

```bash
# Run the application
./run_docker.sh
```

This script will:
1. Build the Docker images with all WebSocket improvements
2. Start the containers for both frontend and backend
3. Make the WebSocket test page available

The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8003
- **WebSocket URL**: ws://localhost:8003/ws
- **WebSocket Test Page**: http://localhost:3000/test_websocket_improved.html

## Docker Configuration

### Environment Variables

The following environment variables are configured in the Docker containers:

#### Backend Settings
```yaml
# WebSocket timeout configuration
APP_WATCHDOG_INACTIVITY_TIMEOUT: 120
APP_WATCHDOG_CHECK_INTERVAL: 5
APP_AUDIO_CONVERSION_TIMEOUT: 20
APP_AUDIO_PROCESSING_TIMEOUT: 15
APP_AI_RESPONSE_TIMEOUT: 60
APP_MESSAGE_SEND_TIMEOUT: 10
```

#### Frontend Settings
```yaml
# WebSocket connection settings
NEXT_PUBLIC_WS_RECONNECT_MAX_ATTEMPTS: 5
NEXT_PUBLIC_WS_RECONNECT_BASE_DELAY: 1000
NEXT_PUBLIC_WS_PING_INTERVAL: 15000
NEXT_PUBLIC_WS_SERVER_TIMEOUT: 60000
```

### Docker Compose Commands

```bash
# Start containers
docker-compose up -d

# View logs
docker-compose logs -f

# Stop containers
docker-compose down

# Rebuild containers
docker-compose build --no-cache

# Restart services
docker-compose restart
```

## Testing WebSocket Connection

### Using the Test Page

1. Open http://localhost:3000/test_websocket_improved.html
2. Click "Connect" to establish a WebSocket connection
3. Enable "Auto-Reconnect" to test the reconnection logic
4. Use the various test buttons to simulate different scenarios:
   - "Test Reconnection" - Simulates a disconnection and tests reconnection
   - "Test Timeout Recovery" - Tests timeout handling with a complex query
   - "Stress Test" - Sends multiple messages rapidly to test stability

### Using the Command Line

Run the WebSocket connection test script:

```bash
./test_websocket_connection.sh
```

This script will attempt to establish a WebSocket connection and display the results.

## Verifying Improvements

To verify that the WebSocket improvements are working:

1. Check that the frontend successfully reconnects after disconnections
2. Verify that long-running AI responses complete without timeouts
3. Confirm that connection errors are properly handled with meaningful messages
4. Test barge-in (interrupting the AI while it's speaking) to ensure it works reliably

## Troubleshooting

If you encounter issues:

1. **Check Docker Logs**:
   ```bash
   docker-compose logs -f
   ```

2. **Verify Container Status**:
   ```bash
   docker ps
   ```

3. **Check Backend Health**:
   ```bash
   curl http://localhost:8003/health
   ```

4. **Inspect WebSocket**:
   - Open browser developer tools (F12)
   - Go to the Network tab
   - Filter by "WS" to see WebSocket connections
   - Check for any errors in the console

## Additional Resources

For more detailed information about the WebSocket improvements:
- See the [WEBSOCKET_IMPROVED_FIX.md](WEBSOCKET_IMPROVED_FIX.md) file
- Examine the updated code in `backend/services/llm_service.py`
- Check the WebSocket handling in the frontend components 