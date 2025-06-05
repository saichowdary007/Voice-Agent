#!/bin/bash

echo "Testing WebSocket Connection to localhost:8003/ws"
echo "================================================="
echo

# Check if curl is installed
if ! command -v curl &> /dev/null; then
    echo "Error: curl is not installed. Please install curl to run this test."
    exit 1
fi

# Check if backend is running
echo "Checking if backend is running..."
HEALTH_CHECK=$(curl -s http://localhost:8003/health)
if [ $? -ne 0 ]; then
    echo "Error: Failed to connect to backend. Make sure it's running on localhost:8003."
    exit 1
fi

echo "Backend health check response: $HEALTH_CHECK"
echo

# Test WebSocket
echo "Testing WebSocket connection..."
echo "Note: This will attempt to establish a WebSocket connection. Press Ctrl+C to exit after a few seconds."
echo

# Using curl to test the WebSocket connection
curl -i -N -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     -H "Host: localhost:8003" \
     -H "Origin: http://localhost:3000" \
     -H "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
     -H "Sec-WebSocket-Version: 13" \
     http://localhost:8003/ws

echo
echo "If you see a '101 Switching Protocols' response, the WebSocket connection is working correctly."
echo "You can also test the connection using the WebSocket test page at:"
echo "http://localhost:3000/test_websocket_improved.html" 