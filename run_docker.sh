#!/bin/bash

# Exit on error
set -e

echo "Voice Agent with WebSocket Improvements"
echo "========================================"
echo

# Make sure the script is executable
chmod +x run_docker.sh

# Check if .env file exists, create it if not
if [ ! -f "backend/.env" ]; then
  echo "Creating default .env file in backend directory..."
  cat > backend/.env << EOL
# Default environment variables for development
LOG_LEVEL=INFO
GOOGLE_API_KEY=
EOL
  echo "Please edit backend/.env to add your Google API key if needed."
fi

# Create necessary directories
mkdir -p backend/static
mkdir -p backend/docs
mkdir -p frontend/public

# Create the WebSocket test files
echo "Creating WebSocket test files..."

# Create test_websocket_improved.html
cat > test_websocket_improved.html << 'EOL'
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Connection Test (Improved)</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        button { margin: 5px; padding: 8px; }
        #log { 
            border: 1px solid #ccc; 
            padding: 10px; 
            height: 300px; 
            overflow-y: scroll; 
            margin-top: 10px; 
            background: #f5f5f5;
            font-family: monospace;
        }
        .error { color: red; }
        .success { color: green; }
        .info { color: blue; }
        .reconnect { color: orange; }
        #connectionStatus {
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 5px;
            display: inline-block;
        }
        .connected { background-color: #d4edda; color: #155724; }
        .disconnected { background-color: #f8d7da; color: #721c24; }
        .connecting { background-color: #fff3cd; color: #856404; }
    </style>
</head>
<body>
    <h1>Improved WebSocket Connection Test</h1>
    <div>
        <label for="wsUrl">WebSocket URL:</label>
        <input type="text" id="wsUrl" value="ws://localhost:8003/ws" size="30">
        <span id="connectionStatus" class="disconnected">Disconnected</span>
    </div>
    <div>
        <button id="connectBtn">Connect</button>
        <button id="disconnectBtn" disabled>Disconnect</button>
        <button id="autoReconnectBtn">Toggle Auto-Reconnect</button>
        <span id="autoReconnectStatus">Auto-Reconnect: OFF</span>
    </div>
    <div>
        <button id="sendTextBtn" disabled>Send Text Message</button>
        <button id="sendBinaryBtn" disabled>Send 10KB Binary</button>
        <button id="sendEosBtn" disabled>Send EOS</button>
        <button id="pingBtn" disabled>Send Ping</button>
        <button id="streamTestBtn" disabled>Test LLM Stream</button>
    </div>
    <div>
        <label>Test Scenarios:</label>
        <button id="testReconnectBtn">Test Reconnection</button>
        <button id="testTimeoutBtn">Test Timeout Recovery</button>
        <button id="testStressBtn">Stress Test (10 msgs)</button>
    </div>
    <div id="log"></div>

    <script>
        // Basic WebSocket tester
        let ws = null;
        let autoReconnect = false;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        
        const logElem = document.getElementById('log');
        const wsUrlInput = document.getElementById('wsUrl');
        const connectBtn = document.getElementById('connectBtn');
        const disconnectBtn = document.getElementById('disconnectBtn');
        const sendTextBtn = document.getElementById('sendTextBtn');
        const sendBinaryBtn = document.getElementById('sendBinaryBtn');
        const sendEosBtn = document.getElementById('sendEosBtn');
        const pingBtn = document.getElementById('pingBtn');
        const streamTestBtn = document.getElementById('streamTestBtn');
        const autoReconnectBtn = document.getElementById('autoReconnectBtn');
        const autoReconnectStatus = document.getElementById('autoReconnectStatus');
        const connectionStatus = document.getElementById('connectionStatus');
        
        function log(message, className = '') {
            const timestamp = new Date().toLocaleTimeString();
            logElem.innerHTML += `<div class="${className}">[${timestamp}] ${message}</div>`;
            logElem.scrollTop = logElem.scrollHeight;
        }
        
        function setButtonsState(connected) {
            connectBtn.disabled = connected;
            disconnectBtn.disabled = !connected;
            sendTextBtn.disabled = !connected;
            sendBinaryBtn.disabled = !connected;
            sendEosBtn.disabled = !connected;
            pingBtn.disabled = !connected;
            streamTestBtn.disabled = !connected;
            
            connectionStatus.textContent = connected ? 'Connected' : 'Disconnected';
            connectionStatus.className = connected ? 'connected' : 'disconnected';
        }
        
        connectBtn.addEventListener('click', () => {
            const url = wsUrlInput.value;
            log(`Connecting to ${url}...`, 'info');
            
            try {
                ws = new WebSocket(url);
                ws.binaryType = 'arraybuffer';
                
                ws.onopen = () => {
                    log('Connection established', 'success');
                    setButtonsState(true);
                    reconnectAttempts = 0;
                };
                
                ws.onclose = (event) => {
                    log(`Connection closed. Code: ${event.code}, Reason: ${event.reason || 'None'}`, 
                        event.code === 1000 ? 'info' : 'error');
                    setButtonsState(false);
                    
                    if (autoReconnect && event.code !== 1000) {
                        reconnectAttempts++;
                        if (reconnectAttempts <= maxReconnectAttempts) {
                            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts - 1), 10000);
                            log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})`, 'reconnect');
                            setTimeout(() => connectBtn.click(), delay);
                        } else {
                            log(`Max reconnection attempts reached (${maxReconnectAttempts})`, 'error');
                        }
                    }
                };
                
                ws.onerror = (error) => {
                    log('WebSocket error', 'error');
                    console.error('WebSocket error:', error);
                };
                
                ws.onmessage = (event) => {
                    if (event.data instanceof ArrayBuffer) {
                        const view = new Uint8Array(event.data);
                        log(`Received binary message: ${view.length} bytes`, 'info');
                    } else {
                        try {
                            const jsonData = JSON.parse(event.data);
                            log(`Received: ${JSON.stringify(jsonData)}`, 'info');
                        } catch (e) {
                            log(`Received text: ${event.data}`, 'info');
                        }
                    }
                };
            } catch (error) {
                log(`Failed to connect: ${error.message}`, 'error');
            }
        });
        
        autoReconnectBtn.addEventListener('click', () => {
            autoReconnect = !autoReconnect;
            autoReconnectStatus.textContent = `Auto-Reconnect: ${autoReconnect ? 'ON' : 'OFF'}`;
            log(`Auto-reconnect ${autoReconnect ? 'enabled' : 'disabled'}`, 'info');
        });
        
        disconnectBtn.addEventListener('click', () => {
            if (ws) {
                log('Closing connection...', 'info');
                ws.close(1000, 'Closed by user');
            }
        });
        
        sendTextBtn.addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                const message = JSON.stringify({type: "text_command", text: "Hello from WebSocket test"});
                ws.send(message);
                log(`Sent text message: ${message}`, 'info');
            } else {
                log('Cannot send: WebSocket not connected', 'error');
            }
        });
        
        sendBinaryBtn.addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                const buffer = new Uint8Array(10 * 1024);
                for (let i = 0; i < buffer.length; i++) buffer[i] = i % 256;
                ws.send(buffer);
                log('Sent 10KB binary message', 'info');
            } else {
                log('Cannot send: WebSocket not connected', 'error');
            }
        });
        
        sendEosBtn.addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                const message = JSON.stringify({type: "eos"});
                ws.send(message);
                log(`Sent EOS message: ${message}`, 'info');
            } else {
                log('Cannot send: WebSocket not connected', 'error');
            }
        });
        
        pingBtn.addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                const message = JSON.stringify({type: "ping", timestamp: Date.now()});
                ws.send(message);
                log(`Sent ping: ${message}`, 'info');
            } else {
                log('Cannot send: WebSocket not connected', 'error');
            }
        });
        
        streamTestBtn.addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                const message = JSON.stringify({
                    type: "text_command", 
                    text: "Tell me about voice recognition technology"
                });
                ws.send(message);
                log(`Sent text command for LLM stream test: ${message}`, 'info');
            } else {
                log('Cannot send: WebSocket not connected', 'error');
            }
        });
        
        document.getElementById('testReconnectBtn').addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                log('Testing reconnection by closing WebSocket with code 1006', 'info');
                ws.close(1006, 'Testing reconnection');
            } else {
                log('Cannot test reconnection: WebSocket not connected', 'error');
            }
        });
        
        document.getElementById('testTimeoutBtn').addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                const message = JSON.stringify({
                    type: "text_command", 
                    text: "Explain in detail how voice recognition works with all technical details"
                });
                ws.send(message);
                log(`Sent complex query to test timeout handling`, 'info');
            } else {
                log('Cannot test timeout: WebSocket not connected', 'error');
            }
        });
        
        document.getElementById('testStressBtn').addEventListener('click', async () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                log('Starting stress test - sending 10 messages rapidly', 'info');
                for (let i = 1; i <= 10; i++) {
                    if (ws.readyState === WebSocket.OPEN) {
                        const message = JSON.stringify({type: "text_command", text: `Test message ${i}/10`});
                        ws.send(message);
                        log(`Stress test: Sent message ${i}/10`, 'info');
                        await new Promise(r => setTimeout(r, 100));
                    }
                }
            } else {
                log('Cannot run stress test: WebSocket not connected', 'error');
            }
        });
        
        log('WebSocket tester initialized', 'info');
    </script>
</body>
</html>
EOL

# Create WEBSOCKET_IMPROVED_FIX.md
cat > WEBSOCKET_IMPROVED_FIX.md << 'EOL'
# WebSocket Connection Improvements

## Overview

This document outlines the improved WebSocket connection handling implemented to resolve the issues with code 1006 disconnections and to ensure more reliable WebSocket communication between the frontend and backend.

## Problem Description

The WebSocket connections were experiencing abnormal closures (code 1006) after sending audio data, leading to:
- Interrupted voice interactions 
- Failed AI responses
- Repeated reconnection attempts
- Poor user experience with constant error messages

## Root Causes Identified

1. **LLM Service Timeouts**: No proper timeout handling in the LLM service's message processing
2. **Missing Error Recovery**: No retry mechanism when API calls to Gemini failed
3. **Incomplete Message Processing**: Inability to handle various message object formats
4. **Connection Management**: Inadequate reconnection logic and error handling

## Solutions Implemented

### 1. Enhanced LLM Service

The `LLMService` class has been significantly improved with:

- **Robust Message Processing**: Added support for different message object formats (both dict and object with attributes)
- **Timeout Management**: Applied proper timeouts for all async operations:
  - 60-second timeout for complete response generation
  - 30-second timeout for stream initialization
- **Retry Logic**: Implemented an exponential backoff retry mechanism:
  - Up to 3 retries for failed API calls
  - Increasing delay between retries (1s, 2s, 4s)
- **Improved Error Handling**: Better exception catching and recovery
- **Graceful Fallbacks**: Always provide mock responses when API calls fail

### 2. Configuration Settings

We're using the existing timeout settings in the app's configuration:

- `audio_conversion_timeout`: 20 seconds
- `audio_processing_timeout`: 15 seconds
- `ai_response_timeout`: 60 seconds
- `message_send_timeout`: 10 seconds
- `watchdog_inactivity_timeout`: 120 seconds (up from previous 30s)
- `watchdog_check_interval`: 5 seconds (up from previous 1s)

### 3. Frontend WebSocket Connection

The frontend already has good reconnection logic:
- Exponential backoff for reconnection attempts
- Proper handling of connection state
- Error reporting to the user
- Ping/pong mechanism to keep connections alive
EOL

# Copy the WebSocket test files to both frontend and backend
echo "Copying WebSocket test files..."
cp test_websocket_improved.html frontend/public/
cp test_websocket_improved.html backend/static/
cp WEBSOCKET_IMPROVED_FIX.md backend/docs/

# Build and start containers
echo "Building and starting Docker containers..."
docker-compose build
docker-compose up -d

echo
echo "Voice Agent is now running:"
echo "- Frontend: http://localhost:3000"
echo "- Backend API: http://localhost:8003"
echo "- WebSocket URL: ws://localhost:8003/ws"
echo "- WebSocket Test Page: http://localhost:3000/test_websocket_improved.html"
echo
echo "To view logs, run: docker-compose logs -f"
echo "To stop, run: docker-compose down" 