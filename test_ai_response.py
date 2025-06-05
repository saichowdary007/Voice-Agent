#!/usr/bin/env python3
import asyncio
import websockets
import json
import time

async def test_ai_response():
    """Test the AI response generation by sending a message through the WebSocket."""
    uri = "ws://localhost:8003/ws"
    
    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as websocket:
        print("Connected!")
        
        # Wait for status message
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Received: {data}")
        
        # Send direct text command
        print("Sending direct text command...")
        await websocket.send(json.dumps({
            "type": "text_command",
            "text": "Hello, can you help me with a test?"
        }))
        
        # Wait for responses
        try:
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=30)
                data = json.loads(response)
                print(f"Received: {data['type']}")
                
                if data.get("type") == "transcript" and "final" in data:
                    print(f"Transcript: {data['final']}")
                
                if data.get("type") == "ai_response":
                    if data.get("token"):
                        print(f"AI token: {data['token']}", end="", flush=True)
                    if data.get("complete") == True:
                        print("\nAI response complete!")
                        break
                        
                if data.get("type") == "error":
                    print(f"Error: {data.get('message')}")
                    break
                    
        except asyncio.TimeoutError:
            print("Timeout waiting for response")
        
        print("Test complete!")

if __name__ == "__main__":
    asyncio.run(test_ai_response()) 