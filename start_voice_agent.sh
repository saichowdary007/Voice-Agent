#!/bin/bash

# Voice Agent Startup Script
# This script starts all required services for the voice agent

set -e

echo "🚀 Starting Voice Agent Services..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    echo -e "${YELLOW}⏳ Waiting for $service_name to be ready...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ $service_name is ready!${NC}"
            return 0
        fi
        echo -e "${YELLOW}   Attempt $attempt/$max_attempts - waiting for $service_name...${NC}"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}❌ $service_name failed to start within timeout${NC}"
    return 1
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found. Please run: python3 -m venv venv${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

# Install/update dependencies
echo -e "${BLUE}📦 Installing dependencies...${NC}"
pip install -q aiohttp

# Check if Kokoro TTS is already running
if check_port 8880; then
    echo -e "${GREEN}✅ Kokoro TTS service already running on port 8880${NC}"
else
    echo -e "${BLUE}🎤 Starting Kokoro TTS service...${NC}"
    
    # Try to start Kokoro TTS with Docker
    if command -v docker >/dev/null 2>&1; then
        echo -e "${YELLOW}   Starting Kokoro TTS with Docker...${NC}"
        docker run -d --name kokoro-tts -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest >/dev/null 2>&1 || {
            echo -e "${YELLOW}   Docker image not found, trying to pull...${NC}"
            docker pull ghcr.io/remsky/kokoro-fastapi-cpu:latest
            docker run -d --name kokoro-tts -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest
        }
        
        # Wait for Kokoro TTS to be ready
        wait_for_service "http://localhost:8880/health" "Kokoro TTS"
    else
        echo -e "${RED}❌ Docker not found. Please install Docker or start Kokoro TTS manually on port 8880${NC}"
        echo -e "${YELLOW}   Manual start: docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest${NC}"
        exit 1
    fi
fi

# Check if backend is already running
if check_port 8080; then
    echo -e "${YELLOW}⚠️  Port 8080 is already in use. Stopping existing service...${NC}"
    # Try to kill any existing process on port 8080
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Start the backend server
echo -e "${BLUE}🖥️  Starting Voice Agent backend on port 8080...${NC}"
export PORT=8080
export HOST=0.0.0.0

# Start the server in the background
python server.py &
SERVER_PID=$!

# Wait for backend to be ready
wait_for_service "http://localhost:8080/health" "Voice Agent Backend"

echo -e "${GREEN}🎉 All services started successfully!${NC}"
echo -e "${GREEN}📡 Backend API: http://localhost:8080${NC}"
echo -e "${GREEN}🎤 Kokoro TTS: http://localhost:8880${NC}"
echo -e "${GREEN}🌐 Frontend should connect to: ws://localhost:8080${NC}"
echo ""
echo -e "${BLUE}📋 Service Status:${NC}"
echo -e "   Backend PID: $SERVER_PID"
echo -e "   Kokoro TTS: $(docker ps --filter name=kokoro-tts --format 'table {{.Names}}\t{{.Status}}' | tail -n +2)"
echo ""
echo -e "${YELLOW}💡 To stop services:${NC}"
echo -e "   Kill backend: kill $SERVER_PID"
echo -e "   Stop Kokoro TTS: docker stop kokoro-tts && docker rm kokoro-tts"
echo ""
echo -e "${GREEN}✨ Voice Agent is ready! Start your frontend and begin talking.${NC}"

# Keep the script running and show logs
echo -e "${BLUE}📊 Backend logs (Ctrl+C to stop):${NC}"
wait $SERVER_PID 