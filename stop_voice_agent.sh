#!/bin/bash

# Voice Agent Stop Script
# This script stops all voice agent services

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 Stopping Voice Agent Services...${NC}"

# Stop backend server on port 8080
echo -e "${YELLOW}📡 Stopping backend server...${NC}"
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null 2>&1; then
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    echo -e "${GREEN}✅ Backend server stopped${NC}"
else
    echo -e "${YELLOW}⚠️  Backend server not running${NC}"
fi

# Stop Kokoro TTS Docker container
echo -e "${YELLOW}🎤 Stopping Kokoro TTS service...${NC}"
if docker ps --filter name=kokoro-tts --format "{{.Names}}" | grep -q kokoro-tts; then
    docker stop kokoro-tts >/dev/null 2>&1
    docker rm kokoro-tts >/dev/null 2>&1
    echo -e "${GREEN}✅ Kokoro TTS service stopped${NC}"
else
    echo -e "${YELLOW}⚠️  Kokoro TTS service not running${NC}"
fi

# Clean up any remaining processes
echo -e "${YELLOW}🧹 Cleaning up remaining processes...${NC}"
pkill -f "python server.py" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true

echo -e "${GREEN}🎉 All Voice Agent services stopped!${NC}"
echo -e "${BLUE}💡 To restart, run: ./start_voice_agent.sh${NC}" 