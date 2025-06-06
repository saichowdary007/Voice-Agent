#!/bin/bash

# Start the Voice Agent frontend

# Check if npm is available
if ! command -v npm &>/dev/null; then
    echo "❌ Error: npm is not installed or not in PATH."
    echo "Please install Node.js and npm, then try again."
    exit 1
fi

# Check if we're in the right directory
if [ ! -d "frontend" ]; then
    echo "❌ Error: 'frontend' directory not found."
    echo "Please run this script from the root of the Voice Agent project."
    exit 1
fi

# Go to the frontend directory
cd frontend || exit 1

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

# Ask for port if needed
echo "Enter port to use for frontend server (default: 3000):"
read PORT
PORT=${PORT:-3000}

# Set environment variables for the frontend
export NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
export NEXT_PUBLIC_API_URL=http://localhost:8000

# Start the frontend server
echo "🚀 Starting Voice Agent frontend on port $PORT..."
echo "=========================================="
npm run dev -- -p $PORT

# Handle exit
echo "👋 Voice Agent frontend stopped." 