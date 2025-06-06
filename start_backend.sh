#!/bin/bash

# Start the Voice Agent backend directly

# Check if Python 3 is available (try multiple possible commands)
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python3.11 &>/dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &>/dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.9 &>/dev/null; then
    PYTHON_CMD="python3.9"
elif command -v python &>/dev/null; then
    # Check if this is Python 3
    PY_VERSION=$(python --version 2>&1)
    if [[ $PY_VERSION == *"Python 3"* ]]; then
        PYTHON_CMD="python"
    else
        echo "❌ Error: Found Python but it's not version 3.x"
        echo "Please install Python 3 and try again."
        exit 1
    fi
else
    echo "❌ Error: No Python 3.x installation found."
    echo "Please install Python 3 and try again."
    exit 1
fi

echo "🐍 Using Python command: $PYTHON_CMD ($(${PYTHON_CMD} --version 2>&1))"

# Activate virtual environment if it exists
if [ -d "env" ] && [ -f "env/bin/activate" ]; then
    echo "🔄 Activating virtual environment..."
    source env/bin/activate
    PYTHON_CMD="env/bin/python"
elif [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
    PYTHON_CMD="venv/bin/python"
else
    echo "⚠️ No virtual environment found. Using system Python."
fi

# Check if requirements are installed
echo "🔍 Checking Python dependencies..."
$PYTHON_CMD -c "import fastapi" &>/dev/null || {
    echo "📦 Installing Python dependencies..."
    $PYTHON_CMD -m pip install -r requirements.txt
}

# Ask for port if needed
echo "Enter port to use for backend server (default: 8000):"
read PORT
PORT=${PORT:-8000}

# Run the backend
echo "🚀 Starting Voice Agent backend on port $PORT..."
echo "=========================================="
PYTHONPATH=. $PYTHON_CMD -m uvicorn backend.app.main:app --host 127.0.0.1 --port $PORT --reload

# Handle exit
echo "👋 Voice Agent backend stopped." 