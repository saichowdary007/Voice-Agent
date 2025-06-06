#!/bin/bash

# Start the Voice Agent application using Docker Compose

# Check if Docker is available
if ! command -v docker &>/dev/null; then
    echo "❌ Error: Docker is not installed or not in PATH."
    echo "Please install Docker and try again."
    exit 1
fi

if ! command -v docker-compose &>/dev/null; then
    echo "❌ Error: docker-compose is not installed or not in PATH."
    echo "Please install docker-compose and try again."
    exit 1
fi

# Stop any existing containers
echo "🛑 Stopping any existing containers..."
docker-compose down

# Build the containers
echo "🏗️  Building Docker containers..."
docker-compose build

# Start the containers
echo "🚀 Starting Voice Agent with Docker..."
echo "=========================================="
docker-compose up

# Handle exit
echo "👋 Voice Agent Docker containers stopped." 