#!/bin/bash
set -e

# Setup environment
export PYTHONPATH=/app

# Print Python path info for debugging
echo "PYTHONPATH: $PYTHONPATH"
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"

# List directories to verify structure
echo "Directory structure:"
ls -la

# Create symbolic link if needed
if [ ! -L "/app/backend" ] && [ -d "/app" ]; then
    echo "Creating symbolic link for backend module"
    ln -s /app /app/backend
fi

# Start the application
exec "$@" 