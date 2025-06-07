#!/bin/sh
# docker-entrypoint.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Run Gunicorn with Uvicorn workers
exec gunicorn "backend.app.main:app" \
    --workers 4 \
    --worker-class "uvicorn.workers.UvicornWorker" \
    --bind "0.0.0.0:8003" 