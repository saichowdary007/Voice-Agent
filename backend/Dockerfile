# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ffmpeg \
    git \
    wget \
    curl \
    pkg-config \
    libasound2-dev \
    libsndfile1-dev \
    libopus-dev \
    libopusfile-dev \
    python3-dev \
    espeak \
    libespeak-dev \
    libssl-dev \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Create models directory
RUN mkdir -p /app/models

# Set environment variables
ENV MODEL_PATH=/app/models
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# WebSocket timeout configuration
ENV APP_WATCHDOG_INACTIVITY_TIMEOUT=120
ENV APP_WATCHDOG_CHECK_INTERVAL=5
ENV APP_AUDIO_CONVERSION_TIMEOUT=20
ENV APP_AUDIO_PROCESSING_TIMEOUT=15
ENV APP_AI_RESPONSE_TIMEOUT=60
ENV APP_MESSAGE_SEND_TIMEOUT=10

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Azure Speech SDK dependencies 
RUN apt-get update && apt-get install -y \
    libssl-dev \
    libasound2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Create app directory structure
RUN mkdir -p app/models
RUN mkdir -p /app/docs
RUN mkdir -p /app/static

# Download models during build (if download script exists)
RUN if [ -f "app/models/download_models.py" ]; then python app/models/download_models.py; fi

# Copy the entrypoint script and make it executable
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app_user && \
    chown -R app_user:app_user /app && \
    chown -R app_user:app_user /app/models

# Switch to non-root user
USER app_user

# Expose port 8003 to match our current setup
EXPOSE 8003

# Health check - check if server is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -s http://localhost:8003/health | grep -q '"status"' || exit 1

# Install gunicorn for more stable process management
RUN pip install --no-cache-dir gunicorn uvicorn

# Set the entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Start the application with improved settings for WebSocket stability
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8003", "--lifespan", "on", "--log-level", "info", "--timeout-keep-alive", "120", "--timeout-graceful-shutdown", "30", "--ws-ping-interval", "15", "--ws-ping-timeout", "60"] 