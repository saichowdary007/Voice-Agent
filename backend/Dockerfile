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
    && rm -rf /var/lib/apt/lists/*

# Create models directory
RUN mkdir -p /opt/render/project/src/models

# Set environment variables
ENV MODEL_PATH=/opt/render/project/src/models
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create app directory structure
RUN mkdir -p app/models

# Download models during build
RUN python app/models/download_models.py

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app_user && \
    chown -R app_user:app_user /app && \
    chown -R app_user:app_user /opt/render/project/src/models

# Switch to non-root user
USER app_user

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"] 