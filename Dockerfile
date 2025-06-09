# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for PyAudio and other libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Inform the user that they need to provide an API key
# This can be done by passing an environment variable during `docker run`
# e.g., docker run -e GEMINI_API_KEY="your_key_here" voice-agent
ENV GEMINI_API_KEY=""
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Command to run the application
CMD ["python", "main.py"]
