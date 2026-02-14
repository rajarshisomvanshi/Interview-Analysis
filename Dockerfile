
# Use an official Python runtime as a parent image
# Using slim version to reduce size, but full version might be safer for complex deps
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
# ffmpeg: for audio processing
# libgl1: for OpenCV (replaces libgl1-mesa-glx in newer Debian)
# build-essential: for compiling some python extensions
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file (if you have one, otherwise we'll install manually or assume poetry)
# Assuming requirements.txt exists at root or we create one
COPY requirements.txt .

# Install dependencies
# Upgrade pip first
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the API port
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py", "server"]
