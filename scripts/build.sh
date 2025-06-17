#!/bin/bash
set -e

echo "Building Faster Whisper OpenAI API Server..."

# Build GPU version
echo "Building GPU version..."
docker build -t faster-whisper-server:latest -t faster-whisper-server:gpu .

# Build CPU version
echo "Building CPU version..."
docker build -f Dockerfile.cpu -t faster-whisper-server:cpu .

echo "Build complete!"
echo "GPU version: faster-whisper-server:latest"
echo "CPU version: faster-whisper-server:cpu"
