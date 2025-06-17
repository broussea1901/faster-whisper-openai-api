#!/bin/bash

# Test transcription endpoint
echo "Testing transcription endpoint..."
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer your-secret-key-1" \
  -F "file=@test.mp3" \
  -F "model=whisper-1" \
  -F "response_format=json"

echo -e "\n\nTesting models endpoint..."
curl -X GET "http://localhost:8000/v1/models" \
  -H "Authorization: Bearer your-secret-key-1"
