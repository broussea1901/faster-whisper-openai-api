#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_URL="http://localhost:8000"
API_KEY="your-secret-key"

echo "🎤 Faster Whisper API Test Script"
echo "================================"

# Check if server is running
echo -n "Checking if server is running... "
if curl -s -o /dev/null -w "%{http_code}" $API_URL > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Server is running${NC}"
else
    echo -e "${RED}✗ Server is not running${NC}"
    echo ""
    echo "Please start the server first:"
    echo "  GPU:   docker-compose up -d"
    echo "  CPU:   docker-compose -f docker-compose.cpu.yml up -d"
    echo ""
    echo "Check logs with: docker-compose logs -f"
    exit 1
fi

# Create a test audio file if it doesn't exist
if [ ! -f "test.mp3" ] && [ ! -f "test.wav" ]; then
    echo ""
    echo -e "${YELLOW}No test audio file found. Creating one...${NC}"
    
    # Option 1: Generate with ffmpeg if available
    if command -v ffmpeg &> /dev/null; then
        echo "Generating test audio with ffmpeg..."
        # Create a 3-second sine wave with speech-like modulation
        ffmpeg -f lavfi -i "sine=frequency=440:duration=3" -af "volume=0.5" -ac 1 -ar 16000 test.wav -y 2>/dev/null && \
        echo -e "${GREEN}✓ Created test.wav${NC}" || {
            echo -e "${YELLOW}Simple audio failed, trying speech synthesis...${NC}"
            # Try to create audio with text-to-speech
            ffmpeg -f lavfi -i "anoisesrc=d=3:c=pink:r=16000:a=0.5" test.wav -y 2>/dev/null
        }
    fi
    
    # Option 2: Download a proper speech sample
    if [ ! -f "test.mp3" ] && [ ! -f "test.wav" ]; then
        echo "Downloading speech sample audio..."
        # Download a public domain speech sample
        curl -L -o test.wav "https://github.com/mozilla/DeepSpeech/raw/master/audio/2830-3980-0043.wav" 2>/dev/null || \
        curl -L -o test.wav "https://github.com/mozilla/DeepSpeech/raw/master/audio/4507-16021-0012.wav" 2>/dev/null || \
        curl -L -o test.mp3 "https://www.kozco.com/tech/organfinale.mp3" 2>/dev/null || {
            echo -e "${RED}Failed to download sample audio${NC}"
            echo "Please provide a test audio file (test.mp3 or test.wav)"
            echo "Requirements: Audio should be at least 1-2 seconds long"
            exit 1
        }
    fi
fi

# Determine which test file to use
TEST_FILE=""
if [ -f "test.wav" ]; then
    TEST_FILE="test.wav"
elif [ -f "test.mp3" ]; then
    TEST_FILE="test.mp3"
fi

echo ""
echo "Using test file: $TEST_FILE"
echo ""

# Test 1: Models endpoint
echo "📋 Test 1: List available models"
echo "--------------------------------"
response=$(curl -s -X GET "$API_URL/v1/models" \
  -H "Authorization: Bearer $API_KEY")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Models endpoint working${NC}"
    echo "Response:"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
else
    echo -e "${RED}✗ Models endpoint failed${NC}"
fi

echo ""

# Test 2: Transcription endpoint (JSON)
echo "🎯 Test 2: Transcription (JSON format - Balanced)"
echo "--------------------------------------"
response=$(curl -s -X POST "$API_URL/v1/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@$TEST_FILE" \
  -F "model=whisper-1" \
  -F "response_format=json")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Transcription endpoint working${NC}"
    echo "Response:"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
else
    echo -e "${RED}✗ Transcription endpoint failed${NC}"
fi

echo ""

# Test 3: Fast transcription
echo "⚡ Test 3: Fast Transcription (whisper-1-fast)"
echo "----------------------------------------------"
start_time=$(date +%s.%N)
response=$(curl -s -X POST "$API_URL/v1/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@$TEST_FILE" \
  -F "model=whisper-1-fast" \
  -F "response_format=text")
end_time=$(date +%s.%N)

if [ $? -eq 0 ]; then
    elapsed=$(echo "$end_time - $start_time" | bc)
    echo -e "${GREEN}✓ Fast transcription working (${elapsed}s)${NC}"
    echo "Transcribed text:"
    echo "$response"
else
    echo -e "${RED}✗ Fast transcription failed${NC}"
fi

echo ""

# Test 4: Quality transcription
echo "🏆 Test 4: Quality Transcription (whisper-1-quality)"
echo "---------------------------------------------------"
start_time=$(date +%s.%N)
response=$(curl -s -X POST "$API_URL/v1/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@$TEST_FILE" \
  -F "model=whisper-1-quality" \
  -F "response_format=json")
end_time=$(date +%s.%N)

if [ $? -eq 0 ]; then
    elapsed=$(echo "$end_time - $start_time" | bc)
    echo -e "${GREEN}✓ Quality transcription working (${elapsed}s)${NC}"
    echo "Response:"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
else
    echo -e "${RED}✗ Quality transcription failed${NC}"
fi

echo ""

# Test 5: Translation endpoint
echo "🌍 Test 5: Translation to English"
echo "---------------------------------"
response=$(curl -s -X POST "$API_URL/v1/audio/translations" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@$TEST_FILE" \
  -F "model=whisper-1")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Translation endpoint working${NC}"
    echo "Response:"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
else
    echo -e "${RED}✗ Translation endpoint failed${NC}"
fi

echo ""

# Test 6: Performance comparison
echo "📊 Test 6: Performance Comparison"
echo "---------------------------------"
if [ -f "$TEST_FILE" ]; then
    audio_duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$TEST_FILE" 2>/dev/null || echo "unknown")
    echo "Audio duration: ${audio_duration}s"
    echo ""
    
    for model in "whisper-1-fast" "whisper-1" "whisper-1-quality"; do
        echo -n "Testing $model... "
        start_time=$(date +%s.%N)
        curl -s -X POST "$API_URL/v1/audio/transcriptions" \
          -H "Authorization: Bearer $API_KEY" \
          -F "file=@$TEST_FILE" \
          -F "model=$model" \
          -F "response_format=text" > /dev/null
        end_time=$(date +%s.%N)
        elapsed=$(echo "$end_time - $start_time" | bc)
        echo "completed in ${elapsed}s"
    done
fi

echo ""

# Test 5: Test with wrong API key
echo "🔒 Test 5: Authentication check"
echo "-------------------------------"
response=$(curl -s -w "\nHTTP_CODE:%{http_code}" -X GET "$API_URL/v1/models" \
  -H "Authorization: Bearer wrong-key")

http_code=$(echo "$response" | grep -o 'HTTP_CODE:[0-9]*' | cut -d':' -f2)

if [ "$http_code" = "401" ]; then
    echo -e "${GREEN}✓ Authentication working correctly (rejected invalid key)${NC}"
else
    echo -e "${YELLOW}⚠ Authentication might not be configured${NC}"
    echo "Make sure API_KEYS environment variable is set"
fi

echo ""
echo "✅ All tests completed!"
echo ""

# Show performance info if available
if command -v docker &> /dev/null; then
    echo "📊 Container stats:"
    docker stats --no-stream faster-whisper-api 2>/dev/null || echo "Container not found"
fi
