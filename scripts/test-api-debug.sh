#!/bin/bash
# Debug version of test script with timeouts and verbose output

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-your-api-key}"
TEST_FILE="${TEST_FILE:-examples/test.wav}"
TIMEOUT=60  # 60 second timeout for transcription

echo -e "${BLUE}🧪 Faster Whisper v2 API Test Suite (Debug Mode)${NC}"
echo "===================================="
echo "API URL: $API_URL"
echo "Test file: $TEST_FILE"
echo "Timeout: ${TIMEOUT}s"
echo ""

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo -e "${YELLOW}Creating test file...${NC}"
    mkdir -p examples
    curl -L -o examples/test.wav "https://github.com/mozilla/DeepSpeech/raw/master/audio/2830-3980-0043.wav" 2>/dev/null || {
        echo -e "${RED}Failed to download test audio${NC}"
        exit 1
    }
fi

# Check file size
echo -e "${BLUE}Test file info:${NC}"
ls -lh "$TEST_FILE"
echo ""

# Health check
echo -e "${YELLOW}1. Health Check${NC}"
curl -s "$API_URL/" | jq . || echo "Failed to get health"
echo ""

# Simple transcription test with timeout and verbose output
echo -e "${YELLOW}2. Transcription Test (with timeout)${NC}"
echo "Sending request to $API_URL/v1/audio/transcriptions"
echo "This may take 1-2 minutes on first run while model loads..."
echo ""

# Run with timeout and show progress
if command -v timeout >/dev/null 2>&1; then
    # GNU timeout
    timeout_cmd="timeout ${TIMEOUT}s"
elif command -v gtimeout >/dev/null 2>&1; then
    # macOS with coreutils
    timeout_cmd="gtimeout ${TIMEOUT}s"
else
    # No timeout command
    timeout_cmd=""
    echo -e "${YELLOW}Warning: timeout command not found${NC}"
fi

echo "Running: curl -X POST '$API_URL/v1/audio/transcriptions' ..."
start_time=$(date +%s)

# Run the actual test with verbose output
$timeout_cmd curl -X POST "$API_URL/v1/audio/transcriptions" \
    -H "Authorization: Bearer $API_KEY" \
    -F "file=@$TEST_FILE" \
    -F "model=whisper-1-fast" \
    -F "response_format=json" \
    -w "\n\nHTTP Status: %{http_code}\nTime: %{time_total}s\n" \
    -m $TIMEOUT \
    --progress-bar

end_time=$(date +%s)
duration=$((end_time - start_time))

echo -e "\n${BLUE}Request took ${duration} seconds${NC}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Transcription successful${NC}"
else
    echo -e "${RED}❌ Transcription failed or timed out${NC}"
    echo -e "\n${YELLOW}Check Docker logs:${NC}"
    echo "docker logs whisper-v2-cpu --tail 50"
fi
