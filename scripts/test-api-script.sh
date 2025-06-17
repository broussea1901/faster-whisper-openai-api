#!/bin/bash
# Test script for Faster Whisper v2 API

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

echo -e "${BLUE}🧪 Faster Whisper v2 API Test Suite${NC}"
echo "===================================="
echo "API URL: $API_URL"
echo "Test file: $TEST_FILE"
echo ""

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${YELLOW}Test file not found. Creating one...${NC}"
        mkdir -p examples
        # Download sample
        curl -L -o examples/test.wav "https://github.com/mozilla/DeepSpeech/raw/master/audio/2830-3980-0043.wav" 2>/dev/null || {
            echo -e "${RED}Failed to download test audio${NC}"
            exit 1
        }
    fi
}

# Function to test endpoint
test_endpoint() {
    local name=$1
    local cmd=$2
    echo -e "${BLUE}Test: $name${NC}"
    if eval "$cmd"; then
        echo -e "${GREEN}✅ Passed${NC}\n"
        return 0
    else
        echo -e "${RED}❌ Failed${NC}\n"
        return 1
    fi
}

# Track failures
FAILED=0

# Check server health
echo -e "${YELLOW}1. Health Check${NC}"
response=$(curl -s "$API_URL/")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Server is running${NC}"
    echo "$response" | jq . 2>/dev/null || echo "$response"
    
    # Check if diarization is enabled
    diarization_enabled=$(echo "$response" | jq -r '.diarization_enabled' 2>/dev/null || echo "false")
    if [ "$diarization_enabled" = "true" ]; then
        echo -e "${GREEN}✅ Diarization is enabled${NC}"
        HAS_DIARIZATION=true
    else
        echo -e "${YELLOW}ℹ️  Diarization is disabled (CPU mode)${NC}"
        HAS_DIARIZATION=false
    fi
else
    echo -e "${RED}❌ Server is not responding${NC}"
    exit 1
fi
echo ""

# Check test file
check_file "$TEST_FILE"

# 2. List models
echo -e "${YELLOW}2. Model Listing${NC}"
test_endpoint "List available models" \
    "curl -s -H 'Authorization: Bearer $API_KEY' '$API_URL/v1/models' | jq -r '.data[].id' 2>/dev/null | grep -E 'whisper-1|whisper-1-fast|whisper-1-quality'" || ((FAILED++))

# 3. Test transcription endpoints
echo -e "${YELLOW}3. Transcription Tests${NC}"

# Test default model
test_endpoint "Default transcription (whisper-1)" \
    "curl -s -X POST '$API_URL/v1/audio/transcriptions' \
        -H 'Authorization: Bearer $API_KEY' \
        -F 'file=@$TEST_FILE' \
        -F 'model=whisper-1' \
        -F 'response_format=json' | jq -r '.text' 2>/dev/null | grep -q '.'" || ((FAILED++))

# Test fast model
test_endpoint "Fast transcription (whisper-1-fast)" \
    "curl -s -X POST '$API_URL/v1/audio/transcriptions' \
        -H 'Authorization: Bearer $API_KEY' \
        -F 'file=@$TEST_FILE' \
        -F 'model=whisper-1-fast' \
        -F 'response_format=text' | grep -q '.'" || ((FAILED++))

# Test quality model
test_endpoint "Quality transcription (whisper-1-quality)" \
    "curl -s -X POST '$API_URL/v1/audio/transcriptions' \
        -H 'Authorization: Bearer $API_KEY' \
        -F 'file=@$TEST_FILE' \
        -F 'model=whisper-1-quality' \
        -F 'response_format=json' | jq -r '.text' 2>/dev/null | grep -q '.'" || ((FAILED++))

# 4. Test response formats
echo -e "${YELLOW}4. Response Format Tests${NC}"

test_endpoint "SRT format" \
    "curl -s -X POST '$API_URL/v1/audio/transcriptions' \
        -H 'Authorization: Bearer $API_KEY' \
        -F 'file=@$TEST_FILE' \
        -F 'model=whisper-1-fast' \
        -F 'response_format=srt' | grep -q -- '-->' " || ((FAILED++))

test_endpoint "VTT format" \
    "curl -s -X POST '$API_URL/v1/audio/transcriptions' \
        -H 'Authorization: Bearer $API_KEY' \
        -F 'file=@$TEST_FILE' \
        -F 'model=whisper-1-fast' \
        -F 'response_format=vtt' | grep -q 'WEBVTT'" || ((FAILED++))

# 5. Test translation
echo -e "${YELLOW}5. Translation Test${NC}"
test_endpoint "Translation to English" \
    "curl -s -X POST '$API_URL/v1/audio/translations' \
        -H 'Authorization: Bearer $API_KEY' \
        -F 'file=@$TEST_FILE' \
        -F 'model=whisper-1' \
        -F 'response_format=json' | jq -r '.text' 2>/dev/null | grep -q '.'" || ((FAILED++))

# 6. Test diarization (if available)
if [ "$HAS_DIARIZATION" = "true" ]; then
    echo -e "${YELLOW}6. Diarization Test${NC}"
    test_endpoint "Transcription with speaker diarization" \
        "curl -s -X POST '$API_URL/v1/audio/transcriptions' \
            -H 'Authorization: Bearer $API_KEY' \
            -F 'file=@$TEST_FILE' \
            -F 'model=whisper-1' \
            -F 'timestamp_granularities=speaker' \
            -F 'response_format=json' | jq '.segments[0].speaker' 2>/dev/null | grep -q 'SPEAKER'" || ((FAILED++))
else
    echo -e "${YELLOW}6. Diarization Test${NC}"
    echo -e "${YELLOW}ℹ️  Skipping diarization test (not available on CPU)${NC}\n"
fi

# 7. Error handling tests
echo -e "${YELLOW}7. Error Handling Tests${NC}"

test_endpoint "Invalid model name" \
    "curl -s -X POST '$API_URL/v1/audio/transcriptions' \
        -H 'Authorization: Bearer $API_KEY' \
        -F 'file=@$TEST_FILE' \
        -F 'model=invalid-model' \
        -w '\n%{http_code}' | tail -n1 | grep -q '400'" || ((FAILED++))

test_endpoint "Missing file" \
    "curl -s -X POST '$API_URL/v1/audio/transcriptions' \
        -H 'Authorization: Bearer $API_KEY' \
        -F 'model=whisper-1' \
        -w '\n%{http_code}' | tail -n1 | grep -q '422'" || ((FAILED++))

# Summary
echo -e "${BLUE}==============================${NC}"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ $FAILED tests failed${NC}"
    exit 1
fi