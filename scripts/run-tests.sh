#!/bin/bash
# Comprehensive test runner for Faster Whisper v2

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🧪 Faster Whisper v2 - Test Runner${NC}"
echo "===================================="

# Check if we should use virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate || . venv/bin/activate
fi

# Function to check if package is installed
check_package() {
    python -c "import $1" 2>/dev/null
}

# 1. Run minimal tests (always works)
echo -e "\n${YELLOW}1. Running minimal tests (no dependencies)${NC}"
python tests/test_minimal.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Minimal tests passed${NC}"
else
    echo -e "${RED}❌ Minimal tests failed${NC}"
fi

# 2. Check if pytest is available
if check_package pytest; then
    echo -e "\n${YELLOW}2. Running unit tests with pytest${NC}"
    
    # Check if all dependencies are installed
    if check_package fastapi && check_package soundfile; then
        pytest tests/test_app.py -v
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Unit tests passed${NC}"
        else
            echo -e "${RED}❌ Unit tests failed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Some test dependencies missing. Running limited tests...${NC}"
        pytest tests/test_minimal.py -v
    fi
    
    # Generate coverage report if possible
    if check_package pytest_cov; then
        echo -e "\n${YELLOW}3. Generating coverage report${NC}"
        pytest tests/ --cov=app --cov-report=term-missing --cov-report=html
        echo -e "${GREEN}✅ Coverage report generated in htmlcov/${NC}"
    fi
else
    echo -e "\n${YELLOW}⚠️  pytest not installed. Install with:${NC}"
    echo "  pip install pytest pytest-asyncio pytest-cov"
fi

# 3. Check if service is running for integration tests
echo -e "\n${YELLOW}4. Checking service status${NC}"
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Service is running${NC}"
    
    # Run API tests
    echo -e "\n${YELLOW}5. Running API integration tests${NC}"
    if [ -x "scripts/test-api.sh" ]; then
        ./scripts/test-api.sh
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ API tests passed${NC}"
        else
            echo -e "${RED}❌ API tests failed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  test-api.sh not found or not executable${NC}"
        echo "  Run: chmod +x scripts/test-api.sh"
    fi
else
    echo -e "${YELLOW}⚠️  Service not running. Skipping integration tests.${NC}"
    echo "  Start with: docker-compose -f docker-compose.cpu.yml up -d"
    
    # Try minimal integration test
    echo -e "\n${YELLOW}Running minimal integration test...${NC}"
    python tests/test_minimal.py --integration
fi

# 4. Code quality checks (if tools available)
echo -e "\n${YELLOW}6. Code quality checks${NC}"

if command -v black > /dev/null 2>&1; then
    echo -n "Running black... "
    if black --check --diff app.py > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC}"
    else
        echo -e "${YELLOW}⚠️  Code formatting issues found${NC}"
