#!/bin/bash
# Setup script for running tests

set -e

echo "🧪 Setting up Faster Whisper v2 tests"
echo "===================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || . venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install base requirements
if [ -f "requirements.txt" ]; then
    echo "Installing base requirements..."
    pip install -r requirements.txt
else
    echo "⚠️  Warning: requirements.txt not found"
fi

# Create requirements-test.txt if it doesn't exist
if [ ! -f "requirements-test.txt" ]; then
    echo "Creating requirements-test.txt..."
    cat > requirements-test.txt << 'EOF'
# Test requirements
pytest==8.3.3
pytest-asyncio==0.24.0
pytest-cov==5.0.0
pytest-timeout==2.3.1
pytest-mock==3.14.0
httpx==0.27.2
requests==2.32.3

# Include base dependencies needed for tests
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.12
pydantic==2.9.2
soundfile==0.12.1
numpy==1.26.4
EOF
fi

# Install test requirements
echo "Installing test requirements..."
pip install -r requirements-test.txt

# Create test directory if it doesn't exist
mkdir -p tests
mkdir -p examples

# Download test audio file if it doesn't exist
if [ ! -f "examples/test.wav" ]; then
    echo "Downloading test audio file..."
    curl -L -o examples/test.wav "https://github.com/mozilla/DeepSpeech/raw/master/audio/2830-3980-0043.wav" 2>/dev/null || {
        echo "⚠️  Failed to download test audio, creating synthetic file..."
        python3 << 'EOF'
import numpy as np
import soundfile as sf
# Create 3 seconds of test audio
sample_rate = 16000
duration = 3.0
t = np.linspace(0, duration, int(sample_rate * duration))
# Generate speech-like frequencies
audio = 0.3 * np.sin(2 * np.pi * 150 * t)  # Fundamental
audio += 0.2 * np.sin(2 * np.pi * 300 * t)  # Harmonic
audio += 0.1 * np.sin(2 * np.pi * 450 * t)  # Harmonic
# Add some noise
audio += 0.05 * np.random.randn(len(t))
# Save
sf.write('examples/test.wav', audio, sample_rate)
print("✓ Created synthetic test audio")
EOF
    }
fi

echo ""
echo "✅ Test setup complete!"
echo ""
echo "To run tests:"
echo "  1. Unit tests:        pytest tests/ -v"
echo "  2. Coverage report:   pytest tests/ --cov=app --cov-report=html"
echo "  3. API tests:         ./scripts/test-api.sh"
echo ""
echo "Note: Make sure the Whisper service is running for API tests:"
echo "  docker-compose -f docker-compose.cpu.yml up -d"
echo ""
