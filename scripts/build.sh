#!/bin/bash
# Build script for Faster Whisper v2

set -e

echo "🚀 Building Faster Whisper v2 Docker Images"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Parse arguments
BUILD_GPU_ONLY=false
BUILD_CPU_ONLY=false
NO_CACHE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu-only) BUILD_GPU_ONLY=true ;;
        --cpu-only) BUILD_CPU_ONLY=true ;;
        --no-cache) NO_CACHE="--no-cache" ;;
        -h|--help) 
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --gpu-only    Build only GPU image with diarization"
            echo "  --cpu-only    Build only CPU image"
            echo "  --no-cache    Build without using cache"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Build CPU image
if [ "$BUILD_GPU_ONLY" != true ]; then
    echo ""
    echo "📦 Building CPU image..."
    docker build $NO_CACHE -f Dockerfile.cpu -t faster-whisper-v2:cpu -t faster-whisper-v2:cpu-latest .
    if [ $? -eq 0 ]; then
        echo "✅ CPU image built successfully"
    else
        echo "❌ CPU build failed"
        exit 1
    fi
fi

# Build GPU image with diarization
if [ "$BUILD_CPU_ONLY" != true ]; then
    echo ""
    echo "🎮 Building GPU image with integrated diarization..."
    
    # Check if requirements-gpu.txt exists, if not create it
    if [ ! -f "requirements-gpu.txt" ]; then
        echo "Creating requirements-gpu.txt..."
        cat > requirements-gpu.txt << 'EOF'
# Core dependencies
faster-whisper==1.0.3
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.12
pydantic==2.9.2
requests==2.32.3

# Audio processing
soundfile==0.12.1
numpy==1.26.4
librosa==0.10.1

# Diarization dependencies (required for GPU build)
nemo-toolkit[asr]==1.23.0
omegaconf>=2.3.0
pytorch-lightning>=2.0.0
tensorboard>=2.14.0
webdataset>=0.2.48
scipy>=1.10.1
scikit-learn>=1.3.0
EOF
    fi
    
    # Check if download_models.py exists, if not create it
    if [ ! -f "download_models.py" ]; then
        echo "Creating download_models.py..."
        cat > download_models.py << 'EOF'
#!/usr/bin/env python3
import os
import sys

print("=== Pre-downloading models for faster startup ===")

os.environ['HF_HOME'] = '/home/whisper/.cache/huggingface'
os.environ['NEMO_CACHE_DIR'] = '/models/nemo_cache'

try:
    print("\n1. Downloading Whisper large-v3 model...")
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3", device="cpu", compute_type="float32")
    print("✓ Whisper model downloaded")
    del model
    
    print("\n2. Downloading NeMo diarization models...")
    import nemo.collections.asr as nemo_asr
    
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        "nvidia/speakerverification_en_titanet_large"
    )
    print("✓ Speaker model downloaded")
    del speaker_model
    
    vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
        "vad_multilingual_marblenet"
    )
    print("✓ VAD model downloaded")
    del vad_model
    
    print("\n✓ All models pre-downloaded successfully!")
    
except Exception as e:
    print(f"\n⚠ Warning: Failed to pre-download some models: {e}")
    sys.exit(0)
EOF
    fi
    
    docker build $NO_CACHE -f Dockerfile.gpu -t faster-whisper-v2:gpu -t faster-whisper-v2:gpu-diarization .
    if [ $? -eq 0 ]; then
        echo "✅ GPU image with diarization built successfully"
    else
        echo "❌ GPU build failed"
        exit 1
    fi
fi

# Display image info
echo ""
echo "📊 Built images:"
docker images | grep -E "faster-whisper-v2|REPOSITORY" | head -4

echo ""
echo "✨ Build complete!"
echo ""
echo "Next steps:"
if [ "$BUILD_GPU_ONLY" != true ]; then
    echo "  CPU: docker-compose -f docker-compose.cpu.yml up -d"
fi
if [ "$BUILD_CPU_ONLY" != true ]; then
    echo "  GPU: docker-compose -f docker-compose.gpu.yml up -d"
fi
echo ""
echo "The GPU build includes:"
echo "  ✓ Whisper large-v3 model"
echo "  ✓ NVIDIA NeMo diarization"
echo "  ✓ Pre-downloaded models for fast startup"
echo "  ✓ CUDA 12.1 optimization"
echo ""