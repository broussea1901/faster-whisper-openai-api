# Faster Whisper v2 - OpenAI-Compatible API with Speaker Diarization

A high-performance, OpenAI-compatible Whisper API server using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with performance profiles and integrated speaker diarization support.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 🚀 What's New in v2

- **🎙️ Speaker Diarization**: Identify who speaks when (GPU only, using NVIDIA NeMo)
- **⚡ Performance Profiles**: Choose speed vs accuracy via model parameter
- **📊 Enhanced Subtitles**: SRT/VTT with speaker labels
- **🔧 Systematic GPU Build**: Diarization integrated by default in GPU builds
- **🏗️ Better Architecture**: Async processing, lifecycle management
- **📦 Pre-downloaded Models**: Faster container startup

## Features

- ✅ **100% OpenAI API Compatible** - Drop-in replacement
- ✅ **3 Performance Profiles** - Balanced, Fast (2-3x), Quality
- ✅ **Speaker Diarization** - Using NVIDIA NeMo (Apache 2.0, no auth required)
- ✅ **GPU & CPU Support** - Optimized for both
- ✅ **Multiple Formats** - JSON, text, SRT, VTT with speakers
- ✅ **API Key Auth** - Secure your endpoints
- ✅ **Production Ready** - Health checks, logging, KServe support

## Quick Start

### 🖥️ CPU Deployment (No Diarization)

```bash
# Using Docker Compose
docker-compose -f docker-compose.cpu.yml up -d

# Or using Make
make run-cpu

# Test
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1"
```

### 🎮 GPU Deployment (With Diarization)

```bash
# Using Docker Compose (recommended)
docker-compose -f docker-compose.gpu.yml up -d

# Or using Make
make build-gpu
make run-gpu

# Test with diarization
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@meeting.mp3" \
  -F "model=whisper-1" \
  -F "timestamp_granularities=speaker"
```

## Performance Profiles

Choose your performance profile using the `model` parameter:

| Model | Speed | Quality | Beam Size | Use Case |
|-------|-------|---------|-----------|----------|
| `whisper-1` | 1x | ★★★★☆ | 5 | Balanced (default) |
| `whisper-1-fast` | 2-3x | ★★★☆☆ | 1 | Real-time, streaming |
| `whisper-1-quality` | 0.5x | ★★★★★ | 10 | Maximum accuracy |

## API Usage

### Python (OpenAI Client)

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:8000/v1"
)

# Fast transcription
result = client.audio.transcriptions.create(
    model="whisper-1-fast",
    file=open("audio.mp3", "rb")
)

# With speaker diarization (GPU only)
result = client.audio.transcriptions.create(
    model="whisper-1",
    file=open("meeting.mp3", "rb"),
    response_format="json",
    timestamp_granularities="speaker"
)

# Quality mode for best accuracy
result = client.audio.transcriptions.create(
    model="whisper-1-quality",
    file=open("important.mp3", "rb")
)
```

### cURL Examples

```bash
# List available models
curl -H "Authorization: Bearer your-api-key" \
  http://localhost:8000/v1/models

# Transcribe with fast profile
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1-fast" \
  -F "response_format=text"

# Transcribe with speaker diarization (GPU only)
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@meeting.mp3" \
  -F "model=whisper-1" \
  -F "timestamp_granularities=speaker" \
  -F "response_format=json"

# Get subtitles with speakers
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@video.mp4" \
  -F "model=whisper-1" \
  -F "timestamp_granularities=speaker" \
  -F "response_format=srt"

# Translate to English
curl -X POST "http://localhost:8000/v1/audio/translations" \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@foreign.mp3" \
  -F "model=whisper-1-fast"
```

## Speaker Diarization

Available only on GPU deployments. The diarization feature:
- Identifies different speakers in the audio
- Assigns speaker labels (SPEAKER_1, SPEAKER_2, etc.)
- Works with all response formats
- No authentication or licensing restrictions (Apache 2.0)

Example response with diarization:

```json
{
  "text": "Full transcription text...",
  "language": "en",
  "duration": 2.5,
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello everyone, welcome to the meeting.",
      "speaker": "SPEAKER_1"
    },
    {
      "start": 2.5,
      "end": 5.0,
      "text": "Thanks for having me, glad to be here.",
      "speaker": "SPEAKER_2"
    }
  ]
}
```

## Building and Deployment

### Using Make (Recommended)

```bash
# Build everything
make build

# Build only GPU image with diarization
make build-gpu

# Run GPU container
make run-gpu

# Run tests
make test

# Test diarization specifically
make test-diarization

# Check status
make status
```

### Using Docker Compose

```bash
# CPU deployment
docker-compose -f docker-compose.cpu.yml up -d

# GPU deployment with diarization
docker-compose -f docker-compose.gpu.yml up -d

# View logs
docker-compose -f docker-compose.gpu.yml logs -f
```

### Manual Docker Build

```bash
# Build GPU image with integrated diarization
docker build -f Dockerfile.gpu -t faster-whisper-v2:gpu .

# Run with GPU support
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -e API_KEYS=your-key \
  -v whisper-models:/home/whisper/.cache \
  -v nemo-models:/models/nemo_cache \
  faster-whisper-v2:gpu
```

### Kubernetes with KServe

```bash
# Update image in kserve-deployment.yaml
kubectl apply -f kserve-deployment.yaml

# Check status
kubectl get inferenceservice whisper-v2-l40s
```

## Environment Variables

| Variable | Description | Default | GPU | CPU |
|----------|-------------|---------|-----|-----|
| `API_KEYS` | Comma-separated API keys | `""` (no auth) | ✅ | ✅ |
| `MODEL_SIZE` | Whisper model size | `large-v3` | ✅ | ✅ |
| `DEVICE` | Computing device | `cuda`/`cpu` | ✅ | ✅ |
| `COMPUTE_TYPE` | Quantization | `float16`/`int8` | ✅ | ✅ |
| `ENABLE_DIARIZATION` | Speaker detection | `true`/`false` | ✅ | ❌ |
| `NUM_WORKERS` | Async workers | `1` | ✅ | ✅ |
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0` | ✅ | - |

## Performance

### NVIDIA L40S GPU (48GB VRAM)
- **Transcription only**: 15-25x realtime
- **With diarization**: 12-20x realtime
- **Memory usage**: ~12GB (transcription) + 8GB (diarization)
- **Concurrent requests**: 2-3 depending on audio length

### CPU (8 cores, int8)
- **Transcription**: 1-2x realtime
- **No diarization support**
- **Memory usage**: 8-12GB RAM
- **Best with `whisper-1-fast` profile

### Model Load Times
- **First run**: 2-5 minutes (downloading models)
- **Subsequent runs**: 30-60 seconds (CPU), 10-20 seconds (GPU)
- **With pre-downloaded models**: <10 seconds

## Project Structure

```
faster-whisper-v2/
├── app.py                    # Main application
├── diarization.py           # NeMo diarization module
├── requirements.txt         # Base dependencies
├── requirements-gpu.txt     # GPU + diarization dependencies
├── download_models.py       # Model pre-download script
├── Dockerfile.cpu          # CPU build
├── Dockerfile.gpu          # GPU build with diarization
├── docker-compose.cpu.yml  # CPU deployment
├── docker-compose.gpu.yml  # GPU deployment
├── kserve-deployment.yaml  # Kubernetes deployment
├── Makefile               # Build automation
├── LICENSE                # MIT License
├── README.md             # This file
├── DEPLOYMENT.md         # Deployment guide
├── PERFORMANCE_TUNING.md # Performance guide
├── examples/
│   └── test.wav         # Test audio
└── scripts/
    ├── build.sh         # Build script
    └── test-api.sh      # Test script
```

## Requirements

- **Docker**: 24.0+
- **Docker Compose**: 2.20+
- **GPU**: NVIDIA GPU with 16GB+ VRAM (for diarization)
- **NVIDIA Driver**: 525+ (for GPU)
- **NVIDIA Container Toolkit** (for GPU)
- **Memory**: 16GB+ RAM recommended

## Troubleshooting

### GPU/Diarization Issues

```bash
# Check if diarization is enabled
curl http://localhost:8000/ | jq '.diarization_enabled'

# Verify GPU is accessible
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check model downloads
docker logs whisper-v2-gpu | grep "downloaded"
```

### Common Issues

1. **Diarization not working**: Ensure you're using GPU build
2. **Out of memory**: Disable diarization or use smaller model
3. **Slow first request**: Models loading, check logs
4. **API key issues**: Set `API_KEYS=""` to disable auth

## License

MIT License. See [LICENSE](LICENSE) file.

### Third-party Licenses
- faster-whisper: MIT
- NVIDIA NeMo: Apache 2.0
- OpenAI Whisper: MIT

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/faster-whisper-v2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/faster-whisper-v2/discussions)
- **Updates**: Watch releases for updates

---

Made with ❤️ for the speech recognition community