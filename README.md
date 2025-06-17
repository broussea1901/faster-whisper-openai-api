# Faster Whisper OpenAI-Compatible Server

A high-performance, OpenAI-compatible Whisper API server using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with Docker support and API key authentication. **GPU-ready out of the box!**

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Deployment Options](#deployment-options)
  - [Local Development](#local-development)
  - [Production with Docker](#production-with-docker)
  - [Kubernetes with KServe](#kubernetes-with-kserve)
- [Model Selection Guide](#model-selection-guide)
- [API Usage](#api-usage)
- [Configuration](#configuration)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Features

- **OpenAI API Compatibility**: Drop-in replacement for OpenAI's Whisper API
- **High Performance**: Uses CTranslate2 for 4x-50x faster inference than OpenAI's Whisper
- **Multiple Models**: Support for all Whisper models from tiny to large-v3
- **GPU-Ready**: NVIDIA GPU support enabled by default for maximum performance
- **API Key Authentication**: Secure your API with multiple API keys
- **Multiple Response Formats**: JSON, text, SRT, VTT, and verbose JSON
- **Production Ready**: Health checks, proper logging, and error handling
- **KServe Compatible**: Ready for deployment on Kubernetes with KServe

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- (Optional) NVIDIA GPU with drivers and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### CPU Deployment (macOS, Windows, Linux without GPU)

```bash
# Start the CPU-optimized version
docker-compose -f docker-compose.cpu.yml up -d

# Check logs
docker-compose -f docker-compose.cpu.yml logs -f

# Test the API
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer your-secret-key" \
  -F "file=@examples/test.wav" \
  -F "model=whisper-1"
```

### GPU Deployment (Linux with NVIDIA GPU)

```bash
# Build the GPU image
docker build -f Dockerfile.gpu -t faster-whisper-server:gpu .

# Run with GPU support
docker run -d \
  --name faster-whisper-api \
  --gpus all \
  -p 8000:8000 \
  -e API_KEYS=your-secret-key \
  -v whisper-cache:/home/whisper/.cache \
  faster-whisper-server:gpu
```

## Deployment Options

### Local Development

Best for testing and development on your local machine.

```bash
# CPU version - uses large-v3 model with int8 optimization
docker-compose -f docker-compose.cpu.yml up -d

# Build both CPU and GPU versions
./scripts/build.sh
```

### Production with Docker

For production deployments with proper resource management.

#### CPU Deployment (large-v3)

```bash
# Build and run CPU-optimized version
docker-compose -f docker-compose.cpu.yml up -d
```

#### GPU Deployment (large-v3)

```bash
# Build GPU-optimized image
docker build -f Dockerfile.gpu -t faster-whisper-server:gpu .

# Run with GPU support
docker run -d \
  --name faster-whisper-api \
  --gpus all \
  -p 8000:8000 \
  -e MODEL_SIZE=large-v3 \
  -e API_KEYS=your-production-key \
  -v whisper-cache:/home/whisper/.cache \
  faster-whisper-server:gpu
```

### Kubernetes with KServe

For scalable deployment on Kubernetes clusters.

1. **Build and push the GPU image:**

```bash
# Build GPU image
docker build -f Dockerfile.gpu -t your-registry/faster-whisper:large-v3 .

# Push to your registry
docker push your-registry/faster-whisper:large-v3
```

2. **Deploy to Kubernetes:**

```bash
# Update image in kserve-deployment.yaml
# Then deploy
kubectl apply -f kserve-deployment.yaml

# Check deployment status
kubectl get inferenceservice whisper-large-v3-l40s

# Get service endpoint
kubectl get svc whisper-large-v3-service
```

## Model Selection Guide

Choose the right model based on your needs:

| Model | Size | Speed (CPU) | Speed (GPU) | Accuracy | Use Case |
|-------|------|-------------|-------------|----------|----------|
| tiny | 39M | ~10x realtime | ~50x realtime | ★★☆☆☆ | Quick drafts, real-time transcription |
| base | 74M | ~7x realtime | ~40x realtime | ★★★☆☆ | Good balance for most applications |
| small | 244M | ~4x realtime | ~25x realtime | ★★★★☆ | Better accuracy, multilingual |
| medium | 769M | ~2x realtime | ~15x realtime | ★★★★☆ | High accuracy, good for podcasts |
| large-v3 | 1550M | ~1x realtime | ~15x realtime | ★★★★★ | Best accuracy, professional use |

### Deployment Recommendations

- **Real-time applications**: Use `tiny` or `base` with GPU
- **Podcasts/Meetings**: Use `small` or `medium`
- **Professional transcription**: Use `large-v3` (default in this setup)
- **Multi-language**: Use `small` or larger

## API Usage

The API is fully compatible with OpenAI's Whisper API.

### Performance Profiles

Choose a model variant based on your needs:

| Model | Description | Speed | Quality | Use Case |
|-------|-------------|-------|---------|----------|
| `whisper-1` | Balanced (default) | 1x | ★★★★☆ | General purpose |
| `whisper-1-fast` | Speed optimized | 2-3x | ★★★☆☆ | Real-time, drafts |
| `whisper-1-quality` | Quality optimized | 0.5x | ★★★★★ | Professional, accuracy-critical |

### Using OpenAI Python Client

```python
from openai import OpenAI

# Point to your local server
client = OpenAI(
    api_key="your-secret-key",
    base_url="http://localhost:8000/v1"
)

# Fast transcription
transcription = client.audio.transcriptions.create(
    model="whisper-1-fast",  # Use fast profile
    file=open("audio.mp3", "rb"),
    response_format="text"
)

# High-quality transcription
transcription = client.audio.transcriptions.create(
    model="whisper-1-quality",  # Use quality profile
    file=open("audio.mp3", "rb"),
    response_format="json"
)
```

### Using cURL

```bash
# Default balanced transcription
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer your-secret-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=json"

# Fast transcription (2-3x faster)
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer your-secret-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1-fast" \
  -F "response_format=text"

# High-quality transcription (2x slower, more accurate)
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer your-secret-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1-quality" \
  -F "response_format=json"
```

### List Available Models

```bash
# See all available models with their performance characteristics
curl -H "Authorization: Bearer your-secret-key" \
  http://localhost:8000/v1/models
```

### Response Formats

- `json` (default): Simple JSON with transcribed text
- `text`: Plain text transcription
- `srt`: SubRip subtitle format
- `vtt`: WebVTT subtitle format  
- `verbose_json`: Detailed JSON with word-level timestamps

## Configuration

### Environment Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `MODEL_SIZE` | Whisper model to use | `large-v3` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `DEVICE` | Compute device | `cuda` (GPU) / `cpu` | `cuda`, `cpu`, `auto` |
| `COMPUTE_TYPE` | Quantization type | `float16` (GPU) / `int8` (CPU) | GPU: `float16`, `int8_float16`<br>CPU: `float32`, `int8` |
| `API_KEYS` | Comma-separated API keys | `""` | Any string values |
| `BEAM_SIZE` | Beam search width | `5` | `1-10` (lower = faster) |
| `PATIENCE` | Early stopping patience | `1.0` | `> 0` |

### Performance Parameters

See [PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md) for detailed performance optimization guide.

## Performance Optimization

### GPU Optimization (NVIDIA L40S optimized)

```yaml
environment:
  - DEVICE=cuda
  - COMPUTE_TYPE=float16
  - BEAM_SIZE=5
  - CUDA_VISIBLE_DEVICES=0
```

### CPU Optimization

```yaml
environment:
  - DEVICE=cpu
  - COMPUTE_TYPE=int8  # 2x faster than float32
  - BEAM_SIZE=1       # Fastest inference
  - OMP_NUM_THREADS=8 # Match your CPU cores
```

### Memory Requirements

- **Model download**: ~3GB for large-v3
- **RAM usage**: 8-12GB for large-v3
- **GPU VRAM**: 8-10GB for large-v3

## Troubleshooting

### Common Issues

**1. Model download is slow**
- First run downloads the model (~3GB for large-v3)
- Models are cached in Docker volume for subsequent runs
- CPU startup: ~30-60 seconds after download

**2. Invalid API key error**
- Default key is `your-secret-key`
- Set custom key: `API_KEYS=mykey docker-compose up -d`
- Disable auth: `API_KEYS="" docker-compose up -d`

**3. GPU not detected**
```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**4. Out of memory errors**
- Use a smaller model
- Enable int8 quantization: `COMPUTE_TYPE=int8`
- Reduce BEAM_SIZE to 1

### Debugging

```bash
# Check container logs
docker-compose -f docker-compose.cpu.yml logs -f

# Check resource usage
docker stats faster-whisper-api

# Test health endpoint
curl http://localhost:8000/

# Run test script
./scripts/test-api.sh
```

## Production Considerations

### Security

1. **Always use API keys in production**
   ```yaml
   environment:
     - API_KEYS=${WHISPER_API_KEYS}
   ```

2. **Run behind a reverse proxy**
   ```nginx
   location /whisper/ {
       proxy_pass http://whisper-api:8000/;
       proxy_set_header Authorization $http_authorization;
   }
   ```

3. **Use HTTPS** - Never expose the API over plain HTTP in production

### Scaling

- **Horizontal scaling**: Run multiple containers behind a load balancer
- **GPU sharing**: Use NVIDIA MPS for multiple containers per GPU
- **Request queuing**: Implement a queue for high-volume deployments

### Monitoring

Recommended metrics to monitor:
- Request latency (p50, p95, p99)
- GPU/CPU utilization
- Memory usage
- Model load time
- Audio processing duration vs. audio length

## License

This server implementation is MIT licensed. Faster-whisper and OpenAI Whisper have their own licenses.
