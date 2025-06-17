# Faster Whisper OpenAI-Compatible Server

A high-performance, OpenAI-compatible Whisper API server using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with Docker support and API key authentication. **GPU-ready out of the box!**

## Features

- **OpenAI API Compatibility**: Drop-in replacement for OpenAI's Whisper API
- **High Performance**: Uses CTranslate2 for 4x faster inference than OpenAI's Whisper
- **GPU-Ready**: NVIDIA GPU support enabled by default for 10x+ speedup
- **API Key Authentication**: Secure your API with multiple API keys
- **Multiple Response Formats**: JSON, text, SRT, VTT, and verbose JSON
- **Docker Support**: Easy deployment with Docker and docker-compose
- **Voice Activity Detection**: Improved accuracy with VAD filtering

## Quick Start

### Prerequisites

For GPU support (recommended):
- NVIDIA GPU with CUDA 11.8+ support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Using Docker Compose

**For GPU (default):**
```bash
docker-compose up -d
```

**For CPU-only:**
```bash
docker-compose -f docker-compose.cpu.yml up -d
```

### Building Manually

```bash
# Build the GPU-ready Docker image
docker build -t faster-whisper-server .

# Run with GPU support
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -e MODEL_SIZE=base \
  -e API_KEYS=your-secret-key \
  faster-whisper-server

# Or run CPU-only version
docker build -f Dockerfile.cpu -t faster-whisper-server:cpu .
docker run -d \
  -p 8000:8000 \
  -e MODEL_SIZE=base \
  -e DEVICE=cpu \
  -e COMPUTE_TYPE=int8 \
  -e API_KEYS=your-secret-key \
  faster-whisper-server:cpu
```

## Configuration

Environment variables:

- `MODEL_SIZE`: Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`)
- `DEVICE`: Device to use (`cuda` for GPU, `cpu` for CPU-only, `auto` for automatic)
- `COMPUTE_TYPE`: 
  - For GPU: `float16` (default), `int8_float16`, `int8`
  - For CPU: `float32`, `int8` (recommended for CPU)
- `API_KEYS`: Comma-separated list of valid API keys (leave empty for no auth)

## API Usage

### Transcription

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer your-secret-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=json"
```

### Translation (to English)

```bash
curl -X POST "http://localhost:8000/v1/audio/translations" \
  -H "Authorization: Bearer your-secret-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1"
```

### Python Client Example

```python
import openai

# Point to your local server
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "your-secret-key"

# Transcribe audio
with open("audio.mp3", "rb") as audio_file:
    transcript = openai.Audio.transcribe(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )
    print(transcript)
```

## Response Formats

- `json` (default): Simple JSON with text
- `verbose_json`: Detailed JSON with segments and metadata
- `text`: Plain text transcript
- `srt`: SubRip subtitle format
- `vtt`: WebVTT subtitle format

## Performance Tips

1. **GPU is 10x+ faster**: The image is GPU-ready by default
2. **Choose appropriate model**: `base` offers good balance of speed/accuracy
3. **Enable VAD**: Already enabled by default for better accuracy
4. **Use int8 quantization**: 
   - GPU: Set `COMPUTE_TYPE=int8_float16` for faster inference
   - CPU: Set `COMPUTE_TYPE=int8` for better CPU performance

## Model Performance Comparison

| Model | Parameters | GPU Speed | CPU Speed | VRAM Usage |
|-------|------------|-----------|-----------|------------|
| tiny  | 39M        | ~32x      | ~10x      | ~1GB       |
| base  | 74M        | ~16x      | ~7x       | ~1GB       |
| small | 244M       | ~12x      | ~4x       | ~2GB       |
| medium| 769M       | ~6x       | ~2x       | ~5GB       |
| large-v3 | 1550M   | ~4x       | ~1x       | ~10GB      |

*Speed relative to real-time on NVIDIA RTX 3090

## GPU Support

The Docker image is **NVIDIA GPU-ready out of the box** using CUDA 11.8.

### Requirements:
1. NVIDIA GPU with CUDA support
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Installation (Ubuntu/Debian):
```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Verify GPU is accessible:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Security

- Always use API keys in production
- Run behind a reverse proxy (nginx, traefik) with HTTPS
- Consider rate limiting for public deployments

## Troubleshooting

**Model download fails**: The first run downloads the model (~150MB for base). Ensure you have internet access.

**Out of memory**: Try a smaller model or reduce `beam_size` in the code.

**Slow on CPU**: Use GPU or try `tiny` model with `int8` compute type.

## License

This server implementation is MIT licensed. Faster-whisper and OpenAI Whisper have their own licenses.
