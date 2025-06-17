# Performance Tuning Guide - Faster Whisper v2

This comprehensive guide covers all aspects of performance optimization for Faster Whisper v2, including the new speaker diarization feature and performance profiles.

## Table of Contents

1. [Quick Optimization Guide](#quick-optimization-guide)
2. [Performance Profiles](#performance-profiles)
3. [Hardware Optimization](#hardware-optimization)
4. [Model Selection](#model-selection)
5. [Diarization Performance](#diarization-performance)
6. [Advanced Tuning](#advanced-tuning)
7. [Benchmarking](#benchmarking)
8. [Monitoring](#monitoring)
9. [Troubleshooting Performance Issues](#troubleshooting-performance-issues)

## Quick Optimization Guide

### Choose the Right Setup

| Use Case | Hardware | Model | Profile | Expected Speed |
|----------|----------|-------|---------|----------------|
| Real-time transcription | GPU | large-v3 | whisper-1-fast | 30-50x realtime |
| Meeting transcription | GPU | large-v3 | whisper-1 + diarization | 12-20x realtime |
| Bulk processing | GPU | large-v3 | whisper-1-quality | 8-15x realtime |
| Low-resource | CPU | large-v3 | whisper-1-fast | 1-3x realtime |

### Quick Commands

```bash
# Fastest transcription (GPU)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1-fast"

# Best quality (GPU)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1-quality" \
  -F "timestamp_granularities=speaker"

# CPU optimized
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1-fast"
```

## Performance Profiles

### Profile Comparison

| Profile | Beam Size | Best Of | Patience | VAD Settings | Speed Multiplier |
|---------|-----------|---------|----------|--------------|------------------|
| `whisper-1-fast` | 1 | 1 | 0.5 | Aggressive | 2-3x |
| `whisper-1` | 5 | 5 | 1.0 | Balanced | 1x (baseline) |
| `whisper-1-quality` | 10 | 10 | 2.0 | Conservative | 0.5x |

### Profile Selection Guide

```python
# Python example for different use cases
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1")

# Live captioning - prioritize speed
result = client.audio.transcriptions.create(
    model="whisper-1-fast",
    file=audio_file,
    response_format="text"
)

# Podcast transcription - balanced
result = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="srt"
)

# Legal/Medical - maximum accuracy
result = client.audio.transcriptions.create(
    model="whisper-1-quality",
    file=audio_file,
    response_format="json",
    timestamp_granularities="word,speaker"  # If GPU
)
```

### Custom Profile Override

While profiles are predefined, you can override settings via environment variables:

```yaml
# docker-compose.override.yml
services:
  whisper-v2-gpu:
    environment:
      # Override default whisper-1 profile settings
      - BEAM_SIZE=3
      - PATIENCE=0.8
      - VAD_THRESHOLD=0.6
```

## Hardware Optimization

### GPU Optimization (NVIDIA L40S)

```yaml
# Optimal L40S configuration
environment:
  - CUDA_VISIBLE_DEVICES=0
  - COMPUTE_TYPE=float16
  - CUDNN_BENCHMARK=1
  - CUDA_LAUNCH_BLOCKING=0
  - TF_ENABLE_CUDNN_TENSOR_OP=1
  - TORCH_CUDA_ARCH_LIST=8.9  # L40S architecture
  
  # Memory optimization
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  - CUDA_MODULE_LOADING=LAZY
```

### Multi-GPU Setup

```bash
# Distribute load across GPUs
# GPU 0: Transcription only
docker run -d \
  --gpus '"device=0"' \
  -e ENABLE_DIARIZATION=false \
  -p 8000:8000 \
  faster-whisper-v2:gpu

# GPU 1: With diarization
docker run -d \
  --gpus '"device=1"' \
  -e ENABLE_DIARIZATION=true \
  -p 8001:8000 \
  faster-whisper-v2:gpu
```

### CPU Optimization

```yaml
# Maximum CPU performance
environment:
  - COMPUTE_TYPE=int8  # Critical: 2x faster than float32
  - OMP_NUM_THREADS=16
  - MKL_NUM_THREADS=16
  - OMP_PROC_BIND=true
  - OMP_PLACES=cores
  - KMP_AFFINITY=granularity=fine,compact,1,0
  - KMP_BLOCKTIME=0
  
  # NUMA optimization
  - OMP_PROC_BIND=spread
  - OMP_PLACES=threads
```

### Memory Optimization

```bash
# GPU memory allocation
docker run -d \
  --gpus all \
  --shm-size="8gb" \
  -e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256" \
  faster-whisper-v2:gpu

# CPU memory optimization
docker run -d \
  --memory="16g" \
  --memory-swap="16g" \
  --memory-swappiness=10 \
  faster-whisper-v2:cpu
```

## Model Selection

### Model Performance Matrix

| Model | Size | GPU VRAM | CPU RAM | Speed (GPU) | Speed (CPU) | Quality |
|-------|------|----------|---------|-------------|-------------|---------|
| tiny | 39M | 1GB | 1GB | 50-100x | 10-20x | ★★☆☆☆ |
| base | 74M | 1GB | 2GB | 40-80x | 7-15x | ★★★☆☆ |
| small | 244M | 2GB | 4GB | 25-50x | 4-8x | ★★★★☆ |
| medium | 769M | 5GB | 8GB | 15-30x | 2-4x | ★★★★☆ |
| large-v3 | 1550M | 10GB | 12GB | 10-25x | 1-2x | ★★★★★ |

### Dynamic Model Loading (Advanced)

```python
# app_dynamic.py modification
MODEL_CONFIGS = {
    "tiny": {"device": "cuda", "compute_type": "int8_float16"},
    "base": {"device": "cuda", "compute_type": "int8_float16"},
    "small": {"device": "cuda", "compute_type": "float16"},
    "medium": {"device": "cuda", "compute_type": "float16"},
    "large-v3": {"device": "cuda", "compute_type": "float16"}
}

# Load model based on available memory
def get_optimal_model():
    import torch
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory
        if vram > 16 * 1024**3:  # 16GB+
            return "large-v3"
        elif vram > 8 * 1024**3:  # 8GB+
            return "medium"
        else:
            return "base"
    return "base"  # CPU default
```

## Diarization Performance

### Diarization Impact

| Configuration | Speed | Memory | Accuracy |
|--------------|-------|---------|----------|
| No diarization | 15-25x | 10GB | N/A |
| With diarization | 12-20x | 18GB | 85-95% |
| Diarization + quality mode | 6-10x | 20GB | 90-98% |

### Optimizing Diarization

```yaml
# Diarization-specific settings
environment:
  # Reduce speaker search space
  - DIARIZATION_MAX_SPEAKERS=4  # Default: 8
  
  # Adjust clustering threshold
  - DIARIZATION_THRESHOLD=0.7  # Higher = fewer speakers detected
  
  # Batch processing
  - DIARIZATION_BATCH_SIZE=32  # L40S can handle large batches
```

### Conditional Diarization

```python
# Only enable diarization for longer audio
async def smart_transcribe(audio_data, sample_rate):
    duration = len(audio_data) / sample_rate
    
    # Skip diarization for short clips
    enable_diarization = duration > 30  # seconds
    
    return await transcribe_audio(
        audio_data,
        sample_rate,
        enable_diarization=enable_diarization
    )
```

## Advanced Tuning

### Request Batching

```python
# Batch multiple requests for efficiency
from asyncio import Queue, gather

class BatchProcessor:
    def __init__(self, batch_size=4, timeout=1.0):
        self.queue = Queue()
        self.batch_size = batch_size
        self.timeout = timeout
    
    async def process_batch(self, requests):
        # Process multiple files in parallel
        tasks = [
            transcribe_audio(req['audio'], req['sample_rate'])
            for req in requests
        ]
        return await gather(*tasks)
```

### Caching Strategy

```python
# Cache frequent transcriptions
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def get_cached_transcription(audio_hash, model, profile):
    # Return cached result if available
    pass

def compute_audio_hash(audio_data):
    return hashlib.md5(audio_data.tobytes()).hexdigest()
```

### Pipeline Optimization

```yaml
# Optimize Docker build for faster startup
# Dockerfile.gpu optimization
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime

# Multi-stage build
FROM python:3.11 AS builder
COPY requirements-gpu.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements-gpu.txt

FROM runtime
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

# Pre-compile Python files
RUN python -m compileall /app
```

## Benchmarking

### Performance Testing Script

```bash
#!/bin/bash
# benchmark.sh

# Test different configurations
for profile in whisper-1-fast whisper-1 whisper-1-quality; do
    echo "Testing $profile..."
    
    # Measure time and memory
    /usr/bin/time -v curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
        -F "file=@benchmark.wav" \
        -F "model=$profile" \
        -o /dev/null 2>&1 | grep -E "Elapsed|Maximum resident"
done

# GPU utilization
nvidia-smi --query-gpu=utilization.gpu,utilization.memory \
    --format=csv,noheader,nounits -l 1 > gpu_usage.log &
```

### Automated Performance Testing

```python
# performance_test.py
import time
import requests
import statistics

def benchmark_profile(profile, audio_file, iterations=10):
    times = []
    
    for _ in range(iterations):
        start = time.time()
        
        response = requests.post(
            "http://localhost:8000/v1/audio/transcriptions",
            files={"file": open(audio_file, "rb")},
            data={"model": profile}
        )
        
        times.append(time.time() - start)
    
    return {
        "profile": profile,
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times),
        "min": min(times),
        "max": max(times)
    }

# Run benchmarks
for profile in ["whisper-1-fast", "whisper-1", "whisper-1-quality"]:
    results = benchmark_profile(profile, "test.wav")
    print(f"{profile}: {results['mean']:.2f}s (±{results['stdev']:.2f}s)")
```

## Monitoring

### Key Metrics to Track

```python
# Add to app.py for Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Metrics
transcription_duration = Histogram(
    'transcription_duration_seconds',
    'Time spent in transcription',
    ['model', 'profile', 'with_diarization']
)

gpu_memory_usage = Gauge(
    'gpu_memory_usage_bytes',
    'Current GPU memory usage'
)

active_requests = Gauge(
    'active_transcription_requests',
    'Number of active transcription requests'
)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Grafana Dashboard Queries

```promql
# Average response time by profile
avg(rate(transcription_duration_seconds_sum[5m])) by (profile) 
/ avg(rate(transcription_duration_seconds_count[5m])) by (profile)

# GPU utilization
avg(gpu_memory_usage_bytes) / (16 * 1024 * 1024 * 1024) * 100

# Requests per second by profile
sum(rate(transcription_duration_seconds_count[1m])) by (profile)

# P95 latency
histogram_quantile(0.95, 
  sum(rate(transcription_duration_seconds_bucket[5m])) by (le, profile)
)
```

## Troubleshooting Performance Issues

### Common Issues and Solutions

1. **Slow First Request**
   ```bash
   # Pre-warm the model
   curl http://localhost:8000/ > /dev/null
   
   # Or in Docker startup
   CMD ["sh", "-c", "python -c 'import app; print(\"Model loaded\")' && uvicorn app:app --host 0.0.0.0"]
   ```

2. **Memory Leaks**
   ```python
   # Add memory profiling
   import tracemalloc
   tracemalloc.start()
   
   # In your endpoint
   current, peak = tracemalloc.get_traced_memory()
   print(f"Current memory usage: {current / 10**6:.1f} MB")
   ```

3. **GPU Underutilization**
   ```bash
   # Check GPU usage
   nvidia-smi dmon -s u -c 10
   
   # Increase batch size or concurrent requests
   -e NUM_WORKERS=2
   ```

4. **CPU Throttling**
   ```bash
   # Check CPU frequency
   watch -n 1 "grep MHz /proc/cpuinfo"
   
   # Set performance governor
   sudo cpupower frequency-set -g performance
   ```

### Performance Debugging Checklist

- [ ] Verify correct compute type (int8 for CPU, float16 for GPU)
- [ ] Check model is loaded on correct device
- [ ] Monitor GPU/CPU utilization during requests
- [ ] Verify no thermal throttling
- [ ] Check Docker resource limits
- [ ] Review VAD settings for audio type
- [ ] Confirm profile selection matches use case
- [ ] Validate audio preprocessing isn't bottleneck
- [ ] Check network latency for remote requests
- [ ] Review concurrent request handling

## Best Practices Summary

1. **Always use performance profiles** - Don't rely on defaults
2. **Match hardware to workload** - Use GPU for diarization
3. **Monitor continuously** - Set up metrics and alerts
4. **Batch when possible** - Group similar requests
5. **Cache strategically** - For repeated content
6. **Pre-warm models** - Avoid cold start penalties
7. **Use appropriate formats** - Text is faster than JSON
8. **Optimize audio input** - Correct sample rate, mono
9. **Scale horizontally** - Multiple containers > one large
10. **Test with real data** - Synthetic benchmarks can mislead