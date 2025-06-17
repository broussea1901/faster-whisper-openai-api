# Performance Tuning Guide for Faster Whisper Server

## Overview

This guide covers all performance-related parameters and optimization strategies for the Faster Whisper OpenAI-compatible server.

## Table of Contents

1. [Model Parameters](#model-parameters)
2. [Inference Parameters](#inference-parameters)
3. [VAD Parameters](#vad-parameters)
4. [Hardware Optimization](#hardware-optimization)
5. [Docker Performance Settings](#docker-performance-settings)
6. [Recommended Configurations](#recommended-configurations)

## Model Parameters

### Basic Model Settings

| Parameter | Default | Description | Performance Impact |
|-----------|---------|-------------|-------------------|
| `MODEL_SIZE` | `base` | Model size (tiny to large-v3) | Larger = slower but more accurate |
| `DEVICE` | `cuda` | Computing device | GPU is 5-10x faster |
| `COMPUTE_TYPE` | `float16` | Quantization type | int8 is ~2x faster with slight quality loss |

### Advanced Model Settings

```yaml
environment:
  # For maximum speed (with quality tradeoff)
  - COMPUTE_TYPE=int8_float16  # 2x faster than float16
  - DEVICE_INDEX=0             # Specific GPU selection
  - CPU_THREADS=8              # For CPU inference
```

## Inference Parameters

### Beam Search Parameters

| Parameter | Default | Range | Description | Speed Impact |
|-----------|---------|-------|-------------|--------------|
| `BEAM_SIZE` | 5 | 1-10 | Number of beams for search | Lower = faster |
| `BEST_OF` | 5 | 1-10 | Number of candidates | Lower = faster |
| `PATIENCE` | 1.0 | 0.0-2.0 | Early stopping patience | Lower = faster |
| `LENGTH_PENALTY` | 1.0 | 0.0-2.0 | Length normalization | Minimal impact |

```yaml
# Speed optimized (2-3x faster)
environment:
  - BEAM_SIZE=1      # Greedy search
  - BEST_OF=1        # Single candidate
  - PATIENCE=0.0     # No patience

# Balanced (default)
environment:
  - BEAM_SIZE=5
  - BEST_OF=5
  - PATIENCE=1.0

# Quality optimized
environment:
  - BEAM_SIZE=10
  - BEST_OF=10
  - PATIENCE=2.0
```

### Temperature and Fallback

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TEMPERATURE_INCREMENT_ON_FALLBACK` | 0.2 | Temperature increase on retry |
| `COMPRESSION_RATIO_THRESHOLD` | 2.4 | Threshold for detecting repetition |
| `LOG_PROB_THRESHOLD` | -1.0 | Minimum average log probability |
| `NO_SPEECH_THRESHOLD` | 0.6 | Threshold for detecting silence |

### Other Performance Parameters

```yaml
environment:
  # Disable for faster processing
  - CONDITION_ON_PREVIOUS_TEXT=false  # Don't use context
  - WORD_TIMESTAMPS=false              # Skip word-level timestamps
  
  # Enable for better quality
  - CONDITION_ON_PREVIOUS_TEXT=true   # Use previous context
  - WORD_TIMESTAMPS=true               # Get word-level timing
```

## VAD Parameters

Voice Activity Detection can significantly speed up processing by skipping silent parts.

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|---------|
| `VAD_MIN_SILENCE_MS` | 500 | Minimum silence duration | Lower = more segments |
| `VAD_THRESHOLD` | 0.5 | Speech detection threshold | Higher = fewer false positives |
| `VAD_MIN_SPEECH_MS` | 250 | Minimum speech duration | Lower = more fragments |
| `VAD_SPEECH_PAD_MS` | 400 | Padding around speech | Higher = better context |

```yaml
# Aggressive VAD (faster)
environment:
  - VAD_MIN_SILENCE_MS=2000   # Skip long silences
  - VAD_THRESHOLD=0.7         # High confidence required
  - VAD_MIN_SPEECH_MS=500     # Ignore short sounds

# Conservative VAD (better quality)
environment:
  - VAD_MIN_SILENCE_MS=200    # Keep short pauses
  - VAD_THRESHOLD=0.3         # Low threshold
  - VAD_MIN_SPEECH_MS=100     # Capture everything
```

## Hardware Optimization

### GPU Optimization

```yaml
environment:
  # NVIDIA GPU settings
  - CUDA_VISIBLE_DEVICES=0           # Use first GPU
  - CUDA_LAUNCH_BLOCKING=0           # Async execution
  - CUDNN_BENCHMARK=1                # Enable cuDNN autotuner
  - TF_ENABLE_CUDNN_TENSOR_OP=1      # Use Tensor Cores
  
  # Multi-GPU (data parallel)
  - CUDA_VISIBLE_DEVICES=0,1         # Use multiple GPUs
```

### CPU Optimization

```yaml
environment:
  # CPU threading
  - OMP_NUM_THREADS=8                # OpenMP threads
  - MKL_NUM_THREADS=8                # Intel MKL threads
  - NUMEXPR_NUM_THREADS=8            # NumExpr threads
  - VECLIB_MAXIMUM_THREADS=8         # macOS Accelerate
  
  # NUMA optimization
  - OMP_PROC_BIND=true               # Bind threads to cores
  - OMP_PLACES=cores                 # Thread placement
```

### Memory Optimization

```yaml
deploy:
  resources:
    limits:
      memory: 16G
      cpu: "8"
    reservations:
      memory: 8G
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Docker Performance Settings

### Build-time Optimization

```dockerfile
# Multi-stage build for smaller image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as builder
# Build dependencies here

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
# Copy only runtime files

# Pre-download model during build
RUN python -c "from faster_whisper import WhisperModel; \
    WhisperModel('${MODEL_SIZE}', device='cpu', compute_type='float32')"
```

### Runtime Optimization

```yaml
services:
  faster-whisper:
    # Shared memory for parallel processing
    shm_size: '2gb'
    
    # Disable logging for performance
    logging:
      driver: none
    
    # Host networking for lower latency
    network_mode: host
    
    # Runtime options
    runtime: nvidia
    
    # Scheduling
    deploy:
      placement:
        constraints:
          - node.labels.gpu == true
```

## Recommended Configurations

### Real-time Transcription

```yaml
# Optimize for lowest latency
environment:
  - MODEL_SIZE=tiny
  - DEVICE=cuda
  - COMPUTE_TYPE=int8_float16
  - BEAM_SIZE=1
  - BEST_OF=1
  - VAD_MIN_SILENCE_MS=500
  - NUM_WORKERS=4
```

**Expected Performance**: 50-100x realtime on GPU

### Balanced Quality/Speed

```yaml
# Good accuracy with reasonable speed
environment:
  - MODEL_SIZE=base
  - DEVICE=cuda
  - COMPUTE_TYPE=float16
  - BEAM_SIZE=5
  - BEST_OF=5
  - VAD_MIN_SILENCE_MS=500
```

**Expected Performance**: 30-40x realtime on GPU

### Maximum Quality

```yaml
# Best possible transcription
environment:
  - MODEL_SIZE=large-v3
  - DEVICE=cuda
  - COMPUTE_TYPE=float16
  - BEAM_SIZE=10
  - BEST_OF=10
  - PATIENCE=2.0
  - CONDITION_ON_PREVIOUS_TEXT=true
  - WORD_TIMESTAMPS=true
```

**Expected Performance**: 8-15x realtime on GPU

### CPU Optimization

```yaml
# Best settings for CPU
environment:
  - MODEL_SIZE=base
  - DEVICE=cpu
  - COMPUTE_TYPE=int8
  - BEAM_SIZE=1
  - OMP_NUM_THREADS=8
  - VAD_MIN_SILENCE_MS=1000
```

**Expected Performance**: 5-10x realtime on modern CPU

## Benchmarking

### Performance Testing Script

```bash
#!/bin/bash
# benchmark.sh

echo "Testing transcription performance..."

# Test file (1 minute audio)
TEST_FILE="test_1min.wav"

# Warm up
curl -s -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer your-key" \
  -F "file=@$TEST_FILE" \
  -F "model=whisper-1" > /dev/null

# Benchmark
echo "Running 10 iterations..."
total_time=0
for i in {1..10}; do
  start=$(date +%s.%N)
  curl -s -X POST "http://localhost:8000/v1/audio/transcriptions" \
    -H "Authorization: Bearer your-key" \
    -F "file=@$TEST_FILE" \
    -F "model=whisper-1" > /dev/null
  end=$(date +%s.%N)
  elapsed=$(echo "$end - $start" | bc)
  total_time=$(echo "$total_time + $elapsed" | bc)
  echo "Iteration $i: ${elapsed}s"
done

avg_time=$(echo "scale=2; $total_time / 10" | bc)
rtf=$(echo "scale=2; 60 / $avg_time" | bc)
echo "Average time: ${avg_time}s"
echo "Real-time factor: ${rtf}x"
```

## Monitoring Performance

### Key Metrics to Monitor

1. **Response Time**: Time to process audio
2. **Real-time Factor**: Audio duration / processing time
3. **GPU Utilization**: `nvidia-smi dmon -s u`
4. **Memory Usage**: Model memory footprint
5. **Queue Length**: For production deployments

### Prometheus Metrics

```python
# Add to app.py for monitoring
from prometheus_client import Counter, Histogram, Gauge

transcription_duration = Histogram(
    'transcription_duration_seconds',
    'Time spent processing transcription',
    ['model_size', 'audio_duration']
)

gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'Current GPU utilization'
)
```

## Troubleshooting Performance Issues

### Common Issues

1. **Slow first request**: Model loading - pre-load during startup
2. **Memory spikes**: Large audio files - implement chunking
3. **GPU underutilization**: Increase batch size or parallel requests
4. **High latency**: Check VAD settings and beam size
5. **Quality degradation**: Balance speed optimizations

### Performance Profiling

```python
# Enable profiling
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... transcription code ...
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats()
```
