# Air-gapped Deployment Guide for Faster Whisper v2

This guide explains how to deploy Faster Whisper v2 in air-gapped environments where the runtime environment has no internet access.

## Overview

The solution pre-downloads all required models during the Docker image build phase, ensuring the service can start and run without any internet connectivity.

## Key Features for Air-gapped Deployment

1. **Pre-downloaded Models**: All models are included in the Docker image
2. **Offline Mode**: Environment variables prevent any download attempts
3. **Model Inventory**: JSON file documenting what's included in the image
4. **No External Dependencies**: Everything needed is bundled

## Building Images for Air-gapped Deployment

### 1. Build CPU Image (No Diarization)

```bash
# Build with pre-downloaded models
docker build -f Dockerfile.cpu -t faster-whisper-v2:cpu-airgapped .

# Verify model is included (about 3GB larger than base)
docker images | grep faster-whisper-v2
```

### 2. Build GPU Image (With Diarization)

```bash
# Build with all models pre-downloaded
docker build -f Dockerfile.gpu -t faster-whisper-v2:gpu-airgapped .

# This image includes:
# - Whisper large-v3 model (~1.5GB)
# - NeMo speaker verification model (~100MB)
# - NeMo VAD model (~50MB)
```

### 3. Verify Offline Readiness

```bash
# Test that image works without internet
docker run --rm \
  --network none \
  -p 8000:8000 \
  faster-whisper-v2:cpu-airgapped

# Check model inventory
docker run --rm faster-whisper-v2:cpu-airgapped \
  cat /app/model_inventory.json
```

## Deployment in Air-gapped Environment

### 1. Export and Transfer Image

```bash
# On build machine (with internet)
docker save faster-whisper-v2:gpu-airgapped | gzip > whisper-v2-airgapped.tar.gz

# Transfer to air-gapped environment via:
# - USB drive
# - Internal registry
# - Secure file transfer

# On deployment machine (air-gapped)
docker load < whisper-v2-airgapped.tar.gz
```

### 2. Run in Air-gapped Mode

```bash
docker run -d \
  --name whisper-v2 \
  --restart unless-stopped \
  -p 8000:8000 \
  -e API_KEYS=${API_KEYS} \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  faster-whisper-v2:gpu-airgapped
```

### 3. Kubernetes/KServe Deployment

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: whisper-v2-airgapped
spec:
  predictor:
    containers:
    - name: whisper-server
      image: your-registry/faster-whisper-v2:gpu-airgapped
      env:
      - name: HF_HUB_OFFLINE
        value: "1"
      - name: TRANSFORMERS_OFFLINE
        value: "1"
      - name: MODEL_SIZE
        value: "large-v3"
      resources:
        requests:
          nvidia.com/gpu: "1"
```

## Alternative: S3 Model Loading

If you prefer to load models from S3 instead of bundling them:

### 1. Modified Startup Script

```python
# s3_model_loader.py
import os
import boto3
from pathlib import Path

def download_models_from_s3():
    """Download models from S3 if not present"""
    s3 = boto3.client('s3',
        endpoint_url=os.getenv('S3_ENDPOINT'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    
    bucket = os.getenv('MODEL_BUCKET', 'whisper-models')
    cache_dir = Path('/home/whisper/.cache/huggingface/hub')
    
    # Download Whisper model
    model_key = f"models/whisper-{MODEL_SIZE}.tar.gz"
    local_path = cache_dir / "whisper-model.tar.gz"
    
    if not local_path.exists():
        print(f"Downloading model from s3://{bucket}/{model_key}")
        s3.download_file(bucket, model_key, str(local_path))
        
        # Extract
        import tarfile
        with tarfile.open(local_path) as tar:
            tar.extractall(cache_dir)
    
    print("Models ready from S3")

# Call before loading model
if os.getenv('USE_S3_MODELS', 'false').lower() == 'true':
    download_models_from_s3()
```

### 2. Docker Compose with S3

```yaml
services:
  whisper-v2:
    image: faster-whisper-v2:gpu
    environment:
      - USE_S3_MODELS=true
      - S3_ENDPOINT=http://minio:9000
      - AWS_ACCESS_KEY_ID=minioadmin
      - AWS_SECRET_ACCESS_KEY=minioadmin
      - MODEL_BUCKET=whisper-models
    volumes:
      - model-cache:/home/whisper/.cache
```

## Model Sizes and Planning

| Component | Size | Required For |
|-----------|------|--------------|
| Whisper large-v3 | ~1.5GB | All deployments |
| Whisper medium | ~750MB | Alternative smaller model |
| NeMo TitaNet | ~100MB | GPU with diarization |
| NeMo VAD | ~50MB | GPU with diarization |
| **Total (GPU)** | **~1.7GB** | Full features |
| **Total (CPU)** | **~1.5GB** | No diarization |

## Troubleshooting Air-gapped Deployments

### 1. Check Offline Mode

```bash
# Verify offline environment variables
docker exec whisper-v2 env | grep -E "HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE"

# Check model inventory
docker exec whisper-v2 cat /app/model_inventory.json
```

### 2. Verify Models Are Loaded

```bash
# Check logs for model loading
docker logs whisper-v2 | grep -i "model"

# Should see:
# "Running in OFFLINE mode"
# "Model loaded successfully"
```

### 3. Test Without Network

```bash
# Completely isolate container
docker run --rm \
  --network none \
  --add-host localhost:127.0.0.1 \
  -p 8000:8000 \
  faster-whisper-v2:cpu-airgapped
```

### 4. Common Issues

**Issue**: "Can't download model"
- **Solution**: Ensure HF_HUB_OFFLINE=1 is set
- **Check**: Model was properly downloaded during build

**Issue**: Image size is same as online version
- **Solution**: Models weren't downloaded during build
- **Check**: Build logs for "Model downloaded successfully"

**Issue**: Diarization fails in air-gapped mode
- **Solution**: Use GPU image with pre-downloaded NeMo models
- **Check**: model_inventory.json shows diarization_enabled: true

## Best Practices

1. **Version Control**: Tag images with model versions
   ```bash
   docker tag whisper-v2:gpu whisper-v2:gpu-large-v3-airgapped
   ```

2. **Size Optimization**: Use multi-stage builds (already implemented)

3. **Testing**: Always test in isolated network before deployment
   ```bash
   docker network create --internal isolated
   docker run --network isolated whisper-v2:gpu-airgapped
   ```

4. **Documentation**: Include model_inventory.json in your deployment docs

5. **Updates**: Plan for model updates by rebuilding images periodically

## Summary

The air-gapped deployment strategy ensures:
- ✅ No runtime downloads required
- ✅ All models included in image
- ✅ Offline mode prevents download attempts
- ✅ Works in completely isolated networks
- ✅ Option to use S3 for model distribution

This approach trades larger image sizes for complete offline capability, perfect for secure or isolated environments.