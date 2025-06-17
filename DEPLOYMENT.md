# Deployment Instructions for Whisper Large-v3

## Performance Profiles

The server supports three performance profiles via the `model` parameter:
- `whisper-1`: Balanced performance (default)
- `whisper-1-fast`: 2-3x faster transcription
- `whisper-1-quality`: Maximum accuracy (2x slower)

## CPU Deployment (Simple)

### 1. Setup

```bash
# Clone or create the project
mkdir whisper-large-v3-deployment
cd whisper-large-v3-deployment

# Create the required files
- Dockerfile.cpu-large-v3
- app_optimized.py
- docker-compose.cpu-large-v3.yml
```

### 2. Build and Run

```bash
# Build the CPU image
docker-compose -f docker-compose.cpu-large-v3.yml build

# Start the service
docker-compose -f docker-compose.cpu-large-v3.yml up -d

# Check logs (first start will download 3GB model)
docker-compose -f docker-compose.cpu-large-v3.yml logs -f

# Test the service
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer your-secret-key" \
  -F "file=@test.mp3" \
  -F "model=whisper-1" \
  -F "response_format=json"
```

### 3. Performance on CPU

Expected performance with large-v3 on CPU:
- **16-core CPU**: ~1-2x realtime (30-60 seconds for 1 minute audio)
- **32-core CPU**: ~2-4x realtime
- **With int8**: 2x faster than float32

## GPU Deployment on KServe (NVIDIA L40S)

### 1. Build GPU Image

```bash
# Build the GPU-optimized image
docker build -f Dockerfile.gpu-l40s -t whisper-large-v3:gpu-l40s .

# Tag for your registry
docker tag whisper-large-v3:gpu-l40s your-registry.com/whisper-large-v3:gpu-l40s

# Push to registry
docker push your-registry.com/whisper-large-v3:gpu-l40s
```

### 2. Update KServe Deployment

```bash
# Edit kserve-l40s-deployment.yaml
# Update the image: your-registry.com/whisper-large-v3:gpu-l40s
# Update API_KEYS in ConfigMap

# Apply to Kubernetes
kubectl apply -f kserve-l40s-deployment.yaml

# Check deployment
kubectl get inferenceservice whisper-large-v3-l40s
kubectl get pods -l serving.kserve.io/inferenceservice=whisper-large-v3-l40s

# Get service endpoint
kubectl get svc whisper-large-v3-service
```

### 3. Performance on L40S GPU

Expected performance with large-v3 on NVIDIA L40S:
- **Transcription speed**: 15-25x realtime
- **Memory usage**: ~8GB VRAM
- **Concurrent requests**: 2-3 per GPU
- **First inference**: ~10-15 seconds (model loading)
- **Subsequent**: ~2-4 seconds for 1 minute audio

### 4. Test KServe Deployment

```bash
# Port forward for testing
kubectl port-forward svc/whisper-large-v3-service 8000:80

# Test transcription
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer your-production-key-1" \
  -F "file=@test.mp3" \
  -F "model=whisper-1" \
  -F "response_format=json"
```

## Optimization Tips

### CPU Optimization
- Use int8 compute type (2x faster)
- Set OMP_NUM_THREADS to your CPU core count
- Use BEAM_SIZE=1 for maximum speed
- Enable aggressive VAD settings

### GPU Optimization (L40S)
- Use float16 for best quality/speed balance
- BEAM_SIZE=5 gives good results
- Enable CUDNN_BENCHMARK=1
- Use persistent volume for model cache

### Memory Requirements
- **Model size**: ~3GB download, ~6GB in memory
- **CPU RAM**: Minimum 8GB, recommended 16GB
- **GPU VRAM**: Minimum 8GB, recommended 16GB

## Monitoring

### Metrics to Track
```bash
# GPU utilization
nvidia-smi dmon -s u

# Memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

# Response times
# Add to your monitoring: p50, p95, p99 latencies
```

### Health Checks
Both deployments include health checks:
- Endpoint: `GET /`
- Expected: 200 OK with model info
- Timeout: 10s
- Start period: 180s (for model download)

## Troubleshooting

### Common Issues

1. **Slow first request**
   - Model loading takes 10-15s
   - Solution: Use readiness probe, pre-warm the model

2. **Out of memory**
   - Large-v3 needs significant memory
   - Solution: Ensure 16GB+ RAM/VRAM available

3. **Poor CPU performance**
   - int8 not enabled or wrong thread count
   - Solution: Check COMPUTE_TYPE=int8 and OMP_NUM_THREADS

4. **GPU not utilized**
   - Wrong CUDA version or driver issues
   - Solution: Check nvidia-smi, ensure CUDA 12.1+

## Production Checklist

- [ ] Set strong API_KEYS
- [ ] Configure proper resource limits
- [ ] Set up monitoring/alerting
- [ ] Test autoscaling under load
- [ ] Configure backup/DR strategy
- [ ] Set up log aggregation
- [ ] Test health checks
- [ ] Document API endpoints
- [ ] Load test before go-live
