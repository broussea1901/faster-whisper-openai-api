# Deployment Guide - Faster Whisper v2

This guide covers all deployment scenarios for Faster Whisper v2, from local development to production Kubernetes deployments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Deployment](#quick-deployment)
3. [CPU Deployment](#cpu-deployment)
4. [GPU Deployment with Diarization](#gpu-deployment-with-diarization)
5. [Kubernetes/KServe Deployment](#kuberneteskserve-deployment)
6. [High Availability Setup](#high-availability-setup)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Security Considerations](#security-considerations)
9. [Migration from v1](#migration-from-v1)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### General Requirements
- Docker 24.0+ and Docker Compose 2.20+
- 16GB+ RAM
- 20GB+ free disk space for models

### GPU Requirements (for diarization)
- NVIDIA GPU with 16GB+ VRAM (L40S recommended)
- NVIDIA Driver 525+
- NVIDIA Container Toolkit
- CUDA 12.1 compatible GPU

### Kubernetes Requirements
- Kubernetes 1.27+
- KServe 0.11+
- GPU operator (for GPU nodes)

## Quick Deployment

### Using Make (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/faster-whisper-v2.git
cd faster-whisper-v2

# Deploy CPU version
make run-cpu

# Deploy GPU version with diarization
make run-gpu

# Check status
make status
```

### Using Docker Compose

```bash
# CPU deployment (no diarization)
docker-compose -f docker-compose.cpu.yml up -d

# GPU deployment (with diarization)
docker-compose -f docker-compose.gpu.yml up -d
```

## CPU Deployment

### Development Setup

```bash
# Create .env file for development
cat > .env << EOF
API_KEYS=dev-key-123
MODEL_SIZE=large-v3
COMPUTE_TYPE=int8
OMP_NUM_THREADS=8
EOF

# Start with development settings
docker-compose -f docker-compose.cpu.yml up
```

### Production CPU Setup

```bash
# Create production configuration
cat > .env.prod << EOF
API_KEYS=$(openssl rand -base64 32),$(openssl rand -base64 32)
MODEL_SIZE=large-v3
COMPUTE_TYPE=int8
OMP_NUM_THREADS=16
NUM_WORKERS=1
EOF

# Deploy with production settings
docker-compose -f docker-compose.cpu.yml --env-file .env.prod up -d

# Enable restart policy
docker update --restart always whisper-v2-cpu
```

### CPU Performance Optimization

```yaml
# docker-compose.cpu.override.yml
services:
  whisper-v2-cpu:
    cpuset: "0-15"  # Pin to specific CPUs
    environment:
      - OMP_NUM_THREADS=16
      - MKL_NUM_THREADS=16
      - OMP_PROC_BIND=true
      - OMP_PLACES=cores
      - COMPUTE_TYPE=int8  # Critical for CPU performance
```

## GPU Deployment with Diarization

### Single GPU Deployment

```bash
# Build GPU image with integrated diarization
make build-gpu

# Or manually
docker build -f Dockerfile.gpu -t faster-whisper-v2:gpu .

# Run with diarization enabled
docker run -d \
  --name whisper-v2-gpu \
  --gpus '"device=0"' \
  --restart unless-stopped \
  -p 8000:8000 \
  -e API_KEYS=${API_KEYS} \
  -e ENABLE_DIARIZATION=true \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v whisper-models:/home/whisper/.cache \
  -v nemo-models:/models/nemo_cache \
  --shm-size="8gb" \
  faster-whisper-v2:gpu
```

### Multi-GPU Deployment

```yaml
# docker-compose.multi-gpu.yml
services:
  whisper-gpu-0:
    extends:
      file: docker-compose.gpu.yml
      service: whisper-v2-gpu
    container_name: whisper-gpu-0
    environment:
      - CUDA_VISIBLE_DEVICES=0
    ports:
      - "8000:8000"

  whisper-gpu-1:
    extends:
      file: docker-compose.gpu.yml
      service: whisper-v2-gpu
    container_name: whisper-gpu-1
    environment:
      - CUDA_VISIBLE_DEVICES=1
    ports:
      - "8001:8000"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - whisper-gpu-0
      - whisper-gpu-1
```

### GPU Resource Management

```bash
# Monitor GPU usage
nvidia-smi dmon -s u -c 10

# Set GPU compute mode to exclusive
nvidia-smi -c EXCLUSIVE_PROCESS

# Limit GPU memory growth
docker run -d \
  --gpus '"device=0"' \
  --runtime=nvidia \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  faster-whisper-v2:gpu
```

## Kubernetes/KServe Deployment

### 1. Prepare Cluster

```bash
# Install GPU operator
kubectl create -f https://nvidia.github.io/gpu-operator/stable/gpu-operator.yaml

# Install KServe
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve.yaml

# Create namespace
kubectl create namespace whisper
```

### 2. Create Secrets and ConfigMaps

```bash
# API keys secret
kubectl create secret generic whisper-api-keys \
  --from-literal=API_KEYS="prod-key-1,prod-key-2" \
  -n whisper

# Model configuration
kubectl create configmap whisper-config \
  --from-literal=MODEL_SIZE=large-v3 \
  --from-literal=ENABLE_DIARIZATION=true \
  -n whisper
```

### 3. Deploy InferenceService

```bash
# Update registry in kserve-deployment.yaml
sed -i 's|your-registry|your-actual-registry.com|g' kserve-deployment.yaml

# Apply deployment
kubectl apply -f kserve-deployment.yaml -n whisper

# Wait for ready state
kubectl wait --for=condition=Ready \
  inferenceservice/whisper-v2-l40s \
  -n whisper \
  --timeout=600s
```

### 4. Configure Autoscaling

```yaml
# hpa-config.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: whisper-v2-hpa
  namespace: whisper
spec:
  scaleTargetRef:
    apiVersion: serving.kserve.io/v1beta1
    kind: InferenceService
    name: whisper-v2-l40s
  minReplicas: 1
  maxReplicas: 4
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "10"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 300
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 60
```

### 5. Expose Service

```bash
# Internal service
kubectl expose inferenceservice whisper-v2-l40s \
  --name=whisper-internal \
  --port=80 \
  --target-port=8000 \
  -n whisper

# External ingress
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: whisper-ingress
  namespace: whisper
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "1024m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  rules:
  - host: whisper.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: whisper-internal
            port:
              number: 80
EOF
```

## High Availability Setup

### Load Balancer Configuration

```nginx
# nginx.conf for HA setup
upstream whisper_backend {
    least_conn;
    
    # Health check
    check interval=5000 rise=2 fall=3 timeout=4000;
    
    # Backend servers
    server whisper-gpu-0:8000 max_fails=3 fail_timeout=30s;
    server whisper-gpu-1:8000 max_fails=3 fail_timeout=30s;
    server whisper-gpu-2:8000 max_fails=3 fail_timeout=30s backup;
}

server {
    listen 80;
    client_max_body_size 1G;
    client_body_timeout 300s;
    
    location / {
        proxy_pass http://whisper_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        
        # Retry on error
        proxy_next_upstream error timeout http_502 http_503 http_504;
        proxy_next_upstream_tries 3;
    }
    
    location /health {
        access_log off;
        proxy_pass http://whisper_backend/;
    }
}
```

### Database for Job Queue (Optional)

```yaml
# redis-queue.yml
services:
  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  whisper-worker:
    image: faster-whisper-v2:gpu
    command: python worker.py
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
```

## Monitoring and Logging

### Prometheus Metrics

```yaml
# prometheus-config.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'whisper'
    static_configs:
      - targets: 
        - 'whisper-gpu-0:8000'
        - 'whisper-gpu-1:8000'
    metrics_path: '/metrics'
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Whisper v2 Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(http_requests_total[5m])"
        }]
      },
      {
        "title": "GPU Utilization",
        "targets": [{
          "expr": "nvidia_gpu_utilization"
        }]
      },
      {
        "title": "Diarization Enabled",
        "targets": [{
          "expr": "whisper_diarization_enabled"
        }]
      },
      {
        "title": "Response Time by Profile",
        "targets": [{
          "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)"
        }]
      }
    ]
  }
}
```

### Log Aggregation

```bash
# Using Fluentd
docker run -d \
  --name fluentd \
  -v /var/lib/docker/containers:/var/lib/docker/containers:ro \
  -v ./fluent.conf:/fluentd/etc/fluent.conf \
  fluent/fluentd:v1.16-1

# fluent.conf
<source>
  @type forward
  port 24224
</source>

<filter whisper.**>
  @type record_transformer
  <record>
    service "whisper-v2"
    hostname "#{Socket.gethostname}"
  </record>
</filter>

<match whisper.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  logstash_format true
  logstash_prefix whisper
</match>
```

## Security Considerations

### API Key Management

```bash
# Generate secure API keys
generate_api_key() {
  openssl rand -base64 32 | tr -d "=+/" | cut -c1-32
}

# Rotate keys without downtime
NEW_KEY=$(generate_api_key)
OLD_KEYS=$(docker exec whisper-v2-gpu printenv API_KEYS)
docker exec whisper-v2-gpu sh -c "export API_KEYS='${OLD_KEYS},${NEW_KEY}'"

# After client migration
docker exec whisper-v2-gpu sh -c "export API_KEYS='${NEW_KEY}'"
```

### TLS Termination

```yaml
# traefik.yml for automatic TLS
providers:
  docker:
    exposedByDefault: false

http:
  routers:
    whisper:
      rule: "Host(`whisper.your-domain.com`)"
      service: whisper
      tls:
        certResolver: letsencrypt
      
  services:
    whisper:
      loadBalancer:
        servers:
          - url: "http://whisper-v2-gpu:8000"

certificatesResolvers:
  letsencrypt:
    acme:
      email: your-email@domain.com
      storage: acme.json
      httpChallenge:
        entryPoint: web
```

### Network Isolation

```bash
# Create isolated network
docker network create --driver bridge --internal whisper-net

# Run with network isolation
docker run -d \
  --network whisper-net \
  --name whisper-v2-gpu \
  faster-whisper-v2:gpu

# Add reverse proxy to bridge networks
docker network connect whisper-net nginx
docker network connect bridge nginx
```

## Migration from v1

### Backup v1 Data

```bash
# Backup model cache
docker run --rm \
  -v whisper-v1-models:/source:ro \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/whisper-v1-models-$(date +%Y%m%d).tar.gz -C /source .

# Export configuration
docker inspect whisper-v1 > whisper-v1-config.json
```

### Migration Steps

```bash
# 1. Deploy v2 alongside v1
docker-compose -f docker-compose.gpu.yml up -d

# 2. Test v2
./scripts/test-api.sh

# 3. Update clients to new endpoint
# Change: http://old-server:8000 → http://new-server:8000

# 4. Monitor both services
docker stats whisper-v1 whisper-v2-gpu

# 5. Gradual traffic shift (if using load balancer)
# Update weights: v1=80%, v2=20% → v1=50%, v2=50% → v1=0%, v2=100%

# 6. Decommission v1
docker stop whisper-v1
docker rm whisper-v1
```

### API Compatibility

v2 is fully backward compatible with v1. New features:
- Performance profiles: Use `model` parameter
- Diarization: Add `timestamp_granularities=speaker`
- All v1 endpoints work unchanged

## Troubleshooting

### Deployment Issues

```bash
# Check container status
docker ps -a | grep whisper

# View detailed logs
docker logs --tail 100 -f whisper-v2-gpu

# Check resource usage
docker stats whisper-v2-gpu

# Verify diarization is working
curl http://localhost:8000/ | jq '.diarization_enabled'

# Test GPU access
docker exec whisper-v2-gpu nvidia-smi
```

### Common Problems and Solutions

1. **Diarization not available**
   ```bash
   # Ensure GPU build
   docker images | grep whisper
   # Should show: faster-whisper-v2:gpu or gpu-diarization tag
   ```

2. **Model download failures**
   ```bash
   # Check disk space
   df -h
   # Clear Docker cache if needed
   docker system prune -a
   ```

3. **GPU out of memory**
   ```bash
   # Check GPU memory
   nvidia-smi
   # Reduce batch size or disable diarization
   -e ENABLE_DIARIZATION=false
   ```

4. **Slow performance**
   ```bash
   # Check if using correct profile
   curl -X POST ... -F "model=whisper-1-fast"
   # Verify compute type
   docker exec whisper-v2-gpu printenv COMPUTE_TYPE
   ```

### Health Check Endpoints

```bash
# Basic health
curl http://localhost:8000/

# Detailed status
curl http://localhost:8000/ | jq

# Test transcription
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer your-key" \
  -F "file=@test.wav" \
  -F "model=whisper-1"
```

## Best Practices

1. **Always use persistent volumes** for model caches
2. **Set resource limits** to prevent OOM
3. **Enable health checks** in production
4. **Use performance profiles** appropriately
5. **Monitor GPU memory** when using diarization
6. **Implement request queuing** for high load
7. **Regular backups** of model caches
8. **API key rotation** schedule
9. **Log aggregation** for troubleshooting
10. **Gradual rollouts** for updates