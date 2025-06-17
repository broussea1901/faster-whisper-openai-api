# Faster Whisper v2.0.0 Release Notes

We're excited to announce Faster Whisper v2, a major update that brings speaker diarization, performance profiles, and air-gapped deployment support!

## ✨ Highlights

### 🎙️ Speaker Diarization (GPU Only)
Automatically identify who's speaking when in your audio files:
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@meeting.mp3" \
  -F "model=whisper-1" \
  -F "timestamp_granularities=speaker"
```

### ⚡ Performance Profiles

Choose the right balance of speed vs accuracy:

whisper-1-fast: 2-3x faster for real-time needs
whisper-1: Balanced performance (default)
whisper-1-quality: Maximum accuracy

🔒 Air-gapped Deployment
Deploy in secure, offline environments:

Models pre-downloaded during build
No internet required at runtime
Perfect for secure enterprise deployments

🚀 Quick Start

```bash
# CPU deployment
docker-compose -f docker-compose.cpu.yml up -d

# GPU deployment with diarization
docker-compose -f docker-compose.gpu.yml up -d

# Test the API
curl http://localhost:8000/
```

📦 What's Included

Production-ready Docker images (CPU & GPU)
Kubernetes/KServe manifests
Comprehensive test suite
Complete documentation
CI/CD pipeline

🔄 Upgrading from v1
The API remains backward compatible. To use new features:

Pull the new images
Use model=whisper-1-fast for faster transcription
Add timestamp_granularities=speaker for diarization (GPU only)

📚 Documentation

Deployment Guide
Performance Tuning
Air-gapped Deployment

🙏 Acknowledgments
Thanks to all contributors and the teams behind:

faster-whisper
NVIDIA NeMo
OpenAI Whisper

📝 Full Changelog
See CHANGELOG.md for detailed changes.
