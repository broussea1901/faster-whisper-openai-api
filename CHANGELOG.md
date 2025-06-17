# Changelog

## [v2.0.0] - 2024-01-XX

### 🎉 Major Features

#### Speaker Diarization (GPU Only)
- Integrated NVIDIA NeMo for speaker identification
- Automatic speaker labeling in transcriptions
- Support for multi-speaker audio files
- Compatible with SRT/VTT subtitle formats

#### Performance Profiles
- **whisper-1-fast**: 2-3x faster transcription for real-time needs
- **whisper-1**: Balanced performance (default)
- **whisper-1-quality**: Maximum accuracy for critical applications

#### Air-gapped Deployment
- Pre-download models during Docker build
- Offline mode support with HF_HUB_OFFLINE
- Model inventory tracking
- Works in completely isolated environments

### 🚀 Improvements

#### API Enhancements
- Full OpenAI Whisper API compatibility
- Better error handling and messages
- Response format options (JSON, text, SRT, VTT)
- Timestamp formatting with speaker labels

#### Infrastructure
- Multi-stage Docker builds for smaller images
- Separate CPU and GPU optimized images
- KServe/Kubernetes deployment support
- Docker Compose configurations

#### Testing & Quality
- Comprehensive test suite (unit, integration, API)
- CI/CD pipeline with GitHub Actions
- Code coverage reporting
- Performance benchmarking

### 📚 Documentation
- Complete deployment guide
- Performance tuning guide
- Air-gapped deployment guide
- API documentation
- Troubleshooting guides

### 🔧 Technical Details

#### Dependencies
- faster-whisper 1.0.3
- FastAPI 0.115.0
- NVIDIA NeMo 1.23.0 (GPU builds)
- Python 3.11

#### Supported Models
- All Whisper models (tiny to large-v3)
- Automatic model selection based on available resources

### 💔 Breaking Changes
- Python 3.11 now required (was 3.8+)
- New response format for segments with speaker information
- Environment variable changes for configuration

### 🐛 Bug Fixes
- Fixed memory leaks in long-running deployments
- Improved error handling for corrupted audio files
- Better GPU memory management

### 📝 Migration Guide

From v1 to v2:
1. Update Docker images to v2 tags
2. No API changes required for basic transcription
3. Enable diarization with `timestamp_granularities=speaker`
4. Use performance profiles via `model` parameter

---

## [v1.0.0] - Previous Release
- Initial release with basic transcription
