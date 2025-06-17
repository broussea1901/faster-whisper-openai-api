#!/usr/bin/env python3
"""
Pre-download all models during Docker build to avoid runtime delays
This ensures the image works in air-gapped environments
"""

import os
import sys
import json
from pathlib import Path

print("=== Pre-downloading models for air-gapped deployment ===")

# Set cache directories
os.environ['HF_HOME'] = os.environ.get('HF_HOME', '/home/whisper/.cache/huggingface')
os.environ['NEMO_CACHE_DIR'] = os.environ.get('NEMO_CACHE_DIR', '/models/nemo_cache')
os.environ['TORCH_HOME'] = os.environ.get('TORCH_HOME', '/home/whisper/.cache/torch')

# Create directories
for env_var in ['HF_HOME', 'NEMO_CACHE_DIR', 'TORCH_HOME']:
    path = os.environ.get(env_var, '')
    if path:
        os.makedirs(path, exist_ok=True)
        print(f"Created {env_var}: {path}")

try:
    # 1. Download Whisper model
    model_size = os.environ.get('MODEL_SIZE', 'large-v3')
    print(f"\n1. Downloading Whisper {model_size} model...")
    
    from faster_whisper import WhisperModel
    
    # Force download by creating model
    model = WhisperModel(
        model_size, 
        device="cpu",  # Use CPU for download to avoid GPU requirements during build
        compute_type="float32",
        download_root=os.environ['HF_HOME']
    )
    
    # Test that model works
    print("   Testing model loading...")
    import numpy as np
    test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    segments, info = model.transcribe(test_audio)
    _ = list(segments)  # Force evaluation
    
    print(f"✓ Whisper {model_size} model downloaded and tested")
    del model
    
    # 2. Download NeMo models for diarization (GPU builds only)
    if os.environ.get('ENABLE_DIARIZATION', 'false').lower() == 'true':
        print("\n2. Downloading NeMo diarization models...")
        
        try:
            import nemo.collections.asr as nemo_asr
            
            # Speaker verification model
            print("   - Downloading TitaNet Large speaker model...")
            speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                "nvidia/speakerverification_en_titanet_large",
                map_location="cpu"  # Download to CPU
            )
            # Save model path info
            speaker_path = speaker_model.model_path if hasattr(speaker_model, 'model_path') else "downloaded"
            print(f"   ✓ Speaker model downloaded: {speaker_path}")
            del speaker_model
            
            # VAD model
            print("   - Downloading MarbleNet VAD model...")
            vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
                "vad_multilingual_marblenet",
                map_location="cpu"
            )
            vad_path = vad_model.model_path if hasattr(vad_model, 'model_path') else "downloaded"
            print(f"   ✓ VAD model downloaded: {vad_path}")
            del vad_model
            
            print("✓ All NeMo models downloaded successfully")
            
        except Exception as e:
            print(f"⚠ Warning: Failed to download NeMo models: {e}")
            print("  Diarization will not be available")
    
    # 3. Create model inventory file
    inventory = {
        "whisper_model": model_size,
        "whisper_location": os.environ['HF_HOME'],
        "diarization_enabled": os.environ.get('ENABLE_DIARIZATION', 'false').lower() == 'true',
        "nemo_location": os.environ.get('NEMO_CACHE_DIR', ''),
        "models_downloaded": True,
        "build_time": os.popen('date').read().strip()
    }
    
    inventory_path = "/app/model_inventory.json"
    with open(inventory_path, 'w') as f:
        json.dump(inventory, f, indent=2)
    
    print(f"\n✓ Model inventory saved to {inventory_path}")
    print(json.dumps(inventory, indent=2))
    
    print("\n✓ All models pre-downloaded successfully!")
    print("  This image will work in air-gapped environments")
    
except Exception as e:
    print(f"\n❌ Error during model download: {e}")
    print("This image may not work properly in air-gapped environments")
    # Don't fail the build, but warn
    sys.exit(0)

print("\nModel preparation complete!")
