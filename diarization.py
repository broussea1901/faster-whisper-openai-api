# diarization.py - Speaker diarization using NVIDIA NeMo (Apache 2.0)

import os
import json
import tempfile
from typing import List, Dict, Optional
import torch

try:
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    print("NeMo not available - diarization disabled")

class NeMoDiarizer:
    """Speaker diarization using NVIDIA NeMo (fully open source)"""
    
    def __init__(self, device='cuda', cache_dir='/models/nemo_cache'):
        if not NEMO_AVAILABLE:
            raise ImportError("NeMo is required for diarization")
            
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load models
        print("Loading NeMo diarization models...")
        self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        )
        self.vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
            "vad_multilingual_marblenet"
        )
        print("Diarization models loaded")
    
    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> List[Dict]:
        """Run speaker diarization"""
        
        config = self._create_config(audio_path, num_speakers)
        
        from nemo.collections.asr.models import ClusteringDiarizer
        diarizer = ClusteringDiarizer(cfg=config)
        
        # Use pre-loaded models
        diarizer._speaker_model = self.speaker_model
        diarizer._vad_model = self.vad_model
        
        # Run diarization
        diarizer.diarize()
        
        # Parse results
        rttm_path = os.path.join(
            config.diarizer.out_dir,
            'pred_rttms',
            os.path.basename(audio_path).replace('.wav', '.rttm')
        )
        
        return self._parse_rttm(rttm_path)
    
    def _create_config(self, audio_path: str, num_speakers: Optional[int]) -> OmegaConf:
        """Create diarization config"""
        
        temp_dir = tempfile.mkdtemp()
        manifest_path = os.path.join(temp_dir, "manifest.json")
        
        # Create manifest
        with open(manifest_path, 'w') as f:
            json.dump({
                'audio_filepath': audio_path,
                'duration': self._get_duration(audio_path),
                'label': 'infer',
                'text': '-'
            }, f)
        
        config = {
            'diarizer': {
                'manifest_filepath': manifest_path,
                'out_dir': temp_dir,
                'speaker_embeddings': {
                    'model_path': None,
                    'parameters': {
                        'window_length_in_sec': [1.5, 1.0, 0.5],
                        'shift_length_in_sec': [0.75, 0.5, 0.25],
                        'multiscale_weights': [1, 1, 1],
                    }
                },
                'clustering': {
                    'parameters': {
                        'oracle_num_speakers': num_speakers is not None,
                        'num_speakers': num_speakers,
                        'max_num_speakers': 8,
                    }
                },
                'vad': {
                    'model_path': None,
                    'parameters': {
                        'window_length_in_sec': 0.15,
                        'shift_length_in_sec': 0.01,
                    }
                }
            }
        }
        
        return OmegaConf.create(config)
    
    def _parse_rttm(self, rttm_path: str) -> List[Dict]:
        """Parse RTTM file"""
        segments = []
        
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] == 'SPEAKER':
                    segments.append({
                        'start': float(parts[3]),
                        'end': float(parts[3]) + float(parts[4]),
                        'speaker': parts[7]
                    })
        
        return sorted(segments, key=lambda x: x['start'])
    
    def _get_duration(self, audio_path: str) -> float:
        """Get audio duration"""
        import soundfile as sf
        return sf.info(audio_path).duration