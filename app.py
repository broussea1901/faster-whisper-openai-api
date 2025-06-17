# app.py - Faster Whisper OpenAI-Compatible API v2
# Includes performance profiles and optional speaker diarization

import os
import io
import time
import json
import asyncio
import tempfile
from typing import Optional, List, Dict, Union
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf

# Configuration
API_KEYS = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []
MODEL_SIZE = os.getenv("MODEL_SIZE", "large-v3")
DEVICE = os.getenv("DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") is not None else "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "false").lower() == "true"

# Check if running in offline mode
OFFLINE_MODE = os.getenv("HF_HUB_OFFLINE", "0") == "1"
if OFFLINE_MODE:
    print("Running in OFFLINE mode - no model downloads will be attempted")
    # Check if models exist
    model_inventory_path = "/app/model_inventory.json"
    if os.path.exists(model_inventory_path):
        with open(model_inventory_path, 'r') as f:
            inventory = json.load(f)
            print(f"Model inventory: {json.dumps(inventory, indent=2)}")
    else:
        print("Warning: No model inventory found")

# Performance profiles
PERFORMANCE_PROFILES = {
    "whisper-1": {
        "name": "Balanced",
        "beam_size": 5,
        "best_of": 5,
        "patience": 1.0,
        "vad_threshold": 0.5,
        "vad_min_silence_ms": 500,
    },
    "whisper-1-fast": {
        "name": "Fast",
        "beam_size": 1,
        "best_of": 1,
        "patience": 0.5,
        "vad_threshold": 0.7,
        "vad_min_silence_ms": 2000,
    },
    "whisper-1-quality": {
        "name": "High Quality",
        "beam_size": 10,
        "best_of": 10,
        "patience": 2.0,
        "vad_threshold": 0.3,
        "vad_min_silence_ms": 200,
    }
}

# Initialize diarization if enabled
diarizer = None
if ENABLE_DIARIZATION:
    if DEVICE == "cuda":
        try:
            from diarization import NeMoDiarizer
            print("Initializing NeMo diarization...")
            diarizer = NeMoDiarizer(device=DEVICE)
            print("✓ Speaker diarization enabled (NeMo)")
        except Exception as e:
            print(f"Warning: Failed to initialize diarization: {e}")
            if OFFLINE_MODE:
                print("Note: Running in offline mode - ensure NeMo models were pre-downloaded")
            ENABLE_DIARIZATION = False
    else:
        print("Info: Diarization is only available on GPU")
        ENABLE_DIARIZATION = False

# Global model and executor
whisper_model = None
executor = ThreadPoolExecutor(max_workers=int(os.getenv("NUM_WORKERS", "1")))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle"""
    global whisper_model
    print(f"Loading Whisper model: {MODEL_SIZE} on {DEVICE} with {COMPUTE_TYPE}")
    if OFFLINE_MODE:
        print("Note: Offline mode enabled - using pre-downloaded models only")
    
    try:
        whisper_model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=os.getenv("HF_HOME", "/home/whisper/.cache/huggingface")
        )
        print("Model loaded successfully!")
        
        # Log model info if in offline mode
        if OFFLINE_MODE:
            print(f"Model cache location: {os.getenv('HF_HOME', '/home/whisper/.cache/huggingface')}")
            
    except Exception as e:
        if OFFLINE_MODE:
            print(f"ERROR: Failed to load model in offline mode: {e}")
            print("Ensure models were pre-downloaded during image build")
            raise
        else:
            print(f"Error loading model: {e}")
            raise
    
    yield
    # Cleanup
    executor.shutdown(wait=True)

app = FastAPI(
    title="Faster Whisper API v2",
    description="OpenAI-compatible Whisper API with performance profiles and speaker diarization",
    version="2.0.0",
    lifespan=lifespan
)

security = HTTPBearer()

# Response models
class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration: float
    segments: Optional[List[Dict]] = None

class TranslationResponse(BaseModel):
    text: str
    language: str
    duration: float

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1677532384
    owned_by: str = "openai"
    capabilities: Dict[str, bool] = {}

# Authentication
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not API_KEYS:
        return True
    if credentials.credentials not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

# Audio processing
def convert_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Convert audio bytes to numpy array"""
    try:
        audio_file = io.BytesIO(audio_bytes)
        data, samplerate = sf.read(audio_file)
        
        # Convert to mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Ensure float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # Normalize
        max_val = np.abs(data).max()
        if max_val > 1.0:
            data = data / max_val
        elif 0 < max_val < 0.1:
            data = data / max_val * 0.5
            
        return data, samplerate
    except Exception as e:
        raise ValueError(f"Audio processing failed: {str(e)}")

async def transcribe_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    task: str = "transcribe",
    language: Optional[str] = None,
    profile: str = "whisper-1",
    enable_diarization: bool = False,
    num_speakers: Optional[int] = None
) -> Dict:
    """Transcribe audio with optional diarization"""
    
    # Get performance settings
    perf = PERFORMANCE_PROFILES.get(profile, PERFORMANCE_PROFILES["whisper-1"])
    
    # Check duration for VAD
    duration = len(audio_data) / sample_rate
    use_vad = duration > 1.0
    
    vad_params = {
        "min_silence_duration_ms": perf["vad_min_silence_ms"],
        "threshold": perf["vad_threshold"],
        "min_speech_duration_ms": 250,
        "speech_pad_ms": 400,
    } if use_vad else None
    
    try:
        # Transcribe
        segments, info = whisper_model.transcribe(
            audio_data,
            language=language,
            task=task,
            beam_size=perf["beam_size"],
            best_of=perf["best_of"],
            patience=perf["patience"],
            temperature=0.0,
            vad_filter=use_vad,
            vad_parameters=vad_params,
            word_timestamps=enable_diarization,  # Need words for alignment
            condition_on_previous_text=True,
            suppress_blank=True,
            suppress_tokens=[-1],
        )
    except Exception as e:
        if OFFLINE_MODE and "download" in str(e).lower():
            raise HTTPException(
                status_code=503,
                detail=f"Model not available in offline mode. Ensure models were pre-downloaded during image build. Error: {str(e)}"
            )
        raise
    
    # Process segments
    processed_segments = []
    full_text = ""
    
    # If diarization is enabled and available
    if enable_diarization and diarizer and DEVICE == "cuda":
        # Save audio temporarily
        temp_path = tempfile.mktemp(suffix='.wav')
        sf.write(temp_path, audio_data, sample_rate)
        
        try:
            # Run diarization
            speaker_segments = await asyncio.to_thread(
                diarizer.diarize,
                temp_path,
                num_speakers
            )
            
            # Align with transcription
            for segment in segments:
                seg_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                }
                
                # Find speaker
                for spk_seg in speaker_segments:
                    if (segment.start >= spk_seg["start"] and 
                        segment.start < spk_seg["end"]):
                        seg_dict["speaker"] = spk_seg["speaker"]
                        break
                
                processed_segments.append(seg_dict)
                full_text += segment.text
                
        except Exception as e:
            if OFFLINE_MODE:
                print(f"Diarization failed in offline mode: {e}")
            raise
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    else:
        # Standard processing without diarization
        for segment in segments:
            processed_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
            full_text += segment.text
    
    return {
        "text": full_text.strip(),
        "language": info.language,
        "segments": processed_segments if processed_segments else None
    }

@app.get("/")
async def health():
    """Health check endpoint"""
    health_info = {
        "status": "healthy",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "diarization_enabled": ENABLE_DIARIZATION,
        "available_profiles": list(PERFORMANCE_PROFILES.keys()),
        "offline_mode": OFFLINE_MODE
    }
    
    # Add model inventory info if available
    if OFFLINE_MODE and os.path.exists("/app/model_inventory.json"):
        try:
            with open("/app/model_inventory.json", 'r') as f:
                health_info["model_inventory"] = json.load(f)
        except:
            pass
    
    return health_info

@app.get("/v1/models")
async def list_models(authorized: bool = Depends(verify_api_key)):
    """List available models"""
    models = []
    
    for model_id in PERFORMANCE_PROFILES:
        models.append(ModelInfo(
            id=model_id,
            capabilities={
                "transcription": True,
                "translation": True,
                "diarization": ENABLE_DIARIZATION
            }
        ).dict())
    
    return {"object": "list", "data": models}

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: Optional[str] = Form(None),
    authorized: bool = Depends(verify_api_key)
):
    """Transcribe audio file"""
    start_time = time.time()
    
    # Validate model
    if model not in PERFORMANCE_PROFILES:
        raise HTTPException(400, f"Invalid model: {model}")
    
    # Check for diarization request
    enable_diarization = (
        timestamp_granularities and 
        "speaker" in timestamp_granularities and
        ENABLE_DIARIZATION
    )
    
    try:
        # Read audio
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(400, "Empty audio file")
        
        # Convert audio
        audio_data, sample_rate = convert_audio(audio_bytes)
        
        # Transcribe
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            asyncio.run,
            transcribe_audio(
                audio_data,
                sample_rate,
                "transcribe",
                language,
                model,
                enable_diarization
            )
        )
        
        processing_time = time.time() - start_time
        
        # Format response
        if response_format == "text":
            return result["text"]
        
        elif response_format == "srt":
            srt = ""
            for i, seg in enumerate(result.get("segments", []), 1):
                start = format_timestamp(seg["start"], "srt")
                end = format_timestamp(seg["end"], "srt")
                speaker = f"[{seg.get('speaker', 'SPEAKER')}] " if 'speaker' in seg else ""
                srt += f"{i}\n{start} --> {end}\n{speaker}{seg['text'].strip()}\n\n"
            return srt
        
        elif response_format == "vtt":
            vtt = "WEBVTT\n\n"
            for seg in result.get("segments", []):
                start = format_timestamp(seg["start"], "vtt")
                end = format_timestamp(seg["end"], "vtt")
                speaker = f"<v {seg.get('speaker', 'SPEAKER')}>" if 'speaker' in seg else ""
                vtt += f"{start} --> {end}\n{speaker}{seg['text'].strip()}\n\n"
            return vtt
        
        else:  # json
            return TranscriptionResponse(
                text=result["text"],
                language=result["language"],
                duration=processing_time,
                segments=result.get("segments")
            )
            
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        if OFFLINE_MODE and "model" in str(e).lower():
            raise HTTPException(503, f"Model error in offline mode: {str(e)}")
        print(f"Transcription error: {e}")
        raise HTTPException(500, "Transcription failed")

@app.post("/v1/audio/translations")
async def translate(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    authorized: bool = Depends(verify_api_key)
):
    """Translate audio to English"""
    start_time = time.time()
    
    # Validate model
    if model not in PERFORMANCE_PROFILES:
        raise HTTPException(400, f"Invalid model: {model}")
    
    try:
        # Read and convert audio
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(400, "Empty audio file")
        
        audio_data, sample_rate = convert_audio(audio_bytes)
        
        # Translate
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            asyncio.run,
            transcribe_audio(
                audio_data,
                sample_rate,
                "translate",
                None,
                model,
                False  # No diarization for translation
            )
        )
        
        processing_time = time.time() - start_time
        
        # Format response
        if response_format == "text":
            return result["text"]
        else:
            return TranslationResponse(
                text=result["text"],
                language=result["language"],
                duration=processing_time
            )
            
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        if OFFLINE_MODE and "model" in str(e).lower():
            raise HTTPException(503, f"Model error in offline mode: {str(e)}")
        print(f"Translation error: {e}")
        raise HTTPException(500, "Translation failed")

def format_timestamp(seconds: float, fmt: str = "srt") -> str:
    """Format timestamp for subtitles"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if fmt == "srt":
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")
    else:  # vtt
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)