# app_optimized.py - Optimized for whisper-large-v3 performance
import os
import io
import time
import asyncio
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf

# Configuration with performance optimization
API_KEYS = os.getenv("API_KEYS", "").split(",")
MODEL_SIZE = os.getenv("MODEL_SIZE", "large-v3")
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")

# Performance parameters
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "5"))
BEST_OF = int(os.getenv("BEST_OF", "5"))
PATIENCE = float(os.getenv("PATIENCE", "1.0"))  # Must be > 0
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "500"))
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "1"))

# Validate patience
if PATIENCE <= 0:
    PATIENCE = 1.0
    print(f"Warning: PATIENCE must be > 0, setting to 1.0")

app = FastAPI(title="Faster Whisper Large-v3 Optimized API")
security = HTTPBearer()

# Initialize model
print(f"Loading {MODEL_SIZE} on {DEVICE} with {COMPUTE_TYPE}")
print(f"Performance: beam_size={BEAM_SIZE}, best_of={BEST_OF}, workers={NUM_WORKERS}")

# Pre-load model for faster first inference
whisper_model = WhisperModel(
    MODEL_SIZE, 
    device=DEVICE, 
    compute_type=COMPUTE_TYPE,
    download_root="/home/whisper/.cache/huggingface",
    local_files_only=False
)

# Thread pool for async processing
executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)

# Response models
class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration: float

# Authentication
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not API_KEYS or API_KEYS == [""]:
        return True
    if credentials.credentials not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

# Optimized audio conversion
def convert_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Convert audio with optimizations for large files"""
    try:
        audio_file = io.BytesIO(audio_bytes)
        data, samplerate = sf.read(audio_file)
        
        # Convert to mono if needed
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Ensure float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # Normalize
        max_val = np.abs(data).max()
        if max_val > 1.0:
            data = data / max_val
        elif max_val < 0.1 and max_val > 0:
            data = data / max_val * 0.5
            
        return data, samplerate
    except Exception as e:
        raise ValueError(f"Audio processing failed: {str(e)}")

# Optimized transcription function
def transcribe_audio(audio_data, sample_rate, language=None, task="transcribe"):
    """Transcribe with optimized settings for large-v3"""
    
    # Check duration
    duration = len(audio_data) / sample_rate
    use_vad = duration > 1.0
    
    # VAD parameters optimized for model
    vad_params = {
        "min_silence_duration_ms": VAD_MIN_SILENCE_MS,
        "threshold": VAD_THRESHOLD,
        "min_speech_duration_ms": 250,
        "speech_pad_ms": 400,
    } if use_vad else None
    
    # Transcribe with optimized parameters
    segments, info = whisper_model.transcribe(
        audio_data,
        language=language,
        task=task,
        beam_size=BEAM_SIZE,
        best_of=BEST_OF,
        patience=PATIENCE,
        length_penalty=1.0,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        initial_prompt=None,
        suppress_blank=True,
        suppress_tokens=[-1],
        without_timestamps=False,
        max_initial_timestamp=1.0,
        word_timestamps=False,
        vad_filter=use_vad,
        vad_parameters=vad_params,
    )
    
    # Collect text
    text = "".join(segment.text for segment in segments)
    return text.strip(), info.language

@app.get("/")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE
    }

@app.get("/v1/models")
async def list_models(authorized: bool = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [{
            "id": "whisper-1",
            "object": "model",
            "created": 1677532384,
            "owned_by": "openai"
        }]
    }

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
    authorized: bool = Depends(verify_api_key)
):
    """Optimized transcription endpoint"""
    start_time = time.time()
    
    try:
        # Read audio
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Convert audio
        audio_data, sample_rate = convert_audio(audio_bytes)
        
        # Run transcription in thread pool
        loop = asyncio.get_event_loop()
        text, detected_language = await loop.run_in_executor(
            executor,
            transcribe_audio,
            audio_data,
            sample_rate,
            language,
            "transcribe"
        )
        
        processing_time = time.time() - start_time
        
        # Return based on format
        if response_format == "text":
            return text
        else:
            return TranscriptionResponse(
                text=text,
                language=detected_language,
                duration=processing_time
            )
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed")

@app.post("/v1/audio/translations")
async def translate(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    response_format: str = Form("json"),
    authorized: bool = Depends(verify_api_key)
):
    """Optimized translation endpoint"""
    start_time = time.time()
    
    try:
        # Read audio
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Convert audio
        audio_data, sample_rate = convert_audio(audio_bytes)
        
        # Run translation in thread pool
        loop = asyncio.get_event_loop()
        text, detected_language = await loop.run_in_executor(
            executor,
            transcribe_audio,
            audio_data,
            sample_rate,
            None,
            "translate"
        )
        
        processing_time = time.time() - start_time
        
        # Return based on format
        if response_format == "text":
            return text
        else:
            return {
                "text": text,
                "language": detected_language,
                "duration": processing_time
            }
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail="Translation failed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
