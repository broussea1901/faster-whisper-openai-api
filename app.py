# app.py
import os
import io
import time
import hashlib
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf

# Configuration
API_KEYS = os.getenv("API_KEYS", "").split(",")  # Comma-separated list of valid API keys
MODEL_SIZE = os.getenv("MODEL_SIZE", "base")
DEVICE = os.getenv("DEVICE", "auto")  # auto, cuda, or cpu
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")  # float16, int8_float16, int8

app = FastAPI(title="Faster Whisper OpenAI-Compatible API")
security = HTTPBearer()

# Initialize model
print(f"Loading Faster Whisper model: {MODEL_SIZE} on {DEVICE} with {COMPUTE_TYPE}")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

# Response models matching OpenAI's format
class TranscriptionSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float

class TranscriptionResponse(BaseModel):
    text: str
    task: str = "transcribe"
    language: str
    duration: float
    segments: Optional[List[TranscriptionSegment]] = None

class TranscriptionVerboseResponse(BaseModel):
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    segments: List[TranscriptionSegment]

# Authentication
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not API_KEYS or API_KEYS == [""]:
        return True  # No authentication if no keys are set
    
    token = credentials.credentials
    if token not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

# Helper function to convert audio to wav
def convert_to_wav(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Convert audio bytes to numpy array and sample rate"""
    audio_file = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(audio_file)
    return data, samplerate

@app.get("/")
async def root():
    return {"message": "Faster Whisper OpenAI-Compatible API Server"}

@app.get("/v1/models")
async def list_models(authorized: bool = Depends(verify_api_key)):
    """List available models (OpenAI-compatible endpoint)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "whisper-1",
                "object": "model",
                "created": 1677532384,
                "owned_by": "openai"
            }
        ]
    }

@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: Optional[str] = Form(None),
    authorized: bool = Depends(verify_api_key)
):
    """
    Transcribe audio file (OpenAI-compatible endpoint)
    
    Supports response_format: json, text, srt, verbose_json, vtt
    """
    start_time = time.time()
    
    # Read and convert audio file
    audio_bytes = await file.read()
    audio_data, sample_rate = convert_to_wav(audio_bytes)
    
    # Transcribe with faster-whisper
    segments, info = model.transcribe(
        audio_data,
        beam_size=5,
        language=language,
        initial_prompt=prompt,
        temperature=temperature if temperature > 0 else None,
        vad_filter=True,  # Voice activity detection for better performance
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    # Process segments
    segments_list = []
    full_text = ""
    
    for segment in segments:
        segments_list.append({
            "id": len(segments_list),
            "seek": int(segment.seek),
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "tokens": segment.tokens,
            "temperature": segment.temperature,
            "avg_logprob": segment.avg_logprob,
            "compression_ratio": segment.compression_ratio,
            "no_speech_prob": segment.no_speech_prob
        })
        full_text += segment.text
    
    duration = time.time() - start_time
    
    # Format response based on requested format
    if response_format == "text":
        return full_text.strip()
    
    elif response_format == "srt":
        srt_content = ""
        for i, seg in enumerate(segments_list):
            start = format_timestamp(seg["start"], fmt="srt")
            end = format_timestamp(seg["end"], fmt="srt")
            srt_content += f"{i+1}\n{start} --> {end}\n{seg['text'].strip()}\n\n"
        return srt_content
    
    elif response_format == "vtt":
        vtt_content = "WEBVTT\n\n"
        for seg in segments_list:
            start = format_timestamp(seg["start"], fmt="vtt")
            end = format_timestamp(seg["end"], fmt="vtt")
            vtt_content += f"{start} --> {end}\n{seg['text'].strip()}\n\n"
        return vtt_content
    
    elif response_format == "verbose_json":
        return TranscriptionVerboseResponse(
            language=info.language,
            duration=duration,
            text=full_text.strip(),
            segments=segments_list
        )
    
    else:  # json (default)
        return TranscriptionResponse(
            text=full_text.strip(),
            language=info.language,
            duration=duration
        )

@app.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    authorized: bool = Depends(verify_api_key)
):
    """
    Translate audio file to English (OpenAI-compatible endpoint)
    """
    start_time = time.time()
    
    # Read and convert audio file
    audio_bytes = await file.read()
    audio_data, sample_rate = convert_to_wav(audio_bytes)
    
    # Translate with faster-whisper (task="translate" forces translation to English)
    segments, info = model.transcribe(
        audio_data,
        beam_size=5,
        task="translate",
        initial_prompt=prompt,
        temperature=temperature if temperature > 0 else None,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    # Process segments
    full_text = ""
    for segment in segments:
        full_text += segment.text
    
    duration = time.time() - start_time
    
    # Format response
    if response_format == "text":
        return full_text.strip()
    else:  # json
        return {
            "text": full_text.strip(),
            "task": "translate",
            "language": info.language,
            "duration": duration
        }

def format_timestamp(seconds: float, fmt: str = "srt") -> str:
    """Format timestamp for SRT or VTT format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if fmt == "srt":
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")
    else:  # vtt
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
