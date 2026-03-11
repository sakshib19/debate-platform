"""
Audio Service — Standalone FastAPI service for transcription + diarization.

Endpoints:
  POST /transcribe  — accepts an audio file, returns speaker-labeled transcript
  GET  /health      — health check
"""

import os
import logging
import tempfile
import shutil
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from app.audio_utils import validate_audio_file, convert_to_wav
from app.transcription import transcribe
from app.diarization import diarize
from app.merger import (
    merge_transcript_and_diarization,
    group_by_speaker,
    format_readable_transcript,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Debate Audio Service", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "healthy", "service": "audio"}


@app.post("/transcribe")
def transcribe_audio(
    file: UploadFile = File(...),
    num_speakers: Optional[int] = Form(None),
):
    """
    Accept an audio file, run transcription + diarization + merge.

    Returns JSON with:
      - merged_segments: list of {start, end, text, speaker, confidence}
      - speakers: per-speaker grouped data
      - readable_transcript: formatted string
    """
    # Save uploaded file to a temp location
    suffix = os.path.splitext(file.filename or "audio.wav")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        audio_path = tmp.name

        if not validate_audio_file(audio_path):
            raise HTTPException(status_code=400, detail="Invalid or unsupported audio file")

        # Convert to WAV for pyannote
        wav_path = convert_to_wav(audio_path)

        # Transcribe
        logger.info("Step 1/3: Transcribing...")
        transcript_segments = transcribe(wav_path)
        if not transcript_segments:
            raise HTTPException(status_code=422, detail="Transcription returned no segments")

        # Diarize
        logger.info("Step 2/3: Diarizing...")
        diarization_segments = diarize(wav_path, num_speakers=num_speakers)

        # Merge
        logger.info("Step 3/3: Merging...")
        merged = merge_transcript_and_diarization(transcript_segments, diarization_segments)
        speakers = group_by_speaker(merged)
        readable = format_readable_transcript(merged)

        return {
            "status": "completed",
            "num_speakers": len(speakers),
            "total_segments": len(merged),
            "merged_segments": merged,
            "speakers": {
                speaker: {
                    "full_transcript": data["full_transcript"],
                    "total_speaking_time": data["total_speaking_time"],
                    "num_segments": data["num_segments"],
                }
                for speaker, data in speakers.items()
            },
            "readable_transcript": readable,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        # Cleanup temp files
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        wav_candidate = os.path.splitext(tmp.name)[0] + ".wav"
        if wav_candidate != tmp.name and os.path.exists(wav_candidate):
            os.unlink(wav_candidate)
