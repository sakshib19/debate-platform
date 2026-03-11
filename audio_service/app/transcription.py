"""
Transcription Module — Uses faster-whisper to convert audio to text.
"""

import logging
from typing import List, Dict

from app.config import settings

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Load Whisper model (cached after first call)."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        logger.info(f"Loading Whisper model: {settings.WHISPER_MODEL_SIZE}")
        _model = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
        logger.info("Whisper model loaded successfully")
    return _model


def transcribe(audio_path: str) -> List[Dict]:
    """
    Transcribe an audio file to text with timestamps.

    Returns:
        List of segments: [{"start": 0.0, "end": 2.48, "text": "...", "confidence": 0.95}, ...]
    """
    model = _get_model()

    logger.info(f"Transcribing: {audio_path}")

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language=None,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    result = []
    for segment in segments:
        result.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
            "confidence": round(segment.avg_logprob, 4),
        })

    logger.info(f"Transcription complete: {len(result)} segments")
    return result
