"""
Transcription Module — Uses faster-whisper to convert audio to text.

Input:  Audio file path (.wav, .mp3, .m4a)
Output: List of segments, each with start_time, end_time, and text

Example output:
[
    {"start": 0.0, "end": 2.48, "text": "Good morning everyone"},
    {"start": 2.48, "end": 5.12, "text": "Today we debate the motion"},
    ...
]

How it works:
1. Load the faster-whisper model (downloads on first run, ~150MB for 'base')
2. Feed the audio file to the model
3. Model returns segments with timestamps and text
4. We format them into a clean list of dicts
"""

import logging
from typing import List, Dict

from app.config import settings

logger = logging.getLogger(__name__)

# Cache the model so it loads once, not on every request
_model = None


def _get_model():
    """
    Load Whisper model (cached after first call).

    Model sizes and tradeoffs:
    - 'tiny'  : ~75MB,  fastest,  lowest accuracy
    - 'base'  : ~150MB, fast,     good for dev/testing
    - 'small' : ~500MB, moderate, good accuracy
    - 'medium': ~1.5GB, slower,   high accuracy
    - 'large-v3': ~3GB, slowest,  best accuracy (use for production)

    We use 'base' for development. Change WHISPER_MODEL_SIZE in .env for production.
    """
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is not installed. "
                "Run: pip install faster-whisper==1.1.0"
            )
        logger.info(f"Loading Whisper model: {settings.WHISPER_MODEL_SIZE}")
        _model = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device="cpu",          # Change to "cuda" if you have NVIDIA GPU
            compute_type="int8",   # Fastest on CPU. Use "float16" for GPU
        )
        logger.info("Whisper model loaded successfully")
    return _model


def transcribe(audio_path: str) -> List[Dict]:
    """
    Transcribe an audio file to text with timestamps.

    Args:
        audio_path: Path to the audio file (wav, mp3, m4a, mp4)

    Returns:
        List of segments:
        [
            {
                "start": 0.0,       # Start time in seconds
                "end": 2.48,        # End time in seconds
                "text": "Hello...", # Transcribed text
                "confidence": 0.95  # How confident the model is (0-1)
            },
            ...
        ]

    How faster-whisper works internally:
    1. Audio is converted to 16kHz mono WAV (done automatically)
    2. Audio is split into 30-second chunks
    3. Each chunk is converted to a mel spectrogram (visual representation of sound)
    4. The transformer model predicts text tokens from the spectrogram
    5. Timestamps are aligned using cross-attention weights
    """
    model = _get_model()

    logger.info(f"Transcribing: {audio_path}")

    # segments is a generator, info contains language detection results
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,              # Higher = more accurate but slower (default: 5)
        language=None,            # Auto-detect language. Set "en" to force English
        vad_filter=True,          # Voice Activity Detection — skips silence
        vad_parameters=dict(
            min_silence_duration_ms=500,  # Ignore silences shorter than 500ms
        ),
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


def transcribe_to_text(audio_path: str) -> str:
    """
    Convenience function — returns just the full text without timestamps.
    Useful for quick testing.
    """
    segments = transcribe(audio_path)
    return " ".join(seg["text"] for seg in segments)