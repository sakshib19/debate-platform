"""
Audio Utilities — Format conversion and validation.

Why we need this:
- Users upload .mp3, .m4a, .mp4, .wav files
- faster-whisper accepts most formats directly
- pyannote works best with 16kHz mono WAV files
- This module converts any input to the format each tool needs

Requires: ffmpeg installed on the system (see README)
"""

import os
import logging
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# Audio formats we accept
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac", ".webm"}


def validate_audio_file(filepath: str) -> bool:
    """Check if the file exists and has a supported extension."""
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return False

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.error(f"Unsupported format: {ext}")
        return False

    # Check file isn't empty
    size = os.path.getsize(filepath)
    if size == 0:
        logger.error(f"File is empty: {filepath}")
        return False

    logger.info(f"Audio file valid: {filepath} ({size / 1024 / 1024:.1f} MB)")
    return True


def convert_to_wav(input_path: str, output_path: str = None) -> str:
    """
    Convert any audio file to 16kHz mono WAV (optimal for pyannote).

    Args:
        input_path:  Path to the input audio file
        output_path: Path for the output WAV file (optional)
                     If not provided, replaces the extension with .wav

    Returns:
        Path to the converted WAV file

    Why 16kHz mono?
    - Pyannote's neural network was trained on 16kHz audio
    - Mono (single channel) — stereo provides no benefit for speech
    - WAV is uncompressed — no decoding overhead during processing
    """
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_converted.wav"

    logger.info(f"Converting {input_path} to WAV...")

    # Load audio (pydub auto-detects format via ffmpeg)
    audio = AudioSegment.from_file(input_path)

    # Convert to 16kHz mono
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)

    # Export as WAV
    audio.export(output_path, format="wav")

    logger.info(f"Converted: {output_path} "
                f"(duration: {len(audio) / 1000:.1f}s, "
                f"size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")

    return output_path


def get_audio_duration(filepath: str) -> float:
    """Get audio duration in seconds."""
    audio = AudioSegment.from_file(filepath)
    return len(audio) / 1000.0  # pydub returns milliseconds