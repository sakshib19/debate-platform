"""
Audio Utilities — Format conversion and validation.
"""

import os
import logging

logger = logging.getLogger(__name__)

# ── FFmpeg path (winget install location) ───────────────────────────
FFMPEG_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft", "WinGet", "Packages",
    "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "ffmpeg-8.0.1-full_build", "bin",
)

if os.path.isdir(FFMPEG_DIR) and FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
    logger.info("Added ffmpeg to PATH: %s", FFMPEG_DIR)


def _get_audio_segment():
    """Lazy import of pydub.AudioSegment."""
    try:
        from pydub import AudioSegment
        return AudioSegment
    except ImportError:
        raise ImportError(
            "pydub is not installed. "
            "Run: pip install pydub==0.25.1\n"
            "Also install ffmpeg: winget install Gyan.FFmpeg"
        )


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

    size = os.path.getsize(filepath)
    if size == 0:
        logger.error(f"File is empty: {filepath}")
        return False

    logger.info(f"Audio file valid: {filepath} ({size / 1024 / 1024:.1f} MB)")
    return True


def convert_to_wav(input_path: str, output_path: str = None) -> str:
    """
    Convert any audio file to 16kHz mono WAV (optimal for pyannote).

    Returns:
        Path to the converted WAV file
    """
    AudioSegment = _get_audio_segment()

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + ".wav"

    if input_path.lower().endswith(".wav"):
        # Still re-export to ensure 16kHz mono
        pass

    logger.info(f"Converting {input_path} → {output_path}")
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")
    logger.info(f"Conversion complete: {output_path}")
    return output_path
