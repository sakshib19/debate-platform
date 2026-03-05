"""
Diarization Module — Uses pyannote.audio to identify speakers.

Input:  Audio file path (.wav)
Output: List of speaker segments with start/end times and speaker labels

Example output:
[
    {"start": 0.0,  "end": 3.21, "speaker": "SPEAKER_00"},
    {"start": 3.21, "end": 6.54, "speaker": "SPEAKER_01"},
    {"start": 6.54, "end": 9.87, "speaker": "SPEAKER_00"},
    ...
]

How diarization works:
1. Audio is split into small overlapping frames (every 16ms)
2. A neural network extracts "speaker embeddings" — a 256-dim vector
   that represents the voice characteristics (pitch, timbre, rhythm)
3. Embeddings are clustered — frames with similar embeddings = same speaker
4. The model outputs "SPEAKER_00 spoke from 0.0s to 3.21s" etc.

Requirements:
- HuggingFace token (free account)
- Accept model terms at:
  https://huggingface.co/pyannote/speaker-diarization-3.1
  https://huggingface.co/pyannote/segmentation-3.0
"""

import logging
from typing import List, Dict, Optional
from pyannote.audio import Pipeline

from app.config import settings

logger = logging.getLogger(__name__)

_pipeline = None


def _get_pipeline() -> Pipeline:
    """
    Load the pyannote diarization pipeline (cached after first call).

    First run downloads ~500MB of model weights from HuggingFace.
    Subsequent runs use the cached version (~/.cache/huggingface/).
    """
    global _pipeline
    if _pipeline is None:
        if not settings.HF_TOKEN or settings.HF_TOKEN.startswith("hf_your"):
            raise ValueError(
                "HuggingFace token not set! "
                "1. Go to https://huggingface.co/settings/tokens and create a token. "
                "2. Go to https://huggingface.co/pyannote/speaker-diarization-3.1 and accept terms. "
                "3. Go to https://huggingface.co/pyannote/segmentation-3.0 and accept terms. "
                "4. Set HF_TOKEN in your .env file."
            )

        logger.info("Loading pyannote diarization pipeline...")
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.HF_TOKEN,
        )
        logger.info("Pyannote pipeline loaded successfully")
    return _pipeline


def diarize(audio_path: str, num_speakers: Optional[int] = None) -> List[Dict]:
    """
    Identify speakers in an audio file.

    Args:
        audio_path:   Path to WAV audio file
        num_speakers: Expected number of speakers (optional but recommended).
                      If provided, helps the model cluster more accurately.
                      For debates, this is usually 2-8 depending on format.

    Returns:
        List of speaker segments:
        [
            {
                "start": 0.0,
                "end": 3.21,
                "speaker": "SPEAKER_00"
            },
            ...
        ]
        Sorted chronologically.

    How num_speakers helps:
    - Without it, pyannote uses a threshold to decide "is this a new speaker?"
      This can over-count or under-count speakers.
    - With it, pyannote knows exactly how many clusters to create.
    - For a 4-person debate, set num_speakers=4.
    """
    pipeline = _get_pipeline()

    logger.info(f"Diarizing: {audio_path} (expected speakers: {num_speakers or 'auto'})")

    # Run diarization
    if num_speakers:
        diarization = pipeline(audio_path, num_speakers=num_speakers)
    else:
        diarization = pipeline(audio_path)

    # Convert pyannote output to list of dicts
    # pyannote returns: (Segment(start, end), track, speaker_label)
    result = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        result.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker,
        })

    logger.info(f"Diarization complete: {len(result)} segments, "
                f"{len(set(r['speaker'] for r in result))} unique speakers")

    return result