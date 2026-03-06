"""
Pipeline Module — Orchestrates the full audio processing flow.

Flow:
1. Validate audio file
2. Convert to WAV (for pyannote compatibility)
3. Transcribe with Whisper → text with timestamps
4. Diarize with Pyannote → speaker labels with timestamps
5. Merge → assign text to speakers
6. Group by speaker → per-speaker transcripts
7. Save results to database (SpeakerResult table)

This is called when a user clicks "Process" on a debate.
"""

import os
import logging
from typing import Dict

from sqlalchemy.orm import Session

from app.models import Debate, SpeakerResult
from app.config import settings

logger = logging.getLogger(__name__)


def process_debate(debate_id: int, db: Session) -> Dict:
    """
    Full processing pipeline for a debate audio file.

    Args:
        debate_id: ID of the debate in the database
        db: SQLAlchemy database session

    Returns:
        Dict with processing results:
        {
            "status": "completed",
            "num_speakers": 4,
            "total_segments": 42,
            "speakers": {
                "SPEAKER_00": {"full_transcript": "...", "total_speaking_time": 120.5},
                ...
            }
        }

    This function:
    1. Loads the debate from DB to get the audio filename
    2. Runs transcription + diarization
    3. Merges results
    4. Saves per-speaker results to the speaker_results table
    5. Updates debate status to "completed"
    """

    # --- Step 1: Load debate and validate ---
    # Lazy imports — only loaded when pipeline actually runs
    from app.ai.audio_utils import validate_audio_file, convert_to_wav
    from app.ai.transcription import transcribe
    from app.ai.diarization import diarize
    from app.ai.merger import (
        merge_transcript_and_diarization,
        group_by_speaker,
        format_readable_transcript,
    )

    debate = db.query(Debate).filter(Debate.id == debate_id).first()
    if not debate:
        raise ValueError(f"Debate {debate_id} not found")

    if not debate.audio_filename:
        raise ValueError(f"Debate {debate_id} has no audio file uploaded")

    audio_path = os.path.join(settings.UPLOAD_DIR, debate.audio_filename)

    if not validate_audio_file(audio_path):
        raise ValueError(f"Invalid audio file: {audio_path}")

    # Update status
    debate.status = "processing"
    db.commit()

    try:
        # --- Step 2: Convert to WAV ---
        logger.info(f"[Pipeline] Step 1/5: Converting audio for debate {debate_id}")
        wav_path = convert_to_wav(audio_path)

        # --- Step 3: Transcribe ---
        logger.info(f"[Pipeline] Step 2/5: Transcribing debate {debate_id}")
        transcript_segments = transcribe(wav_path)

        if not transcript_segments:
            raise ValueError("Transcription returned no segments — audio may be empty or corrupt")

        # --- Step 4: Diarize ---
        logger.info(f"[Pipeline] Step 3/5: Diarizing debate {debate_id}")
        diarization_segments = diarize(wav_path, num_speakers=debate.num_speakers)

        # --- Step 5: Merge ---
        logger.info(f"[Pipeline] Step 4/5: Merging transcript and diarization")
        merged = merge_transcript_and_diarization(transcript_segments, diarization_segments)
        speaker_data = group_by_speaker(merged)
        readable_transcript = format_readable_transcript(merged)

        # --- Step 6: Save to database ---
        logger.info(f"[Pipeline] Step 5/5: Saving results for debate {debate_id}")

        # Delete old results if re-processing
        db.query(SpeakerResult).filter(SpeakerResult.debate_id == debate_id).delete()

        for speaker_label, data in speaker_data.items():
            result = SpeakerResult(
                debate_id=debate_id,
                speaker_label=speaker_label,
                transcript=data["full_transcript"],
                # Scores will be filled in by evaluator.py (Week 3)
                score_content=None,
                score_style=None,
                score_structure=None,
                score_rebuttal=None,
                score_strategy=None,
                score_total=None,
                feedback=None,
                strengths=None,
                weaknesses=None,
                suggestions=None,
            )
            db.add(result)

        # Update debate status
        debate.status = "transcribed"
        db.commit()

        # --- Cleanup temp WAV file ---
        if wav_path != audio_path and os.path.exists(wav_path):
            os.remove(wav_path)

        logger.info(f"[Pipeline] Debate {debate_id} processed successfully: "
                    f"{len(speaker_data)} speakers, {len(merged)} segments")

        return {
            "status": "completed",
            "debate_id": debate_id,
            "num_speakers": len(speaker_data),
            "total_segments": len(merged),
            "readable_transcript": readable_transcript,
            "speakers": {
                speaker: {
                    "full_transcript": data["full_transcript"][:200] + "...",
                    "total_speaking_time": data["total_speaking_time"],
                    "num_segments": data["num_segments"],
                }
                for speaker, data in speaker_data.items()
            },
        }

    except Exception as e:
        logger.error(f"[Pipeline] Failed for debate {debate_id}: {e}")
        debate.status = "failed"
        db.commit()
        raise