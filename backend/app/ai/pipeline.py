"""
Pipeline Module — Orchestrates the full audio processing flow.

Sends the audio file to the audio_service via HTTP for transcription
and diarization, then saves results to the database.

Flow:
1. Load debate from DB, validate audio exists
2. POST audio to audio_service /transcribe endpoint
3. Receive merged speaker-labeled transcript
4. Save per-speaker results to database (SpeakerResult table)
"""

import os
import logging
from typing import Dict

import httpx
from sqlalchemy.orm import Session

from app.models import Debate, SpeakerResult
from app.config import settings

logger = logging.getLogger(__name__)

# Generous timeout: transcription of long audio can take minutes
AUDIO_SERVICE_TIMEOUT = httpx.Timeout(timeout=600.0)


def process_debate(debate_id: int, db: Session) -> Dict:
    """
    Full processing pipeline for a debate audio file.

    Sends audio to audio_service for transcription + diarization,
    then saves per-speaker results to the database.
    """

    debate = db.query(Debate).filter(Debate.id == debate_id).first()
    if not debate:
        raise ValueError(f"Debate {debate_id} not found")

    if not debate.audio_filename:
        raise ValueError(f"Debate {debate_id} has no audio file uploaded")

    audio_path = os.path.join(settings.UPLOAD_DIR, debate.audio_filename)

    if not os.path.exists(audio_path):
        raise ValueError(f"Audio file not found: {audio_path}")

    # Update status
    debate.status = "processing"
    db.commit()

    try:
        # --- Step 1: Send audio to audio_service ---
        logger.info(f"[Pipeline] Sending audio to audio_service for debate {debate_id}")

        url = f"{settings.AUDIO_SERVICE_URL}/transcribe"
        with open(audio_path, "rb") as f:
            files = {"file": (debate.audio_filename, f)}
            data = {}
            if debate.num_speakers:
                data["num_speakers"] = str(debate.num_speakers)

            response = httpx.post(url, files=files, data=data, timeout=AUDIO_SERVICE_TIMEOUT)

        if response.status_code != 200:
            detail = response.text[:500]
            raise RuntimeError(f"Audio service returned {response.status_code}: {detail}")

        result = response.json()
        speaker_data = result["speakers"]
        readable_transcript = result.get("readable_transcript", "")

        # --- Step 2: Save to database ---
        logger.info(f"[Pipeline] Saving results for debate {debate_id}")

        # Delete old results if re-processing
        db.query(SpeakerResult).filter(SpeakerResult.debate_id == debate_id).delete()

        for speaker_label, data in speaker_data.items():
            sr = SpeakerResult(
                debate_id=debate_id,
                speaker_label=speaker_label,
                transcript=data["full_transcript"],
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
            db.add(sr)

        # Update debate status
        debate.status = "transcribed"
        db.commit()

        logger.info(f"[Pipeline] Debate {debate_id} processed successfully: "
                    f"{len(speaker_data)} speakers")

        return {
            "status": "completed",
            "debate_id": debate_id,
            "num_speakers": len(speaker_data),
            "total_segments": result.get("total_segments", 0),
            "readable_transcript": readable_transcript,
            "speakers": {
                speaker: {
                    "full_transcript": d["full_transcript"][:200] + "...",
                    "total_speaking_time": d["total_speaking_time"],
                    "num_segments": d["num_segments"],
                }
                for speaker, d in speaker_data.items()
            },
        }

    except Exception as e:
        logger.error(f"[Pipeline] Failed for debate {debate_id}: {e}")
        debate.status = "failed"
        db.commit()
        raise