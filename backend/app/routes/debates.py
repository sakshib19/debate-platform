import os
import shutil
import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.auth import get_current_user
from app.models import User, Debate
from app.schemas import DebateCreateRequest, DebateResponse, SpeakerResultResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debates", tags=["Debates"])

AUDIO_SERVICE_TIMEOUT = httpx.Timeout(timeout=10.0)


@router.post("/", response_model=DebateResponse, status_code=201)
def create_debate(
    body: DebateCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    debate = Debate(
        user_id=current_user.id,
        title=body.title,
        format=body.format,
        motion=body.motion,
        num_speakers=body.num_speakers,
        status="created",
    )
    db.add(debate)
    db.commit()
    db.refresh(debate)
    return debate


@router.post("/{debate_id}/upload-audio")
def upload_audio(
    debate_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    debate = db.query(Debate).filter(
        Debate.id == debate_id, Debate.user_id == current_user.id
    ).first()
    if not debate:
        raise HTTPException(status_code=404, detail="Debate not found")

    allowed_types = [
        "audio/mpeg", "audio/wav", "audio/mp4", "audio/x-m4a",
        "audio/ogg", "audio/flac", "audio/webm", "video/mp4",
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"File type {file.content_type} not supported")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    filename = f"debate_{debate_id}_{file.filename}"
    filepath = os.path.join(settings.UPLOAD_DIR, filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    debate.audio_filename = filename
    debate.status = "uploaded"
    db.commit()

    return {"message": "Audio uploaded successfully", "debate_id": debate_id, "filename": filename}


@router.post("/{debate_id}/process")
def process_debate_endpoint(
    debate_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Trigger audio processing: transcription + diarization + merge.

    This runs in the background so the API responds immediately.
    Check debate status with GET /api/debates/{debate_id} to see progress.

    Status flow: uploaded → processing → transcribed → (later: evaluated)
    """
    debate = db.query(Debate).filter(
        Debate.id == debate_id, Debate.user_id == current_user.id
    ).first()
    if not debate:
        raise HTTPException(status_code=404, detail="Debate not found")

    if not debate.audio_filename:
        raise HTTPException(status_code=400, detail="No audio file uploaded yet")

    if debate.status == "processing":
        raise HTTPException(status_code=409, detail="Already processing")

    # Check audio service is reachable before starting
    try:
        health_resp = httpx.get(
            f"{settings.AUDIO_SERVICE_URL}/health",
            timeout=AUDIO_SERVICE_TIMEOUT,
        )
        if health_resp.status_code != 200:
            raise HTTPException(
                status_code=503,
                detail="Audio service is not healthy",
            )
    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Audio service is unreachable. Ensure it is running at "
                   f"{settings.AUDIO_SERVICE_URL}",
        )

    # Verify the audio file actually exists on disk
    audio_path = os.path.join(settings.UPLOAD_DIR, debate.audio_filename)
    if not os.path.exists(audio_path):
        raise HTTPException(
            status_code=400,
            detail=f"Audio file missing from disk: {debate.audio_filename}"
        )

    # Import pipeline (now just uses httpx — no heavy AI deps)
    from app.ai.pipeline import process_debate
    from app.database import SessionLocal

    debate.status = "processing"
    db.commit()

    def _run_pipeline():
        """Run in background with its own DB session."""
        pipeline_db = SessionLocal()
        try:
            process_debate(debate_id, pipeline_db)
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Mark debate as failed
            try:
                d = pipeline_db.query(Debate).filter(Debate.id == debate_id).first()
                if d:
                    d.status = "failed"
                    pipeline_db.commit()
            except Exception:
                pass
        finally:
            pipeline_db.close()

    background_tasks.add_task(_run_pipeline)

    return {
        "message": "Processing started",
        "debate_id": debate_id,
        "status": "processing",
        "note": "Poll GET /api/debates/{debate_id} to check status",
    }


@router.get("/", response_model=list[DebateResponse])
def list_debates(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return db.query(Debate).filter(Debate.user_id == current_user.id).all()


@router.get("/{debate_id}", response_model=DebateResponse)
def get_debate(
    debate_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    debate = db.query(Debate).filter(Debate.id == debate_id).first()
    if not debate:
        raise HTTPException(status_code=404, detail="Debate not found")
    return debate


@router.get("/{debate_id}/results", response_model=list[SpeakerResultResponse])
def get_results(
    debate_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    debate = db.query(Debate).filter(Debate.id == debate_id).first()
    if not debate:
        raise HTTPException(status_code=404, detail="Debate not found")
    return debate.speaker_results