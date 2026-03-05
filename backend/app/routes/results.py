from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_user
from app.models import User, SpeakerResult
from app.schemas import SpeakerResultResponse

router = APIRouter(prefix="/results", tags=["Results"])


@router.get("/{result_id}", response_model=SpeakerResultResponse)
def get_speaker_result(
    result_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get detailed feedback for a single speaker."""
    result = db.query(SpeakerResult).filter(SpeakerResult.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Speaker result not found")
    return result
