from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_user
from app.models import User, SpeakerResult, QAMessage
from app.schemas import QARequest, QAMessageResponse

router = APIRouter(prefix="/qa", tags=["Q&A"])


@router.post("/{result_id}", response_model=QAMessageResponse)
def ask_question(
    result_id: int,
    body: QARequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Ask a follow-up question about a speaker's performance."""
    result = db.query(SpeakerResult).filter(SpeakerResult.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Speaker result not found")

    # Save user question
    user_msg = QAMessage(
        speaker_result_id=result_id,
        role="user",
        content=body.question,
    )
    db.add(user_msg)
    db.commit()

    # TODO: Generate AI answer using RAG (Week 3)
    # For now, return a placeholder
    ai_answer = (
        f"[Placeholder] This will use RAG to answer: '{body.question}' "
        f"based on {result.speaker_label}'s transcript and debate manuals."
    )

    assistant_msg = QAMessage(
        speaker_result_id=result_id,
        role="assistant",
        content=ai_answer,
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return assistant_msg


@router.get("/{result_id}/history", response_model=list[QAMessageResponse])
def get_qa_history(
    result_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get full Q&A conversation history for a speaker result."""
    messages = (
        db.query(QAMessage)
        .filter(QAMessage.speaker_result_id == result_id)
        .order_by(QAMessage.created_at)
        .all()
    )
    return messages
