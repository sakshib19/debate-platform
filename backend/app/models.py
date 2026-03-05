from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(150), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    debates = relationship("Debate", back_populates="user")


class Debate(Base):
    __tablename__ = "debates"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=False)
    format = Column(String(50), nullable=False)  # asian_parl, british_parl, wsdc
    motion = Column(Text, nullable=True)
    num_speakers = Column(Integer, default=2)
    audio_filename = Column(String(500), nullable=True)
    status = Column(String(50), default="uploaded")  # uploaded, processing, done, failed
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="debates")
    speaker_results = relationship("SpeakerResult", back_populates="debate")


class SpeakerResult(Base):
    __tablename__ = "speaker_results"

    id = Column(Integer, primary_key=True, index=True)
    debate_id = Column(Integer, ForeignKey("debates.id"), nullable=False)
    speaker_label = Column(String(50), nullable=False)  # "Speaker 1", "Speaker 2"
    transcript = Column(Text, nullable=True)

    # Scores (each out of 10)
    score_content = Column(Float, nullable=True)
    score_style = Column(Float, nullable=True)
    score_structure = Column(Float, nullable=True)
    score_rebuttal = Column(Float, nullable=True)
    score_strategy = Column(Float, nullable=True)
    score_total = Column(Float, nullable=True)

    # Feedback
    feedback = Column(Text, nullable=True)       # JSON string with detailed feedback
    strengths = Column(Text, nullable=True)
    weaknesses = Column(Text, nullable=True)
    suggestions = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    debate = relationship("Debate", back_populates="speaker_results")
    qa_messages = relationship("QAMessage", back_populates="speaker_result")


class QAMessage(Base):
    __tablename__ = "qa_messages"

    id = Column(Integer, primary_key=True, index=True)
    speaker_result_id = Column(Integer, ForeignKey("speaker_results.id"), nullable=False)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    speaker_result = relationship("SpeakerResult", back_populates="qa_messages")
