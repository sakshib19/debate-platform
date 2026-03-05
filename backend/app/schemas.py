from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, EmailStr


# --- Auth ---
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# --- User ---
class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    created_at: datetime

    class Config:
        from_attributes = True


# --- Debate ---
class DebateCreateRequest(BaseModel):
    title: str
    format: str  # asian_parl, british_parl, wsdc
    motion: Optional[str] = None
    num_speakers: int = 2


class DebateResponse(BaseModel):
    id: int
    title: str
    format: str
    motion: Optional[str]
    num_speakers: int
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


# --- Speaker Results ---
class SpeakerScores(BaseModel):
    score_content: Optional[float] = None
    score_style: Optional[float] = None
    score_structure: Optional[float] = None
    score_rebuttal: Optional[float] = None
    score_strategy: Optional[float] = None
    score_total: Optional[float] = None


class SpeakerResultResponse(BaseModel):
    id: int
    speaker_label: str
    transcript: Optional[str]
    score_content: Optional[float]
    score_style: Optional[float]
    score_structure: Optional[float]
    score_rebuttal: Optional[float]
    score_strategy: Optional[float]
    score_total: Optional[float]
    feedback: Optional[str]
    strengths: Optional[str]
    weaknesses: Optional[str]
    suggestions: Optional[str]

    class Config:
        from_attributes = True


# --- Q&A ---
class QARequest(BaseModel):
    question: str


class QAMessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True
