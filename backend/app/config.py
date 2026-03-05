from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Debate AI Platform"
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/debate_platform"
    JWT_SECRET: str = "change-this-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_HOURS: int = 24
    OPENAI_API_KEY: str = ""
    HF_TOKEN: str = ""  # HuggingFace token for Pyannote
    WHISPER_MODEL_SIZE: str = "base" 
    UPLOAD_DIR: str = "uploads"
    RAG_DOCS_DIR: str = "rag_documents"

    class Config:
        env_file = ".env"


settings = Settings()
