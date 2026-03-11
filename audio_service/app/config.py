import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HF_TOKEN: str = ""
    WHISPER_MODEL_SIZE: str = "base"

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")


settings = Settings()
