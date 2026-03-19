"""
config.py — Application configuration via environment variables.

Uses pydantic-settings so every value can be overridden with an
environment variable or a ``.env`` file (12-factor style).
This ensures configuration is type-safe and centralised in one place.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Centralised, type-safe configuration.

    Attributes
    ----------
    audio_dir : Path
        Directory where downloaded / uploaded audio files are stored.
    whisper_model : str
        OpenAI Whisper model size to use for transcription
        (``tiny``, ``base``, ``small``, ``medium``, ``large``).
    db_path : Path
        File path for the SQLite database.
    groq_api_key : Optional[str]
        API key for the Groq service, required for LLM-based
        transcript structuring and conflict detection.
    """

    audio_dir: Path = Path("audio_files")
    whisper_model: str = "base"
    db_path: Path = Path("tellia.db")
    groq_api_key: Optional[str] = None

    # pydantic-settings config: load values from a .env file
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton instance imported by the rest of the application
settings = Settings()
