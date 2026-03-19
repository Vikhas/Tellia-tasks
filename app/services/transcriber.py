"""
transcriber.py — Audio transcription service.

Converts audio files to text using OpenAI Whisper.
If Whisper is not installed, the module gracefully falls back to a
mock transcript so the rest of the pipeline can still be developed
and tested without a GPU or the Whisper dependency.
"""

import logging
from pathlib import Path

from app.exceptions import TranscriptionError

logger = logging.getLogger(__name__)

# ── Whisper availability check ────────────────────────────────────
# Performed once at import time.  If the ``whisper`` package is not
# installed the flag ``_WHISPER_AVAILABLE`` is set to False and all
# calls to ``transcribe()`` will return a mock string instead.
try:
    import whisper  # type: ignore
    _WHISPER_AVAILABLE = True
    logger.info("OpenAI Whisper is available — real transcription enabled.")
except ImportError:
    whisper = None
    _WHISPER_AVAILABLE = False


def transcribe(file_path: Path, model_name: str = "base") -> str:
    """
    Transcribe an audio file to plain text.

    Parameters
    ----------
    file_path : Path
        Path to the audio file on disk.
    model_name : str
        Whisper model size to load (``tiny``, ``base``, ``small``,
        ``medium``, ``large``).  Larger models are more accurate
        but require more memory and time.

    Returns
    -------
    str
        The transcribed text, or a mock placeholder when Whisper
        is not installed.

    Raises
    ------
    TranscriptionError
        If the audio file doesn't exist or Whisper fails internally.
    """
    # Guard: ensure the file actually exists before attempting transcription
    if not file_path.exists():
        raise TranscriptionError(
            message="Audio file not found.",
            detail=str(file_path),
        )

    # ── Real transcription path ───────────────────────────────────
    if _WHISPER_AVAILABLE:
        try:
            assert whisper is not None
            # Load the specified Whisper model (cached after first load)
            model = whisper.load_model(model_name)

            import typing
            # Run the speech-to-text inference on the audio file
            result: typing.Any = model.transcribe(str(file_path))

            # Extract the "text" field from the Whisper result dict
            return str(result.get("text", "")).strip()
        except Exception as exc:
            raise TranscriptionError(
                message="Whisper transcription failed.",
                detail=str(exc),
            ) from exc

    # ── Mock fallback path ────────────────────────────────────────
    logger.warning("Whisper is not installed. Returning mock.")
    return f"[Mock transcript for {file_path.name}]"
