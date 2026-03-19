"""
exceptions.py — Custom exception hierarchy.

All domain-specific errors inherit from ``VoiceNoteError`` so they
can be caught by a single FastAPI exception handler in ``main.py``.
Each exception carries a human-readable ``message`` and an optional
``detail`` string for debugging context.
"""

from typing import Optional


class VoiceNoteError(Exception):
    """
    Base exception for all Tellia voice-note processing errors.

    Parameters
    ----------
    message : str
        Short, user-facing error message.
    detail : Optional[str]
        Extended information (e.g. the underlying exception string)
        useful for debugging but not necessarily shown to end-users.
    """

    def __init__(self, message: str, detail: Optional[str] = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(message)


class DownloadError(VoiceNoteError):
    """Raised when fetching audio from a remote URL fails (HTTP error, timeout, etc.).
    Mapped to HTTP 502 Bad Gateway by the exception handler."""
    pass


class TranscriptionError(VoiceNoteError):
    """Raised when the Whisper model fails to transcribe the audio file.
    Mapped to HTTP 500 Internal Server Error by the exception handler."""
    pass
