"""
schemas.py — Pydantic models for API request / response validation.

These models are used by FastAPI to:
  • Automatically validate incoming JSON payloads.
  • Serialize outgoing responses to JSON.
  • Generate the interactive OpenAPI (Swagger) documentation.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, HttpUrl


class VoiceNoteRequest(BaseModel):
    """
    Incoming payload for the ``POST /voice-note`` endpoint.

    Attributes
    ----------
    deviceId : str
        Identifier of the device that recorded the voice note
        (e.g. ``"device-abc-123"``).
    timestamp : str
        ISO-8601 datetime string indicating when the note was recorded.
    audioUrl : HttpUrl
        Public URL pointing to the audio file to download.
    structuringApproach : str
        Controls how the LLM structures the transcript:
        ``"predefined"`` = fixed schema  |  ``"dynamic"`` = AI-generated schema.
    """

    deviceId: str
    timestamp: str
    audioUrl: HttpUrl
    structuringApproach: str = "predefined"


class VoiceNoteResponse(BaseModel):
    """
    Response returned after processing a voice note.

    Attributes
    ----------
    deviceId : str
        Echo of the source device identifier.
    timestamp : str
        Echo of the recording timestamp.
    transcript : str
        Plain text produced by Whisper (or the mock fallback).
    audioPath : str
        Relative path to the saved audio file on disk
        (e.g. ``"audio_files/a1b2c3d4.mp3"``).
    structuredData : Optional[Dict[str, Any]]
        JSON object produced by the Groq LLM — structured
        extraction of the transcript content.
    conflictId : Optional[str]
        UUID hex string shared between two notes that contradict
        each other.  ``None`` if no conflict was detected.
    conflictReasoning : Optional[str]
        LLM-generated explanation of the semantic conflict.
        ``None`` if no conflict was detected.
    """

    deviceId: str
    timestamp: str
    transcript: str
    audioPath: str
    structuredData: Optional[Dict[str, Any]] = None
    conflictId: Optional[str] = None
    conflictReasoning: Optional[str] = None
