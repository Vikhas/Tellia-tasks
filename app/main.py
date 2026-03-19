"""
main.py — FastAPI application entry point.

Defines all API routes, exception handlers, and the application
lifespan (startup) logic.  This is the central orchestrator that
wires together downloading, transcription, structuring, conflict
detection, and persistence.
"""

import logging
import uuid
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ── Internal modules ──────────────────────────────────────────────
from app.config import settings
from app.exceptions import DownloadError, TranscriptionError, VoiceNoteError
from app.schemas import VoiceNoteRequest, VoiceNoteResponse
from app.services.downloader import download_audio
from app.services.transcriber import transcribe
from app.services.structurer import structure_transcript, detect_conflict
from app.services.storage import (
    init_db,
    save_voice_note,
    get_voice_notes,
    get_recent_notes,
    update_conflict,
    delete_voice_note,
)

# ── Logging configuration ────────────────────────────────────────
# Sets up a human-readable log format with timestamps for all loggers.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Application Lifespan — runs once on server start
# ══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """
    Startup hook: ensures the audio storage directory exists
    and initialises the SQLite database schema before the first
    request is served.
    """
    settings.audio_dir.mkdir(parents=True, exist_ok=True)  # Create audio_files/ if absent
    init_db()                                               # Create DB table if absent
    yield  # Application runs; nothing to tear down on shutdown


# ══════════════════════════════════════════════════════════════════
# FastAPI app instance
# ══════════════════════════════════════════════════════════════════

app = FastAPI(title="Tellia — Voice Note Service", lifespan=lifespan)

# Mount the audio directory so saved files are directly playable
# via URLs like /audio_files/<filename>.mp3
app.mount("/audio_files", StaticFiles(directory="audio_files"), name="audio_files")


# ══════════════════════════════════════════════════════════════════
# Helper — Conflict Detection
# ══════════════════════════════════════════════════════════════════

def check_and_handle_conflicts(
    transcript: str, structured_data: dict
) -> tuple[Optional[str], Optional[str]]:
    """
    Compare a newly transcribed note against the 10 most recent
    notes in the database using the Groq LLM.

    If a semantic contradiction is detected (e.g. conflicting
    observations about the same crop/location), both the new note
    and the older conflicting note are tagged with a shared
    ``conflict_id`` and the LLM's reasoning string.

    Returns
    -------
    tuple[Optional[str], Optional[str]]
        (conflict_id, conflict_reasoning) or (None, None).
    """
    # Retrieve recent notes to serve as context for the LLM
    recent_notes = get_recent_notes(limit=10)

    # Ask the LLM whether the new note contradicts any recent note
    conflict_data = detect_conflict(transcript, structured_data or {}, recent_notes)

    if conflict_data.get("conflict"):
        # Generate a unique ID shared by both conflicting notes
        conflict_id = uuid.uuid4().hex
        conflict_reasoning = conflict_data.get("reasoning")
        conflicting_note_id = conflict_data.get("conflicting_note_id")

        # Flag the *older* note in the DB with the same conflict info
        if conflicting_note_id and conflict_reasoning:
            update_conflict(conflicting_note_id, conflict_id, str(conflict_reasoning))

        return conflict_id, str(conflict_reasoning) if conflict_reasoning else None

    return None, None


# ══════════════════════════════════════════════════════════════════
# Global Exception Handler
# ══════════════════════════════════════════════════════════════════

@app.exception_handler(VoiceNoteError)
async def handle_voicenote_error(_req: Request, exc: VoiceNoteError) -> JSONResponse:
    """
    Catch any VoiceNoteError (or subclass) and return a clean
    JSON error response instead of a raw server traceback.

    - DownloadError   → HTTP 502  (upstream fetch failure)
    - Other errors    → HTTP 500  (internal processing failure)
    """
    status = 502 if isinstance(exc, DownloadError) else 500
    return JSONResponse(
        status_code=status,
        content={"error": exc.message, "detail": exc.detail},
    )


# ══════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════

# ── Health check ──────────────────────────────────────────────────

@app.get("/health")
async def health_check() -> dict:
    """Simple liveness probe — returns ``{"status": "ok"}``."""
    return {"status": "ok"}


# ── POST /voice-note — URL-based audio processing ────────────────

@app.post("/voice-note", response_model=VoiceNoteResponse)
async def process_voice_note(payload: VoiceNoteRequest) -> VoiceNoteResponse:
    """
    Accept a JSON body with an ``audioUrl``, download the file,
    run the full pipeline (transcribe → structure → conflict check),
    persist the result, and return it.
    """
    # 1. Download audio from the remote URL to local disk
    audio_path = await download_audio(
        url=str(payload.audioUrl), dest_dir=settings.audio_dir
    )

    # 2. Transcribe the audio file → plain text
    transcript = transcribe(file_path=audio_path, model_name=settings.whisper_model)

    # 3. Send transcript to Groq LLM for structured extraction
    structured_data = structure_transcript(
        transcript, approach=payload.structuringApproach
    )

    # 4. Check for semantic conflicts with recent notes
    conflict_id, conflict_reasoning = check_and_handle_conflicts(
        transcript, structured_data
    )

    # 5. Build the response object
    response_obj = VoiceNoteResponse(
        deviceId=payload.deviceId,
        timestamp=payload.timestamp,
        transcript=transcript,
        audioPath=str(audio_path),
        structuredData=structured_data,
        conflictId=conflict_id,
        conflictReasoning=conflict_reasoning,
    )

    # 6. Persist in SQLite
    save_voice_note(response_obj)

    return response_obj


# ── GET / — Serve the web UI ─────────────────────────────────────

@app.get("/")
async def serve_ui() -> FileResponse:
    """Return the single-page HTML frontend."""
    return FileResponse(Path("app/static/index.html"))


# ── POST /api/upload — File-upload audio processing ──────────────

@app.post("/api/upload", response_model=VoiceNoteResponse)
async def upload_voice_note_file(
    file: UploadFile = File(...),
    deviceId: str = Form("web-browser"),
    timestamp: str = Form(...),
    structuringApproach: str = Form("predefined"),
) -> VoiceNoteResponse:
    """
    Accept a multipart file upload from the web UI (or any client),
    run the full pipeline, persist, and return the result.

    The pipeline is identical to ``POST /voice-note`` except the
    audio bytes come directly in the request instead of being
    downloaded from a URL.
    """
    # 1. Determine file extension and generate a unique filename
    ext = Path(file.filename).suffix if file.filename else ".audio"
    audio_path = settings.audio_dir / f"{uuid.uuid4().hex}{ext}"

    # 2. Stream the uploaded bytes to disk
    with audio_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3. Transcribe → Structure → Conflict-check  (same as /voice-note)
    transcript = transcribe(file_path=audio_path, model_name=settings.whisper_model)
    structured_data = structure_transcript(transcript, approach=structuringApproach)
    conflict_id, conflict_reasoning = check_and_handle_conflicts(
        transcript, structured_data
    )

    # 4. Build response and persist
    response_obj = VoiceNoteResponse(
        deviceId=deviceId,
        timestamp=timestamp,
        transcript=transcript,
        audioPath=str(audio_path),
        structuredData=structured_data,
        conflictId=conflict_id,
        conflictReasoning=conflict_reasoning,
    )
    save_voice_note(response_obj)

    return response_obj


# ── GET /api/notes — Retrieve history ────────────────────────────

@app.get("/api/notes")
async def fetch_history() -> dict:
    """Return all saved voice notes from the database (most recent first)."""
    return {"notes": get_voice_notes()}


# ── DELETE /api/notes/{note_id} — Delete a note ──────────────────

@app.delete("/api/notes/{note_id}")
async def delete_history(note_id: int) -> JSONResponse:
    """
    Delete a voice note by its database ID.

    Performs a two-step cleanup:
      1. Remove the row from the SQLite database.
      2. Delete the corresponding audio file from disk.
    """
    # Step 1 — Remove from DB and retrieve the audio file path
    audio_path = delete_voice_note(note_id)
    if not audio_path:
        return JSONResponse(status_code=404, content={"error": "Note not found"})

    # Step 2 — Clean up the audio file from local storage
    full_path = Path(audio_path)
    if full_path.exists():
        import os
        os.remove(full_path)

    return JSONResponse(content={"message": "Deleted successfully"})
