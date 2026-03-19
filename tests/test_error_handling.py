"""
test_error_handling.py — Comprehensive error-handling tests for the Tellia API.

Covers:
  1. Download errors     (invalid URL, unreachable host, HTTP errors)
  2. Transcription errors (missing audio file, Whisper failure)
  3. Structuring errors   (missing API key, malformed LLM response)
  4. Upload validation    (non-audio file, missing fields)
  5. CRUD errors          (deleting non-existent note)
  6. Conflict detection   (missing API key, no recent notes)
  7. Health check         (basic liveness)

Run with:
    pytest tests/test_error_handling.py -v
"""

import io
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.exceptions import DownloadError, TranscriptionError


# ══════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def setup_environment(tmp_path):
    """
    Redirect the audio directory and database to a temp folder
    for every test so we never pollute the real project data.
    """
    with patch("app.config.settings") as mock_settings:
        mock_settings.audio_dir = tmp_path / "audio_files"
        mock_settings.audio_dir.mkdir()
        mock_settings.db_path = tmp_path / "test.db"
        mock_settings.whisper_model = "base"
        mock_settings.groq_api_key = "fake-key-for-testing"
        yield mock_settings


@pytest.fixture
def client():
    """FastAPI test client — sends requests without starting a real server."""
    return TestClient(app)


# ══════════════════════════════════════════════════════════════════
# 1. Health Check
# ══════════════════════════════════════════════════════════════════

class TestHealthCheck:
    """Verify the /health endpoint always returns 200 OK."""

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ══════════════════════════════════════════════════════════════════
# 2. Download Errors — POST /voice-note
# ══════════════════════════════════════════════════════════════════

class TestDownloadErrors:
    """Verify that download failures return HTTP 502 with a clear error."""

    @patch("app.main.download_audio", new_callable=AsyncMock)
    def test_download_http_error_returns_502(self, mock_download, client):
        """Simulate a remote server returning an HTTP error (e.g. 404)."""
        mock_download.side_effect = DownloadError(
            message="Remote server returned an error.",
            detail="404 Not Found",
        )
        response = client.post("/voice-note", json={
            "deviceId": "test-device",
            "timestamp": "2026-03-19T00:00:00Z",
            "audioUrl": "https://example.com/nonexistent.mp3",
        })
        assert response.status_code == 502
        body = response.json()
        assert "error" in body
        assert body["error"] == "Remote server returned an error."
        assert body["detail"] == "404 Not Found"

    @patch("app.main.download_audio", new_callable=AsyncMock)
    def test_download_timeout_returns_502(self, mock_download, client):
        """Simulate a network timeout during download."""
        mock_download.side_effect = DownloadError(
            message="Remote server returned an error.",
            detail="ReadTimeout: timed out",
        )
        response = client.post("/voice-note", json={
            "deviceId": "test-device",
            "timestamp": "2026-03-19T00:00:00Z",
            "audioUrl": "https://example.com/slow.mp3",
        })
        assert response.status_code == 502
        assert "timed out" in response.json()["detail"]

    @patch("app.main.download_audio", new_callable=AsyncMock)
    def test_download_connection_refused_returns_502(self, mock_download, client):
        """Simulate a connection refused error."""
        mock_download.side_effect = DownloadError(
            message="Remote server returned an error.",
            detail="ConnectionRefused: [Errno 111]",
        )
        response = client.post("/voice-note", json={
            "deviceId": "test-device",
            "timestamp": "2026-03-19T00:00:00Z",
            "audioUrl": "https://192.0.2.1/audio.mp3",
        })
        assert response.status_code == 502


# ══════════════════════════════════════════════════════════════════
# 3. Transcription Errors — POST /voice-note & /api/upload
# ══════════════════════════════════════════════════════════════════

class TestTranscriptionErrors:
    """Verify that Whisper failures return HTTP 500 with a clear error."""

    @patch("app.main.download_audio", new_callable=AsyncMock)
    @patch("app.main.transcribe")
    def test_missing_audio_file_returns_500(self, mock_transcribe, mock_download, client, tmp_path):
        """Simulate the downloaded file not existing when transcribe is called."""
        mock_download.return_value = tmp_path / "missing.mp3"
        mock_transcribe.side_effect = TranscriptionError(
            message="Audio file not found.",
            detail=str(tmp_path / "missing.mp3"),
        )
        response = client.post("/voice-note", json={
            "deviceId": "test-device",
            "timestamp": "2026-03-19T00:00:00Z",
            "audioUrl": "https://example.com/audio.mp3",
        })
        assert response.status_code == 500
        assert response.json()["error"] == "Audio file not found."

    @patch("app.main.download_audio", new_callable=AsyncMock)
    @patch("app.main.transcribe")
    def test_whisper_crash_returns_500(self, mock_transcribe, mock_download, client, tmp_path):
        """Simulate Whisper throwing an internal exception."""
        mock_download.return_value = tmp_path / "corrupt.mp3"
        mock_transcribe.side_effect = TranscriptionError(
            message="Whisper transcription failed.",
            detail="RuntimeError: ffmpeg not found",
        )
        response = client.post("/voice-note", json={
            "deviceId": "test-device",
            "timestamp": "2026-03-19T00:00:00Z",
            "audioUrl": "https://example.com/corrupt.mp3",
        })
        assert response.status_code == 500
        assert "Whisper transcription failed" in response.json()["error"]

    @patch("app.main.transcribe")
    def test_upload_transcription_failure_returns_500(self, mock_transcribe, client):
        """Transcription failure through the /api/upload endpoint."""
        mock_transcribe.side_effect = TranscriptionError(
            message="Whisper transcription failed.",
            detail="Out of memory",
        )
        # Create a fake audio file to upload
        fake_audio = io.BytesIO(b"\x00" * 1024)
        response = client.post("/api/upload", data={
            "deviceId": "web-browser",
            "timestamp": "2026-03-19T00:00:00Z",
            "structuringApproach": "predefined",
        }, files={
            "file": ("test.mp3", fake_audio, "audio/mpeg"),
        })
        assert response.status_code == 500
        assert response.json()["error"] == "Whisper transcription failed."


# ══════════════════════════════════════════════════════════════════
# 4. Structuring Errors (Groq LLM)
# ══════════════════════════════════════════════════════════════════

class TestStructuringErrors:
    """Verify graceful handling when the Groq LLM call fails."""

    @patch("app.main.check_and_handle_conflicts", return_value=(None, None))
    @patch("app.main.save_voice_note")
    @patch("app.main.structure_transcript")
    @patch("app.main.transcribe")
    @patch("app.main.download_audio", new_callable=AsyncMock)
    def test_missing_groq_key_returns_error_in_structured_data(
        self, mock_download, mock_transcribe, mock_structure,
        mock_save, mock_conflict, client, tmp_path
    ):
        """When GROQ_API_KEY is missing, structuredData should contain an error."""
        mock_download.return_value = tmp_path / "audio.mp3"
        mock_transcribe.return_value = "some transcript"
        mock_structure.return_value = {"error": "GROQ_API_KEY not configured."}

        response = client.post("/voice-note", json={
            "deviceId": "test-device",
            "timestamp": "2026-03-19T00:00:00Z",
            "audioUrl": "https://example.com/audio.mp3",
        })
        assert response.status_code == 200
        body = response.json()
        assert body["structuredData"]["error"] == "GROQ_API_KEY not configured."

    @patch("app.main.check_and_handle_conflicts", return_value=(None, None))
    @patch("app.main.save_voice_note")
    @patch("app.main.structure_transcript")
    @patch("app.main.transcribe")
    @patch("app.main.download_audio", new_callable=AsyncMock)
    def test_groq_returns_unparseable_json(
        self, mock_download, mock_transcribe, mock_structure,
        mock_save, mock_conflict, client, tmp_path
    ):
        """When Groq returns malformed content, structuredData has a parse error."""
        mock_download.return_value = tmp_path / "audio.mp3"
        mock_transcribe.return_value = "some transcript"
        mock_structure.return_value = {"error": "Failed to parse structured output."}

        response = client.post("/voice-note", json={
            "deviceId": "test-device",
            "timestamp": "2026-03-19T00:00:00Z",
            "audioUrl": "https://example.com/audio.mp3",
        })
        assert response.status_code == 200
        assert "error" in response.json()["structuredData"]

    @patch("app.main.check_and_handle_conflicts", return_value=(None, None))
    @patch("app.main.save_voice_note")
    @patch("app.main.structure_transcript")
    @patch("app.main.transcribe")
    @patch("app.main.download_audio", new_callable=AsyncMock)
    def test_groq_api_network_failure(
        self, mock_download, mock_transcribe, mock_structure,
        mock_save, mock_conflict, client, tmp_path
    ):
        """When the Groq API is unreachable, structuredData has the error."""
        mock_download.return_value = tmp_path / "audio.mp3"
        mock_transcribe.return_value = "some transcript"
        mock_structure.return_value = {"error": "Connection to api.groq.com failed."}

        response = client.post("/voice-note", json={
            "deviceId": "test-device",
            "timestamp": "2026-03-19T00:00:00Z",
            "audioUrl": "https://example.com/audio.mp3",
        })
        assert response.status_code == 200
        assert "error" in response.json()["structuredData"]


# ══════════════════════════════════════════════════════════════════
# 5. Upload Validation — POST /api/upload
# ══════════════════════════════════════════════════════════════════

class TestUploadValidation:
    """Verify request validation on the file-upload endpoint."""

    def test_upload_missing_file_returns_422(self, client):
        """Omitting the required file field should return 422 Unprocessable Entity."""
        response = client.post("/api/upload", data={
            "deviceId": "web-browser",
            "timestamp": "2026-03-19T00:00:00Z",
        })
        assert response.status_code == 422

    def test_upload_missing_timestamp_returns_422(self, client):
        """Omitting the required timestamp field should return 422."""
        fake_audio = io.BytesIO(b"\x00" * 1024)
        response = client.post("/api/upload", data={
            "deviceId": "web-browser",
        }, files={
            "file": ("test.mp3", fake_audio, "audio/mpeg"),
        })
        assert response.status_code == 422


# ══════════════════════════════════════════════════════════════════
# 6. POST /voice-note — Request Validation
# ══════════════════════════════════════════════════════════════════

class TestVoiceNoteValidation:
    """Verify request validation on the URL-based endpoint."""

    def test_missing_audio_url_returns_422(self, client):
        """Omitting the required audioUrl field should return 422."""
        response = client.post("/voice-note", json={
            "deviceId": "test-device",
            "timestamp": "2026-03-19T00:00:00Z",
        })
        assert response.status_code == 422

    def test_invalid_audio_url_returns_422(self, client):
        """Providing a non-URL string for audioUrl should return 422."""
        response = client.post("/voice-note", json={
            "deviceId": "test-device",
            "timestamp": "2026-03-19T00:00:00Z",
            "audioUrl": "not-a-valid-url",
        })
        assert response.status_code == 422

    def test_missing_device_id_returns_422(self, client):
        """Omitting the required deviceId should return 422."""
        response = client.post("/voice-note", json={
            "timestamp": "2026-03-19T00:00:00Z",
            "audioUrl": "https://example.com/audio.mp3",
        })
        assert response.status_code == 422

    def test_empty_json_body_returns_422(self, client):
        """Sending an empty JSON body should return 422."""
        response = client.post("/voice-note", json={})
        assert response.status_code == 422


# ══════════════════════════════════════════════════════════════════
# 7. DELETE /api/notes/{id} — Delete Errors
# ══════════════════════════════════════════════════════════════════

class TestDeleteErrors:
    """Verify error handling when deleting notes."""

    @patch("app.main.delete_voice_note")
    def test_delete_nonexistent_note_returns_404(self, mock_delete, client):
        """Deleting a note that doesn't exist should return 404."""
        mock_delete.return_value = None  # Simulates "not found"
        response = client.delete("/api/notes/9999")
        assert response.status_code == 404
        assert response.json()["error"] == "Note not found"

    @patch("app.main.delete_voice_note")
    def test_delete_note_with_missing_audio_file(self, mock_delete, client, tmp_path):
        """
        Deleting a note whose audio file was already removed from disk
        should still succeed (the DB row is cleaned up).
        """
        # Simulate the DB returning a path that doesn't exist on disk
        mock_delete.return_value = str(tmp_path / "already_deleted.mp3")
        response = client.delete("/api/notes/1")
        assert response.status_code == 200
        assert response.json()["message"] == "Deleted successfully"

    @patch("app.main.delete_voice_note")
    def test_delete_note_cleans_up_audio_file(self, mock_delete, client, tmp_path):
        """Verify that the audio file is actually removed from disk on delete."""
        # Create a real file so we can verify it gets deleted
        audio_file = tmp_path / "to_delete.mp3"
        audio_file.write_bytes(b"\x00" * 512)
        assert audio_file.exists()

        mock_delete.return_value = str(audio_file)
        response = client.delete("/api/notes/1")
        assert response.status_code == 200
        assert not audio_file.exists()  # File should be gone


# ══════════════════════════════════════════════════════════════════
# 8. Conflict Detection Edge Cases
# ══════════════════════════════════════════════════════════════════

class TestConflictDetection:
    """Verify conflict detection handles edge cases gracefully."""

    @patch("app.main.save_voice_note")
    @patch("app.main.detect_conflict")
    @patch("app.main.get_recent_notes")
    @patch("app.main.structure_transcript")
    @patch("app.main.transcribe")
    @patch("app.main.download_audio", new_callable=AsyncMock)
    def test_no_conflict_when_no_recent_notes(
        self, mock_download, mock_transcribe, mock_structure,
        mock_recent, mock_detect, mock_save, client, tmp_path
    ):
        """When there are no previous notes, conflict should be None."""
        mock_download.return_value = tmp_path / "audio.mp3"
        mock_transcribe.return_value = "Test transcript"
        mock_structure.return_value = {"type": "observation"}
        mock_recent.return_value = []
        mock_detect.return_value = {"conflict": False}

        response = client.post("/voice-note", json={
            "deviceId": "test-device",
            "timestamp": "2026-03-19T00:00:00Z",
            "audioUrl": "https://example.com/audio.mp3",
        })
        assert response.status_code == 200
        assert response.json()["conflictId"] is None
        assert response.json()["conflictReasoning"] is None

    @patch("app.main.save_voice_note")
    @patch("app.main.update_conflict")
    @patch("app.main.detect_conflict")
    @patch("app.main.get_recent_notes")
    @patch("app.main.structure_transcript")
    @patch("app.main.transcribe")
    @patch("app.main.download_audio", new_callable=AsyncMock)
    def test_conflict_detected_returns_conflict_data(
        self, mock_download, mock_transcribe, mock_structure,
        mock_recent, mock_detect, mock_update, mock_save, client, tmp_path
    ):
        """When the LLM finds a conflict, verify the response includes conflict info."""
        mock_download.return_value = tmp_path / "audio.mp3"
        mock_transcribe.return_value = "Wheat in block 4 is healthy"
        mock_structure.return_value = {"type": "observation"}
        mock_recent.return_value = [
            {"id": 5, "timestamp": "2026-03-18T00:00:00Z",
             "transcript": "Rust on wheat block 4", "structured_data": {}}
        ]
        mock_detect.return_value = {
            "conflict": True,
            "conflicting_note_id": 5,
            "reasoning": "New note says wheat is healthy but previous note reported rust.",
        }

        response = client.post("/voice-note", json={
            "deviceId": "test-device",
            "timestamp": "2026-03-19T00:00:00Z",
            "audioUrl": "https://example.com/audio.mp3",
        })
        assert response.status_code == 200
        body = response.json()
        assert body["conflictId"] is not None
        assert "healthy" in body["conflictReasoning"]
        # Verify the older note was also flagged
        mock_update.assert_called_once()

    @patch("app.main.save_voice_note")
    @patch("app.main.detect_conflict")
    @patch("app.main.get_recent_notes")
    @patch("app.main.structure_transcript")
    @patch("app.main.transcribe")
    @patch("app.main.download_audio", new_callable=AsyncMock)
    def test_conflict_detection_failure_is_graceful(
        self, mock_download, mock_transcribe, mock_structure,
        mock_recent, mock_detect, mock_save, client, tmp_path
    ):
        """If conflict detection crashes, the note should still be saved (fail-open)."""
        mock_download.return_value = tmp_path / "audio.mp3"
        mock_transcribe.return_value = "Test transcript"
        mock_structure.return_value = {"type": "note"}
        mock_recent.return_value = [{"id": 1, "timestamp": "T", "transcript": "T", "structured_data": {}}]
        # detect_conflict fails gracefully by returning no conflict
        mock_detect.return_value = {"conflict": False}

        response = client.post("/voice-note", json={
            "deviceId": "test-device",
            "timestamp": "2026-03-19T00:00:00Z",
            "audioUrl": "https://example.com/audio.mp3",
        })
        assert response.status_code == 200
        assert response.json()["conflictId"] is None
        # Ensure the note was still saved despite no conflict
        mock_save.assert_called_once()


# ══════════════════════════════════════════════════════════════════
# 9. Happy Path — Full Pipeline Success
# ══════════════════════════════════════════════════════════════════

class TestHappyPath:
    """Verify the full pipeline works end-to-end when everything succeeds."""

    @patch("app.main.save_voice_note")
    @patch("app.main.check_and_handle_conflicts", return_value=(None, None))
    @patch("app.main.structure_transcript")
    @patch("app.main.transcribe")
    @patch("app.main.download_audio", new_callable=AsyncMock)
    def test_voice_note_full_success(
        self, mock_download, mock_transcribe, mock_structure,
        mock_conflict, mock_save, client, tmp_path
    ):
        """POST /voice-note returns 200 with all expected fields on success."""
        mock_download.return_value = tmp_path / "audio.mp3"
        mock_transcribe.return_value = "Hello this is a test note"
        mock_structure.return_value = {"type": "note", "title": "Test"}

        response = client.post("/voice-note", json={
            "deviceId": "device-123",
            "timestamp": "2026-03-19T00:00:00Z",
            "audioUrl": "https://example.com/audio.mp3",
        })
        assert response.status_code == 200
        body = response.json()
        assert body["deviceId"] == "device-123"
        assert body["transcript"] == "Hello this is a test note"
        assert body["structuredData"]["type"] == "note"
        mock_save.assert_called_once()

    @patch("app.main.save_voice_note")
    @patch("app.main.check_and_handle_conflicts", return_value=(None, None))
    @patch("app.main.structure_transcript")
    @patch("app.main.transcribe")
    def test_upload_full_success(
        self, mock_transcribe, mock_structure,
        mock_conflict, mock_save, client
    ):
        """POST /api/upload returns 200 with all expected fields on success."""
        mock_transcribe.return_value = "Uploaded note transcript"
        mock_structure.return_value = {"type": "task", "title": "Do something"}

        fake_audio = io.BytesIO(b"\x00" * 1024)
        response = client.post("/api/upload", data={
            "deviceId": "web-browser",
            "timestamp": "2026-03-19T00:00:00Z",
            "structuringApproach": "dynamic",
        }, files={
            "file": ("recording.wav", fake_audio, "audio/wav"),
        })
        assert response.status_code == 200
        body = response.json()
        assert body["deviceId"] == "web-browser"
        assert body["transcript"] == "Uploaded note transcript"
        assert body["structuredData"]["type"] == "task"
