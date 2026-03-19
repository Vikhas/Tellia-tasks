"""
downloader.py — Async audio download service.

Downloads an audio file from a remote URL using ``httpx`` and
saves it to the local ``audio_files/`` directory with a unique
UUID-based filename to avoid collisions.
"""

import logging
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.exceptions import DownloadError

logger = logging.getLogger(__name__)


async def download_audio(url: str, dest_dir: Path) -> Path:
    """
    Download an audio file from a remote URL and save it locally.

    Parameters
    ----------
    url : str
        The public URL of the audio file to download.
    dest_dir : Path
        Local directory to save the downloaded file into.

    Returns
    -------
    Path
        The full path to the saved audio file on disk.

    Raises
    ------
    DownloadError
        If the HTTP request fails (network error, 4xx/5xx response, timeout).
    """
    # Extract the original file extension (e.g. ".mp3", ".wav")
    # from the URL path; default to ".audio" if none is found.
    ext = Path(urlparse(url).path).suffix or ".audio"

    # Generate a collision-safe filename using a random UUID
    dest_path = dest_dir / f"{uuid.uuid4().hex}{ext}"

    logger.info("Downloading audio from %s → %s", url, dest_path)

    try:
        # Use an async HTTP client with redirect support and a
        # generous 60-second timeout for large audio files.
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()           # Raise on 4xx / 5xx
            dest_path.write_bytes(response.content)  # Persist to disk
    except httpx.HTTPError as exc:
        # Wrap the httpx error in our domain-specific DownloadError
        # so the global exception handler can return a proper 502.
        raise DownloadError(
            message="Remote server returned an error.",
            detail=str(exc),
        ) from exc

    return dest_path
