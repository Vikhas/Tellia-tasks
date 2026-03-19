"""
structurer.py — LLM-based transcript structuring and conflict detection.

Uses the Groq API (LLaMA 3.3 70B Versatile) to:
  1. Convert plain-text transcripts into structured JSON payloads.
  2. Detect semantic contradictions between a new note and recent
     historical notes (especially useful in agriculture / field work).

Two structuring approaches are supported:
  • **Predefined**: Forces the LLM to classify into fixed types
    (task, observation, reminder, note).
  • **Dynamic**: Lets the LLM invent the best-fit JSON schema
    on the fly (e.g. meeting_summary, shopping_list, diary_entry).
"""

import json
import logging
from typing import Any, Dict

from groq import Groq

from app.config import settings
from app.exceptions import VoiceNoteError

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Transcript Structuring
# ══════════════════════════════════════════════════════════════════

def structure_transcript(
    transcript: str, approach: str = "predefined"
) -> Dict[str, Any]:
    """
    Send a plain-text transcript to the Groq LLM and receive a
    structured JSON extraction.

    Parameters
    ----------
    transcript : str
        The plain text extracted from the audio voice note.
    approach : str
        ``"predefined"`` — map to fixed types (task / observation / etc.)
        ``"dynamic"``    — let the AI generate the best schema.

    Returns
    -------
    dict
        Parsed JSON object with at least a ``"type"`` key.
        On error, returns ``{"error": "<description>"}``.
    """
    # ── Guard: API key must be configured ─────────────────────────
    if not settings.groq_api_key:
        logger.warning(
            "GROQ_API_KEY environment variable is not set. "
            "Returning empty structured data."
        )
        return {"error": "GROQ_API_KEY not configured."}

    # Initialise the Groq client with the configured API key
    client = Groq(api_key=settings.groq_api_key)

    # ── Build the system prompt based on the chosen approach ──────
    if approach == "predefined":
        # Constrain the LLM to a fixed set of output types
        system_prompt = (
            "You are a helpful data-extraction assistant. "
            "Analyze the transcript and strictly map it to one of these types: "
            "'task', 'observation', 'reminder', or 'note'. "
            "Include relevant supporting properties for that type "
            "(e.g. 'scheduledTime', 'title', 'issue', 'location', 'entities'). "
            "Your entire output MUST be a valid JSON object starting with a 'type' key."
        )
    else:
        # Let the LLM freely determine the best schema
        system_prompt = (
            "You are an AI tasked with converting raw voice-note transcripts "
            "into highly structured data. "
            "Invent the best possible JSON schema to logically represent the user's input. "
            "Determine the core 'type' (e.g., 'observation', 'meeting_summary', "
            "'shopping_list', 'diary_entry', 'code_idea') "
            "and create an 'entities' object containing any relevant extracted data points. "
            "Your entire output MUST be a valid JSON object starting with a 'type' key."
        )

    # ── Call the Groq API ─────────────────────────────────────────
    try:
        logger.info("Structuring transcript using approach '%s' on Groq ...", approach)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            # Force the model to return valid JSON (no markdown wrapping)
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript: {transcript}"},
            ],
            temperature=0.1,  # Low temperature for deterministic, consistent output
        )

        # Extract the text content from the first choice
        content = response.choices[0].message.content
        if not content:
            return {"error": "Empty response from Groq."}

        # Parse the JSON string into a Python dict
        return json.loads(content)

    except json.JSONDecodeError as e:
        logger.error("Failed to parse Groq response as JSON: %s", e)
        return {"error": "Failed to parse structured output."}
    except Exception as e:
        logger.error("Failed to structure transcript via Groq: %s", e)
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════
# Semantic Conflict Detection
# ══════════════════════════════════════════════════════════════════

def detect_conflict(
    new_transcript: str,
    new_structured_data: Dict[str, Any],
    recent_notes: list,
) -> Dict[str, Any]:
    """
    Use the Groq LLM to determine whether a new voice note
    semantically contradicts any of the recent historical notes.

    This is tailored for agricultural field observations where
    phonetic transcription errors are common (e.g. "weed" vs "wheat").

    Parameters
    ----------
    new_transcript : str
        Plain text of the newly created note.
    new_structured_data : Dict[str, Any]
        Structured JSON extraction of the new note.
    recent_notes : list
        List of recent note dicts from the database, each containing
        ``id``, ``timestamp``, ``transcript``, and ``structured_data``.

    Returns
    -------
    dict
        ``{"conflict": False}`` if no contradiction found, or
        ``{"conflict": True, "conflicting_note_id": <int>,
        "reasoning": "<explanation>"}`` if one is found.
    """
    # ── Guard: API key must be configured ─────────────────────────
    if not settings.groq_api_key:
        logger.warning(
            "GROQ_API_KEY environment variable is not set. "
            "Cannot detect conflicts."
        )
        return {"conflict": False, "error": "GROQ_API_KEY not configured."}

    # Initialise the Groq client
    client = Groq(api_key=settings.groq_api_key)

    # No history to compare against → no possible conflict
    if not recent_notes:
        return {"conflict": False}

    # ── Build the conflict-detection prompts ──────────────────────
    system_prompt = (
        "You are an intelligent agricultural data assistant. "
        "Your job is to compare a NEW observation against a list of RECENT observations "
        "and determine if there is a semantic conflict (a contradiction in facts about the same entity/location). "
        "CRITICAL INSTRUCTION: Transcripts are generated by voice AI and often contain phonetic errors "
        "(e.g., 'weed' vs 'wheat', 'block' vs 'flock'). "
        "If two entities sound similar and occupy the same location in context, "
        "assume they refer to the exact same subject. "
        "Return a JSON object with 'conflict' (boolean). If a contradiction is found, "
        "also include 'conflicting_note_id' (integer, the id of the older contradicted note) "
        "and 'reasoning' (string explaining the contradiction).\n"
        'Example output: {"conflict": true, "conflicting_note_id": 12, '
        '"reasoning": "New note says wheat is healthy, but previous note reported rust '
        '(spelled as weed due to typo)."}\n'
        'Example output: {"conflict": false}'
    )

    # Compose the user prompt with the new note + all recent notes
    user_prompt = (
        f"NEW NOTE TRANSCRIPT: {new_transcript}\n"
        f"NEW NOTE STRUCTURED DATA: {json.dumps(new_structured_data)}\n\n"
        f"RECENT NOTES HISTORY:\n"
    )

    # Append each historical note as context for the LLM
    for note in recent_notes:
        user_prompt += (
            f"ID: {note['id']}, "
            f"Timestamp: {note['timestamp']}, "
            f"Transcript: {note['transcript']}, "
            f"Structured Data: {json.dumps(note.get('structured_data'))}\n"
        )

    # ── Call the Groq API ─────────────────────────────────────────
    try:
        logger.info(
            "Checking for semantic conflicts against %d recent notes...",
            len(recent_notes),
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for reliable, consistent detection
        )

        content = response.choices[0].message.content
        if not content:
            return {"conflict": False}

        # Parse the LLM's JSON response
        return json.loads(content)

    except Exception as e:
        # On any failure, fail open — assume no conflict rather than
        # blocking the pipeline.
        logger.error("Failed to detect conflict via Groq: %s", e)
        return {"conflict": False}
