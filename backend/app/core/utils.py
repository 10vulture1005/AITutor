"""
Shared utility functions for Eduverse.

Provides:
  - detect_file_type(mime_type, file_name)  → normalized file type
  - detect_source_type(file_name)           → citation source label
  - validate_groq_key(key)                  → raises ValueError if invalid
  - create_groq_client(api_key)             → pre-configured Groq client
  - retrieve_and_filter(retriever, query)   → docs filtered by relevance threshold

Used by: tools.py, file_service.py, nodes.py, chat.py, files.py
"""

import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def detect_file_type(mime_type: str, file_name: str) -> str:
    """
    Detect normalized file type from MIME type and filename.

    Returns:
        "pdf", "video", "audio", "image", "text", or "unknown"
    """
    mime_lower = (mime_type or "").lower()
    name_lower = file_name.lower()

    if "pdf" in mime_lower or name_lower.endswith(".pdf"):
        return "pdf"

    if "video" in mime_lower or any(
        name_lower.endswith(ext)
        for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv")
    ):
        return "video"

    if "audio" in mime_lower or any(
        name_lower.endswith(ext)
        for ext in (".mp3", ".wav", ".m4a", ".ogg", ".aac", ".flac")
    ):
        return "audio"

    if "image" in mime_lower or any(
        name_lower.endswith(ext)
        for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff")
    ):
        return "image"

    if any(
        keyword in mime_lower
        for keyword in ("text", "document", "msword", "wordprocessing")
    ):
        return "text"

    return "unknown"


def detect_source_type(file_name: str) -> str:
    """
    Detect source type label for citations.

    Returns:
        "pdf", "video", "audio", or "document"
    """
    name = file_name.lower()
    if name.endswith(".pdf"):
        return "pdf"
    if any(name.endswith(e) for e in (".mp4", ".avi", ".mkv", ".mov", ".webm")):
        return "video"
    if any(name.endswith(e) for e in (".mp3", ".wav", ".m4a", ".ogg", ".aac")):
        return "audio"
    return "document"


# ── Groq helpers ──────────────────────────────────────────────────

def validate_groq_key(key: str) -> None:
    """Raise ValueError if *key* is empty or not a valid Groq key."""
    if not key or not key.startswith("gsk_"):
        raise ValueError("Invalid Groq API key. Must start with 'gsk_'.")


def create_groq_client(api_key: str):
    """Return a ready-to-use :class:`groq.Groq` client."""
    from groq import Groq  # late import keeps module lightweight
    return Groq(api_key=api_key)


# ── RAG helpers ───────────────────────────────────────────────────

def retrieve_and_filter(
    retriever,
    query: str,
    threshold: float = None,
) -> List[Document]:
    """Invoke *retriever* and drop results below the relevance threshold.

    Returns an empty list when nothing passes the filter.
    """
    from app.core.config import settings  # late import to avoid circular

    threshold = threshold if threshold is not None else settings.RAG_RELEVANCE_THRESHOLD
    docs = retriever.invoke(query)
    if not docs:
        return []

    relevant = [
        doc for doc in docs
        if doc.metadata.get("relevance_score", 0.0) >= threshold
    ]
    if relevant:
        logger.info(
            f"Relevance filter: {len(docs)} → {len(relevant)} docs "
            f"(threshold={threshold})"
        )
    else:
        scores = [round(d.metadata.get("relevance_score", 0.0), 3) for d in docs]
        logger.warning(
            f"Relevance filter: ALL {len(docs)} docs below threshold "
            f"{threshold} — scores: {scores}"
        )
    return relevant
