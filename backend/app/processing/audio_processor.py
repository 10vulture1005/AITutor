import asyncio
import logging
import os
import tempfile
from typing import List, Optional

from langchain_core.documents import Document

from app.core.utils import create_groq_client
from app.processing.text_cleaner import clean_transcription

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm', '.mp4'}
MAX_FILE_SIZE = 25 * 1024 * 1024  
SEGMENT_DURATION = 30  

def _transcribe_sync(audio_path: str, groq_api_key: str) -> dict:
    """Synchronous Whisper transcription — called via asyncio.to_thread()."""
    client = create_groq_client(groq_api_key)

    with open(audio_path, 'rb') as f:
        response = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f),
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments = []
    if hasattr(response, 'segments') and response.segments:
        for seg in response.segments:
            segments.append({
                "start": getattr(seg, 'start', 0),
                "end": getattr(seg, 'end', 0),
                "text": getattr(seg, 'text', '').strip(),
            })

    return {
        "text": response.text or "",
        "segments": segments,
        "duration": getattr(response, 'duration', None),
    }


async def _transcribe(audio_path: str, groq_api_key: str) -> dict:
    """Transcribe audio via Groq Whisper API with timestamps (non-blocking)."""
    try:
        return await asyncio.to_thread(_transcribe_sync, audio_path, groq_api_key)
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return {"text": "", "segments": [], "duration": None}


def _group_segments(segments: List[dict], target_duration: int = SEGMENT_DURATION) -> List[dict]:
    """Group short Whisper segments into ~30s chunks for meaningful embeddings."""
    if not segments:
        return []

    groups, current = [], {"start": segments[0]["start"], "end": segments[0]["end"], "texts": [segments[0]["text"]]}

    for seg in segments[1:]:
        if seg["end"] - current["start"] > target_duration and current["texts"]:
            groups.append({"start": current["start"], "end": current["end"], "text": " ".join(current["texts"])})
            current = {"start": seg["start"], "end": seg["end"], "texts": [seg["text"]]}
        else:
            current["end"] = seg["end"]
            current["texts"].append(seg["text"])

    if current["texts"]:
        groups.append({"start": current["start"], "end": current["end"], "text": " ".join(current["texts"])})

    return groups


async def process_audio(
    audio_path: str,
    groq_api_key: str,
    file_name: Optional[str] = None,
    course_id: Optional[str] = None,
    source_id: Optional[str] = None,
) -> List[Document]:
    """Transcribe audio → time-segmented LangChain Documents."""
    file_name = file_name or os.path.basename(audio_path)
    source = source_id or file_name

    result = await _transcribe(audio_path, groq_api_key)
    if not result["text"]:
        return []

    documents = []
    grouped = _group_segments(result["segments"]) if result["segments"] else []

    if grouped:
        for g in grouped:
            text = clean_transcription(g["text"])
            if text:
                documents.append(Document(
                    page_content=f"[AUDIO] {text}",
                    metadata={
                        "source_type": "audio", "source_id": source, "file_name": file_name,
                        "course_id": course_id, "page_number": None,
                        "start_time": round(g["start"], 1), "end_time": round(g["end"], 1),
                        "contains_visual": False,
                    }
                ))
    else:
        text = clean_transcription(result["text"])
        if text:
            documents.append(Document(
                page_content=f"[AUDIO] {text}",
                metadata={
                    "source_type": "audio", "source_id": source, "file_name": file_name,
                    "course_id": course_id, "page_number": None,
                    "start_time": 0, "end_time": result.get("duration"),
                    "contains_visual": False,
                }
            ))

    logger.info(f"Processed audio '{file_name}': {len(documents)} segments")
    return documents


async def process_audio_bytes(
    audio_bytes: bytes, groq_api_key: str,
    file_name: str = "audio.mp3", course_id: Optional[str] = None, source_id: Optional[str] = None,
) -> List[Document]:
    """Process audio from bytes (uploaded file)."""
    ext = os.path.splitext(file_name)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        return await process_audio(tmp_path, groq_api_key, file_name, course_id, source_id)
    finally:
        os.unlink(tmp_path)