import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status

from app.api.routes.auth import get_current_user
from app.core.utils import validate_groq_key
from app.processing.audio_processor import process_audio_bytes, SUPPORTED_AUDIO_FORMATS
from app.processing.image_processor import process_image, SUPPORTED_IMAGE_FORMATS
from app.processing.pdf_processor import process_pdf
from app.processing.semantic_merger import SemanticMerger
from app.processing.video_processor import process_video_bytes, SUPPORTED_VIDEO_FORMATS

logger = logging.getLogger(__name__)
router = APIRouter()

ALL_SUPPORTED = {'.pdf'} | SUPPORTED_IMAGE_FORMATS | SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS
MAX_UPLOAD = 100 * 1024 * 1024  

def _file_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.pdf': return 'pdf'
    if ext in SUPPORTED_IMAGE_FORMATS: return 'image'
    if ext in SUPPORTED_AUDIO_FORMATS: return 'audio'
    if ext in SUPPORTED_VIDEO_FORMATS: return 'video'
    return 'unknown'

@router.post("/process")
async def process_file(
    file: UploadFile = File(...),
    course_id: Optional[str] = Form(None),
    course_name: Optional[str] = Form(None),
    x_groq_api_key: str = Header(..., alias="X-Groq-Api-Key"),
    _user = Depends(get_current_user),
):
    """
    Upload any file (PDF/image/audio/video) → auto-detect → process → chunked documents.
    Requires Groq API key in X-Groq-Api-Key header.
    """
    try:
        validate_groq_key(x_groq_api_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    ftype = _file_type(file.filename or "unknown")
    if ftype == 'unknown':
        raise HTTPException(status_code=400, detail=f"Unsupported format. Supported: {sorted(ALL_SUPPORTED)}")

    content = await file.read()
    if len(content) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {MAX_UPLOAD // 1024 // 1024}MB")
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    fname = file.filename or "uploaded_file"

    try:
        if ftype == 'pdf':
            docs = await process_pdf(content, x_groq_api_key, fname, course_id)
        elif ftype == 'image':
            docs = [await process_image(content, x_groq_api_key, fname, course_id)]
        elif ftype == 'audio':
            docs = await process_audio_bytes(content, x_groq_api_key, fname, course_id)
        elif ftype == 'video':
            docs = await process_video_bytes(content, x_groq_api_key, fname, course_id)
        else:
            raise HTTPException(status_code=400, detail=f"Cannot process: {ftype}")

        chunked = SemanticMerger().merge_and_chunk(docs, course_id, course_name)

        return {
            "status": "success",
            "file_name": fname,
            "file_type": ftype,
            "raw_documents": len(docs),
            "chunked_documents": len(chunked),
            "documents": [{"text": d.page_content, "metadata": d.metadata} for d in chunked],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed for '{fname}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


@router.get("/supported-formats")
async def get_supported_formats():
    return {
        "pdf": [".pdf"],
        "image": sorted(SUPPORTED_IMAGE_FORMATS),
        "audio": sorted(SUPPORTED_AUDIO_FORMATS),
        "video": sorted(SUPPORTED_VIDEO_FORMATS),
    }