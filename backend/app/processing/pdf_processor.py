import asyncio
import logging
from typing import List, Optional

from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

from app.core.config import settings
from app.processing.image_processor import create_vision_llm

logger = logging.getLogger(__name__)


def _build_loader(
    file_path: str,
    groq_api_key: str,
    vision_model: str = None,
) -> PyMuPDF4LLMLoader:
    """Build a PyMuPDF4LLMLoader with Groq Vision for image analysis.

    Produces true Markdown output (# headings, **bold**, |tables|,
    multi-column reflow) which works with the markdown-aware
    separators in SemanticMerger for structure-preserving chunking.
    """
    vision_llm = create_vision_llm(groq_api_key, vision_model)

    return PyMuPDF4LLMLoader(
        file_path,
        mode="page",
        extract_images=True,
        images_parser=LLMImageBlobParser(model=vision_llm),
        # pymupdf4llm kwargs — "lines" is less strict than default
        # "lines_strict" and handles more table styles in student PDFs
        table_strategy="lines",
    )


def _enrich_metadata(doc: Document, file_name: str, course_id: Optional[str], source_id: Optional[str]) -> Document:
    """Add Eduverse schema fields to PyMuPDFLoader's metadata."""
    content = doc.page_content
    has_visual = "[VISUAL]" in content or "![" in content

    doc.metadata.update({
        "source_type": "pdf",
        "source_id": source_id or file_name,
        "file_name": file_name,
        "course_id": course_id,
        "page_number": doc.metadata.get("page", 0) + 1,
        "start_time": None,
        "end_time": None,
        "contains_visual": has_visual,
    })
    return doc


async def process_pdf(
    file_path: str,
    groq_api_key: str,
    file_name: str = "document.pdf",
    course_id: Optional[str] = None,
    source_id: Optional[str] = None,
) -> List[Document]:
    """
    Process PDF using LangChain PyMuPDFLoader + ChatGroq Vision.
    Extracts text per page, analyzes images with Groq Vision automatically.

    Args:
        file_path: Path to the PDF file on disk (avoids temp-file round trip).

    Returns:
        List of LangChain Documents with per-page content + metadata
    """
    try:
        loader = _build_loader(file_path, groq_api_key)
        # loader.load() is synchronous (PyMuPDF + Groq Vision) — run in thread
        docs = await asyncio.to_thread(loader.load)

        enriched = [
            _enrich_metadata(doc, file_name, course_id, source_id)
            for doc in docs
            if doc.page_content.strip()
        ]

        logger.info(f"Processed PDF '{file_name}': {len(enriched)} pages")
        return enriched

    except Exception as e:
        logger.error(f"PDF processing failed for '{file_name}': {e}")
        raise RuntimeError(f"PDF processing failed: {e}") from e
