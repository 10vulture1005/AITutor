import asyncio
import logging
import os
import tempfile
from typing import List, Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_groq import ChatGroq
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _build_loader(
    file_path: str,
    groq_api_key: str,
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
) -> PyMuPDFLoader:
    """Build a PyMuPDFLoader with Groq Vision for image analysis."""
    vision_llm = ChatGroq(
        model=vision_model,
        api_key=groq_api_key,
        max_tokens=1024,
        temperature=0.2,
    )

    return PyMuPDFLoader(
        file_path,
        mode="page",
        images_inner_format="markdown-img",
        images_parser=LLMImageBlobParser(model=vision_llm),
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
    file_content: bytes,
    groq_api_key: str,
    file_name: str = "document.pdf",
    course_id: Optional[str] = None,
    source_id: Optional[str] = None,
) -> List[Document]:
    """
    Process PDF using LangChain PyMuPDFLoader + ChatGroq Vision.
    Extracts text per page, analyzes images with Groq Vision automatically.

    Returns:
        List of LangChain Documents with per-page content + metadata
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
        loader = _build_loader(tmp_path, groq_api_key)
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
    finally:
        os.unlink(tmp_path)
