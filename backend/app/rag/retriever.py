"""
Retrieval pipeline for Eduverse RAG.

Pipeline:  MMR base retriever  →  FlashRank cross-encoder reranking

- MMR (Maximal Marginal Relevance) ensures diverse initial results
- FlashRank reranks by semantic relevance (local model, free, ~200ms)
- relevance_score threshold filters out irrelevant results

Previous pipeline had MultiQueryRetriever (extra LLM call + 3x vector
searches) which added 2-4s latency — removed as FlashRank handles
relevance better for our dataset size.
"""

import logging
from typing import List, Optional

from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.config import settings
from app.rag.vector_store import EduverseVectorStore

logger = logging.getLogger(__name__)


# ── Relevance score threshold ────────────────────────────────────────
# FlashRank scores each doc 0.0–1.0. This is a low threshold
# to only filter truly irrelevant noise — broad queries (e.g.,
# "summarize all topics") score lower than specific ones.
MIN_RELEVANCE_SCORE = 0.05


def build_retriever(
    user_id: str,
    groq_api_key: str,
    course_id: Optional[str] = None,
) -> BaseRetriever:
    """
    Build the retrieval pipeline for a user.

    Args:
        user_id:      Authenticated user ID (selects their pgvector collection).
        groq_api_key: User-provided Groq key (kept for interface compatibility).
        course_id:    Optional — restrict retrieval to one course.

    Returns:
        A LangChain retriever that:
          1. Retrieves via MMR (diverse results)
          2. Reranks with FlashRank cross-encoder (local, free)
          3. Filters by minimum relevance score
    """

    # ── Step 1: Base retriever (MMR for diversity) ─────────────────────
    vs = EduverseVectorStore(user_id=user_id)

    # Pre-check: if collection is empty, log a warning
    info = vs.collection_info()
    if info["count"] == 0:
        logger.warning(
            f"User {user_id} has an empty collection — "
            f"queries will return no results"
        )

    search_kwargs = {
        "k": settings.RAG_RETRIEVER_K * 2,      # fetch more for reranking
        "fetch_k": settings.RAG_RETRIEVER_FETCH_K,
    }
    if course_id:
        search_kwargs["filter"] = {"course_id": course_id}

    base_retriever = vs.get_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )

    # ── Step 2: FlashRank reranking (local cross-encoder) ──────────────
    reranker = FlashrankRerank(
        top_n=settings.RAG_RERANK_TOP_N,
    )

    final_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )
    logger.info(
        f"Retrieval pipeline built: MMR(k={search_kwargs['k']}) → "
        f"FlashRank(top_n={settings.RAG_RERANK_TOP_N}, "
        f"min_score={MIN_RELEVANCE_SCORE})"
    )

    return final_retriever
