"""
Advanced retrieval pipeline for Eduverse RAG.

Pipeline:  [BM25 (keyword) + MMR (semantic)] → Merge → FlashRank reranking

- BM25 catches exact keyword matches (function names, technical terms)
- MMR (Maximal Marginal Relevance) captures semantic meaning + diversity
- EnsembleRetriever merges results with configurable weights (0.3/0.7)
- FlashRank cross-encoder reranks the merged list by true relevance

Caching:
- Built retrievers are cached per (user_id, course_id) with a 5-min TTL.
- Cache is invalidated automatically by TTL or manually via invalidate_retriever_cache().
"""

import logging
import time
from typing import Optional

from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever

from app.core.config import settings
from app.rag.vector_store import EduverseVectorStore

logger = logging.getLogger(__name__)


# ── Retriever cache with TTL ─────────────────────────────────────
# Avoids rebuilding BM25 index + FlashRank on every chat query.
# Key: (user_id, course_id)  Value: (retriever, timestamp)
_retriever_cache: dict[tuple, tuple[BaseRetriever, float]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes


def invalidate_retriever_cache(user_id: str, course_id: Optional[str] = None) -> None:
    """
    Invalidate cached retriever for a user.

    Called after indexing completes to ensure fresh results.
    If course_id is None, invalidates ALL caches for this user.
    """
    if course_id is not None:
        _retriever_cache.pop((user_id, course_id), None)
    else:
        # Invalidate all entries for this user
        keys_to_remove = [k for k in _retriever_cache if k[0] == user_id]
        for key in keys_to_remove:
            _retriever_cache.pop(key, None)
    logger.info(f"Invalidated retriever cache for user={user_id}, course={course_id}")


def build_retriever(
    user_id: str,
    groq_api_key: str,
    course_id: Optional[str] = None,
) -> BaseRetriever:
    """
    Build the hybrid retrieval pipeline for a user.

    Uses a TTL-based cache (5 min) to avoid rebuilding the BM25 index
    and FlashRank reranker on every query. Cache is invalidated
    automatically by TTL or manually after re-indexing.

    Args:
        user_id:      Authenticated user ID (selects their pgvector collection).
        groq_api_key: User-provided Groq key (kept for interface compatibility).
        course_id:    Optional — restrict retrieval to one course.

    Returns:
        A LangChain retriever that:
          1. Retrieves via BM25 (keyword matching)
          2. Retrieves via MMR (semantic + diverse)
          3. Merges results (30% BM25 / 70% semantic)
          4. Reranks with FlashRank cross-encoder (local, free)
    """
    cache_key = (user_id, course_id)

    # Check cache
    if cache_key in _retriever_cache:
        cached_retriever, cached_at = _retriever_cache[cache_key]
        if time.time() - cached_at < _CACHE_TTL_SECONDS:
            logger.debug(f"Using cached retriever for user={user_id}")
            return cached_retriever
        else:
            # Expired
            del _retriever_cache[cache_key]

    # Build fresh retriever
    retriever = _build_retriever_pipeline(user_id, course_id)

    # Cache it
    _retriever_cache[cache_key] = (retriever, time.time())

    return retriever


def _build_retriever_pipeline(
    user_id: str,
    course_id: Optional[str] = None,
) -> BaseRetriever:
    """Internal: build the full retriever pipeline (uncached)."""

    vs = EduverseVectorStore(user_id=user_id)

    # Pre-check: if collection is empty, return a simple vector retriever
    info = vs.collection_info()
    if info["count"] == 0:
        logger.warning(
            f"User {user_id} has an empty collection — "
            f"queries will return no results"
        )
        return vs.get_retriever(search_type="mmr", search_kwargs={"k": 5})

    # ── Step 1: Vector retriever (MMR for diversity) ───────────────
    search_kwargs = {
        "k": settings.RAG_RETRIEVER_K * 2,      # fetch more for reranking
        "fetch_k": settings.RAG_RETRIEVER_FETCH_K,
    }
    if course_id:
        search_kwargs["filter"] = {"course_id": course_id}

    vector_retriever = vs.get_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )

    # ── Step 2: BM25 retriever (keyword matching) ──────────────────
    # Load docs for BM25 indexing (in-memory, fast for <500 docs)
    all_docs = vs.get_all_documents(limit=500)

    if all_docs:
        bm25_retriever = BM25Retriever.from_documents(
            all_docs,
            k=settings.RAG_RETRIEVER_K,
        )

        # ── Step 3: Merge BM25 + Vector with ensemble ─────────────
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.3, 0.7],  # favor semantic, include keyword matches
        )
        base_retriever = ensemble_retriever
        logger.info(
            f"Hybrid retriever built: BM25({len(all_docs)} docs, w=0.3) + "
            f"MMR(k={search_kwargs['k']}, w=0.7)"
        )
    else:
        # Fallback: vector-only if BM25 loading fails
        base_retriever = vector_retriever
        logger.info(f"Vector-only retriever built: MMR(k={search_kwargs['k']})")

    # ── Step 4: FlashRank reranking (local cross-encoder) ──────────
    reranker = FlashrankRerank(
        top_n=settings.RAG_RERANK_TOP_N,
    )

    final_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )
    logger.info(
        f"Retrieval pipeline complete → FlashRank(top_n={settings.RAG_RERANK_TOP_N})"
    )

    return final_retriever
