"""
Agent tools for the Eduverse AI tutor.

Four tools that the LangGraph ReAct agent can call:
1. search_course_materials — RAG retrieval with relevance filtering
2. search_web             — Groq compound-mini web search (user-consented only)
3. generate_flashcards    — study flashcard generation
4. summarize_topic        — structured topic summaries
"""

import json
import logging
from typing import Optional

from groq import Groq
from langchain_core.tools import tool

from app.core.config import settings
from app.core.utils import detect_source_type
from app.rag.retriever import build_retriever

logger = logging.getLogger(__name__)

# ── Citation cache ────────────────────────────────────────────────
# Stores structured citations from the last search_course_materials call
# per user. Read by chat.py after agent completes.
_citation_cache: dict[str, list] = {}


def get_citations(user_id: str) -> list:
    """Get and clear cached citations for a user (called by chat.py)."""
    return _citation_cache.pop(user_id, [])


# ── Tool 1: Search course materials (with relevance filtering) ───

def _make_search_course_materials(user_id: str, groq_api_key: str, course_id: Optional[str] = None):
    """Factory: returns a tool bound to the user's vector store."""

    @tool
    def search_course_materials(query: str) -> str:
        """Search the student's indexed course materials (PDFs, videos, audio, images).
        Returns numbered source blocks with citations and relevance scores.
        Use this when the student asks about their specific course content,
        lectures, or assignments. Results below the relevance threshold are
        automatically filtered out."""
        try:
            retriever = build_retriever(user_id, groq_api_key, course_id)
            docs = retriever.invoke(query)
            if not docs:
                _citation_cache[user_id] = []
                return "No relevant information found in course materials."

            # ── Filter by relevance score ─────────────────────────
            # FlashRank reranker adds 'relevance_score' to metadata.
            # Filter out documents below the threshold to prevent
            # citing irrelevant content.
            threshold = settings.RAG_RELEVANCE_THRESHOLD
            relevant_docs = [
                doc for doc in docs
                if doc.metadata.get("relevance_score", 0.0) >= threshold
            ]

            if not relevant_docs:
                _citation_cache[user_id] = []
                logger.info(
                    f"All {len(docs)} results below relevance threshold "
                    f"({threshold}) for query: {query[:80]}"
                )
                return (
                    "No relevant information found in course materials. "
                    "The search returned some results but none were relevant "
                    "enough to the query."
                )

            logger.info(
                f"Relevance filter: {len(docs)} → {len(relevant_docs)} docs "
                f"(threshold={threshold})"
            )

            # Build formatted text for the LLM
            # Use parent_content (richer context) if available, else chunk content
            blocks = []
            for i, doc in enumerate(relevant_docs, 1):
                meta = doc.metadata
                source = meta.get("file_name", "unknown")
                page = meta.get("page_number")
                score = meta.get("relevance_score", 0.0)
                header = f"[{i}] (source: {source}"
                if page is not None:
                    header += f", page {page}"
                header += f", relevance: {score:.2f})"
                # Parent content gives 800-char context vs 300-char child chunk
                content = meta.get("parent_content", doc.page_content)
                blocks.append(f"{header}\n{content}")

            # Store structured citations in cache (read by chat.py)
            _citation_cache[user_id] = [
                {
                    "id": i,
                    "file_name": doc.metadata.get("file_name", "unknown"),
                    "source_type": detect_source_type(doc.metadata.get("file_name", "")),
                    "page_number": doc.metadata.get("page_number"),
                    "start_time": doc.metadata.get("start_time"),
                    "end_time": doc.metadata.get("end_time"),
                    "relevance_score": round(doc.metadata.get("relevance_score", 0.0), 3),
                    "content": doc.page_content[:200],
                }
                for i, doc in enumerate(relevant_docs, 1)
            ]

            return "\n\n".join(blocks)
        except Exception as e:
            logger.error(f"Course search failed: {e}")
            return f"Course material search failed: {e}"

    return search_course_materials


# ── Tool 2: Web search via Groq compound-mini ────────────────────

def _make_search_web(groq_api_key: str):
    """Factory: returns a web search tool using Groq's compound-mini."""

    @tool
    def search_web(query: str) -> str:
        """Search the internet for information. ONLY use this tool when the
        student has EXPLICITLY asked to search the web (e.g., 'search online',
        'yes look it up', 'search the web for it'). NEVER call this tool
        automatically — always ask the student first."""
        try:
            client = Groq(api_key=groq_api_key)
            response = client.chat.completions.create(
                model=settings.WEB_SEARCH_MODEL,
                messages=[{"role": "user", "content": query}],
            )
            content = response.choices[0].message.content
            executed = getattr(response.choices[0].message, "executed_tools", None)
            if executed:
                content += "\n\n[Sources from web search]"
            return content or "No web results found."
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Web search failed: {e}"

    return search_web


# ── Tool 3: Flashcard generation ─────────────────────────────────

FLASHCARD_PROMPT = """You are creating study flashcards from educational content.
Based on the following material, create {n} flashcards on the topic "{topic}".

Source material:
{context}

Return ONLY a JSON array with this exact structure (no markdown, no extra text):
[
  {{
    "front": "Term or question",
    "back": "Definition or answer"
  }}
]"""


def _make_generate_flashcards(user_id: str, groq_api_key: str, course_id: Optional[str] = None):
    """Factory: returns a flashcard generation tool."""

    @tool
    def generate_flashcards(topic: str, num_cards: int = 10) -> str:
        """Generate study flashcards from the student's course materials.
        Use this when the student asks for flashcards, key terms, or
        vocabulary review for a topic.
        Args:
            topic: The subject to create flashcards for
            num_cards: Number of flashcards (default 10)"""
        try:
            retriever = build_retriever(user_id, groq_api_key, course_id)
            docs = retriever.invoke(topic)
            if not docs:
                return "No course materials found on this topic."

            # Filter by relevance for flashcards too
            threshold = settings.RAG_RELEVANCE_THRESHOLD
            relevant_docs = [
                doc for doc in docs
                if doc.metadata.get("relevance_score", 0.0) >= threshold
            ]
            if not relevant_docs:
                return "No sufficiently relevant course materials found for this topic."

            context = "\n\n".join(doc.page_content for doc in relevant_docs)

            client = Groq(api_key=groq_api_key)
            response = client.chat.completions.create(
                model=settings.JSON_MODEL,
                messages=[{
                    "role": "user",
                    "content": FLASHCARD_PROMPT.format(
                        n=num_cards, topic=topic, context=context,
                    ),
                }],
                temperature=0.3,
            )

            raw = response.choices[0].message.content
            cards = json.loads(raw)
            if isinstance(cards, dict) and "flashcards" in cards:
                cards = cards["flashcards"]

            lines = [f"🃏 **Flashcards: {topic}** ({len(cards)} cards)\n"]
            for i, card in enumerate(cards, 1):
                lines.append(f"**Card {i}**")
                lines.append(f"   📋 **Front:** {card['front']}")
                lines.append(f"   ✅ **Back:** {card['back']}\n")
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Flashcard generation failed: {e}")
            return f"Flashcard generation failed: {e}"

    return generate_flashcards


# ── Tool 4: Topic summarization ──────────────────────────────────

SUMMARY_PROMPT = """You are summarizing educational content for a student.
Based on the following course materials, create a clear, structured summary
of the topic "{topic}".

Include:
- Key concepts and definitions
- Important relationships between ideas
- Any formulas, rules, or frameworks mentioned

Source material:
{context}

Write a concise, student-friendly summary in markdown format."""


def _make_summarize_topic(user_id: str, groq_api_key: str, course_id: Optional[str] = None):
    """Factory: returns a topic summarization tool."""

    @tool
    def summarize_topic(topic: str) -> str:
        """Summarize a topic from the student's course materials.
        Use this when the student asks to summarize, explain, or review
        a chapter, lecture, or topic from their course.
        Args:
            topic: The subject or chapter to summarize"""
        try:
            retriever = build_retriever(user_id, groq_api_key, course_id)
            docs = retriever.invoke(topic)
            if not docs:
                return "No course materials found on this topic."

            # Filter by relevance for summaries too
            threshold = settings.RAG_RELEVANCE_THRESHOLD
            relevant_docs = [
                doc for doc in docs
                if doc.metadata.get("relevance_score", 0.0) >= threshold
            ]
            if not relevant_docs:
                return "No sufficiently relevant course materials found for this topic."

            context = "\n\n".join(doc.page_content for doc in relevant_docs)

            client = Groq(api_key=groq_api_key)
            response = client.chat.completions.create(
                model=settings.AGENT_MODEL,
                messages=[{
                    "role": "user",
                    "content": SUMMARY_PROMPT.format(topic=topic, context=context),
                }],
                temperature=0.3,
            )

            summary = response.choices[0].message.content
            source_names = set(
                d.metadata.get("file_name", "unknown") for d in relevant_docs
            )
            summary += f"\n\n📚 _Sources: {', '.join(source_names)}_"
            return summary

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Summary generation failed: {e}"

    return summarize_topic


# ── Public: build all tools for a user ────────────────────────────

def build_agent_tools(
    user_id: str,
    groq_api_key: str,
    course_id: Optional[str] = None,
) -> list:
    """Build the complete tool set (4 tools) for the tutor agent."""
    return [
        _make_search_course_materials(user_id, groq_api_key, course_id),
        _make_search_web(groq_api_key),
        _make_generate_flashcards(user_id, groq_api_key, course_id),
        _make_summarize_topic(user_id, groq_api_key, course_id),
    ]
