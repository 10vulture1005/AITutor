"""
LangGraph ReAct agent for the Eduverse AI tutor.

Autonomous agent with 4 tools: search_course_materials, search_web,
generate_flashcards, and summarize_topic.

Uses PostgresSaver with connection pooling for persistent memory.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from psycopg_pool import ConnectionPool

from app.core.config import settings
from app.rag.prompts import AGENT_SYSTEM_PROMPT
from app.rag.tools import build_agent_tools

logger = logging.getLogger(__name__)

# ── Connection pool (created once, reused across requests) ────────

_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    """Lazy-init a module-level connection pool with keepalive."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.PG_CONNINFO,
            min_size=2,
            max_size=10,
            kwargs={
                "autocommit": True,
                "keepalives": 1,
                "keepalives_idle": 60,
                "keepalives_interval": 15,
                "keepalives_count": 3,
            },
            max_idle=300,       # close connections idle > 5 min
            reconnect_timeout=60,
        )
    return _pool


def _get_checkpointer() -> PostgresSaver:
    """
    Create a PostgresSaver backed by the shared connection pool.

    Retries setup() up to 3 times — Supabase kills idle connections,
    causing 'server closed connection unexpectedly' on first attempt.
    The pool auto-discards bad connections, so retry usually succeeds.
    """
    global _pool
    pool = _get_pool()

    for attempt in range(3):
        try:
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
            return checkpointer
        except Exception as e:
            logger.warning(
                f"Checkpointer setup failed (attempt {attempt + 1}/3): {e}"
            )
            if attempt < 2:
                # Pool already discarded the bad connection,
                # but if all connections are stale, reset the pool
                try:
                    _pool.check()  # force health check
                except Exception:
                    _pool.close()
                    _pool = None
                    pool = _get_pool()
                import time
                time.sleep(0.5 * (attempt + 1))
            else:
                raise


# ── Agent builder ─────────────────────────────────────────────────

def build_tutor_agent(
    user_id: str,
    groq_api_key: str,
    course_id: Optional[str] = None,
):
    """
    Build the LangGraph ReAct tutor agent.

    Conversation history is persisted in PostgreSQL via PostgresSaver
    with connection pooling, so sessions survive server restarts.
    """
    llm = ChatGroq(
        model=settings.AGENT_MODEL,
        api_key=groq_api_key,
        temperature=settings.RAG_LLM_TEMPERATURE,
    )

    tools = build_agent_tools(user_id, groq_api_key, course_id)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=AGENT_SYSTEM_PROMPT,
        checkpointer=_get_checkpointer(),
    )

    return agent


# ── Invoke (full response) ───────────────────────────────────────

MAX_RETRIES = 3

async def invoke_agent(
    agent,
    query: str,
    session_id: str,
) -> dict:
    """
    Invoke the tutor agent and return the complete response.

    Retries up to MAX_RETRIES times on Groq tool_use_failed errors
    (intermittent 400s when model generates malformed tool calls).

    Returns:
        {"answer": str, "messages": list}
    """
    config = {"configurable": {"thread_id": session_id}}
    inputs = {"messages": [HumanMessage(content=query)]}

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            result = await asyncio.to_thread(agent.invoke, inputs, config)
            messages = result.get("messages", [])
            answer = _extract_final_answer(messages)
            return {"answer": answer, "messages": messages}
        except Exception as e:
            last_error = e
            error_str = str(e)
            if "tool_use_failed" in error_str or "failed_generation" in error_str:
                logger.warning(
                    f"Groq tool_use_failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying..."
                )
                await asyncio.sleep(1 * (attempt + 1))
                continue
            raise

    raise last_error


# ── Stream (Server-Sent Events) ──────────────────────────────────

async def stream_agent(
    agent,
    query: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    Stream the agent's response as Server-Sent Events (SSE).

    Yields SSE-formatted strings: 'data: {"type": ..., "content": ...}\n\n'
    """
    config = {"configurable": {"thread_id": session_id}}
    inputs = {"messages": [HumanMessage(content=query)]}

    def _stream_sync():
        """Run agent.stream() synchronously (called via to_thread)."""
        chunks = []
        for event in agent.stream(inputs, config, stream_mode="updates"):
            chunks.append(event)
        return chunks

    events = await asyncio.to_thread(_stream_sync)

    for event in events:
        for node_name, node_output in event.items():
            messages = node_output.get("messages", [])
            for msg in messages:
                if hasattr(msg, "tool_call_id"):
                    # Tool result
                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': getattr(msg, 'name', 'unknown'), 'content': msg.content[:200]})}\n\n"
                elif hasattr(msg, "tool_calls") and msg.tool_calls:
                    # Agent deciding to call a tool
                    for tc in msg.tool_calls:
                        yield f"data: {json.dumps({'type': 'tool_call', 'tool': tc['name'], 'args': str(tc.get('args', {}))[:200]})}\n\n"
                elif hasattr(msg, "content") and msg.content:
                    # Final answer or intermediate reasoning
                    yield f"data: {json.dumps({'type': 'answer', 'content': msg.content})}\n\n"

    yield "data: [DONE]\n\n"


# ── Helper ────────────────────────────────────────────────────────

def _extract_final_answer(messages: list) -> str:
    """Extract the last AI message content (skip ToolMessages)."""
    for msg in reversed(messages):
        if hasattr(msg, "content") and not hasattr(msg, "tool_call_id"):
            return msg.content
    return ""
