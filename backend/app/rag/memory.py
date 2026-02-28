"""
Session utilities for the Eduverse AI tutor.

The agent stores conversation state in PostgresSaver checkpoint tables.
This module provides helper functions to list and clear sessions,
using PostgresSaver's API for proper deserialization of message data.
"""

import logging
from typing import List

from sqlalchemy import text

from app.core.config import settings
from app.core.sync_db import get_sync_engine

logger = logging.getLogger(__name__)


def _get_checkpointer():
    """Get a PostgresSaver instance for reading checkpoint data."""
    from app.rag.agent import _get_checkpointer as agent_checkpointer
    return agent_checkpointer()


def list_user_sessions(user_id: str) -> List[str]:
    """
    List all chat session thread_ids for a user.

    Queries the checkpoints table where thread_id
    starts with the user's ID prefix (format: {user_id}_{uuid}).
    """
    try:
        engine = get_sync_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT DISTINCT thread_id FROM checkpoints "
                    "WHERE thread_id LIKE :prefix "
                    "ORDER BY thread_id"
                ),
                {"prefix": f"{user_id}_%"},
            )
            return [row[0] for row in result]
    except Exception as e:
        logger.warning(f"Could not list sessions: {e}")
        return []


def clear_session(session_id: str) -> bool:
    """
    Clear a session's checkpoint data.

    Returns True if any rows were deleted.
    """
    try:
        engine = get_sync_engine()
        with engine.begin() as conn:
            r1 = conn.execute(
                text("DELETE FROM checkpoint_writes WHERE thread_id = :tid"),
                {"tid": session_id},
            )
            r2 = conn.execute(
                text("DELETE FROM checkpoint_blobs WHERE thread_id = :tid"),
                {"tid": session_id},
            )
            r3 = conn.execute(
                text("DELETE FROM checkpoints WHERE thread_id = :tid"),
                {"tid": session_id},
            )
            deleted = (r1.rowcount or 0) + (r2.rowcount or 0) + (r3.rowcount or 0)

        if deleted:
            logger.info(f"Cleared session: {session_id} ({deleted} rows)")
        return deleted > 0
    except Exception as e:
        logger.warning(f"Could not clear session {session_id}: {e}")
        return False


def get_session_messages(session_id: str) -> List[dict]:
    """
    Get conversation messages for a session.

    Uses PostgresSaver.get_tuple() to properly deserialize checkpoint
    blobs (stored as msgpack in checkpoint_blobs table).

    Returns list of {"role": "human"|"ai", "content": "..."}.
    """
    try:
        checkpointer = _get_checkpointer()
        config = {"configurable": {"thread_id": session_id}}
        checkpoint_tuple = checkpointer.get_tuple(config)

        if not checkpoint_tuple:
            return []

        cp = checkpoint_tuple.checkpoint
        channel_values = cp.get("channel_values", {})
        raw_messages = channel_values.get("messages", [])

        messages = []
        for msg in raw_messages:
            # LangChain message objects have .type and .content attributes
            msg_type = getattr(msg, "type", None)
            content = getattr(msg, "content", None)

            if msg_type == "human" and content:
                messages.append({"role": "human", "content": content})
            elif msg_type == "ai" and content:
                # Skip tool-call-only messages (no user-facing content)
                if not getattr(msg, "tool_calls", None) or content:
                    messages.append({"role": "ai", "content": content})

        return messages
    except Exception as e:
        logger.warning(f"Could not get messages for {session_id}: {e}")
        return []
