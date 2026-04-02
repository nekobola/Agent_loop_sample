"""Memory store for agent sessions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import logging
logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry."""

    role: str
    content: str
    timestamp: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class SessionMemory:
    """In-memory store for a single session."""

    session_key: str
    messages: list[MemoryEntry] = field(default_factory=list)
    summary: str = ""

    def add(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        self.messages.append(MemoryEntry(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
        ))

    def get_messages(self, max_count: int | None = None) -> list[dict[str, Any]]:
        msgs = [m.to_dict() for m in self.messages]
        if max_count:
            msgs = msgs[-max_count:]
        return msgs

    def clear(self) -> None:
        self.messages.clear()
        self.summary = ""


class MemoryStore:
    """Persistent memory store for agent sessions.

    Provides session isolation, automatic cleanup,
    and optional disk persistence.
    """

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        *,
        max_messages_per_session: int = 1000,
        max_age_seconds: float = 86400 * 7,
    ) -> None:
        if storage_dir:
            self._storage_dir = Path(storage_dir).expanduser().resolve()
            self._storage_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._storage_dir = None

        self._sessions: dict[str, SessionMemory] = {}
        self._max_messages = max_messages_per_session
        self._max_age = max_age_seconds
        self._access_times: dict[str, float] = {}

    def get_session(self, session_key: str) -> SessionMemory:
        """Get or create a session memory."""
        if session_key not in self._sessions:
            self._sessions[session_key] = SessionMemory(session_key=session_key)
        self._access_times[session_key] = time.time()
        return self._sessions[session_key]

    def add_message(
        self,
        session_key: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a message to a session."""
        session = self.get_session(session_key)
        session.add(role, content, metadata)

        # Trim if over limit
        if len(session.messages) > self._max_messages:
            excess = len(session.messages) - self._max_messages
            session.messages = session.messages[excess:]

    def get_messages(
        self,
        session_key: str,
        max_count: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get messages for a session."""
        session = self.get_session(session_key)
        return session.get_messages(max_count)

    def clear_session(self, session_key: str) -> None:
        """Clear a session."""
        if session_key in self._sessions:
            self._sessions[session_key].clear()
        self._access_times.pop(session_key, None)

    def list_sessions(self) -> list[str]:
        """List all active session keys."""
        self._cleanup_stale()
        return list(self._sessions.keys())

    def _cleanup_stale(self) -> None:
        """Remove stale sessions."""
        now = time.time()
        stale = [
            k for k, t in self._access_times.items()
            if now - t > self._max_age
        ]
        for k in stale:
            self._sessions.pop(k, None)
            self._access_times.pop(k, None)
            logger.info("Cleaned up stale session: %s", k)

    def persist_session(self, session_key: str) -> None:
        """Persist a session to disk."""
        if not self._storage_dir:
            return

        session = self._sessions.get(session_key)
        if not session:
            return

        path = self._storage_dir / f"{session_key}.json"
        data = {
            "session_key": session.session_key,
            "messages": [m.to_dict() for m in session.messages],
            "summary": session.summary,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.debug("Persisted session %s to %s", session_key, path)

    def load_session(self, session_key: str) -> bool:
        """Load a session from disk. Returns True if found."""
        if not self._storage_dir:
            return False

        path = self._storage_dir / f"{session_key}.json"
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text())
            session = SessionMemory(session_key=session_key)
            session.messages = [
                MemoryEntry(
                    role=m["role"],
                    content=m["content"],
                    timestamp=m["timestamp"],
                    metadata=m.get("metadata", {}),
                )
                for m in data.get("messages", [])
            ]
            session.summary = data.get("summary", "")
            self._sessions[session_key] = session
            self._access_times[session_key] = time.time()
            return True
        except Exception as e:
            logger.warning("Failed to load session %s: %s", session_key, e)
            return False
