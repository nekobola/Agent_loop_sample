"""Tests for MemoryStore and related classes."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from agent.memory import MemoryEntry, MemoryStore, SessionMemory


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_to_dict(self) -> None:
        """Test converting MemoryEntry to dict."""
        entry = MemoryEntry(
            role="user",
            content="Hello",
            timestamp=1234567890.0,
            metadata={"key": "value"},
        )
        result = entry.to_dict()
        assert result["role"] == "user"
        assert result["content"] == "Hello"
        assert result["timestamp"] == 1234567890.0
        assert result["metadata"] == {"key": "value"}

    def test_to_dict_empty_metadata(self) -> None:
        """Test converting MemoryEntry with empty metadata."""
        entry = MemoryEntry(
            role="assistant",
            content="Hi there",
            timestamp=1234567890.0,
            metadata={},
        )
        result = entry.to_dict()
        assert result["metadata"] == {}


class TestSessionMemory:
    """Tests for SessionMemory class."""

    def test_init(self) -> None:
        """Test SessionMemory initialization."""
        session = SessionMemory(session_key="test-session")
        assert session.session_key == "test-session"
        assert session.messages == []
        assert session.summary == ""

    def test_add_message(self) -> None:
        """Test adding a message to a session."""
        session = SessionMemory(session_key="test")
        session.add("user", "Hello")
        assert len(session.messages) == 1
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "Hello"

    def test_add_message_with_metadata(self) -> None:
        """Test adding a message with metadata."""
        session = SessionMemory(session_key="test")
        session.add("user", "Hello", metadata={"source": "test"})
        assert session.messages[0].metadata == {"source": "test"}

    def test_get_messages(self) -> None:
        """Test getting messages from a session."""
        session = SessionMemory(session_key="test")
        session.add("user", "Hello")
        session.add("assistant", "Hi there")
        messages = session.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_get_messages_with_max_count(self) -> None:
        """Test getting limited number of messages."""
        session = SessionMemory(session_key="test")
        for i in range(5):
            session.add("user", f"Message {i}")
        messages = session.get_messages(max_count=2)
        # get_messages returns the LAST max_count messages (tail), not the first
        assert len(messages) == 2
        assert messages[0]["content"] == "Message 3"
        assert messages[1]["content"] == "Message 4"

    def test_clear(self) -> None:
        """Test clearing a session."""
        session = SessionMemory(session_key="test")
        session.add("user", "Hello")
        session.add("assistant", "Hi")
        session.clear()
        assert len(session.messages) == 0
        assert session.summary == ""


class TestMemoryStore:
    """Tests for MemoryStore class."""

    def test_init_without_storage_dir(self) -> None:
        """Test MemoryStore initialization without storage dir."""
        store = MemoryStore()
        assert store._storage_dir is None
        assert store._sessions == {}

    def test_init_with_storage_dir(self) -> None:
        """Test MemoryStore initialization with storage dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(storage_dir=tmpdir)
            assert store._storage_dir is not None
            assert store._storage_dir.exists()

    def test_init_custom_limits(self) -> None:
        """Test MemoryStore with custom limits."""
        store = MemoryStore(
            max_messages_per_session=100,
            max_age_seconds=3600,
        )
        assert store._max_messages == 100
        assert store._max_age == 3600

    def test_get_session_creates_new(self) -> None:
        """Test that get_session creates a new session if not exists."""
        store = MemoryStore()
        session = store.get_session("new-session")
        assert session is not None
        assert session.session_key == "new-session"

    def test_get_session_returns_same(self) -> None:
        """Test that get_session returns the same session."""
        store = MemoryStore()
        session1 = store.get_session("test")
        session2 = store.get_session("test")
        assert session1 is session2

    def test_add_message(self) -> None:
        """Test adding a message to a session via store."""
        store = MemoryStore()
        store.add_message("test", "user", "Hello")
        messages = store.get_messages("test")
        assert len(messages) == 1
        assert messages[0]["content"] == "Hello"

    def test_add_message_trims_excess(self) -> None:
        """Test that messages are trimmed when exceeding max."""
        store = MemoryStore(max_messages_per_session=3)
        for i in range(5):
            store.add_message("test", "user", f"Message {i}")
        messages = store.get_messages("test")
        assert len(messages) == 3
        assert messages[0]["content"] == "Message 2"

    def test_get_messages_nonexistent_session(self) -> None:
        """Test getting messages from non-existent session creates empty session."""
        store = MemoryStore()
        messages = store.get_messages("nonexistent")
        assert messages == []

    def test_clear_session(self) -> None:
        """Test clearing a session."""
        store = MemoryStore()
        store.add_message("test", "user", "Hello")
        store.clear_session("test")
        assert store.get_messages("test") == []

    def test_list_sessions(self) -> None:
        """Test listing all sessions."""
        store = MemoryStore()
        store.get_session("session1")
        store.get_session("session2")
        sessions = store.list_sessions()
        assert "session1" in sessions
        assert "session2" in sessions

    def test_list_sessions_excludes_stale(self) -> None:
        """Test that list_sessions excludes stale sessions."""
        store = MemoryStore(max_age_seconds=0.001)
        store.get_session("stale")
        time.sleep(0.01)
        store.get_session("fresh")
        sessions = store.list_sessions()
        assert "stale" not in sessions
        assert "fresh" in sessions


class TestMemoryStorePersistence:
    """Tests for MemoryStore persistence functionality."""

    def test_persist_session(self) -> None:
        """Test persisting a session to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(storage_dir=tmpdir)
            store.add_message("test", "user", "Hello")
            store.persist_session("test")

            path = Path(tmpdir) / "test.json"
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["session_key"] == "test"
            assert len(data["messages"]) == 1

    def test_persist_session_no_storage_dir(self) -> None:
        """Test that persist_session does nothing without storage_dir."""
        store = MemoryStore()
        store.add_message("test", "user", "Hello")
        store.persist_session("test")  # Should not raise

    def test_load_session(self) -> None:
        """Test loading a session from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a session file manually
            path = Path(tmpdir) / "test.json"
            path.write_text(json.dumps({
                "session_key": "test",
                "messages": [
                    {"role": "user", "content": "Hello", "timestamp": 1234567890.0, "metadata": {}},
                ],
                "summary": "",
            }))

            store = MemoryStore(storage_dir=tmpdir)
            result = store.load_session("test")
            assert result is True
            messages = store.get_messages("test")
            assert len(messages) == 1
            assert messages[0]["content"] == "Hello"

    def test_load_session_not_found(self) -> None:
        """Test loading a session that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(storage_dir=tmpdir)
            result = store.load_session("nonexistent")
            assert result is False

    def test_load_session_invalid_json(self) -> None:
        """Test loading a session with invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            path.write_text("invalid json")

            store = MemoryStore(storage_dir=tmpdir)
            result = store.load_session("test")
            assert result is False
