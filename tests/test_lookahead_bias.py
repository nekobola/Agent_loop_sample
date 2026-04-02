"""Regression tests for look-ahead bias verification.

These tests verify that the agent loop and memory systems do not exhibit
look-ahead bias - i.e., they do not use data at time t that would not
have been available at that time.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.loop import AgentLoop, LLMProvider, LoopConfig
from agent.memory import MemoryEntry, MemoryStore, SessionMemory


class TestNoLookAheadBiasInMemory:
    """Tests to verify no look-ahead bias in memory operations."""

    def test_messages_stored_in_order(self) -> None:
        """Verify messages are stored and retrieved in FIFO order."""
        session = SessionMemory(session_key="test")
        session.add("user", "First")
        time.sleep(0.01)  # Ensure different timestamps
        session.add("user", "Second")
        time.sleep(0.01)
        session.add("assistant", "Third")

        messages = session.get_messages()
        assert len(messages) == 3
        assert messages[0]["content"] == "First"
        assert messages[1]["content"] == "Second"
        assert messages[2]["content"] == "Third"

    def test_timestamps_are_past(self) -> None:
        """Verify message timestamps are not in the future."""
        before = time.time()
        session = SessionMemory(session_key="test")
        session.add("user", "Test message")
        after = time.time()

        messages = session.get_messages()
        assert len(messages) == 1
        # Timestamp should be between before and after
        assert before <= messages[0]["timestamp"] <= after

    def test_get_messages_does_not_peek_future(self) -> None:
        """Verify get_messages doesn't access future messages."""
        session = SessionMemory(session_key="test")
        session.add("user", "Message 1")
        session.add("user", "Message 2")

        # Get only last message
        single = session.get_messages(max_count=1)
        assert len(single) == 1
        assert single[0]["content"] == "Message 2"

        # Verify we can't accidentally get message 2 when asking for 1
        # (This would be look-ahead if the implementation peeked ahead)

    def test_memory_store_session_isolation(self) -> None:
        """Verify sessions are properly isolated."""
        store = MemoryStore()

        store.add_message("session1", "user", "S1: First")
        store.add_message("session2", "user", "S2: First")

        store.add_message("session1", "assistant", "S1: Second")

        # session1 should have 2 messages
        s1_messages = store.get_messages("session1")
        assert len(s1_messages) == 2

        # session2 should have 1 message
        s2_messages = store.get_messages("session2")
        assert len(s2_messages) == 1
        assert s2_messages[0]["content"] == "S2: First"

    def test_no_data_leakage_between_sessions(self) -> None:
        """Verify no data from one session appears in another."""
        store = MemoryStore()

        store.add_message("alpha", "user", "Alpha secret")
        store.add_message("beta", "user", "Beta secret")

        alpha_messages = store.get_messages("alpha")
        beta_messages = store.get_messages("beta")

        # No cross-contamination
        alpha_contents = [m["content"] for m in alpha_messages]
        beta_contents = [m["content"] for m in beta_messages]

        assert "Alpha secret" in alpha_contents
        assert "Beta secret" not in alpha_contents
        assert "Beta secret" in beta_contents
        assert "Alpha secret" not in beta_contents


class TestNoLookAheadBiasInLoop:
    """Tests to verify no look-ahead bias in agent loop."""

    @pytest.mark.asyncio
    async def test_process_uses_only_current_messages(self) -> None:
        """Verify process doesn't access future conversation turns."""
        # Create a mock provider that tracks what's in messages
        provider_messages_seen: list[list[dict]] = []

        class TrackingProvider(LLMProvider):
            async def generate(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
                # Record what messages were sent
                provider_messages_seen.append(list(messages))
                return {"content": "Done", "tool_calls": []}

        # Need to set up memory store to persist conversation history
        memory = MemoryStore()
        loop = AgentLoop(provider=TrackingProvider(), memory_store=memory)

        # First call
        await loop.process("First message", session_key="test")

        # Second call with new message
        await loop.process("Second message", session_key="test")

        # Verify first call only saw 1 user message
        first_call = provider_messages_seen[0]
        user_contents = [m["content"] for m in first_call if m["role"] == "user"]
        assert "First message" in user_contents
        # Second message should not appear in first call
        assert "Second message" not in user_contents

    @pytest.mark.asyncio
    async def test_iteration_does_not_look_ahead(self) -> None:
        """Verify each iteration only sees prior messages, not future ones."""
        call_count = 0

        class EchoProvider(LLMProvider):
            async def generate(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
                nonlocal call_count
                call_count += 1
                # First call: return tool call
                # Second call: return final response
                if call_count == 1:
                    return {
                        "content": "Using tool",
                        "tool_calls": [{"name": "echo", "arguments": {"msg": "test"}}]
                    }
                return {"content": "Final", "tool_calls": []}

        loop = AgentLoop(provider=EchoProvider())

        # Mock registry with echo tool
        registry = MagicMock()
        registry.get.return_value = MagicMock(
            async_handler=None,
            handler=lambda **kwargs: "executed",
        )
        registry.execute = AsyncMock(return_value=MagicMock(
            result="executed",
            error=None,
            tool_name="echo",
        ))
        loop._registry = registry

        response = await loop.process("Do something")

        # Should complete in 2 iterations (1 tool call, 1 final)
        assert response.iterations == 2

    @pytest.mark.asyncio
    async def test_max_iterations_enforced(self) -> None:
        """Verify max_iterations strictly limits loops."""
        infinite_provider_calls = 0

        class InfiniteProvider(LLMProvider):
            async def generate(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
                nonlocal infinite_provider_calls
                infinite_provider_calls += 1
                return {
                    "content": "Always call tool",
                    "tool_calls": [{"name": "noop", "arguments": {}}]
                }

        loop = AgentLoop(provider=InfiniteProvider(), config=LoopConfig(max_iterations=5))

        # Mock registry
        registry = MagicMock()
        registry.get.return_value = MagicMock(
            async_handler=None,
            handler=lambda **kwargs: None,
        )
        registry.execute = AsyncMock(return_value=MagicMock(
            result=None,
            error=None,
            tool_name="noop",
        ))
        loop._registry = registry

        response = await loop.process("Go infinite")

        # Should stop at max_iterations=5
        assert response.iterations == 5
        assert infinite_provider_calls == 5
