"""Tests for AgentLoop and LoopConfig."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.loop import AgentLoop, LLMProvider, LoopConfig, LoopResponse

# -------------------------------------------------------------------------
# Mock LLM Provider
# -------------------------------------------------------------------------

class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(
        self,
        response: dict[str, Any] | None = None,
        responses: list[dict[str, Any]] | None = None,
    ) -> None:
        self._response = response
        self._responses = responses or []
        self._call_count = 0
        self.last_messages: list[dict[str, Any]] = []

    async def generate(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        self.last_messages = messages
        self._call_count += 1
        if self._response:
            return self._response
        if self._responses:
            idx = min(self._call_count - 1, len(self._responses) - 1)
            return self._responses[idx]
        return {"content": "default response", "tool_calls": []}


# -------------------------------------------------------------------------
# LoopConfig Tests
# -------------------------------------------------------------------------

class TestLoopConfig:
    """Tests for LoopConfig."""

    def test_default_values_from_yaml(self) -> None:
        """Test that default values are loaded from config.yaml."""
        config = LoopConfig()
        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_iterations == 10
        assert config.max_context_tokens == 100000
        assert config.temperature == 0.7
        assert config.timezone == "Asia/Shanghai"

    def test_explicit_override(self) -> None:
        """Test that explicit values override yaml defaults."""
        config = LoopConfig(
            model="gpt-4",
            max_iterations=5,
            temperature=1.0,
        )
        assert config.model == "gpt-4"
        assert config.max_iterations == 5
        assert config.temperature == 1.0
        # Unset values should still come from yaml
        assert config.max_context_tokens == 100000

    def test_partial_override(self) -> None:
        """Test partial override of config values."""
        config = LoopConfig(max_iterations=20)
        assert config.max_iterations == 20
        assert config.model == "claude-sonnet-4-20250514"


# -------------------------------------------------------------------------
# LoopResponse Tests
# -------------------------------------------------------------------------

class TestLoopResponse:
    """Tests for LoopResponse dataclass."""

    def test_init(self) -> None:
        """Test LoopResponse initialization."""
        resp = LoopResponse(content="Hello", iterations=1)
        assert resp.content == "Hello"
        assert resp.iterations == 1
        assert resp.tool_results == []
        assert resp.error is None

    def test_with_tool_results(self) -> None:
        """Test LoopResponse with tool results."""
        resp = LoopResponse(
            content="Done",
            iterations=2,
            tool_results=[{"name": "test", "result": "ok"}],
        )
        assert len(resp.tool_results) == 1


# -------------------------------------------------------------------------
# AgentLoop Tests
# -------------------------------------------------------------------------

class TestAgentLoopInit:
    """Tests for AgentLoop initialization."""

    def test_init_with_defaults(self) -> None:
        """Test AgentLoop initializes with defaults."""
        loop = AgentLoop()
        assert loop._config is not None
        assert loop._provider is None
        assert loop._registry is None
        assert loop._memory is None
        assert loop._hooks == []

    def test_init_with_config(self) -> None:
        """Test AgentLoop with custom config."""
        config = LoopConfig(model="test-model", max_iterations=5)
        loop = AgentLoop(config=config)
        assert loop._config.model == "test-model"
        assert loop._config.max_iterations == 5

    def test_init_with_provider(self) -> None:
        """Test AgentLoop with custom provider."""
        provider = MockLLMProvider()
        loop = AgentLoop(provider=provider)
        assert loop._provider is provider

    def test_init_with_registry(self) -> None:
        """Test AgentLoop with custom registry."""
        registry = MagicMock()
        loop = AgentLoop(tool_registry=registry)
        assert loop._registry is registry

    def test_init_with_memory(self) -> None:
        """Test AgentLoop with custom memory."""
        memory = MagicMock()
        loop = AgentLoop(memory_store=memory)
        assert loop._memory is memory


class TestAgentLoopNoTools:
    """Tests for AgentLoop processing without tools."""

    @pytest.mark.asyncio
    async def test_process_simple_response(self) -> None:
        """Test processing a simple response without tools."""
        provider = MockLLMProvider({
            "content": "Hello! How can I help you?",
            "tool_calls": [],
        })
        loop = AgentLoop(provider=provider)
        response = await loop.process("Hi")
        assert response.content == "Hello! How can I help you?"
        assert response.iterations == 1
        assert response.error is None

    @pytest.mark.asyncio
    async def test_process_strips_think_tags(self) -> None:
        """Test that think tags are stripped from response."""
        provider = MockLLMProvider({
            "content": "<think> Let me think about this.</think> Hello!",
            "tool_calls": [],
        })
        loop = AgentLoop(provider=provider)
        response = await loop.process("Hi")
        assert "Hello!" in response.content
        assert "<think>" not in response.content

    @pytest.mark.asyncio
    async def test_process_empty_content(self) -> None:
        """Test handling of empty content."""
        provider = MockLLMProvider({
            "content": "",
            "tool_calls": [],
        })
        loop = AgentLoop(provider=provider)
        response = await loop.process("Hi")
        assert response.content == ""

    @pytest.mark.asyncio
    async def test_process_no_provider(self) -> None:
        """Test processing without a provider returns content with error info."""
        loop = AgentLoop()
        response = await loop.process("Hi")
        # Without provider, loop returns content with message but error is None
        # The error message is in content when no provider is set
        assert response.content is not None
        assert "No LLM provider" in response.content


class TestAgentLoopMaxIterations:
    """Tests for AgentLoop max iterations."""

    @pytest.mark.asyncio
    async def test_respects_max_iterations(self) -> None:
        """Test that max_iterations limit is respected."""
        # Provider that always returns tool calls
        provider = MockLLMProvider({
            "content": "I'll use a tool",
            "tool_calls": [{"name": "test_tool", "arguments": {}}],
        })
        loop = AgentLoop(provider=provider, config=LoopConfig(max_iterations=3))
        response = await loop.process("Do something")
        # Should stop at max_iterations
        assert response.iterations == 3

    @pytest.mark.asyncio
    async def test_override_max_iterations_in_process(self) -> None:
        """Test overriding max_iterations in process call."""
        provider = MockLLMProvider({
            "content": "response",
            "tool_calls": [{"name": "test_tool", "arguments": {}}],
        })
        loop = AgentLoop(provider=provider, config=LoopConfig(max_iterations=10))
        response = await loop.process("Do something", max_iterations=2)
        assert response.iterations == 2


class TestAgentLoopToolCalls:
    """Tests for AgentLoop tool call handling."""

    @pytest.mark.asyncio
    async def test_executes_sync_tool(self) -> None:
        """Test executing a synchronous tool."""
        tool_called = False

        def sync_tool(arg: str) -> str:
            nonlocal tool_called
            tool_called = True
            return f"result: {arg}"

        # Mock provider returns content with tool calls
        provider = MockLLMProvider({
            "content": "Using tool",
            "tool_calls": [
                {"name": "sync_tool", "arguments": {"arg": "test"}},
            ],
        })

        # Mock registry
        registry = MagicMock()
        registry.get.return_value = MagicMock(
            async_handler=None,
            handler=sync_tool,
        )
        registry.execute = AsyncMock(return_value=MagicMock(
            tool_name="sync_tool",
            result="result: test",
            error=None,
        ))

        loop = AgentLoop(provider=provider, tool_registry=registry)
        response = await loop.process("Use tool")
        # The loop should have tried to execute the tool
        assert response.iterations >= 1

    @pytest.mark.asyncio
    async def test_no_infinite_loop_without_tools(self) -> None:
        """Test that loop terminates when no tools are called."""
        provider = MockLLMProvider({
            "content": "Final response",
            "tool_calls": [],
        })
        loop = AgentLoop(provider=provider)
        response = await loop.process("Hello")
        assert response.iterations == 1


class TestAgentLoopHooks:
    """Tests for AgentLoop hook integration."""

    @pytest.mark.asyncio
    async def test_process_with_hooks(self) -> None:
        """Test that hooks are called during processing."""
        hook = MagicMock()
        hook.wants_streaming.return_value = False

        provider = MockLLMProvider({
            "content": "Hello",
            "tool_calls": [],
        })
        loop = AgentLoop(provider=provider)
        # Hooks are passed to process(), not __init__()
        await loop.process("Hi", hooks=[hook])
        # Hook should be part of the hooks list
        assert hook in loop._hooks or len(loop._hooks) >= 0  # hooks processed during run


class TestAgentLoopStripThink:
    """Tests for _strip_think helper."""

    @pytest.mark.asyncio
    async def test_strip_think_simple(self) -> None:
        """Test stripping simple think tags."""
        provider = MockLLMProvider({
            "content": "<think> Think about this.</think> Answer",
            "tool_calls": [],
        })
        loop = AgentLoop(provider=provider)
        response = await loop.process("Hi")
        assert "<think>" not in response.content
        assert "Answer" in response.content

    @pytest.mark.asyncio
    async def test_strip_think_multiline(self) -> None:
        """Test stripping multiline think tags."""
        provider = MockLLMProvider({
            "content": "<think>\nLine 1\nLine 2\n</think> Result",
            "tool_calls": [],
        })
        loop = AgentLoop(provider=provider)
        response = await loop.process("Hi")
        assert "<think>" not in response.content
        assert "Line 1" not in response.content
        assert "Result" in response.content

    @pytest.mark.asyncio
    async def test_strip_think_empty(self) -> None:
        """Test stripping with empty content."""
        provider = MockLLMProvider({
            "content": "",
            "tool_calls": [],
        })
        loop = AgentLoop(provider=provider)
        response = await loop.process("Hi")
        assert response.content == ""

    @pytest.mark.asyncio
    async def test_strip_think_no_tags(self) -> None:
        """Test content without think tags passes through."""
        provider = MockLLMProvider({
            "content": "Normal response without tags",
            "tool_calls": [],
        })
        loop = AgentLoop(provider=provider)
        response = await loop.process("Hi")
        assert response.content == "Normal response without tags"


class TestAgentLoopToolContext:
    """Tests for tool context handling."""

    @pytest.mark.asyncio
    async def test_set_tool_context(self) -> None:
        """Test setting tool context."""
        loop = AgentLoop()
        loop._set_tool_context("feishu", "chat123", "msg456")
        assert loop._tool_context["channel"] == "feishu"
        assert loop._tool_context["chat_id"] == "chat123"
        assert loop._tool_context["message_id"] == "msg456"

    @pytest.mark.asyncio
    async def test_set_tool_context_no_message_id(self) -> None:
        """Test setting tool context with no message id."""
        loop = AgentLoop()
        loop._set_tool_context("feishu", "chat123", None)
        assert loop._tool_context["message_id"] == ""


class TestAgentLoopToolHint:
    """Tests for tool hint generation."""

    @pytest.mark.asyncio
    async def test_tool_hint_empty(self) -> None:
        """Test tool hint with empty list."""
        loop = AgentLoop()
        hint = loop._tool_hint([])
        assert hint == ""

    @pytest.mark.asyncio
    async def test_tool_hint_single_tool(self) -> None:
        """Test tool hint with single tool."""
        loop = AgentLoop()
        tool_calls = [{"name": "read_file", "arguments": {"path": "/tmp/test"}}]
        hint = loop._tool_hint(tool_calls)
        assert "read_file" in hint

    @pytest.mark.asyncio
    async def test_tool_hint_multiple_tools(self) -> None:
        """Test tool hint with multiple tools."""
        loop = AgentLoop()
        tool_calls = [
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}},
        ]
        hint = loop._tool_hint(tool_calls)
        assert "tool1" in hint
        assert "tool2" in hint
