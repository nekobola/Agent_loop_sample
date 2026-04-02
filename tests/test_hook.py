"""Tests for hook system."""

from __future__ import annotations

from typing import Any

import pytest

from agent.hook import (
    AgentHook,
    CompositeHook,
    HookBuilder,
    Response,
    ToolCall,
)


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_init(self) -> None:
        """Test ToolCall initialization."""
        tc = ToolCall(name="test", arguments={"arg": "value"})
        assert tc.name == "test"
        assert tc.arguments == {"arg": "value"}
        assert tc.result is None
        assert tc.error is None

    def test_with_result_and_error(self) -> None:
        """Test ToolCall with result and error."""
        tc = ToolCall(
            name="test",
            arguments={},
            result="success",
            error=None,
        )
        assert tc.result == "success"
        assert tc.error is None


class TestResponse:
    """Tests for Response dataclass."""

    def test_init(self) -> None:
        """Test Response initialization."""
        resp = Response(content="Hello", tool_calls=[])
        assert resp.content == "Hello"
        assert resp.tool_calls == []

    def test_with_tool_calls(self) -> None:
        """Test Response with tool calls."""
        tc = ToolCall(name="test", arguments={})
        resp = Response(content="", tool_calls=[tc])
        assert len(resp.tool_calls) == 1


class TestAgentHook:
    """Tests for AgentHook base class."""

    def test_wants_streaming_default_false(self) -> None:
        """Test default wants_streaming returns False."""
        hook = AgentHook()
        assert hook.wants_streaming() is False

    @pytest.mark.asyncio
    async def test_on_stream_noop(self) -> None:
        """Test on_stream does nothing by default."""
        hook = AgentHook()
        await hook.on_stream(None, "delta")  # type: ignore

    @pytest.mark.asyncio
    async def test_on_stream_end_noop(self) -> None:
        """Test on_stream_end does nothing by default."""
        hook = AgentHook()
        await hook.on_stream_end(None)  # type: ignore

    @pytest.mark.asyncio
    async def test_before_execute_tools_noop(self) -> None:
        """Test before_execute_tools does nothing by default."""
        hook = AgentHook()
        await hook.before_execute_tools(None)  # type: ignore

    @pytest.mark.asyncio
    async def test_after_tools_executed_noop(self) -> None:
        """Test after_tools_executed does nothing by default."""
        hook = AgentHook()
        await hook.after_tools_executed(None)  # type: ignore

    def test_finalize_content_returns_input(self) -> None:
        """Test finalize_content returns input unchanged."""
        hook = AgentHook()
        result = hook.finalize_content(None, "content")  # type: ignore
        assert result == "content"

    @pytest.mark.asyncio
    async def test_on_error_noop(self) -> None:
        """Test on_error does nothing by default."""
        hook = AgentHook()
        await hook.on_error(None, Exception("test"))  # type: ignore


class TestCompositeHook:
    """Tests for CompositeHook class."""

    def test_empty_hooks(self) -> None:
        """Test CompositeHook with no hooks."""
        composite = CompositeHook([])
        assert composite.wants_streaming() is False

    def test_wants_streaming_single_hook(self) -> None:
        """Test wants_streaming with a single hook that wants streaming."""
        hook = AgentHook()
        hook.wants_streaming = lambda: True  # type: ignore
        composite = CompositeHook([hook])
        assert composite.wants_streaming() is True

    def test_wants_streaming_no_hook_wants_it(self) -> None:
        """Test wants_streaming returns False when no hook wants it."""
        hook = AgentHook()
        composite = CompositeHook([hook])
        assert composite.wants_streaming() is False

    @pytest.mark.asyncio
    async def test_on_stream_calls_all(self) -> None:
        """Test on_stream calls all hooks."""
        calls: list[str] = []

        class MockHook(AgentHook):
            def __init__(self, name: str) -> None:
                self._name = name

            async def on_stream(self, ctx: Any, delta: str) -> None:
                calls.append(f"{self._name}:{delta}")

        composite = CompositeHook([MockHook("h1"), MockHook("h2")])
        await composite.on_stream(None, "delta")  # type: ignore
        assert calls == ["h1:delta", "h2:delta"]

    @pytest.mark.asyncio
    async def test_on_stream_end_calls_all(self) -> None:
        """Test on_stream_end calls all hooks."""
        calls: list[str] = []

        class MockHook(AgentHook):
            def __init__(self, name: str) -> None:
                self._name = name

            async def on_stream_end(self, ctx: Any, *, resuming: bool = False) -> None:
                calls.append(f"{self._name}:resuming={resuming}")

        composite = CompositeHook([MockHook("h1"), MockHook("h2")])
        await composite.on_stream_end(None, resuming=True)  # type: ignore
        assert calls == ["h1:resuming=True", "h2:resuming=True"]

    @pytest.mark.asyncio
    async def test_before_execute_tools_calls_all(self) -> None:
        """Test before_execute_tools calls all hooks."""
        calls: list[str] = []

        class MockHook(AgentHook):
            def __init__(self, name: str) -> None:
                self._name = name

            async def before_execute_tools(self, ctx: Any) -> None:
                calls.append(self._name)

        composite = CompositeHook([MockHook("h1"), MockHook("h2")])
        await composite.before_execute_tools(None)  # type: ignore
        assert calls == ["h1", "h2"]

    @pytest.mark.asyncio
    async def test_after_tools_executed_calls_all(self) -> None:
        """Test after_tools_executed calls all hooks."""
        calls: list[str] = []

        class MockHook(AgentHook):
            def __init__(self, name: str) -> None:
                self._name = name

            async def after_tools_executed(self, ctx: Any) -> None:
                calls.append(self._name)

        composite = CompositeHook([MockHook("h1"), MockHook("h2")])
        await composite.after_tools_executed(None)  # type: ignore
        assert calls == ["h1", "h2"]

    def test_finalize_content_chains(self) -> None:
        """Test finalize_content chains through hooks."""
        class MockHook(AgentHook):
            def __init__(self, prefix: str) -> None:
                self._prefix = prefix

            def finalize_content(self, ctx: Any, content: str | None) -> str | None:
                if content is None:
                    return None
                return f"{self._prefix}:{content}"

        composite = CompositeHook([MockHook("h1"), MockHook("h2")])
        result = composite.finalize_content(None, "content")  # type: ignore
        # h1 processes first, then h2 processes h1's output: h2:h1:content
        assert result == "h2:h1:content"

    def test_finalize_content_with_none(self) -> None:
        """Test finalize_content handles None."""
        class MockHook(AgentHook):
            def finalize_content(self, ctx: Any, content: str | None) -> str | None:
                return None

        composite = CompositeHook([MockHook()])
        result = composite.finalize_content(None, "content")  # type: ignore
        assert result is None

    @pytest.mark.asyncio
    async def test_on_error_calls_all(self) -> None:
        """Test on_error calls all hooks."""
        errors: list[str] = []

        class MockHook(AgentHook):
            def __init__(self, name: str) -> None:
                self._name = name

            async def on_error(self, ctx: Any, error: Exception) -> None:
                errors.append(f"{self._name}:{error}")

        composite = CompositeHook([MockHook("h1"), MockHook("h2")])
        await composite.on_error(None, Exception("test"))  # type: ignore
        assert errors == ["h1:test", "h2:test"]


class TestHookBuilder:
    """Tests for HookBuilder class."""

    def test_empty_builder(self) -> None:
        """Test empty HookBuilder."""
        builder = HookBuilder()
        composite = builder.build()
        assert len(composite._hooks) == 0

    def test_add_hook(self) -> None:
        """Test adding a hook."""
        builder = HookBuilder()
        hook = AgentHook()
        result = builder.add(hook)
        assert result is builder
        composite = builder.build()
        assert len(composite._hooks) == 1

    def test_add_multiple_hooks(self) -> None:
        """Test adding multiple hooks."""
        builder = HookBuilder()
        builder.add(AgentHook())
        builder.add(AgentHook())
        composite = builder.build()
        assert len(composite._hooks) == 2

    def test_with_streaming(self) -> None:
        """Test with_streaming adds a streaming hook."""
        builder = HookBuilder()
        stream_called: list[str] = []

        def on_stream(delta: str) -> None:
            stream_called.append(delta)

        builder.with_streaming(on_stream=on_stream)
        composite = builder.build()
        assert composite.wants_streaming() is True

    def test_with_streaming_no_callback(self) -> None:
        """Test with_streaming with no callback doesn't want streaming."""
        builder = HookBuilder()
        builder.with_streaming()
        composite = builder.build()
        assert composite.wants_streaming() is False

    def test_with_logging(self) -> None:
        """Test with_logging adds a logging hook."""
        builder = HookBuilder()
        builder.with_logging()
        composite = builder.build()
        assert len(composite._hooks) == 1

    def test_fluent_api(self) -> None:
        """Test fluent API chain."""
        builder = HookBuilder()
        result = (
            builder
            .add(AgentHook())
            .with_logging()
            .with_streaming()
        )
        assert result is builder
        composite = builder.build()
        assert len(composite._hooks) == 3
