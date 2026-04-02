"""Lifecycle hook system for agent loop."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.loop import AgentLoop

import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentHookContext:
    """Context passed to hook callbacks."""

    loop: AgentLoop
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    response: Response | None = None
    session_key: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    def strip_think(self, text: str | None) -> str:
        """Strip <think> tags from text."""
        if not text:
            return ""
        import re
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


@dataclass
class ToolCall:
    """Represents a single tool invocation."""

    name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None


@dataclass
class Response:
    """LLM response."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str | None = None


class AgentHook:
    """Base class for agent lifecycle hooks.

    Subclass and override specific methods to hook into the agent loop.
    """

    def wants_streaming(self) -> bool:
        """Return True if this hook wants streaming callbacks."""
        return False

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        """Called for each streaming delta from the LLM."""
        pass

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool = False) -> None:
        """Called when streaming finishes."""
        pass

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        """Called before tools are executed."""
        pass

    async def after_tools_executed(self, context: AgentHookContext) -> None:
        """Called after tools have been executed."""
        pass

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        """Called to finalize/modify content before it's returned."""
        return content

    async def on_error(self, context: AgentHookContext, error: Exception) -> None:
        """Called when an error occurs in the loop."""
        pass


class CompositeHook(AgentHook):
    """Chain multiple hooks together."""

    def __init__(self, hooks: list[AgentHook]) -> None:
        self._hooks = hooks

    def wants_streaming(self) -> bool:
        return any(h.wants_streaming() for h in self._hooks)

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        for h in self._hooks:
            await h.on_stream(context, delta)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool = False) -> None:
        for h in self._hooks:
            await h.on_stream_end(context, resuming=resuming)

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        for h in self._hooks:
            await h.before_execute_tools(context)

    async def after_tools_executed(self, context: AgentHookContext) -> None:
        for h in self._hooks:
            await h.after_tools_executed(context)

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        for h in self._hooks:
            content = h.finalize_content(context, content)
        return content

    async def on_error(self, context: AgentHookContext, error: Exception) -> None:
        for h in self._hooks:
            await h.on_error(context, error)


class HookBuilder:
    """Helper to compose hooks with a fluent API."""

    def __init__(self) -> None:
        self._hooks: list[AgentHook] = []

    def add(self, hook: AgentHook) -> HookBuilder:
        self._hooks.append(hook)
        return self

    def with_streaming(
        self,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> HookBuilder:
        """Add a streaming hook."""
        self._hooks.append(_StreamingHook(on_stream, on_stream_end))
        return self

    def with_progress(
        self, on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> HookBuilder:
        """Add a progress reporting hook."""
        self._hooks.append(_ProgressHook(on_progress))
        return self

    def with_logging(self) -> HookBuilder:
        """Add a logging hook."""
        self._hooks.append(_LoggingHook())
        return self

    def build(self) -> CompositeHook:
        return CompositeHook(self._hooks)


class _StreamingHook(AgentHook):
    """Internal hook for streaming support."""

    def __init__(
        self,
        on_stream: Callable[[str], Awaitable[None]] | None,
        on_stream_end: Callable[..., Awaitable[None]] | None,
    ) -> None:
        self._on_stream = on_stream
        self._on_stream_end = on_stream_end
        self._buf = ""

    def wants_streaming(self) -> bool:
        return self._on_stream is not None

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        if self._on_stream:
            self._buf += delta
            await self._on_stream(delta)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool = False) -> None:
        if self._on_stream_end:
            await self._on_stream_end(resuming=resuming)
        self._buf = ""


class _ProgressHook(AgentHook):
    """Internal hook for progress reporting."""

    def __init__(self, on_progress: Callable[[str], Awaitable[None]] | None) -> None:
        self._on_progress = on_progress

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        if self._on_progress:
            response = context.response
            if response:
                thought = context.strip_think(response.content)
                if thought:
                    await self._on_progress(thought)


class _LoggingHook(AgentHook):
    """Internal hook for tool call logging."""

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        for tc in context.tool_calls:
            args_str = json.dumps(tc.arguments, ensure_ascii=False)
            logger.info("Tool call: %s(%s)", tc.name, args_str[:200])

    async def on_error(self, context: AgentHookContext, error: Exception) -> None:
        logger.error("Agent loop error: %s", error)
