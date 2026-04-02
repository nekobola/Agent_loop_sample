"""AgentLoop: the core async processing engine for workspace-coder.

Inspired by nanobot's ultra-lightweight agent architecture.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default config path relative to this file's directory
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# Sentinel for unset fields
_UNSET = object()


def _load_config_dict() -> dict[str, Any]:
    """Load config from config.yaml if it exists."""
    if _DEFAULT_CONFIG_PATH.exists():
        try:
            with open(_DEFAULT_CONFIG_PATH) as f:
                data = yaml.safe_load(f)
                if data and "agent" in data:
                    return data["agent"]
        except Exception as exc:
            logger.warning("Failed to load config.yaml: %s", exc)
    return {}


# Load config once at module level
_yaml_config: dict[str, Any] = _load_config_dict()


@dataclass
class LoopConfig:
    """Configuration for the agent loop.

    Loads defaults from config.yaml in the project root.
    Override individual fields to override config values.
    """

    model: str = _UNSET  # type: ignore[assignment]
    max_iterations: int = _UNSET  # type: ignore[assignment]
    max_context_tokens: int = _UNSET  # type: ignore[assignment]
    temperature: float = _UNSET  # type: ignore[assignment]
    timezone: str = _UNSET  # type: ignore[assignment]
    workspace_path: str | Path = _UNSET  # type: ignore[assignment]

    def __post_init__(self) -> None:
        defaults = _yaml_config
        if self.model is _UNSET:
            object.__setattr__(self, "model", defaults.get("model", "claude-sonnet-4-20250514"))
        if self.max_iterations is _UNSET:
            object.__setattr__(self, "max_iterations", defaults.get("max_iterations", 10))
        if self.max_context_tokens is _UNSET:
            object.__setattr__(self, "max_context_tokens", defaults.get("max_context_tokens", 100000))
        if self.temperature is _UNSET:
            object.__setattr__(self, "temperature", defaults.get("temperature", 0.7))
        if self.timezone is _UNSET:
            object.__setattr__(self, "timezone", defaults.get("timezone", "Asia/Shanghai"))
        if self.workspace_path is _UNSET:
            object.__setattr__(self, "workspace_path", defaults.get("workspace_path", "."))


@dataclass
class LoopResponse:
    """Response from a loop execution."""

    content: str | None
    iterations: int = 0
    tool_results: list[Any] = field(default_factory=list)
    error: str | None = None


class AgentLoop:
    """Core async agent loop.

    Manages the fetch → think → act → reflect cycle.
    Integrates with LLM providers, tool registry, memory, and hooks.
    """

    def __init__(
        self,
        config: LoopConfig | None = None,
        *,
        provider: LLMProvider | None = None,
        tool_registry: Any | None = None,
        memory_store: Any | None = None,
    ) -> None:
        self._config = config or LoopConfig()
        self._provider = provider
        self._registry = tool_registry
        self._memory = memory_store
        self._hooks: list[Any] = []
        self._stream_handlers: list[Callable[[str], Awaitable[None]]] = []
        self._tool_context: dict[str, str] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def process(
        self,
        message: str,
        *,
        session_key: str = "default",
        hooks: list[Any] | None = None,
        system_prompt: str | None = None,
        max_iterations: int | None = None,
    ) -> LoopResponse:
        """Process a single user message through the agent loop.

        Args:
            message: The user message
            session_key: Session identifier for memory isolation
            hooks: Lifecycle hooks for this run
            system_prompt: Override system prompt
            max_iterations: Override max tool-call iterations

        Returns:
            LoopResponse with the agent's final content
        """
        from agent.hook import AgentHookContext

        max_iterations = max_iterations or self._config.max_iterations
        effective_hooks = hooks or self._hooks

        # Build context
        memory = self._memory
        if memory:
            memory.get_session(session_key)

        # Add user message to memory
        if memory:
            memory.add_message(session_key, "user", message)

        # Get conversation history
        history = []
        if memory:
            history = memory.get_messages(session_key)

        # Build messages for LLM
        messages = history.copy()
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        iteration = 0
        all_tool_results: list[Any] = []
        content = ""

        while iteration < max_iterations:
            iteration += 1
            logger.debug("Iteration %d/%d", iteration, max_iterations)

            # Call LLM
            response = await self._call_llm(messages)
            if not response:
                return LoopResponse(content="", error="No response from LLM", iterations=iteration)

            content = response.get("content", "") or ""
            content = self._strip_think(content)

            # Check for tool calls
            tool_calls = self._extract_tool_calls(response)
            if not tool_calls:
                # No tools, we're done
                if memory:
                    memory.add_message(session_key, "assistant", content)
                return LoopResponse(content=content, iterations=iteration, tool_results=all_tool_results)

            # Build hook context
            context = AgentHookContext(
                loop=self,
                messages=messages,
                tool_calls=tool_calls,
                response=response,
                session_key=session_key,
            )

            # before_execute_tools hooks
            for hook in effective_hooks:
                if hasattr(hook, 'before_execute_tools'):
                    await hook.before_execute_tools(context)

            # Execute tools
            tool_results = []
            for tc in tool_calls:
                result = await self._execute_tool(tc)
                tool_results.append(result)
                all_tool_results.append(result)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.name,
                    "content": str(result.result) if result.error is None else f"Error: {result.error}",
                })

            # after_tools_executed hooks
            for hook in effective_hooks:
                if hasattr(hook, 'after_tools_executed'):
                    await hook.after_tools_executed(context)

        # Max iterations reached
        if memory:
            memory.add_message(session_key, "assistant", content)
        return LoopResponse(content=content, iterations=iteration, tool_results=all_tool_results)

    async def process_direct(
        self,
        message: str,
        *,
        session_key: str = "sdk:default",
    ) -> LoopResponse:
        """Direct process for SDK usage (no hooks)."""
        return await self.process(message, session_key=session_key)

    def get_messages(self, session_key: str) -> list[dict[str, Any]]:
        """Get messages for a session."""
        if self._memory:
            return self._memory.get_messages(session_key)
        return []

    # -------------------------------------------------------------------------
    # LLM Integration
    # -------------------------------------------------------------------------

    async def _call_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Call the configured LLM provider.

        Override this or inject a provider to use different LLMs.
        """
        if self._provider is None:
            # Default: try to use openai-compatible API
            return await self._call_openaiCompatible(messages)

        # Use injected provider
        return await self._provider.generate(messages)

    async def _call_openaiCompatible(self, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Default LLM call using OpenAI-compatible API."""
        import os

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            # Fallback: return a canned response for testing
            return {
                "content": "No LLM provider configured. Set OPENAI_API_KEY.",
                "tool_calls": [],
            }

        try:
            import openai
            client = openai.OpenAI(api_key=api_key)

            # Build tools from registry
            tools = None
            if self._registry:
                tools = self._registry.get_schemas()

            response = client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                tools=tools,
                temperature=self._config.temperature,
            )

            msg = response.choices[0].message
            return {
                "content": msg.content or "",
                "tool_calls": msg.tool_calls or [],
            }

        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return {"content": f"LLM error: {e}", "tool_calls": []}

    # -------------------------------------------------------------------------
    # Tool Execution
    # -------------------------------------------------------------------------

    async def _execute_tool(self, tool_call: Any) -> Any:
        """Execute a single tool call."""
        from agent.registry import ToolResult

        name = tool_call.name if hasattr(tool_call, 'name') else str(tool_call.get("name", ""))
        arguments = tool_call.arguments if hasattr(tool_call, 'arguments') else tool_call.get("arguments", {})

        logger.info("Executing tool: %s(%s)", name, json.dumps(arguments, ensure_ascii=False)[:200])

        if self._registry:
            return await self._registry.execute(name, arguments)

        return ToolResult(tool_name=name, result=None, error=f"No registry, cannot execute {name}")

    def _extract_tool_calls(self, response: dict[str, Any]) -> list[Any]:
        """Extract tool calls from LLM response."""
        raw = response.get("tool_calls", [])
        if not raw:
            return []

        # Handle both dict format and object format
        tool_calls = []
        for tc in raw:
            if isinstance(tc, dict):
                tool_calls.append(_DictToolCall(tc))
            else:
                tool_calls.append(tc)
        return tool_calls

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _strip_think(text: str | None) -> str:
        """Strip <think> tags from text."""
        if not text:
            return ""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _tool_hint(self, tool_calls: list[Any]) -> str:
        """Generate a hint string for pending tool calls."""
        if not tool_calls:
            return ""
        names = [tc.name if hasattr(tc, 'name') else tc.get("name", "?") for tc in tool_calls]
        return f"[Calling tools: {', '.join(names)}]"

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None) -> None:
        """Set context for tool execution (for logging/callbacks)."""
        self._tool_context = {
            "channel": channel,
            "chat_id": chat_id,
            "message_id": message_id or "",
        }


class _DictToolCall:
    """Wrapper to make a dict look like a tool call object."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @property
    def name(self) -> str:
        return self._data.get("name", "")

    @property
    def arguments(self) -> dict[str, Any]:
        return self._data.get("arguments", {})


# -------------------------------------------------------------------------
# Provider Interface (for dependency injection)
# -------------------------------------------------------------------------

class LLMProvider:
    """Abstract LLM provider interface.

    Implement this to add custom LLM backends.
    """

    async def generate(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate a response from the LLM.

        Returns:
            dict with 'content' and 'tool_calls' keys
        """
        raise NotImplementedError
