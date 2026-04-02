"""Context builder for constructing LLM prompts."""

from __future__ import annotations

import re
from typing import Any


class ContextBuilder:
    """Builds context/messages for the LLM.

    Handles system prompt composition, message formatting,
    and context window management.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        max_context_tokens: int = 100000,
    ) -> None:
        self._system_prompt = system_prompt or ""
        self._max_context_tokens = max_context_tokens
        self._tools: list[dict[str, Any]] = []

    def set_system_prompt(self, prompt: str) -> ContextBuilder:
        """Set the system prompt."""
        self._system_prompt = prompt
        return self

    def add_system_prompt(self, text: str) -> ContextBuilder:
        """Append to the system prompt."""
        self._system_prompt += "\n" + text
        return self

    def set_tools(self, tools: list[dict[str, Any]]) -> ContextBuilder:
        """Set available tools for the LLM."""
        self._tools = tools
        return self

    def build(
        self,
        messages: list[dict[str, Any]],
        *,
        prepend_system: bool = True,
    ) -> list[dict[str, Any]]:
        """Build the final message list for the LLM.

        Args:
            messages: Conversation history messages
            prepend_system: Whether to prepend system prompt

        Returns:
            Formatted message list ready for LLM consumption
        """
        result: list[dict[str, Any]] = []

        if prepend_system and self._system_prompt:
            system_content = self._system_prompt
            if self._tools:
                system_content += "\n\n" + self._format_tools(self._tools)
            result.append({"role": "system", "content": system_content})

        result.extend(messages)
        return result

    def _format_tools(self, tools: list[dict[str, Any]]) -> str:
        """Format tools into a string for the system prompt."""
        lines = ["## Available Tools", ""]
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")
            params = tool.get("parameters", {})
            lines.append(f"### {name}")
            lines.append(f"{desc}")
            if params:
                lines.append(f"Parameters: {params}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def strip_think(text: str | None) -> str:
        """Strip <think> tags from text."""
        if not text:
            return ""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    @staticmethod
    def format_tool_result(tool_name: str, result: Any) -> dict[str, Any]:
        """Format a tool execution result as a tool message."""
        content = str(result) if result is not None else ""
        return {
            "role": "tool",
            "tool_call_id": tool_name,
            "content": content,
        }
