"""Tool registry for agent tools."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import logging
logger = logging.getLogger(__name__)


@dataclass
class ToolDef:
    """Definition of a registered tool."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., Any] = field(default=None)
    async_handler: Callable[..., Awaitable[Any]] | None = field(default=None)
    workspace_only: bool = True

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    """Result of a tool execution."""

    tool_name: str
    result: Any
    error: str | None = None
    execution_ms: float = 0.0


class ToolRegistry:
    """Registry for agent tools.

    Tools can be registered with sync or async handlers.
    The registry handles execution, error catching, and schema generation.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}
        self._workspace_path: str | None = None

    def set_workspace(self, path: str) -> None:
        """Set the workspace path for workspace-restricted tools."""
        self._workspace_path = path

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        *,
        workspace_only: bool = True,
    ) -> Callable:
        """Decorator to register a tool.

        Usage:
            @registry.register("read_file", "Read a file from disk")
            async def read_file(path: str) -> str:
                ...
        """
        def decorator(func: Callable) -> Callable:
            is_async = asyncio.iscoroutinefunction(func)
            self._tools[name] = ToolDef(
                name=name,
                description=description,
                parameters=parameters or {},
                handler=func if not is_async else None,
                async_handler=func if is_async else None,
                workspace_only=workspace_only,
            )
            return func
        return decorator

    def register_sync(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        *,
        workspace_only: bool = True,
    ) -> Callable:
        """Decorator for sync tool handlers."""
        def decorator(func: Callable) -> Callable:
            self._tools[name] = ToolDef(
                name=name,
                description=description,
                parameters=parameters or {},
                handler=func,
                async_handler=None,
                workspace_only=workspace_only,
            )
            return func
        return decorator

    def register_async(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        *,
        workspace_only: bool = True,
    ) -> Callable:
        """Decorator for async tool handlers."""
        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable:
            self._tools[name] = ToolDef(
                name=name,
                description=description,
                parameters=parameters or {},
                handler=None,
                async_handler=func,
                workspace_only=workspace_only,
            )
            return func
        return decorator

    def get(self, name: str) -> ToolDef | None:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDef]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI function-calling schemas for all tools."""
        return [t.to_openai_schema() for t in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with given arguments.

        Returns:
            ToolResult with the execution outcome
        """
        import time
        start = time.monotonic()

        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                tool_name=name,
                result=None,
                error=f"Unknown tool: {name}",
                execution_ms=0.0,
            )

        try:
            if tool.async_handler:
                result = await tool.async_handler(**arguments)
            elif tool.handler:
                result = tool.handler(**arguments)
            else:
                result = None

            execution_ms = (time.monotonic() - start) * 1000
            return ToolResult(
                tool_name=name,
                result=result,
                execution_ms=execution_ms,
            )

        except Exception as e:
            execution_ms = (time.monotonic() - start) * 1000
            logger.exception("Tool %s failed: %s", name, e)
            return ToolResult(
                tool_name=name,
                result=None,
                error=str(e),
                execution_ms=execution_ms,
            )

    def execute_sync(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Sync version of execute for non-async contexts."""
        import time
        start = time.monotonic()

        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                tool_name=name,
                result=None,
                error=f"Unknown tool: {name}",
                execution_ms=0.0,
            )

        try:
            if tool.handler:
                result = tool.handler(**arguments)
            elif tool.async_handler:
                raise RuntimeError(f"Tool {name} is async, use execute()")
            else:
                result = None

            execution_ms = (time.monotonic() - start) * 1000
            return ToolResult(
                tool_name=name,
                result=result,
                execution_ms=execution_ms,
            )

        except Exception as e:
            execution_ms = (time.monotonic() - start) * 1000
            logger.exception("Tool %s failed: %s", name, e)
            return ToolResult(
                tool_name=name,
                result=None,
                error=str(e),
                execution_ms=execution_ms,
            )


# Global registry instance
_default_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry
