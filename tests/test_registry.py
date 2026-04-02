"""Tests for ToolRegistry."""

from __future__ import annotations

import asyncio

import pytest

from agent.registry import ToolDef, ToolRegistry, get_registry


class TestToolDef:
    """Tests for ToolDef dataclass."""

    def test_to_openai_schema(self) -> None:
        """Test OpenAI schema generation."""
        tool = ToolDef(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
        )
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool"
        assert schema["function"]["parameters"] == {
            "type": "object",
            "properties": {"arg": {"type": "string"}},
        }

    def test_default_workspace_only(self) -> None:
        """Test default workspace_only is True."""
        tool = ToolDef(name="test", description="test")
        assert tool.workspace_only is True


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register_sync_tool(self) -> None:
        """Test registering a synchronous tool."""
        registry = ToolRegistry()

        @registry.register_sync("echo", "Echo back the input")
        def echo(text: str) -> str:
            return text

        tool = registry.get("echo")
        assert tool is not None
        assert tool.name == "echo"
        assert tool.description == "Echo back the input"
        assert tool.handler is not None
        assert tool.async_handler is None

    def test_register_async_tool(self) -> None:
        """Test registering an asynchronous tool."""
        registry = ToolRegistry()

        @registry.register_async("async_echo", "Async echo")
        async def async_echo(text: str) -> str:
            return text

        tool = registry.get("async_echo")
        assert tool is not None
        assert tool.name == "async_echo"
        assert tool.handler is None
        assert tool.async_handler is not None

    def test_register_decorator(self) -> None:
        """Test register decorator (auto-detects async)."""
        registry = ToolRegistry()

        @registry.register("greet", "Greet someone")
        def greet(name: str) -> str:
            return f"Hello, {name}"

        @registry.register("async_greet", "Async greet")
        async def async_greet(name: str) -> str:
            return f"Hello, {name}"

        assert registry.get("greet") is not None
        assert registry.get("async_greet") is not None

    def test_register_with_parameters(self) -> None:
        """Test registering tool with parameters schema."""
        registry = ToolRegistry()
        params = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The name"},
            },
            "required": ["name"],
        }

        @registry.register("greet", "Greet someone", parameters=params)
        def greet(name: str) -> str:
            return f"Hello, {name}"

        tool = registry.get("greet")
        assert tool is not None
        assert tool.parameters == params

    def test_register_workspace_only_false(self) -> None:
        """Test registering tool with workspace_only=False."""
        registry = ToolRegistry()

        @registry.register("global_tool", "Global tool", workspace_only=False)
        def global_tool() -> str:
            return "global"

        tool = registry.get("global_tool")
        assert tool is not None
        assert tool.workspace_only is False

    def test_get_nonexistent_tool(self) -> None:
        """Test getting a tool that doesn't exist."""
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_list_tools(self) -> None:
        """Test listing all registered tools."""
        registry = ToolRegistry()

        @registry.register("tool1", "First tool")
        def tool1() -> str:
            return "1"

        @registry.register("tool2", "Second tool")
        def tool2() -> str:
            return "2"

        tools = registry.list_tools()
        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "tool1" in names
        assert "tool2" in names

    def test_get_schemas(self) -> None:
        """Test getting OpenAI schemas for all tools."""
        registry = ToolRegistry()

        @registry.register("tool1", "First tool")
        def tool1() -> str:
            return "1"

        @registry.register("tool2", "Second tool")
        def tool2() -> str:
            return "2"

        schemas = registry.get_schemas()
        assert len(schemas) == 2
        assert all(s["type"] == "function" for s in schemas)

    def test_set_workspace(self) -> None:
        """Test setting workspace path."""
        registry = ToolRegistry()
        registry.set_workspace("/tmp/workspace")
        assert registry._workspace_path == "/tmp/workspace"


class TestToolRegistryExecute:
    """Tests for ToolRegistry.execute method."""

    @pytest.mark.asyncio
    async def test_execute_sync_tool(self) -> None:
        """Test executing a synchronous tool."""
        registry = ToolRegistry()

        @registry.register_sync("add", "Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        result = await registry.execute("add", {"a": 2, "b": 3})
        assert result.tool_name == "add"
        assert result.result == 5
        assert result.error is None
        assert result.execution_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_async_tool(self) -> None:
        """Test executing an asynchronous tool."""
        registry = ToolRegistry()

        @registry.register_async("async_add", "Async add")
        async def async_add(a: int, b: int) -> int:
            await asyncio.sleep(0.001)
            return a + b

        result = await registry.execute("async_add", {"a": 2, "b": 3})
        assert result.tool_name == "async_add"
        assert result.result == 5
        assert result.error is None

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self) -> None:
        """Test executing a tool that doesn't exist."""
        registry = ToolRegistry()
        result = await registry.execute("nonexistent", {})
        assert result.tool_name == "nonexistent"
        assert result.result is None
        assert result.error == "Unknown tool: nonexistent"

    @pytest.mark.asyncio
    async def test_execute_with_exception(self) -> None:
        """Test tool execution that raises an exception."""
        registry = ToolRegistry()

        @registry.register_sync("fail", "A failing tool")
        def fail() -> None:
            raise ValueError("intentional failure")

        result = await registry.execute("fail", {})
        assert result.tool_name == "fail"
        assert result.result is None
        assert result.error == "intentional failure"


class TestToolRegistryExecuteSync:
    """Tests for ToolRegistry.execute_sync method."""

    def test_execute_sync_tool(self) -> None:
        """Test sync execution of a sync tool."""
        registry = ToolRegistry()

        @registry.register_sync("multiply", "Multiply two numbers")
        def multiply(a: int, b: int) -> int:
            return a * b

        result = registry.execute_sync("multiply", {"a": 3, "b": 4})
        assert result.tool_name == "multiply"
        assert result.result == 12
        assert result.error is None

    def test_execute_sync_nonexistent_tool(self) -> None:
        """Test sync execution of nonexistent tool."""
        registry = ToolRegistry()
        result = registry.execute_sync("nonexistent", {})
        assert result.error == "Unknown tool: nonexistent"

    def test_execute_sync_async_tool_returns_error(self) -> None:
        """Test that executing async tool via sync returns error result."""
        registry = ToolRegistry()

        @registry.register_async("async_only", "Async only")
        async def async_only() -> str:
            return "async"

        result = registry.execute_sync("async_only", {})
        assert result.error is not None
        assert "async" in result.error.lower()


class TestGetRegistry:
    """Tests for get_registry function."""

    def test_get_registry_returns_instance(self) -> None:
        """Test that get_registry returns a ToolRegistry instance."""
        registry = get_registry()
        assert isinstance(registry, ToolRegistry)

    def test_get_registry_returns_same_instance(self) -> None:
        """Test that get_registry returns the same instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2
