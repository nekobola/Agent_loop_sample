"""Agent runner and run result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.hook import AgentHook


@dataclass
class RunResult:
    """Result of a single agent run."""

    content: str
    tools_used: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    iterations: int = 0


@dataclass
class AgentRunSpec:
    """Specification for a single agent run."""

    message: str
    session_key: str = "default"
    hooks: list[AgentHook] | None = None
    system_prompt: str | None = None
    max_iterations: int = 10
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRunner:
    """Runs a single agent execution based on a spec.

    This is a thin wrapper that delegates to the AgentLoop.
    """

    def __init__(self, loop: Any) -> None:
        self._loop = loop

    async def run(self, spec: AgentRunSpec) -> RunResult:
        """Execute a single agent run."""
        from agent.hook import CompositeHook

        hooks = spec.hooks or []
        composite = CompositeHook(hooks) if len(hooks) > 1 else (hooks[0] if hooks else None)

        try:
            response = await self._loop.process(
                message=spec.message,
                session_key=spec.session_key,
                hooks=[composite] if composite else [],
                system_prompt=spec.system_prompt,
                max_iterations=spec.max_iterations,
            )

            content = response.content if response else ""
            return RunResult(
                content=content or "",
                tools_used=[],  # Populated by loop
                messages=self._loop.get_messages(spec.session_key),
                iterations=response.iterations if response else 0,
            )

        except Exception as e:
            return RunResult(
                content="",
                error=str(e),
                iterations=0,
            )
