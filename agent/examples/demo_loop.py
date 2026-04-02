#!/usr/bin/env python3
"""Demo: Minimal agent loop example.

Run with:
    python3 -m agent.examples.demo_loop
"""

import asyncio
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.hook import HookBuilder
from agent.loop import AgentLoop, LoopConfig
from agent.memory import MemoryStore
from agent.registry import get_registry

# ---------------------------------------------------------------------------
# Register some tools
# -------------------------------------------------------------------------

registry = get_registry()
registry.set_workspace(".")

@registry.register_async(
    "echo",
    "Echo back the input text with a prefix",
    {"type": "object", "properties": {"text": {"type": "string"}}},
)
async def echo(text: str) -> str:
    return f"[echo] {text}"


@registry.register_async(
    "add",
    "Add two numbers together",
    {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
)
async def add(a: float, b: float) -> float:
    return a + b


# ---------------------------------------------------------------------------
# Demo
# -------------------------------------------------------------------------

async def main():
    print("🚀 nanobot-style AgentLoop demo\n")

    # Config
    config = LoopConfig(
        model="claude-sonnet-4-20250514",
        max_iterations=3,
        temperature=0.7,
    )

    # Memory
    memory = MemoryStore()

    # Create loop
    loop = AgentLoop(
        config=config,
        tool_registry=registry,
        memory_store=memory,
    )

    # Hooks
    hooks = HookBuilder() \
        .with_logging() \
        .with_progress(lambda msg: print(f"📤 {msg[:80]}...")) \
        .build()

    # System prompt
    system_prompt = """You are a helpful assistant running in workspace-coder.
You have access to tools. Use them when needed.
Be concise and practical."""

    # Run
    print("Running: 'What is 2 + 3?'\n")
    result = await loop.process(
        message="What is 2 + 3?",
        session_key="demo",
        hooks=[hooks],
        system_prompt=system_prompt,
    )

    print(f"\n✅ Iterations: {result.iterations}")
    print(f"📄 Content: {result.content}")
    print(f"🔧 Tool results: {len(result.tool_results)}")
    for tr in result.tool_results:
        print(f"   - {tr.tool_name}: {tr.result} ({tr.execution_ms:.1f}ms)")

    print("\nMemory messages:", len(memory.get_messages("demo")))


if __name__ == "__main__":
    asyncio.run(main())
