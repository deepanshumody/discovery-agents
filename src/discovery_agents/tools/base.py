"""Tool protocol and registry for agent tool-use.

A tool advertises a name + JSON-schema parameters and runs against a dict of
arguments (the shape a model emits in a tool call), returning a `ToolResult`
that carries data plus citation ids for grounded, auditable answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..llm.types import ToolSpec


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    citations: list[str] = field(default_factory=list)
    error: str = ""


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]

    def run(self, arguments: dict[str, Any]) -> ToolResult: ...


class ToolRegistry:
    """Holds tools and exposes them as model-facing specs."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(name=t.name, description=t.description, parameters=t.parameters)
            for t in self._tools.values()
        ]

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"unknown tool: {name}")
        try:
            return tool.run(arguments)
        except Exception as exc:  # tools must never crash the agent loop
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
