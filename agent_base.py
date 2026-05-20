"""Agent base classes and tracing helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


@dataclass
class TraceEvent:
    agent: str
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AgentTrace:
    """Simple trace collector for explainable multi-agent runs."""

    def __init__(self) -> None:
        self.events: List[TraceEvent] = []

    def log(self, agent: str, message: str, **payload: Any) -> None:
        self.events.append(TraceEvent(agent=agent, message=message, payload=payload))

    def as_markdown(self) -> str:
        lines = ["# Agent Trace", ""]
        for event in self.events:
            lines.append(f"## {event.agent}")
            lines.append(f"- Time: `{event.timestamp}`")
            lines.append(f"- Message: {event.message}")
            if event.payload:
                lines.append(f"- Payload: `{event.payload}`")
            lines.append("")
        return "\n".join(lines)


class BaseAgent:
    """Small base class so every agent has a name and a trace."""

    name = "BaseAgent"

    def __init__(self, trace: AgentTrace | None = None) -> None:
        self.trace = trace or AgentTrace()

    def log(self, message: str, **payload: Any) -> None:
        self.trace.log(self.name, message, **payload)
