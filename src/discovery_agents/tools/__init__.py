"""Tools the agents can call (registry + concrete tools)."""

from __future__ import annotations

from .base import Tool, ToolRegistry, ToolResult
from .calculator import CalculatorTool
from .evidence_search import EvidenceSearchTool
from .web_search import WebSearchTool

__all__ = [
    "CalculatorTool",
    "EvidenceSearchTool",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "WebSearchTool",
]
