"""Tests for the tool registry and the evidence_search RAG tool."""

from __future__ import annotations

from discovery_agents.retrieval import EvidenceIndex
from discovery_agents.sample_data import SAMPLE_EVIDENCE
from discovery_agents.tools import (
    CalculatorTool,
    EvidenceSearchTool,
    ToolRegistry,
    WebSearchTool,
)


def _registry() -> ToolRegistry:
    index = EvidenceIndex.from_evidence(SAMPLE_EVIDENCE)
    return ToolRegistry([EvidenceSearchTool(index)])


def test_calculator_respects_operator_precedence() -> None:
    result = CalculatorTool().run({"expression": "2 + 2 * 3"})
    assert result.ok
    assert result.data["result"] == 8.0  # multiplication binds tighter than addition


def test_calculator_rejects_unsafe_input() -> None:
    result = CalculatorTool().run({"expression": "__import__('os').system('ls')"})
    assert not result.ok
    assert "cannot evaluate" in result.error


def test_calculator_handles_division_by_zero() -> None:
    result = CalculatorTool().run({"expression": "1/0"})
    assert not result.ok


def test_web_search_is_deterministic_and_cited() -> None:
    tool = WebSearchTool()
    a = tool.run({"query": "enterprise agentic workflows", "k": 2})
    b = tool.run({"query": "enterprise agentic workflows", "k": 2})
    assert a.ok and a.data == b.data  # deterministic default backend
    assert len(a.citations) == 2
    assert all(url.startswith("https://") for url in a.citations)


def test_tool_conforms_to_protocol_shape() -> None:
    tool = EvidenceSearchTool(EvidenceIndex.from_evidence(SAMPLE_EVIDENCE))
    assert tool.name == "evidence_search"
    assert tool.parameters["type"] == "object"
    assert "query" in tool.parameters["properties"]


def test_registry_specs_and_membership() -> None:
    registry = _registry()
    assert "evidence_search" in registry
    specs = registry.specs()
    assert len(specs) == 1
    assert specs[0].name == "evidence_search"


def test_evidence_search_returns_cited_results() -> None:
    registry = _registry()
    result = registry.run("evidence_search", {"query": "implementation handoff", "k": 2})
    assert result.ok
    assert len(result.citations) == 2
    assert all(isinstance(cid, str) for cid in result.citations)
    assert "E4" in result.citations


def test_evidence_search_requires_query() -> None:
    registry = _registry()
    result = registry.run("evidence_search", {"query": "  "})
    assert not result.ok
    assert "query" in result.error


def test_unknown_tool_is_handled_gracefully() -> None:
    registry = _registry()
    result = registry.run("nope", {})
    assert not result.ok
    assert "unknown tool" in result.error
