"""Tests for the MCP server tool logic (keyless; SDK wiring is skipped if absent)."""

from __future__ import annotations

import importlib.util

import pytest

from discovery_agents import mcp_server


def test_tool_specs_are_well_formed() -> None:
    names = {spec["name"] for spec in mcp_server.TOOL_SPECS}
    assert names == {"discovery_run", "evidence_search", "eval_run"}
    for spec in mcp_server.TOOL_SPECS:
        assert spec["input_schema"]["type"] == "object"
        assert spec["description"]


def test_run_discovery_returns_summary() -> None:
    result = mcp_server.run_discovery()
    assert result["selected_direction_id"] is not None
    assert len(result["directions"]) >= 5
    assert result["trace"]["spans"] > 0
    assert all("evidence_ids" in d for d in result["directions"])


def test_evidence_search_returns_cited_snippets() -> None:
    result = mcp_server.evidence_search("implementation handoff", k=2)
    assert result["query"] == "implementation handoff"
    assert len(result["results"]) == 2
    assert any(r["id"] == "E4" for r in result["results"])


def test_run_eval_reports_regression_verdict() -> None:
    result = mcp_server.run_eval()
    assert result["regression_passed"] is True
    assert "citation_precision" in result["metrics"]


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="mcp extra not installed")
def test_build_server_registers_tools() -> None:  # pragma: no cover - needs mcp
    server = mcp_server.build_server()
    assert server is not None
