"""Model Context Protocol (MCP) server exposing the discovery workflow.

Lets an MCP client (Claude Desktop, Claude Code, etc.) drive the pipeline as
tools: run a full discovery, search the evidence corpus, or run the eval gate.

Design: the tool *logic* lives in plain, importable, keyless functions
(`run_discovery`, `search_evidence`, `run_eval`) that are fully unit-tested. The
MCP SDK wiring (`build_server`) is a thin layer imported lazily, so the core has
no hard dependency on `mcp` and the tests run without it. Install the server with
``pip install 'discovery-agents[mcp]'`` and run ``discovery-agents-mcp``.
"""

from __future__ import annotations

from typing import Any

from .config import RunConfig
from .pipeline import ProductDiscoveryPipeline
from .retrieval.index import EvidenceIndex
from .sample_data import SAMPLE_BRIEF, SAMPLE_EVIDENCE

# Machine-readable specs (also used to register tools and to test schemas).
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "discovery_run",
        "description": "Run the full product-discovery workflow on the built-in enterprise "
        "sample and return the selected direction, all directions, evals, and trace totals.",
        "input_schema": {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "description": "mock|anthropic|cohere|openai"}
            },
        },
    },
    {
        "name": "evidence_search",
        "description": "Search the customer-evidence corpus and return cited snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "eval_run",
        "description": "Run the evaluation harness and report the scorecard + regression gate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "description": "mock|anthropic|cohere|openai"}
            },
        },
    },
]


def run_discovery(provider: str = "mock") -> dict[str, Any]:
    """Run the discovery pipeline and return a JSON-serializable summary."""
    pipeline = ProductDiscoveryPipeline(RunConfig(provider=provider))
    run = pipeline.run(SAMPLE_BRIEF, SAMPLE_EVIDENCE)
    return {
        "selected_direction_id": run.selected_direction_id,
        "directions": [
            {
                "id": d.id,
                "title": d.title,
                "one_liner": d.one_liner,
                "evidence_ids": d.evidence_ids,
            }
            for d in run.directions
        ],
        "evals": [{"metric": e.metric, "score": e.score} for e in run.evals],
        "trace": {
            "spans": len(pipeline.trace.spans),
            "tokens": pipeline.trace.total_usage.total_tokens,
            "cost_usd": pipeline.trace.total_cost_usd,
        },
    }


def evidence_search(query: str, k: int = 3) -> dict[str, Any]:
    """Search the built-in evidence corpus; return cited snippets."""
    index = EvidenceIndex.from_evidence(SAMPLE_EVIDENCE)
    hits = index.search(query, k=max(1, k))
    return {
        "query": query,
        "results": [
            {"id": h.chunk.id, "text": h.chunk.text, "score": round(h.score, 4)} for h in hits
        ],
    }


def run_eval(provider: str = "mock") -> dict[str, Any]:
    """Run the eval harness and return the scorecard + regression verdict."""
    from .eval import EvalHarness, load_baseline

    harness = EvalHarness(RunConfig(provider=provider))
    scorecard = harness.evaluate()
    report = harness.check_regression(scorecard, load_baseline())
    return {
        "metrics": scorecard.metrics,
        "ops": scorecard.ops,
        "regression_passed": report.passed,
        "regressions": report.regressions,
    }


def build_server() -> Any:
    """Build a FastMCP server exposing the three tools (requires the mcp extra)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - only when extra is absent
        raise ImportError(
            "The MCP server requires the 'mcp' extra: pip install 'discovery-agents[mcp]'"
        ) from exc

    server = FastMCP("discovery-agents")  # pragma: no cover - exercised only with mcp installed
    server.add_tool(run_discovery, name="discovery_run", description=TOOL_SPECS[0]["description"])
    server.add_tool(
        evidence_search, name="evidence_search", description=TOOL_SPECS[1]["description"]
    )
    server.add_tool(run_eval, name="eval_run", description=TOOL_SPECS[2]["description"])
    return server


def main() -> None:  # pragma: no cover - process entry point
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
