"""Tests for the ReAct agent loop and the LangGraph adapter (skipped w/o extra)."""

from __future__ import annotations

import importlib.util

import pytest

from discovery_agents.llm.mock import MockLLMClient
from discovery_agents.llm.types import LLMResponse, ToolInvocation
from discovery_agents.retrieval import EvidenceIndex
from discovery_agents.runtime.agent import LLMAgent
from discovery_agents.runtime.graph import build_discovery_graph
from discovery_agents.sample_data import SAMPLE_EVIDENCE
from discovery_agents.tools import EvidenceSearchTool, ToolRegistry


def _registry() -> ToolRegistry:
    return ToolRegistry([EvidenceSearchTool(EvidenceIndex.from_evidence(SAMPLE_EVIDENCE))])


def test_react_terminates_on_final_answer_with_mock() -> None:
    agent = LLMAgent(MockLLMClient(), _registry())
    result = agent.run("What should we build?")
    assert result.stop_reason == "final_answer"
    assert result.steps == []
    assert result.answer  # mock returns deterministic text


def test_react_runs_a_tool_then_finishes() -> None:
    script = [
        LLMResponse(
            tool_calls=[
                ToolInvocation(name="evidence_search", arguments={"query": "handoff", "k": 1})
            ]
        ),
        LLMResponse(text="Build the handoff packet."),
    ]
    agent = LLMAgent(MockLLMClient(script=script), _registry())
    result = agent.run("Investigate the handoff gap.")
    assert result.stop_reason == "final_answer"
    assert result.answer == "Build the handoff packet."
    assert len(result.steps) == 1
    assert result.steps[0].tool == "evidence_search"
    assert result.steps[0].ok
    assert result.citations  # tool returned citation ids
    tool_spans = [s for s in agent.trace.spans if s.op == "tool"]
    assert len(tool_spans) == 1


def test_react_stops_at_max_steps() -> None:
    # Every response asks for a tool, so it never produces a final answer.
    loop = [
        LLMResponse(tool_calls=[ToolInvocation(name="evidence_search", arguments={"query": "x"})])
    ]
    agent = LLMAgent(MockLLMClient(script=loop * 5), _registry(), max_steps=3)
    result = agent.run("loop forever")
    assert result.stop_reason == "max_steps"
    assert len(result.steps) == 3


def test_discovery_graph_has_expected_nodes() -> None:
    from discovery_agents.agents import (
        CanvasAgent,
        CritiqueAgent,
        DecisionMemoryAgent,
        EvalAgent,
        EvidenceInsightAgent,
        HandoffAgent,
        IdeationAgent,
        ProductStrategyAgent,
        SelectionAgent,
    )
    from discovery_agents.observability.trace import Trace

    trace = Trace()
    graph = build_discovery_graph(
        trace=trace,
        evidence_agent=EvidenceInsightAgent(trace),
        strategy_agent=ProductStrategyAgent(trace),
        ideation_agent=IdeationAgent(trace),
        critique_agent=CritiqueAgent(trace),
        canvas_agent=CanvasAgent(trace),
        selection_agent=SelectionAgent(trace),
        handoff_agent=HandoffAgent(trace),
        eval_agent=EvalAgent(trace),
        memory_agent=DecisionMemoryAgent(trace),
    )
    names = [n.name for n in graph.ordered_nodes({"brief", "evidence", "index"})]
    assert names[0] == "EvidenceInsightAgent"
    assert names[-1] in {"EvalAgent", "DecisionMemoryAgent"}
    assert len(names) == 9


@pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None, reason="langgraph extra not installed"
)
def test_langgraph_adapter_matches_state_machine() -> None:  # pragma: no cover
    from discovery_agents import ProductDiscoveryPipeline, RunConfig
    from discovery_agents.sample_data import SAMPLE_BRIEF

    baseline = ProductDiscoveryPipeline().run(SAMPLE_BRIEF, SAMPLE_EVIDENCE)
    via_lg = ProductDiscoveryPipeline(RunConfig(use_langgraph=True)).run(
        SAMPLE_BRIEF, SAMPLE_EVIDENCE
    )
    assert via_lg.selected_direction_id == baseline.selected_direction_id
    assert len(via_lg.directions) == len(baseline.directions)
