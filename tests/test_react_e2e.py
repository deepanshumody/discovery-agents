"""End-to-end ReAct test: the agent calls multiple tools, then answers."""

from __future__ import annotations

from discovery_agents.llm.mock import MockLLMClient
from discovery_agents.llm.types import LLMResponse, ToolInvocation
from discovery_agents.retrieval import EvidenceIndex
from discovery_agents.runtime.agent import LLMAgent
from discovery_agents.sample_data import SAMPLE_EVIDENCE
from discovery_agents.tools import (
    CalculatorTool,
    EvidenceSearchTool,
    ToolRegistry,
    WebSearchTool,
)


def test_react_uses_evidence_search_then_calculator_then_answers() -> None:
    registry = ToolRegistry(
        [
            EvidenceSearchTool(EvidenceIndex.from_evidence(SAMPLE_EVIDENCE)),
            CalculatorTool(),
            WebSearchTool(),
        ]
    )
    script = [
        LLMResponse(
            tool_calls=[
                ToolInvocation(name="evidence_search", arguments={"query": "handoff", "k": 1})
            ]
        ),
        LLMResponse(
            tool_calls=[ToolInvocation(name="calculator", arguments={"expression": "3 * 2"})]
        ),
        LLMResponse(text="The handoff gap matters; estimated impact factor is 6."),
    ]
    agent = LLMAgent(MockLLMClient(script=script), registry, max_steps=6)
    result = agent.run("Assess the handoff gap and compute 3*2.")

    assert result.stop_reason == "final_answer"
    assert [step.tool for step in result.steps] == ["evidence_search", "calculator"]
    assert result.steps[0].ok and result.steps[1].ok
    assert result.steps[1].observation["result"] == 6.0
    assert "E4" in result.citations  # evidence_search returned a real citation id
    # One LLM span per model turn (3) + one tool span per tool call (2).
    assert sum(1 for s in agent.trace.spans if s.op == "react.step") == 3
    assert sum(1 for s in agent.trace.spans if s.op == "tool") == 2
