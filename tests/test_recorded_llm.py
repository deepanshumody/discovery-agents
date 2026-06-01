"""Recorded-response tests: exercise the real LLM parsing path (key-free).

These drive the agents with realistic captured model output (the code path the
deterministic mock skips), proving prompts/JSON parsing actually work.
"""

from __future__ import annotations

from discovery_agents import ProductDiscoveryPipeline, RunConfig
from discovery_agents.agents.ideation import IdeationAgent
from discovery_agents.llm._parsing import extract_json
from discovery_agents.llm.mock import MockLLMClient
from discovery_agents.llm.types import LLMResponse
from discovery_agents.models import Insight, ProductBrief
from discovery_agents.retrieval import EvidenceIndex
from discovery_agents.sample_data import SAMPLE_BRIEF, SAMPLE_EVIDENCE

# A realistic model reply: prose + a fenced JSON block (what a real provider returns).
RECORDED_RAW = """Here are the directions:
```json
{"directions": [
  {"title": "Self-Serve Card Controls", "one_liner": "Let customers freeze/replace cards in-app.",
   "evidence_ids": ["E1", "E4"], "differentiator": "Cuts card-related support volume.",
   "core_loop": "User opens card -> freeze/replace -> done", "why_now": "Top support driver",
   "implementation_notes": ["card state API"], "risks": ["fraud edge cases"]}
]}
```
Hope this helps."""


def test_extract_json_from_realistic_model_text() -> None:
    parsed = extract_json(RECORDED_RAW)
    assert parsed is not None
    assert parsed["directions"][0]["title"] == "Self-Serve Card Controls"


def test_ideation_uses_recorded_model_directions_not_fallback() -> None:
    structured = extract_json(RECORDED_RAW)
    recorded = LLMResponse(text=RECORDED_RAW, structured=structured, model="recorded")
    index = EvidenceIndex.from_evidence(SAMPLE_EVIDENCE)
    agent = IdeationAgent(llm=MockLLMClient(script=[recorded]))
    insights = [Insight("I1", "t", "s", ["E1", "E4"], 0.8, ["handoff"])]

    directions = agent.run(SAMPLE_BRIEF, insights, ["opportunity"], index=index)
    # The model's direction is used (not the curated deterministic baseline).
    assert directions[0].title == "Self-Serve Card Controls"
    assert set(directions[0].evidence_ids) <= {"E1", "E4"}  # citations parsed + corpus-validated


def test_parse_directions_drops_hallucinated_citations() -> None:
    agent = IdeationAgent(llm=MockLLMClient())
    value = {
        "directions": [
            {"title": "X", "one_liner": "y", "evidence_ids": ["E1", "E999"], "differentiator": "d"}
        ]
    }
    parsed = agent._parse_directions(value, ProductBrief("c", "p", "u", "g"), valid_evidence={"E1"})
    assert parsed is not None
    assert parsed[0].evidence_ids == ["E1"]  # E999 is not in the corpus -> dropped


def test_pipeline_uses_recorded_model_output() -> None:
    structured = extract_json(RECORDED_RAW)
    pipeline = ProductDiscoveryPipeline(RunConfig(provider="mock"))
    # Inject a recorded structured response into the ideation agent's client.
    pipeline.ideation_agent.llm = MockLLMClient(script=[LLMResponse(structured=structured)])
    run = pipeline.run(SAMPLE_BRIEF, SAMPLE_EVIDENCE)
    assert any(d.title == "Self-Serve Card Controls" for d in run.directions)
