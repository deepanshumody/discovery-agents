"""Recorded-response tests for the reasoning agents wired to the LLM.

Each test injects a realistic captured structured response (the path the keyless
mock skips) and asserts the agent uses the model's output, then a second case
asserts the deterministic fallback still fires when the model returns nothing.
This proves the prompt/parse/overlay code in each agent actually works without a
key, while the `--eval` gate proves the mock baseline is unchanged.
"""

from __future__ import annotations

from discovery_agents.agents.critique import CritiqueAgent
from discovery_agents.agents.evidence_insight import EvidenceInsightAgent
from discovery_agents.agents.handoff import HandoffAgent
from discovery_agents.agents.memory import DecisionMemoryAgent
from discovery_agents.agents.strategy import ProductStrategyAgent
from discovery_agents.llm.mock import MockLLMClient
from discovery_agents.llm.types import LLMResponse
from discovery_agents.models import CritiqueScore, Insight, ProductBrief, ProductDirection
from discovery_agents.retrieval import EvidenceIndex
from discovery_agents.sample_data import SAMPLE_EVIDENCE


def _scripted(structured: dict | None) -> MockLLMClient:
    return MockLLMClient(script=[LLMResponse(structured=structured)])


def _direction(did: str, title: str) -> ProductDirection:
    return ProductDirection(
        id=did,
        title=title,
        one_liner=f"{title} in one line.",
        target_user="product team",
        core_loop="open -> act -> done",
        why_now="now",
        evidence_ids=["E1"],
        differentiator="not generic",
        implementation_notes=["a", "b", "c"],
        risks=["scope creep"],
    )


# --- Strategy ---------------------------------------------------------------


def test_strategy_uses_recorded_opportunities() -> None:
    brief = ProductBrief("c", "p", "u", "Ship faster")
    insights = [Insight("I1", "t", "s", ["E1"], 0.8, ["ideation"])]
    agent = ProductStrategyAgent(llm=_scripted({"opportunities": ["Bundle onboarding flows"]}))

    out = agent.run(brief, insights)

    assert out[0] == "Primary goal: Ship faster"  # goal always leads
    assert "Bundle onboarding flows" in out  # model opportunity used, not the rule set


def test_strategy_falls_back_when_model_silent() -> None:
    brief = ProductBrief("c", "p", "u", "Ship faster")
    insights = [Insight("I1", "t", "s", ["E1"], 0.8, ["ideation"])]
    agent = ProductStrategyAgent(llm=_scripted(None))

    out = agent.run(brief, insights)

    assert out[0] == "Primary goal: Ship faster"
    assert any("blank-state" in line for line in out)  # deterministic rule fired


# --- Critique ---------------------------------------------------------------


def test_critique_uses_recorded_scorecard() -> None:
    directions = [_direction("D1", "Alpha"), _direction("D2", "Beta")]
    scores = dict.fromkeys(
        ("customer_alignment", "novelty", "feasibility", "strategic_fit", "clarity", "risk_level"),
        4,
    )
    structured = {
        "critiques": [
            {"direction_id": "D1", **scores, "summary": "Solid.", "recommended_next_step": "ship it"},
            {"direction_id": "D2", **{**scores, "risk_level": 2}},
        ]
    }
    agent = CritiqueAgent(llm=_scripted(structured))

    out = agent.run(directions, SAMPLE_EVIDENCE)

    assert {c.direction_id for c in out} == {"D1", "D2"}
    d1 = next(c for c in out if c.direction_id == "D1")
    assert d1.summary == "Solid." and d1.recommended_next_step == "ship it"


def test_critique_falls_back_when_a_direction_is_missing() -> None:
    directions = [_direction("D1", "Alpha"), _direction("D2", "Beta")]
    # Model only scored D1 -> incomplete coverage -> deterministic fallback for all.
    structured = {"critiques": [{"direction_id": "D1", "customer_alignment": 5}]}
    agent = CritiqueAgent(llm=_scripted(structured))

    out = agent.run(directions, SAMPLE_EVIDENCE)

    assert {c.direction_id for c in out} == {"D1", "D2"}
    assert all(isinstance(c, CritiqueScore) for c in out)


# --- Memory -----------------------------------------------------------------


def test_memory_uses_recorded_decision_log() -> None:
    directions = [_direction("D1", "Alpha"), _direction("D2", "Beta")]
    critiques = [
        CritiqueScore("D1", 5, 5, 4, 5, 5, 2, "s", "n"),
        CritiqueScore("D2", 3, 3, 3, 3, 3, 4, "s", "n"),
    ]
    agent = DecisionMemoryAgent(llm=_scripted({"decision_log": ["Chose Alpha for reach."]}))

    out = agent.run("D1", directions, critiques)

    assert out == ["Chose Alpha for reach."]


def test_memory_falls_back_when_model_silent() -> None:
    directions = [_direction("D1", "Alpha"), _direction("D2", "Beta")]
    critiques = [
        CritiqueScore("D1", 5, 5, 4, 5, 5, 2, "s", "n"),
        CritiqueScore("D2", 3, 3, 3, 3, 3, 4, "s", "n"),
    ]
    agent = DecisionMemoryAgent(llm=_scripted(None))

    out = agent.run("D1", directions, critiques)

    assert len(out) == 3 and out[0].startswith("Selected 'Alpha'")


# --- Handoff ----------------------------------------------------------------


def test_handoff_overlays_recorded_prose_but_keeps_data_contract() -> None:
    selected = _direction("D1", "Alpha")
    structured = {
        "user_story": "As a PM, I want X so that Y.",
        "functional_requirements": ["Do the thing"],
        "acceptance_criteria": ["It does the thing"],
    }
    agent = HandoffAgent(llm=_scripted(structured))

    spec = agent.run(selected)

    assert spec.user_story == "As a PM, I want X so that Y."
    assert spec.functional_requirements == ["Do the thing"]
    assert "ProductBrief" in spec.data_contract  # structural fields stay deterministic
    assert spec.analytics_events  # unchanged


def test_handoff_falls_back_when_model_silent() -> None:
    selected = _direction("D1", "Alpha")
    agent = HandoffAgent(llm=_scripted(None))

    spec = agent.run(selected)

    assert spec.user_story.startswith("As a product team")
    assert len(spec.functional_requirements) == 5


# --- EvidenceInsight --------------------------------------------------------


def test_insight_uses_recorded_summary_for_one_cluster() -> None:
    index = EvidenceIndex.from_evidence(SAMPLE_EVIDENCE)
    structured = {"summaries": [{"id": "I1", "summary": "Teams want instant divergent options."}]}
    agent = EvidenceInsightAgent(llm=_scripted(structured))

    insights = agent.run(SAMPLE_EVIDENCE, index=index)

    by_id = {i.id: i for i in insights}
    assert by_id["I1"].summary == "Teams want instant divergent options."
    # A cluster the model didn't summarize keeps the deterministic text.
    other = next(i for i in insights if i.id != "I1")
    assert other.summary.startswith("This theme is supported by")


def test_insight_falls_back_when_model_silent() -> None:
    index = EvidenceIndex.from_evidence(SAMPLE_EVIDENCE)
    agent = EvidenceInsightAgent(llm=_scripted(None))

    insights = agent.run(SAMPLE_EVIDENCE, index=index)

    assert insights and all(i.summary.startswith("This theme is supported by") for i in insights)
