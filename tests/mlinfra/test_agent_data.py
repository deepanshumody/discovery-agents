"""Real Banking77 evidence flows through the agent pipeline (data-driven clustering)."""

from __future__ import annotations

from discovery_agents import ProductDiscoveryPipeline
from discovery_agents.agents.evidence_insight import EvidenceInsightAgent
from discovery_agents.mlinfra.agent_data import evidence_from_banking77


def test_evidence_from_banking77_shape() -> None:
    brief, evidence = evidence_from_banking77(intents=5, per_intent=3)
    assert len(evidence) == 15
    assert brief.company == "NeoBank"
    assert all(item.tags and item.text for item in evidence)  # each tagged with its intent
    assert len({item.tags[0] for item in evidence}) == 5  # 5 distinct intents


def test_data_driven_clustering_on_real_tags() -> None:
    # Real intent tags don't match the curated THEMES, so clustering falls back to
    # the evidence's own tags -> one insight per intent present.
    _, evidence = evidence_from_banking77(intents=4, per_intent=3)
    insights = EvidenceInsightAgent().run(evidence)
    assert len(insights) == 4
    intents = {item.tags[0] for item in evidence}
    assert all(set(i.tags) <= intents for i in insights)


def test_pipeline_runs_on_real_banking77_evidence() -> None:
    brief, evidence = evidence_from_banking77(intents=6, per_intent=3)
    run = ProductDiscoveryPipeline().run(brief, evidence)
    assert run.selected_direction_id is not None
    assert len(run.directions) >= 5
    assert run.insights  # produced real insights from real messages
