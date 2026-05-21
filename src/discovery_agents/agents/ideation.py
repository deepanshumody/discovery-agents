"""Ideation agent — emits multiple divergent product directions."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from ..agent_base import BaseAgent
from ..models import Insight, ProductBrief, ProductDirection


class IdeationAgent(BaseAgent):
    """Generates multiple product directions instead of one polished answer."""

    name = "IdeationAgent"

    def run(
        self,
        brief: ProductBrief,
        insights: List[Insight],
        opportunities: List[str],
    ) -> List[ProductDirection]:
        evidence_by_tag: Dict[str, List[str]] = defaultdict(list)
        for insight in insights:
            for tag in insight.tags:
                evidence_by_tag[tag].extend(insight.evidence_ids)

        directions = [
            ProductDirection(
                id="D1",
                title="Opportunity Map Canvas",
                one_liner="A visual map of possible product bets, grouped by customer pain, risk, and payoff.",
                target_user=brief.target_user,
                core_loop=(
                    "User enters product context → agents cluster evidence → canvas shows several "
                    "opportunity zones → team compares and selects a direction."
                ),
                why_now="Teams are overwhelmed by text-heavy ideation and need faster visual alignment.",
                evidence_ids=_pick_evidence(evidence_by_tag, ["blank_state", "alignment", "decision"], fallback=["E1", "E2"]),
                differentiator="Optimizes for deciding what to build, not generating a single polished mockup.",
                implementation_notes=[
                    "Represent opportunities as cards with evidence, score, risk, and next step.",
                    "Add clustering by user segment, severity, and strategic theme.",
                    "Persist accepted/rejected directions as decision memory.",
                ],
                risks=[
                    "Could become noisy if too many directions are shown.",
                    "Requires thoughtful visual hierarchy to prevent decision fatigue.",
                ],
                canvas_column="Explore",
                canvas_row=1,
            ),
            ProductDirection(
                id="D2",
                title="Multi-Agent Critique Room",
                one_liner="A panel of agents critiques each idea from customer, product, design, and engineering perspectives.",
                target_user=brief.target_user,
                core_loop=(
                    "Team selects a candidate idea → agents critique it from multiple lenses → system "
                    "summarizes risks, missing context, and next experiments."
                ),
                why_now="Prompt-to-prototype tools create artifacts, but teams still need tradeoff reasoning.",
                evidence_ids=_pick_evidence(evidence_by_tag, ["critique", "differentiation", "implementation"], fallback=["E4", "E6"]),
                differentiator="Makes product reasoning visible before design or engineering commitment.",
                implementation_notes=[
                    "Use role-specific prompts and structured scorecards.",
                    "Require every critique to cite evidence or mark uncertainty.",
                    "Track recurring objections as future memory features.",
                ],
                risks=[
                    "LLM critique can sound confident without evidence.",
                    "Needs calibrated scoring to avoid fake precision.",
                ],
                canvas_column="Critique",
                canvas_row=1,
            ),
            ProductDirection(
                id="D3",
                title="Product Memory Graph",
                one_liner="A memory layer that remembers customer evidence, team taste, rejected ideas, and shipped outcomes.",
                target_user=brief.target_user,
                core_loop=(
                    "Every ideation session writes decisions → memory graph updates company taste and "
                    "constraints → future generations become more context-aware."
                ),
                why_now="Generic AI design tools often miss company-specific judgment and previous decisions.",
                evidence_ids=_pick_evidence(evidence_by_tag, ["brand_fit", "context", "learning_loop"], fallback=["E3", "E5"]),
                differentiator="Remembers product judgment, not just a design system.",
                implementation_notes=[
                    "Store entities: customer pain, persona, decision, feature, rejected idea, shipped result.",
                    "Use retrieval plus graph traversal for context injection.",
                    "Expose memory citations in every generated direction.",
                ],
                risks=[
                    "Memory can reinforce stale assumptions if not refreshed.",
                    "Requires clear controls for privacy and data boundaries.",
                ],
                canvas_column="Memory",
                canvas_row=1,
            ),
            ProductDirection(
                id="D4",
                title="Coding-Agent Handoff Packet",
                one_liner="Turns selected product directions into structured implementation specs with acceptance criteria.",
                target_user="product engineers and coding agents",
                core_loop=(
                    "Team selects direction → handoff agent generates requirements, data contracts, edge cases, "
                    "analytics events, and acceptance criteria → engineer or coding agent builds."
                ),
                why_now="AI prototypes are only valuable if they convert into buildable product work.",
                evidence_ids=_pick_evidence(evidence_by_tag, ["handoff", "implementation"], fallback=["E4"]),
                differentiator="Bridges product discovery and implementation instead of stopping at mockups.",
                implementation_notes=[
                    "Generate API/data assumptions and UI states.",
                    "Include analytics plan and eval criteria.",
                    "Create implementation-ready Markdown, JSON, and ticket format outputs.",
                ],
                risks=[
                    "Spec quality depends on selected direction clarity.",
                    "May need integration with project management or code tools.",
                ],
                canvas_column="Handoff",
                canvas_row=1,
            ),
            ProductDirection(
                id="D5",
                title="Feedback-to-Eval Loop",
                one_liner="Uses selections, edits, shares, and build-through to evaluate which ideas are actually useful.",
                target_user="founders and product teams",
                core_loop=(
                    "User compares directions → system tracks selection/edit/share/build-through signals → "
                    "eval agent updates future scoring and generation strategy."
                ),
                why_now="Creative AI quality is hard to measure unless user behavior becomes part of evaluation.",
                evidence_ids=_pick_evidence(evidence_by_tag, ["selection", "feedback", "learning_loop"], fallback=["E5"]),
                differentiator="Measures product usefulness, not only output polish.",
                implementation_notes=[
                    "Instrument idea_selected, idea_edited, idea_shared, spec_exported, feature_built events.",
                    "Blend automated checks with behavioral signals.",
                    "Keep a per-workspace eval dashboard.",
                ],
                risks=[
                    "Behavioral metrics can be sparse early.",
                    "Needs guardrails to avoid optimizing for superficial engagement.",
                ],
                canvas_column="Evaluate",
                canvas_row=1,
            ),
        ]
        self.log("Generated divergent product directions", direction_count=len(directions))
        return directions


def _pick_evidence(evidence_by_tag: Dict[str, List[str]], tags: Iterable[str], fallback: List[str]) -> List[str]:
    picked: List[str] = []
    for tag in tags:
        picked.extend(evidence_by_tag.get(tag, []))
    deduped: List[str] = []
    for item in picked or fallback:
        if item not in deduped:
            deduped.append(item)
    return deduped[:3]
