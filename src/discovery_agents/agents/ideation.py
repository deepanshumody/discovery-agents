"""Ideation agent — emits multiple divergent product directions.

With a real provider, the agent asks the model for several evidence-grounded
directions as structured JSON and parses them. With the keyless mock (or if the
model returns invalid output), it falls back to a curated deterministic baseline
so the demo and tests stay reproducible.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from ..agent_base import BaseAgent
from ..models import Insight, ProductBrief, ProductDirection
from ..retrieval.index import EvidenceIndex

# JSON schema advertised to the model when requesting structured output.
DIRECTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "directions": {
            "type": "array",
            "minItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "one_liner": {"type": "string"},
                    "target_user": {"type": "string"},
                    "core_loop": {"type": "string"},
                    "why_now": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "differentiator": {"type": "string"},
                    "implementation_notes": {"type": "array", "items": {"type": "string"}},
                    "risks": {"type": "array", "items": {"type": "string"}},
                    "canvas_column": {"type": "string"},
                },
                "required": ["title", "one_liner", "evidence_ids", "differentiator"],
            },
        }
    },
    "required": ["directions"],
}


class IdeationAgent(BaseAgent):
    """Generates multiple product directions instead of one polished answer."""

    name = "IdeationAgent"

    def run(
        self,
        brief: ProductBrief,
        insights: list[Insight],
        opportunities: list[str],
        index: EvidenceIndex | None = None,
    ) -> list[ProductDirection]:
        baseline = self._deterministic_directions(brief, insights, opportunities)
        # Validate citations against the FULL evidence corpus (via the index), not just
        # the insight-covered subset, so legitimate grounded citations aren't dropped.
        valid_evidence = set(index.ids) if index is not None else set()
        valid_evidence |= {eid for insight in insights for eid in insight.evidence_ids}

        system, user = self._prompt(brief, insights, opportunities, index)
        response = self._chat(
            op="ideation.generate",
            system=system,
            user=user,
            schema=DIRECTIONS_SCHEMA,
            message="generate divergent directions",
        )

        directions = self._parse_directions(response.structured, brief, valid_evidence)
        source = "llm"
        if not directions:
            directions = baseline
            source = "deterministic"

        self.log(
            "Generated divergent product directions",
            direction_count=len(directions),
            source=source,
        )
        return directions

    # -- LLM path ------------------------------------------------------------
    def _prompt(
        self,
        brief: ProductBrief,
        insights: list[Insight],
        opportunities: list[str],
        index: EvidenceIndex | None,
    ) -> tuple[str, str]:
        system = (
            "You are a senior product strategist on an enterprise AI team. Given customer "
            "evidence and strategic opportunities, propose at least five DIVERGENT, buildable "
            "product directions. Each must cite the evidence ids that support it, name a clear "
            "differentiator, and surface real risks. Prefer deciding-what-to-build over polishing "
            "a single artifact."
        )
        insight_lines = "\n".join(
            f"- {i.id} {i.title}: {i.summary} (evidence: {', '.join(i.evidence_ids)})"
            for i in insights
        )
        opp_lines = "\n".join(f"- {o}" for o in opportunities)
        retrieved_block = self._retrieved_block(brief, opportunities, index)
        user = (
            f"Company: {brief.company}\n"
            f"Product: {brief.product}\n"
            f"Target user: {brief.target_user}\n"
            f"Goal: {brief.goal}\n\n"
            f"Insights:\n{insight_lines}\n\n"
            f"Strategic opportunities:\n{opp_lines}\n"
            f"{retrieved_block}\n"
            "Return JSON with a 'directions' array. Only cite evidence ids that appear above."
        )
        return system, user

    def _retrieved_block(
        self, brief: ProductBrief, opportunities: list[str], index: EvidenceIndex | None
    ) -> str:
        if index is None:
            return ""
        query = f"{brief.goal} {' '.join(opportunities)}"
        hits = index.search(query, k=4)
        if not hits:
            return ""
        lines = "\n".join(f"- {h.chunk.id}: {h.chunk.text}" for h in hits)
        return f"\nRetrieved evidence snippets (cite these ids where relevant):\n{lines}\n"

    def _parse_directions(
        self,
        value: dict[str, Any] | None,
        brief: ProductBrief,
        valid_evidence: set[str],
    ) -> list[ProductDirection] | None:
        if not value:
            return None
        raw = value.get("directions")
        if not isinstance(raw, list) or not raw:
            return None

        directions: list[ProductDirection] = []
        for index, item in enumerate(raw, start=1):
            if not isinstance(item, dict) or not item.get("title"):
                continue
            evidence_ids = [
                eid for eid in _as_str_list(item.get("evidence_ids")) if eid in valid_evidence
            ]
            directions.append(
                ProductDirection(
                    id=str(item.get("id") or f"D{index}"),
                    title=str(item["title"]),
                    one_liner=str(item.get("one_liner", "")),
                    target_user=str(item.get("target_user") or brief.target_user),
                    core_loop=str(item.get("core_loop", "")),
                    why_now=str(item.get("why_now", "")),
                    evidence_ids=evidence_ids,
                    differentiator=str(item.get("differentiator", "")),
                    implementation_notes=_as_str_list(item.get("implementation_notes")),
                    risks=_as_str_list(item.get("risks")),
                    canvas_column=str(item.get("canvas_column") or "Ideas"),
                    canvas_row=index,
                )
            )
        return directions or None

    # -- deterministic baseline ---------------------------------------------
    def _deterministic_directions(
        self, brief: ProductBrief, insights: list[Insight], opportunities: list[str]
    ) -> list[ProductDirection]:
        evidence_by_tag: dict[str, list[str]] = defaultdict(list)
        for insight in insights:
            for tag in insight.tags:
                evidence_by_tag[tag].extend(insight.evidence_ids)

        return [
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
                evidence_ids=_pick_evidence(
                    evidence_by_tag, ["blank_state", "alignment", "decision"], fallback=["E1", "E2"]
                ),
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
                evidence_ids=_pick_evidence(
                    evidence_by_tag,
                    ["critique", "differentiation", "implementation"],
                    fallback=["E4", "E6"],
                ),
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
                evidence_ids=_pick_evidence(
                    evidence_by_tag,
                    ["brand_fit", "context", "learning_loop"],
                    fallback=["E3", "E5"],
                ),
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
                evidence_ids=_pick_evidence(
                    evidence_by_tag, ["handoff", "implementation"], fallback=["E4"]
                ),
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
                evidence_ids=_pick_evidence(
                    evidence_by_tag, ["selection", "feedback", "learning_loop"], fallback=["E5"]
                ),
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


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if isinstance(v, (str, int, float))]


def _pick_evidence(
    evidence_by_tag: dict[str, list[str]], tags: Iterable[str], fallback: list[str]
) -> list[str]:
    picked: list[str] = []
    for tag in tags:
        picked.extend(evidence_by_tag.get(tag, []))
    deduped: list[str] = []
    for item in picked or fallback:
        if item not in deduped:
            deduped.append(item)
    return deduped[:3]
