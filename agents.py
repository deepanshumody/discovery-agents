"""Agent implementations for product-discovery workflows.

Each agent is intentionally deterministic for the demo. In production, the same
interfaces could call an LLM, retrieval system, design model, or product analytics
store. Determinism makes this project easy to run in interviews and code reviews.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List

from .agent_base import AgentTrace, BaseAgent
from .models import (
    CanvasCard,
    CodingSpec,
    CritiqueScore,
    EvalResult,
    EvidenceItem,
    Insight,
    ProductBrief,
    ProductDirection,
)


STOPWORDS = {
    "the", "and", "or", "a", "an", "to", "of", "in", "for", "with", "on", "that", "this",
    "it", "we", "our", "is", "are", "as", "by", "be", "from", "into", "but", "not", "just",
}


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z_\-]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


class EvidenceInsightAgent(BaseAgent):
    """Clusters customer evidence into actionable product insights."""

    name = "EvidenceInsightAgent"

    THEMES = [
        ("Divergent Product Exploration", ["blank_state", "ideation", "speed"]),
        ("Team Alignment And Decision-Making", ["alignment", "stakeholders", "decision"]),
        ("Context And Product Taste", ["visual_quality", "brand_fit", "context"]),
        ("Implementation Handoff", ["handoff", "implementation", "engineering"]),
        ("Feedback-To-Eval Loop", ["selection", "feedback", "learning_loop"]),
        ("Tradeoff Critique", ["differentiation", "generic_outputs", "critique"]),
    ]

    def run(self, evidence: List[EvidenceItem]) -> List[Insight]:
        insights: List[Insight] = []
        for idx, (title, tags) in enumerate(self.THEMES, start=1):
            items = [item for item in evidence if set(item.tags) & set(tags)]
            if not items:
                continue
            evidence_ids = [i.id for i in items]
            summary = self._summarize_cluster(title, tags, items)
            confidence = round(min(0.95, 0.50 + 0.07 * len(items) + 0.02 * sum(i.severity for i in items)), 2)
            insights.append(
                Insight(
                    id=f"I{idx}",
                    title=title,
                    summary=summary,
                    evidence_ids=evidence_ids,
                    confidence=confidence,
                    tags=tags,
                )
            )

        self.log("Clustered evidence into product-discovery themes", insight_count=len(insights))
        return insights

    def _summarize_cluster(self, title: str, tags: List[str], items: List[EvidenceItem]) -> str:
        segment_counts = Counter(i.user_segment for i in items)
        top_segments = ", ".join(s for s, _ in segment_counts.most_common(3))
        common_words = Counter(itertools.chain.from_iterable(tokenize(i.text) for i in items))
        keywords = ", ".join(w for w, _ in common_words.most_common(6))
        tag_text = ", ".join(tags)
        return (
            f"This theme is supported by {len(items)} evidence item(s) across {top_segments}. "
            f"Relevant tags: {tag_text}. Recurring language centers on: {keywords}."
        )


class ProductStrategyAgent(BaseAgent):
    """Converts insights into strategic opportunity areas."""

    name = "ProductStrategyAgent"

    def run(self, brief: ProductBrief, insights: List[Insight]) -> List[str]:
        opportunities = []
        if any("blank_state" in i.tags or "ideation" in i.tags for i in insights):
            opportunities.append("Reduce blank-state anxiety by showing several concrete paths immediately.")
        if any("alignment" in i.tags or "decision" in i.tags for i in insights):
            opportunities.append("Turn abstract debate into a visual decision workflow with explicit tradeoffs.")
        if any("handoff" in i.tags or "implementation" in i.tags for i in insights):
            opportunities.append("Convert selected ideas into implementation-ready specs for engineers or coding agents.")
        if any("brand_fit" in i.tags or "context" in i.tags for i in insights):
            opportunities.append("Use product memory so generated directions reflect brand, taste, and prior decisions.")
        if any("feedback" in i.tags or "learning_loop" in i.tags for i in insights):
            opportunities.append("Learn from what users select, edit, reject, and share.")

        # Always tie back to the stated goal.
        opportunities.insert(0, f"Primary goal: {brief.goal}")
        self.log("Generated strategic opportunity map", opportunities=len(opportunities))
        return opportunities


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


class CritiqueAgent(BaseAgent):
    """Scores directions with transparent, product-specific criteria."""

    name = "CritiqueAgent"

    def run(self, directions: List[ProductDirection], evidence: List[EvidenceItem]) -> List[CritiqueScore]:
        evidence_ids = {item.id for item in evidence}
        critiques: List[CritiqueScore] = []
        for direction in directions:
            coverage = len(set(direction.evidence_ids) & evidence_ids)
            customer_alignment = min(5, 2 + coverage)
            novelty = 5 if "not" in direction.differentiator.lower() or "judgment" in direction.differentiator.lower() else 4
            feasibility = 4 if len(direction.implementation_notes) >= 3 else 3
            strategic_fit = 5 if any(word in direction.one_liner.lower() for word in ["product", "decision", "handoff", "memory", "evaluate"]) else 4
            clarity = 5 if len(direction.one_liner) < 140 else 4
            risk_level = min(5, max(2, len(direction.risks)))
            next_step = _next_step_for(direction.id)
            critiques.append(
                CritiqueScore(
                    direction_id=direction.id,
                    customer_alignment=customer_alignment,
                    novelty=novelty,
                    feasibility=feasibility,
                    strategic_fit=strategic_fit,
                    clarity=clarity,
                    risk_level=risk_level,
                    summary=(
                        f"{direction.title} is strong on customer evidence and strategic fit. "
                        f"Primary risk: {direction.risks[0]}"
                    ),
                    recommended_next_step=next_step,
                )
            )
        self.log("Critiqued candidate directions", critique_count=len(critiques))
        return critiques


class CanvasAgent(BaseAgent):
    """Maps directions into a simple visual canvas structure."""

    name = "CanvasAgent"

    def run(self, directions: List[ProductDirection], critiques: List[CritiqueScore]) -> List[CanvasCard]:
        score_by_id = {c.direction_id: c.weighted_score for c in critiques}
        cards = []
        for index, direction in enumerate(directions, start=1):
            body = (
                f"{direction.one_liner}\n\n"
                f"Core loop: {direction.core_loop}\n\n"
                f"Differentiator: {direction.differentiator}"
            )
            cards.append(
                CanvasCard(
                    id=direction.id,
                    title=direction.title,
                    column=direction.canvas_column,
                    row=direction.canvas_row or index,
                    body=body,
                    evidence=direction.evidence_ids,
                    score=score_by_id.get(direction.id, 0.0),
                    tags=[direction.canvas_column.lower(), "agent-generated"],
                )
            )
        self.log("Mapped directions to canvas cards", card_count=len(cards))
        return cards


class SelectionAgent(BaseAgent):
    """Selects the strongest direction for an implementation-ready handoff."""

    name = "SelectionAgent"

    def run(self, critiques: List[CritiqueScore]) -> str:
        ranked = sorted(critiques, key=lambda c: c.weighted_score, reverse=True)
        selected = ranked[0].direction_id
        self.log("Selected top direction", selected_direction_id=selected, score=ranked[0].weighted_score)
        return selected


class HandoffAgent(BaseAgent):
    """Creates a coding-agent-ready implementation spec."""

    name = "HandoffAgent"

    def run(self, selected: ProductDirection) -> CodingSpec:
        feature_name = selected.title
        spec = CodingSpec(
            direction_id=selected.id,
            feature_name=feature_name,
            user_story=(
                f"As a {selected.target_user}, I want {selected.one_liner.lower()} "
                "so that I can decide what to build with less ambiguity."
            ),
            functional_requirements=[
                "Accept a product brief, constraints, and customer evidence as input.",
                "Generate at least five distinct product directions with evidence citations.",
                "Score each direction on customer alignment, novelty, feasibility, strategic fit, clarity, and risk.",
                "Render directions on a multiplayer-friendly canvas with comparison metadata.",
                "Export the selected direction as Markdown and JSON handoff artifacts.",
            ],
            non_functional_requirements=[
                "Pipeline should be traceable agent-by-agent.",
                "Every generated direction should cite evidence or explicitly mark missing evidence.",
                "The demo should run without external model dependencies.",
                "Production version should allow LLM provider swaps and eval regression tests.",
            ],
            data_contract={
                "ProductBrief": ["company", "product", "target_user", "goal", "constraints", "strategic_themes"],
                "EvidenceItem": ["id", "source", "text", "user_segment", "severity", "tags"],
                "ProductDirection": ["id", "title", "one_liner", "core_loop", "evidence_ids", "risks"],
                "CritiqueScore": ["customer_alignment", "novelty", "feasibility", "strategic_fit", "clarity", "risk_level"],
            },
            acceptance_criteria=[
                "Given sample evidence, the system produces at least five distinct directions.",
                "Each direction has at least one evidence citation.",
                "The top direction is selected by a transparent weighted score.",
                "The generated HTML canvas contains every direction and score.",
                "The Markdown handoff includes requirements, risks, analytics events, and open questions.",
            ],
            analytics_events=[
                "direction_generated",
                "direction_selected",
                "direction_edited",
                "direction_rejected",
                "canvas_shared",
                "handoff_exported",
                "feature_built_from_direction",
            ],
            open_questions=[
                "Which user behavior should matter most: selection, edit depth, sharing, or build-through?",
                "How should product memory be scoped across personal, team, and company workspaces?",
                "What is the right balance between divergent exploration and focused recommendation?",
            ],
        )
        self.log("Generated coding-agent handoff", direction_id=selected.id)
        return spec


class EvalAgent(BaseAgent):
    """Evaluates the overall agent run."""

    name = "EvalAgent"

    def run(
        self,
        brief: ProductBrief,
        evidence: List[EvidenceItem],
        directions: List[ProductDirection],
        critiques: List[CritiqueScore],
        spec: CodingSpec,
    ) -> List[EvalResult]:
        evals = [
            self._direction_count_eval(directions),
            self._evidence_coverage_eval(evidence, directions),
            self._distinctiveness_eval(directions),
            self._constraint_coverage_eval(brief, directions, spec),
            self._handoff_completeness_eval(spec),
            self._risk_visibility_eval(directions, critiques),
        ]
        self.log("Evaluated agent run", eval_count=len(evals))
        return evals

    def _direction_count_eval(self, directions: List[ProductDirection]) -> EvalResult:
        score = min(1.0, len(directions) / 5)
        return EvalResult("direction_diversity_count", score, f"Generated {len(directions)} directions; target is 5+.")

    def _evidence_coverage_eval(self, evidence: List[EvidenceItem], directions: List[ProductDirection]) -> EvalResult:
        cited = set(itertools.chain.from_iterable(d.evidence_ids for d in directions))
        available = {e.id for e in evidence}
        score = len(cited & available) / max(1, len(available))
        return EvalResult("evidence_coverage", round(score, 2), f"Cited {len(cited & available)} of {len(available)} evidence items.")

    def _distinctiveness_eval(self, directions: List[ProductDirection]) -> EvalResult:
        if len(directions) < 2:
            return EvalResult("semantic_distinctiveness", 0.0, "Need at least two directions to compare.")
        token_sets = [set(tokenize(d.title + " " + d.one_liner + " " + d.differentiator)) for d in directions]
        similarities = []
        for a, b in itertools.combinations(token_sets, 2):
            similarities.append(len(a & b) / max(1, len(a | b)))
        avg_similarity = sum(similarities) / len(similarities)
        score = round(max(0.0, 1.0 - avg_similarity), 2)
        return EvalResult("semantic_distinctiveness", score, f"Average pairwise Jaccard similarity is {avg_similarity:.2f}.")

    def _constraint_coverage_eval(self, brief: ProductBrief, directions: List[ProductDirection], spec: CodingSpec) -> EvalResult:
        combined = " ".join(
            [d.one_liner + " " + d.core_loop + " " + d.differentiator for d in directions]
            + spec.functional_requirements
            + spec.non_functional_requirements
        ).lower()
        hits = 0
        for constraint in brief.constraints:
            keywords = [w for w in tokenize(constraint) if len(w) > 4]
            if any(k in combined for k in keywords):
                hits += 1
        score = hits / max(1, len(brief.constraints))
        return EvalResult("constraint_coverage", round(score, 2), f"Covered {hits} of {len(brief.constraints)} constraints.")

    def _handoff_completeness_eval(self, spec: CodingSpec) -> EvalResult:
        sections = [
            spec.user_story,
            spec.functional_requirements,
            spec.non_functional_requirements,
            spec.data_contract,
            spec.acceptance_criteria,
            spec.analytics_events,
            spec.open_questions,
        ]
        complete = sum(1 for section in sections if section)
        score = complete / len(sections)
        return EvalResult("handoff_completeness", round(score, 2), f"Completed {complete} of {len(sections)} handoff sections.")

    def _risk_visibility_eval(self, directions: List[ProductDirection], critiques: List[CritiqueScore]) -> EvalResult:
        directions_with_risks = sum(1 for d in directions if d.risks)
        critiques_with_next_steps = sum(1 for c in critiques if c.recommended_next_step)
        score = (directions_with_risks + critiques_with_next_steps) / max(1, len(directions) + len(critiques))
        return EvalResult("risk_visibility", round(score, 2), "Checks whether risks and next steps are explicit.")


class DecisionMemoryAgent(BaseAgent):
    """Writes a lightweight decision log that could become product memory."""

    name = "DecisionMemoryAgent"

    def run(self, selected_id: str, directions: List[ProductDirection], critiques: List[CritiqueScore]) -> List[str]:
        direction_by_id = {d.id: d for d in directions}
        critique_by_id = {c.direction_id: c for c in critiques}
        selected = direction_by_id[selected_id]
        selected_critique = critique_by_id[selected_id]
        rejected = [d for d in directions if d.id != selected_id]
        log = [
            f"Selected '{selected.title}' because it scored {selected_critique.weighted_score} and directly supports: {selected.differentiator}",
            "Rejected alternatives for now: " + ", ".join(d.title for d in rejected),
            "Memory update: future generations should emphasize evidence-backed exploration, visible tradeoffs, and implementation handoff.",
        ]
        self.log("Wrote decision memory", selected=selected.title, rejected_count=len(rejected))
        return log


def _pick_evidence(evidence_by_tag: Dict[str, List[str]], tags: Iterable[str], fallback: List[str]) -> List[str]:
    picked: List[str] = []
    for tag in tags:
        picked.extend(evidence_by_tag.get(tag, []))
    # Dedupe while preserving order.
    deduped = []
    for item in picked or fallback:
        if item not in deduped:
            deduped.append(item)
    return deduped[:3]


def _next_step_for(direction_id: str) -> str:
    return {
        "D1": "Prototype a canvas with 5–8 opportunity cards and test whether teams can choose a direction faster.",
        "D2": "Run the critique panel against three real customer problems and compare to founder judgment.",
        "D3": "Build a tiny memory graph over evidence, decisions, and rejected ideas; measure context recall quality.",
        "D4": "Generate a handoff packet for one selected direction and ask an engineer/coding agent to implement from it.",
        "D5": "Instrument user behavior and create an eval dashboard for selection, edit depth, and build-through.",
    }.get(direction_id, "Run a small user test and collect qualitative feedback.")
