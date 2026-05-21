"""Handoff agent — builds a coding-agent-ready implementation spec."""

from __future__ import annotations

from ..agent_base import BaseAgent
from ..models import CodingSpec, ProductDirection


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
