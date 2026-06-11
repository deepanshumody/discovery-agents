"""Handoff agent — builds a coding-agent-ready implementation spec.

The structural parts of the spec (the data contract and analytics events) are
fixed by the system's own schema and stay deterministic. The prose — user story,
requirements, acceptance criteria, open questions — is written by the model when a
real provider is configured, and falls back to the deterministic template under the
keyless mock (or on invalid output).
"""

from __future__ import annotations

from typing import Any

from ..agent_base import BaseAgent
from ..models import CodingSpec, ProductDirection

HANDOFF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "user_story": {"type": "string"},
        "functional_requirements": {"type": "array", "items": {"type": "string"}},
        "non_functional_requirements": {"type": "array", "items": {"type": "string"}},
        "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["user_story", "functional_requirements", "acceptance_criteria"],
}


class HandoffAgent(BaseAgent):
    """Creates a coding-agent-ready implementation spec."""

    name = "HandoffAgent"

    def run(self, selected: ProductDirection) -> CodingSpec:
        spec = self._deterministic(selected)
        system, user = self._prompt(selected)
        response = self._chat(
            op="handoff.spec",
            system=system,
            user=user,
            schema=HANDOFF_SCHEMA,
            message="write handoff spec",
        )
        source = "deterministic"
        if self._overlay(spec, response.structured):  # mutate prose fields in place when valid
            source = "llm"
        self.log("Generated coding-agent handoff", direction_id=selected.id, source=source)
        return spec

    def _prompt(self, selected: ProductDirection) -> tuple[str, str]:
        system = (
            "You are a staff engineer writing an implementation-ready handoff for a coding agent. "
            "Given a selected product direction, write a crisp user_story, concrete "
            "functional_requirements and non_functional_requirements, testable acceptance_criteria, "
            "and the most important open_questions. Be specific and buildable. Return JSON."
        )
        user = (
            f"Direction: {selected.title}\n"
            f"One-liner: {selected.one_liner}\n"
            f"Target user: {selected.target_user}\n"
            f"Core loop: {selected.core_loop}\n"
            f"Differentiator: {selected.differentiator}\n"
            f"Implementation notes: {'; '.join(selected.implementation_notes)}\n"
            f"Risks: {'; '.join(selected.risks)}\n\n"
            "Return JSON with keys user_story, functional_requirements, "
            "non_functional_requirements, acceptance_criteria, open_questions."
        )
        return system, user

    def _overlay(self, spec: CodingSpec, value: dict[str, Any] | None) -> bool:
        """Overlay model-written prose onto the deterministic spec. Returns True if applied."""
        if not value:
            return False
        story = value.get("user_story")
        if not isinstance(story, str) or not story.strip():
            return False
        spec.user_story = story.strip()
        for field_name in (
            "functional_requirements",
            "non_functional_requirements",
            "acceptance_criteria",
            "open_questions",
        ):
            cleaned = _string_list(value.get(field_name))
            if cleaned:
                setattr(spec, field_name, cleaned)
        return True

    def _deterministic(self, selected: ProductDirection) -> CodingSpec:
        return CodingSpec(
            direction_id=selected.id,
            feature_name=selected.title,
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
                "ProductBrief": [
                    "company",
                    "product",
                    "target_user",
                    "goal",
                    "constraints",
                    "strategic_themes",
                ],
                "EvidenceItem": ["id", "source", "text", "user_segment", "severity", "tags"],
                "ProductDirection": [
                    "id",
                    "title",
                    "one_liner",
                    "core_loop",
                    "evidence_ids",
                    "risks",
                ],
                "CritiqueScore": [
                    "customer_alignment",
                    "novelty",
                    "feasibility",
                    "strategic_fit",
                    "clarity",
                    "risk_level",
                ],
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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(x).strip() for x in value if isinstance(x, (str, int, float)) and str(x).strip()]
