"""Decision-memory agent — produces a lightweight decision log.

The model writes the rationale for the selection; with the keyless mock (or invalid
output) it falls back to the deterministic log.
"""

from __future__ import annotations

from typing import Any

from ..agent_base import BaseAgent
from ..models import CritiqueScore, ProductDirection

DECISION_LOG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"decision_log": {"type": "array", "items": {"type": "string"}}},
    "required": ["decision_log"],
}


class DecisionMemoryAgent(BaseAgent):
    """Writes a lightweight decision log that could become product memory."""

    name = "DecisionMemoryAgent"

    def run(
        self, selected_id: str, directions: list[ProductDirection], critiques: list[CritiqueScore]
    ) -> list[str]:
        direction_by_id = {d.id: d for d in directions}
        critique_by_id = {c.direction_id: c for c in critiques}
        selected = direction_by_id[selected_id]
        selected_critique = critique_by_id[selected_id]
        rejected = [d for d in directions if d.id != selected_id]

        system, user = self._prompt(selected, selected_critique, rejected)
        response = self._chat(
            op="memory.log",
            system=system,
            user=user,
            schema=DECISION_LOG_SCHEMA,
            message="decision rationale",
        )
        log = self._parse(response.structured)
        source = "llm"
        if log is None:
            log = self._deterministic(selected, selected_critique, rejected)
            source = "deterministic"
        self.log(
            "Wrote decision memory", selected=selected.title, rejected_count=len(rejected), source=source
        )
        return log

    def _prompt(
        self,
        selected: ProductDirection,
        critique: CritiqueScore,
        rejected: list[ProductDirection],
    ) -> tuple[str, str]:
        system = (
            "You record concise decision memory. In 2-4 short lines, explain why the selected "
            "direction was chosen over the alternatives and what future runs should remember. "
            "Return JSON."
        )
        user = (
            f"Selected: {selected.title} (score {critique.weighted_score}) — {selected.differentiator}\n"
            f"Rejected: {', '.join(d.title for d in rejected)}\n\n"
            "Return JSON {\"decision_log\": [\"line\", ...]}."
        )
        return system, user

    def _parse(self, value: dict[str, Any] | None) -> list[str] | None:
        if not value:
            return None
        raw = value.get("decision_log")
        if not isinstance(raw, list):
            return None
        out = [str(x).strip() for x in raw if isinstance(x, (str, int, float)) and str(x).strip()]
        return out or None

    def _deterministic(
        self,
        selected: ProductDirection,
        selected_critique: CritiqueScore,
        rejected: list[ProductDirection],
    ) -> list[str]:
        return [
            f"Selected '{selected.title}' because it scored {selected_critique.weighted_score} and directly supports: {selected.differentiator}",
            "Rejected alternatives for now: " + ", ".join(d.title for d in rejected),
            "Memory update: future generations should emphasize evidence-backed exploration, visible tradeoffs, and implementation handoff.",
        ]
