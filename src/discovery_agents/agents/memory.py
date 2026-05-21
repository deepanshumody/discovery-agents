"""Decision-memory agent — produces a lightweight decision log."""

from __future__ import annotations

from typing import List

from ..agent_base import BaseAgent
from ..models import CritiqueScore, ProductDirection


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
