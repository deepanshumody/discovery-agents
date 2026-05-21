"""Selection agent — picks the highest-scoring direction."""

from __future__ import annotations

from typing import List

from ..agent_base import BaseAgent
from ..models import CritiqueScore


class SelectionAgent(BaseAgent):
    """Selects the strongest direction for an implementation-ready handoff."""

    name = "SelectionAgent"

    def run(self, critiques: List[CritiqueScore]) -> str:
        ranked = sorted(critiques, key=lambda c: c.weighted_score, reverse=True)
        selected = ranked[0].direction_id
        self.log("Selected top direction", selected_direction_id=selected, score=ranked[0].weighted_score)
        return selected
