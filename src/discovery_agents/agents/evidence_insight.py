"""Evidence-clustering agent."""

from __future__ import annotations

import itertools
from collections import Counter
from typing import List

from ..agent_base import BaseAgent
from ..models import EvidenceItem, Insight
from ._utils import tokenize


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
