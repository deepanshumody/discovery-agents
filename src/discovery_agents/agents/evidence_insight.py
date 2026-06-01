"""Evidence-clustering agent.

Clusters evidence into themes by tag, then augments each theme with
retrieval: a RAG query over the evidence index surfaces semantically related
passages (which may live under different tags), and those citation ids are merged
in. Retrieval is deterministic (HashingEmbedder), so keyless runs stay stable.
"""

from __future__ import annotations

import itertools
from collections import Counter

from ..agent_base import BaseAgent
from ..models import EvidenceItem, Insight
from ..observability.trace import TraceSpan
from ..retrieval.index import EvidenceIndex
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

    # Minimum cosine score for a retrieved passage to be merged as a citation.
    RETRIEVAL_THRESHOLD = 0.05

    def run(
        self, evidence: list[EvidenceItem], index: EvidenceIndex | None = None
    ) -> list[Insight]:
        index = index or EvidenceIndex.from_evidence(evidence)
        insights: list[Insight] = []
        for idx, (title, tags) in enumerate(self._themes_for(evidence), start=1):
            items = [item for item in evidence if set(item.tags) & set(tags)]
            if not items:
                continue
            tag_ids = [i.id for i in items]
            retrieved_ids = self._retrieve(index, title, tags, exclude=set(tag_ids))
            evidence_ids = tag_ids + retrieved_ids
            summary = self._summarize_cluster(title, tags, items, retrieved_ids)
            confidence = round(
                min(0.95, 0.50 + 0.07 * len(items) + 0.02 * sum(i.severity for i in items)), 2
            )
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

    def _themes_for(self, evidence: list[EvidenceItem]) -> list[tuple[str, list[str]]]:
        """Use the curated THEMES when they cover the evidence; otherwise derive themes
        from the evidence's own most-frequent tags (so arbitrary real data still clusters)."""
        covered = sum(
            1 for item in evidence if any(set(item.tags) & set(tags) for _, tags in self.THEMES)
        )
        if evidence and covered >= 0.5 * len(evidence):
            return self.THEMES
        counts: Counter[str] = Counter(tag for item in evidence for tag in item.tags)
        return [(tag.replace("_", " ").title(), [tag]) for tag, _ in counts.most_common(6)]

    def _retrieve(
        self, index: EvidenceIndex, title: str, tags: list[str], exclude: set[str]
    ) -> list[str]:
        query = f"{title} {' '.join(tags)}"
        hits = index.search(query, k=3)
        retrieved = [
            hit.chunk.id
            for hit in hits
            if hit.score >= self.RETRIEVAL_THRESHOLD and hit.chunk.id not in exclude
        ]
        self.trace.record(
            TraceSpan(
                agent=self.name,
                op="retrieval",
                message=f"evidence search for theme '{title}'",
                payload={
                    "query": query,
                    "hits": [(h.chunk.id, round(h.score, 4)) for h in hits],
                    "merged": retrieved,
                },
            )
        )
        return retrieved

    def _summarize_cluster(
        self, title: str, tags: list[str], items: list[EvidenceItem], retrieved_ids: list[str]
    ) -> str:
        segment_counts = Counter(i.user_segment for i in items)
        top_segments = ", ".join(s for s, _ in segment_counts.most_common(3))
        common_words = Counter(itertools.chain.from_iterable(tokenize(i.text) for i in items))
        keywords = ", ".join(w for w, _ in common_words.most_common(6))
        tag_text = ", ".join(tags)
        summary = (
            f"This theme is supported by {len(items)} evidence item(s) across {top_segments}. "
            f"Relevant tags: {tag_text}. Recurring language centers on: {keywords}."
        )
        if retrieved_ids:
            summary += f" Retrieval also surfaced related evidence: {', '.join(retrieved_ids)}."
        return summary
