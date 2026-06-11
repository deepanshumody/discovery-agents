"""Evidence-clustering agent.

Clusters evidence into themes by tag, then augments each theme with
retrieval: a RAG query over the evidence index surfaces semantically related
passages (which may live under different tags), and those citation ids are merged
in. Retrieval is deterministic (HashingEmbedder), so keyless runs stay stable.

The clustering and retrieval are deterministic; only the human-readable theme
summary is written by the model when a real provider is configured. A single
batched call summarizes all clusters at once and falls back, per cluster, to the
deterministic summary under the keyless mock (or on invalid output).
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..agent_base import BaseAgent
from ..models import EvidenceItem, Insight
from ..observability.trace import TraceSpan
from ..retrieval.index import EvidenceIndex
from ._utils import tokenize

SUMMARIES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summaries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["id", "summary"],
            },
        }
    },
    "required": ["summaries"],
}


@dataclass
class _Cluster:
    """A theme plus the evidence that supports it (built deterministically)."""

    id: str
    title: str
    tags: list[str]
    items: list[EvidenceItem]
    retrieved_ids: list[str]
    confidence: float

    @property
    def evidence_ids(self) -> list[str]:
        return [i.id for i in self.items] + self.retrieved_ids


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
        clusters = self._build_clusters(evidence, index)
        summaries = self._summaries(clusters)
        insights = [
            Insight(
                id=cluster.id,
                title=cluster.title,
                summary=summaries[cluster.id],
                evidence_ids=cluster.evidence_ids,
                confidence=cluster.confidence,
                tags=cluster.tags,
            )
            for cluster in clusters
        ]
        self.log("Clustered evidence into product-discovery themes", insight_count=len(insights))
        return insights

    def _build_clusters(
        self, evidence: list[EvidenceItem], index: EvidenceIndex
    ) -> list[_Cluster]:
        clusters: list[_Cluster] = []
        for title, tags in self._themes_for(evidence):
            items = [item for item in evidence if set(item.tags) & set(tags)]
            if not items:
                continue
            tag_ids = {i.id for i in items}
            retrieved_ids = self._retrieve(index, title, tags, exclude=tag_ids)
            confidence = round(
                min(0.95, 0.50 + 0.07 * len(items) + 0.02 * sum(i.severity for i in items)), 2
            )
            clusters.append(
                _Cluster(
                    id=f"I{len(clusters) + 1}",
                    title=title,
                    tags=tags,
                    items=items,
                    retrieved_ids=retrieved_ids,
                    confidence=confidence,
                )
            )
        return clusters

    def _summaries(self, clusters: list[_Cluster]) -> dict[str, str]:
        """Return {cluster_id: summary}, LLM-written when available, deterministic otherwise."""
        deterministic = {c.id: self._summarize_cluster(c) for c in clusters}
        if not clusters:
            return deterministic
        system, user = self._prompt(clusters)
        response = self._chat(
            op="insight.summarize",
            system=system,
            user=user,
            schema=SUMMARIES_SCHEMA,
            message="summarize evidence clusters",
        )
        llm = self._parse(response.structured)
        # Per-cluster fallback: keep the deterministic summary unless the model supplied one.
        return {c.id: (llm.get(c.id) or deterministic[c.id]) for c in clusters}

    def _prompt(self, clusters: list[_Cluster]) -> tuple[str, str]:
        system = (
            "You analyze clustered customer evidence. For each theme, write a concise, specific "
            "insight summary (1-2 sentences) grounded in the supplied evidence text. Return JSON "
            "with a 'summaries' array of {id, summary}, one per theme id."
        )
        blocks = []
        for c in clusters:
            quotes = "\n".join(f"    - ({i.user_segment}) {i.text}" for i in c.items)
            blocks.append(f"  {c.id} — {c.title} [tags: {', '.join(c.tags)}]\n{quotes}")
        user = "Themes:\n" + "\n".join(blocks) + "\n\nReturn JSON {\"summaries\": [{id, summary}]}."
        return system, user

    def _parse(self, value: dict[str, Any] | None) -> dict[str, str]:
        if not value:
            return {}
        raw = value.get("summaries")
        if not isinstance(raw, list):
            return {}
        out: dict[str, str] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("id") or "").strip()
            summary = str(item.get("summary") or "").strip()
            if cid and summary:
                out[cid] = summary
        return out

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

    def _summarize_cluster(self, cluster: _Cluster) -> str:
        items = cluster.items
        segment_counts = Counter(i.user_segment for i in items)
        top_segments = ", ".join(s for s, _ in segment_counts.most_common(3))
        common_words = Counter(itertools.chain.from_iterable(tokenize(i.text) for i in items))
        keywords = ", ".join(w for w, _ in common_words.most_common(6))
        tag_text = ", ".join(cluster.tags)
        summary = (
            f"This theme is supported by {len(items)} evidence item(s) across {top_segments}. "
            f"Relevant tags: {tag_text}. Recurring language centers on: {keywords}."
        )
        if cluster.retrieved_ids:
            summary += f" Retrieval also surfaced related evidence: {', '.join(cluster.retrieved_ids)}."
        return summary
