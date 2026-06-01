"""RAG tool: search the customer-evidence corpus and return cited snippets."""

from __future__ import annotations

from typing import Any

from ..retrieval.index import EvidenceIndex
from .base import ToolResult


class EvidenceSearchTool:
    """Retrieve evidence passages relevant to a query (with citation ids)."""

    name = "evidence_search"
    description = (
        "Search the customer/product evidence corpus for passages relevant to a "
        "query. Returns the most similar snippets with their citation ids and scores."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search the evidence for."},
            "k": {"type": "integer", "description": "Number of snippets to return.", "default": 3},
        },
        "required": ["query"],
    }

    def __init__(self, index: EvidenceIndex) -> None:
        self.index = index

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, error="query is required")
        try:
            k = int(arguments.get("k", 3))
        except (TypeError, ValueError):
            k = 3
        hits = self.index.search(query, k=max(1, k))
        data = [
            {
                "id": hit.chunk.id,
                "text": hit.chunk.text,
                "score": round(hit.score, 4),
                "source": hit.chunk.metadata.get("source"),
            }
            for hit in hits
        ]
        return ToolResult(ok=True, data=data, citations=[hit.chunk.id for hit in hits])
