"""A web-search tool with a deterministic, keyless default backend.

The default backend returns deterministic synthetic results (so demos and tests
are reproducible without network access). Inject a real `backend` callable
(e.g. wrapping a search API) to make it live.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from .base import ToolResult

SearchBackend = Callable[[str, int], list[dict[str, Any]]]


def _deterministic_backend(query: str, k: int) -> list[dict[str, Any]]:
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()
    results: list[dict[str, Any]] = []
    for i in range(k):
        slug = f"{digest[i * 4 : i * 4 + 8]}"
        results.append(
            {
                "title": f"{query.strip().title()} — reference {i + 1}",
                "url": f"https://example.com/{slug}",
                "snippet": f"Synthetic result {i + 1} for '{query.strip()}' (deterministic mock).",
            }
        )
    return results


class WebSearchTool:
    """Search the web for context relevant to a query (mockable external API)."""

    name = "web_search"
    description = "Search the web for context relevant to a query. Returns titles, urls, snippets."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "default": 3},
        },
        "required": ["query"],
    }

    def __init__(self, backend: SearchBackend | None = None) -> None:
        self.backend: SearchBackend = backend or _deterministic_backend

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, error="query is required")
        try:
            k = max(1, int(arguments.get("k", 3)))
        except (TypeError, ValueError):
            k = 3
        results = self.backend(query, k)
        return ToolResult(ok=True, data=results, citations=[r["url"] for r in results])
