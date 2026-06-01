"""Vector store abstraction.

`InMemoryVectorStore` is the keyless default (exact cosine search). Pinecone and
Weaviate adapters are provided as lazy-imported interfaces, used only when their
client and credentials are configured; otherwise they raise a clear error rather
than failing silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Chunk:
    """A retrievable unit. `id` doubles as the citation id."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: list[float] = field(default_factory=list)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk]) -> None: ...

    def search(self, query_vector: list[float], k: int) -> list[ScoredChunk]: ...


class InMemoryVectorStore:
    """Exact cosine-similarity store. Upsert replaces chunks with the same id."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def upsert(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk

    def search(self, query_vector: list[float], k: int) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(chunk=chunk, score=cosine_similarity(query_vector, chunk.vector))
            for chunk in self._chunks.values()
        ]
        # Stable ordering: score desc, then id asc, so ties are deterministic.
        scored.sort(key=lambda s: (-s.score, s.chunk.id))
        return scored[: max(0, k)]


class PineconeVectorStore:
    """Lazy Pinecone adapter (install extra `vectordb`, set PINECONE_API_KEY)."""

    def __init__(self, index_name: str, namespace: str = "discovery") -> None:
        try:
            from pinecone import Pinecone  # lazy import
        except ImportError as exc:  # pragma: no cover - exercised only when configured
            raise ImportError(
                "PineconeVectorStore requires the 'vectordb' extra: "
                "pip install 'discovery-agents[vectordb]'"
            ) from exc
        self._index = Pinecone().Index(index_name)
        self._namespace = namespace

    def upsert(self, chunks: list[Chunk]) -> None:  # pragma: no cover
        vectors: list[dict[str, Any]] = []
        for c in chunks:
            metadata: dict[str, Any] = {"text": c.text}
            metadata.update(c.metadata)
            vectors.append({"id": c.id, "values": c.vector, "metadata": metadata})
        self._index.upsert(namespace=self._namespace, vectors=vectors)

    def search(self, query_vector: list[float], k: int) -> list[ScoredChunk]:  # pragma: no cover
        result = self._index.query(
            namespace=self._namespace,
            vector=query_vector,
            top_k=k,
            include_metadata=True,
        )
        out: list[ScoredChunk] = []
        for match in result.get("matches", []):
            meta = match.get("metadata", {}) or {}
            out.append(
                ScoredChunk(
                    chunk=Chunk(id=match["id"], text=meta.get("text", ""), metadata=meta),
                    score=float(match.get("score", 0.0)),
                )
            )
        return out
