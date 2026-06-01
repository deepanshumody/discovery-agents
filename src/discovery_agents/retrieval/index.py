"""Evidence index: embed customer evidence and retrieve cited passages."""

from __future__ import annotations

from ..models import EvidenceItem
from .embeddings import Embedder, HashingEmbedder
from .vector_store import Chunk, InMemoryVectorStore, ScoredChunk, VectorStore


class EvidenceIndex:
    """A small RAG index over the evidence corpus.

    Each evidence item becomes one chunk whose id is the evidence id, so search
    results carry citation ids directly.
    """

    def __init__(self, embedder: Embedder | None = None, store: VectorStore | None = None) -> None:
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.store: VectorStore = store or InMemoryVectorStore()

    @classmethod
    def from_evidence(
        cls,
        evidence: list[EvidenceItem],
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ) -> EvidenceIndex:
        index = cls(embedder, store)
        index.add_evidence(evidence)
        return index

    def add_evidence(self, evidence: list[EvidenceItem]) -> None:
        chunks = [
            Chunk(
                id=item.id,
                text=item.text,
                metadata={
                    "source": item.source,
                    "segment": item.user_segment,
                    "severity": item.severity,
                    "tags": list(item.tags),
                },
            )
            for item in evidence
        ]
        vectors = self.embedder.embed_batch([c.text for c in chunks])
        for chunk, vector in zip(chunks, vectors):
            chunk.vector = vector
        self.store.upsert(chunks)

    def search(self, query: str, k: int = 3) -> list[ScoredChunk]:
        query_vector = self.embedder.embed(query)
        return self.store.search(query_vector, k)

    def citations(self, query: str, k: int = 3) -> list[str]:
        return [hit.chunk.id for hit in self.search(query, k)]
