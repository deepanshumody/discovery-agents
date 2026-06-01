"""Retrieval / RAG: embeddings, a vector store, and an evidence index."""

from __future__ import annotations

from .embeddings import Embedder, HashingEmbedder, get_embedder
from .index import EvidenceIndex
from .vector_store import (
    Chunk,
    InMemoryVectorStore,
    PineconeVectorStore,
    ScoredChunk,
    VectorStore,
    cosine_similarity,
)

__all__ = [
    "Chunk",
    "Embedder",
    "EvidenceIndex",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "PineconeVectorStore",
    "ScoredChunk",
    "VectorStore",
    "cosine_similarity",
    "get_embedder",
]
