"""Tests for embeddings, the vector store, and the evidence index."""

from __future__ import annotations

from discovery_agents.retrieval import (
    Chunk,
    EvidenceIndex,
    HashingEmbedder,
    InMemoryVectorStore,
    cosine_similarity,
)
from discovery_agents.sample_data import SAMPLE_EVIDENCE


def test_hashing_embedder_is_deterministic_and_fixed_dim() -> None:
    emb = HashingEmbedder(dim=128)
    a = emb.embed("handoff requirements and acceptance criteria")
    b = emb.embed("handoff requirements and acceptance criteria")
    assert a == b
    assert len(a) == 128


def test_identical_text_is_maximally_similar() -> None:
    emb = HashingEmbedder()
    v = emb.embed("evidence about alignment and decisions")
    assert cosine_similarity(v, v) == 1.0 or abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_related_text_more_similar_than_unrelated() -> None:
    emb = HashingEmbedder()
    base = emb.embed("the team debates alternatives and needs to align on a decision")
    related = emb.embed("we must align stakeholders and decide what to pilot")
    unrelated = emb.embed("acceptance criteria data contracts and analytics events")
    assert cosine_similarity(base, related) > cosine_similarity(base, unrelated)


def test_in_memory_store_returns_top_k_descending() -> None:
    emb = HashingEmbedder()
    store = InMemoryVectorStore()
    chunks = [
        Chunk(id="a", text="alignment and decisions", vector=emb.embed("alignment and decisions")),
        Chunk(id="b", text="handoff and engineering", vector=emb.embed("handoff and engineering")),
        Chunk(id="c", text="feedback learning loop", vector=emb.embed("feedback learning loop")),
    ]
    store.upsert(chunks)
    results = store.search(emb.embed("align on a decision"), k=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert results[0].chunk.id == "a"


def test_evidence_index_search_returns_citation_ids() -> None:
    index = EvidenceIndex.from_evidence(SAMPLE_EVIDENCE)
    citations = index.citations("handoff requirements and acceptance criteria", k=2)
    assert "E4" in citations  # the handoff-tagged evidence item


def test_upsert_is_idempotent_by_id() -> None:
    emb = HashingEmbedder()
    store = InMemoryVectorStore()
    store.upsert([Chunk(id="x", text="one", vector=emb.embed("one"))])
    store.upsert([Chunk(id="x", text="two", vector=emb.embed("two"))])
    results = store.search(emb.embed("two"), k=5)
    assert len(results) == 1
    assert results[0].chunk.text == "two"
