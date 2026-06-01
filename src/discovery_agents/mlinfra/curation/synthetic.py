"""Deterministic synthetic corpus generator.

Lets the platform demonstrate scale without shipping real data: given a seed it
produces the same documents every time, so curation and training stay reproducible.
"""

from __future__ import annotations

import random

_WORDS = [
    "evidence",
    "handoff",
    "alignment",
    "decision",
    "agent",
    "retrieval",
    "pipeline",
    "enterprise",
    "customer",
    "workflow",
    "model",
    "training",
    "tensor",
    "throughput",
    "latency",
    "checkpoint",
    "embedding",
    "cluster",
    "dataset",
    "gradient",
    "distributed",
    "benchmark",
    "reliability",
    "observability",
    "guardrail",
    "evaluation",
    "inference",
    "scalable",
    "stakeholder",
    "requirement",
    "criteria",
    "analytics",
    "context",
    "memory",
]


def generate_corpus(n: int, seed: int = 7, min_len: int = 6, max_len: int = 16) -> list[str]:
    """Generate ``n`` deterministic pseudo-documents from a fixed word bank."""
    rng = random.Random(seed)
    docs: list[str] = []
    for _ in range(n):
        length = rng.randint(min_len, max_len)
        docs.append(" ".join(rng.choice(_WORDS) for _ in range(length)))
    return docs
