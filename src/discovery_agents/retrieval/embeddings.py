"""Text embedders.

`HashingEmbedder` is the keyless default: a deterministic, dependency-free
feature-hashing embedding (signed hashing trick) that maps text to a fixed-dim
L2-normalized vector. It uses `hashlib` (not the salted built-in `hash`) so
vectors are stable across processes and CI runs.

Real embedders (Cohere, OpenAI) are lazy-imported and used only when a key and
the corresponding extra are present.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 2]


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]


@runtime_checkable
class Embedder(Protocol):
    """Maps text to a fixed-dimensional vector."""

    dim: int

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic feature-hashing embedder (no API key, no dependencies).

    The default dimension is large enough that hash collisions stay negligible on
    small corpora, so lexical overlap drives ranking (a 1024-dim space keeps
    short-query retrieval reliable in practice).
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _tokenize(text):
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        return _l2_normalize(vector)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
