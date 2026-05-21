"""Shared text helpers used by multiple agents."""

from __future__ import annotations

import re
from typing import List


STOPWORDS = {
    "the", "and", "or", "a", "an", "to", "of", "in", "for", "with", "on", "that", "this",
    "it", "we", "our", "is", "are", "as", "by", "be", "from", "into", "but", "not", "just",
}


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z_\-]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]
