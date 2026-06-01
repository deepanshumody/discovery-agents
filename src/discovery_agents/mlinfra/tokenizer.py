"""A small, deterministic word-level tokenizer (no external tokenizer deps).

Builds a vocabulary from a corpus with reserved PAD/UNK ids, then encodes text to
fixed-length id sequences for the tensor archive. Deterministic given the same
corpus, so curation and training are reproducible.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

_WORD = re.compile(r"[a-z0-9]+")

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


def _split(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class WordVocab:
    """Maps words to ids (PAD=0, UNK=1) with deterministic ordering."""

    def __init__(self, tokens: list[str]) -> None:
        # tokens excludes the two reserved entries, which always lead.
        self.id_to_token: list[str] = [PAD_TOKEN, UNK_TOKEN, *tokens]
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(self.id_to_token)}

    def __len__(self) -> int:
        return len(self.id_to_token)

    @classmethod
    def build(cls, corpus: Iterable[str], max_size: int | None = None) -> WordVocab:
        counts: Counter[str] = Counter()
        for doc in corpus:
            counts.update(_split(doc))
        # Deterministic order: most frequent first, ties broken alphabetically.
        ordered = [tok for tok, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        if max_size is not None:
            ordered = ordered[: max(0, max_size - 2)]
        return cls(ordered)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(tok, UNK_ID) for tok in _split(text)]

    def encode_padded(self, text: str, seq_len: int) -> list[int]:
        ids = self.encode(text)[:seq_len]
        return ids + [PAD_ID] * (seq_len - len(ids))

    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(
            self.id_to_token[i] for i in ids if 0 <= i < len(self.id_to_token) and i != PAD_ID
        )

    def to_dict(self) -> dict[str, object]:
        return {"tokens": self.id_to_token[2:]}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> WordVocab:
        raw = data.get("tokens", [])
        tokens = raw if isinstance(raw, list) else []
        return cls([str(t) for t in tokens])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> WordVocab:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
