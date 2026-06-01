"""Distributed corpus curation: tokenize documents and write the tensor archive.

The `Executor` protocol abstracts *how* the per-document work is distributed
(sequentially, via Dask, or — future — Ray), so the curator is backend-agnostic.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np

from ..store.base import ArrayStore
from ..tokenizer import WordVocab

T = TypeVar("T")
R = TypeVar("R")


@runtime_checkable
class Executor(Protocol):
    def map(self, fn: Callable[[T], R], items: list[T]) -> list[R]: ...


def _encode(doc: str, vocab: WordVocab, seq_len: int) -> list[int]:
    """Top-level (picklable) encoder so Dask/Ray can ship it to workers."""
    return vocab.encode_padded(doc, seq_len)


class CorpusCurator:
    """Maps documents → padded token rows (via an executor) → an ArrayStore."""

    def __init__(self, vocab: WordVocab, seq_len: int, executor: Executor) -> None:
        self.vocab = vocab
        self.seq_len = seq_len
        self.executor = executor

    def run(self, docs: list[str], store: ArrayStore, name: str = "tokens") -> int:
        encoder: Callable[[str], list[int]] = functools.partial(
            _encode, vocab=self.vocab, seq_len=self.seq_len
        )
        rows = self.executor.map(encoder, docs)
        array = np.asarray(rows, dtype="int64")
        store.create(name, (0, self.seq_len), "int64")
        store.append(name, array)
        return int(array.shape[0])
