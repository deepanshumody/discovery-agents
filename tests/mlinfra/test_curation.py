"""Tests for synthetic corpus generation and curation (Local vs Dask parity)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from discovery_agents.mlinfra.curation import (  # noqa: E402
    CorpusCurator,
    DaskExecutor,
    LocalExecutor,
    generate_corpus,
)
from discovery_agents.mlinfra.store import open_store  # noqa: E402
from discovery_agents.mlinfra.tokenizer import WordVocab  # noqa: E402


def test_generate_corpus_is_deterministic() -> None:
    assert generate_corpus(10, seed=7) == generate_corpus(10, seed=7)
    assert generate_corpus(10, seed=7) != generate_corpus(10, seed=8)


def test_local_executor_preserves_order() -> None:
    out = LocalExecutor().map(lambda x: x * 2, [1, 2, 3])
    assert out == [2, 4, 6]


def test_curation_writes_expected_archive(tmp_path) -> None:
    docs = generate_corpus(32, seed=7)
    vocab = WordVocab.build(docs)
    store = open_store("numpy", str(tmp_path / "corpus"), mode="w")
    curator = CorpusCurator(vocab, seq_len=16, executor=LocalExecutor())
    n = curator.run(docs, store)
    assert n == 32
    rows = store.read("tokens", 0, 32)
    assert rows.shape == (32, 16)
    # First row decodes back to (a prefix of) the first document.
    assert vocab.decode(rows[0]).split()[0] == docs[0].split()[0]


def test_local_and_dask_produce_identical_archive(tmp_path) -> None:
    pytest.importorskip("dask")
    docs = generate_corpus(40, seed=11)
    vocab = WordVocab.build(docs)

    local_store = open_store("numpy", str(tmp_path / "local"), mode="w")
    CorpusCurator(vocab, 16, LocalExecutor()).run(docs, local_store)
    local_rows = local_store.read("tokens", 0, 40)

    dask_store = open_store("numpy", str(tmp_path / "dask"), mode="w")
    CorpusCurator(vocab, 16, DaskExecutor(npartitions=4)).run(docs, dask_store)
    dask_rows = dask_store.read("tokens", 0, 40)

    np.testing.assert_array_equal(local_rows, dask_rows)
