"""Tests for the archive-backed dataset and the data loader."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

from discovery_agents.mlinfra.curation import (  # noqa: E402
    CorpusCurator,
    LocalExecutor,
    generate_corpus,
)
from discovery_agents.mlinfra.data import (  # noqa: E402
    ArrayStoreDataset,
    build_dataloader,
    to_device,
)
from discovery_agents.mlinfra.store import open_store  # noqa: E402
from discovery_agents.mlinfra.tokenizer import WordVocab  # noqa: E402

SEQ_LEN = 16


def _build_store(tmp_path, n=64):
    docs = generate_corpus(n, seed=7)
    vocab = WordVocab.build(docs)
    store = open_store("numpy", str(tmp_path / "corpus"), mode="w")
    CorpusCurator(vocab, SEQ_LEN, LocalExecutor()).run(docs, store)
    return store


def test_dataset_len_and_item_shape(tmp_path) -> None:
    store = _build_store(tmp_path, n=40)
    dataset = ArrayStoreDataset(store)
    assert len(dataset) == 40
    item = dataset[0]
    assert item.dtype == torch.long
    assert item.shape == (SEQ_LEN,)


def test_dataloader_batch_shape_and_determinism(tmp_path) -> None:
    store = _build_store(tmp_path, n=64)
    dataset = ArrayStoreDataset(store)
    loader_a = build_dataloader(dataset, batch_size=8, seed=123)
    loader_b = build_dataloader(dataset, batch_size=8, seed=123)
    batch_a = next(iter(loader_a))
    batch_b = next(iter(loader_b))
    assert batch_a.shape == (8, SEQ_LEN)
    assert torch.equal(batch_a, batch_b)  # same seed -> same shuffle


def test_to_device_cpu_is_noop(tmp_path) -> None:
    store = _build_store(tmp_path, n=16)
    batch = next(iter(build_dataloader(ArrayStoreDataset(store), batch_size=4, seed=1)))
    moved = to_device(batch, "cpu")
    assert moved.device.type == "cpu"
    assert torch.equal(moved, batch)
