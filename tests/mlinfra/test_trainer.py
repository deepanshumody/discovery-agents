"""The training loop reduces contrastive loss on a tiny corpus."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

from discovery_agents.mlinfra.config import ModelConfig, TrainConfig  # noqa: E402
from discovery_agents.mlinfra.curation import (  # noqa: E402
    CorpusCurator,
    LocalExecutor,
    generate_corpus,
)
from discovery_agents.mlinfra.data import ArrayStoreDataset, build_dataloader  # noqa: E402
from discovery_agents.mlinfra.model import TextEncoder  # noqa: E402
from discovery_agents.mlinfra.store import open_store  # noqa: E402
from discovery_agents.mlinfra.tokenizer import WordVocab  # noqa: E402
from discovery_agents.mlinfra.train import Trainer, set_seed  # noqa: E402

SEQ_LEN = 16


def _build(tmp_path, n=128):
    docs = generate_corpus(n, seed=7)
    vocab = WordVocab.build(docs)
    store = open_store("numpy", str(tmp_path / "corpus"), mode="w")
    CorpusCurator(vocab, SEQ_LEN, LocalExecutor()).run(docs, store)
    loader = build_dataloader(ArrayStoreDataset(store), batch_size=16, seed=0)
    model = TextEncoder(
        ModelConfig(dim=32, num_layers=2, num_heads=4, max_seq_len=SEQ_LEN, vocab_size=len(vocab))
    )
    return model, loader


def test_training_reduces_loss(tmp_path) -> None:
    set_seed(0)
    model, loader = _build(tmp_path)
    config = TrainConfig(
        steps=80,
        lr=1e-3,
        warmup_steps=10,
        checkpoint_every=0,
        log_every=10_000,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    trainer = Trainer(model, config, device="cpu")
    result = trainer.fit(loader, steps=80)
    losses = result["losses"]
    assert result["step"] == 80
    first = sum(losses[:10]) / 10
    last = sum(losses[-10:]) / 10
    assert last < first  # the embedder learns -> contrastive loss drops
