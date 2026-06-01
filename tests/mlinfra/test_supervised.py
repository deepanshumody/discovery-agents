"""Tests for SupCon loss, the label-aware sampler, and supervised training."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from discovery_agents.mlinfra.config import ModelConfig, TrainConfig  # noqa: E402
from discovery_agents.mlinfra.data import LabelBatchSampler  # noqa: E402
from discovery_agents.mlinfra.model import TextEncoder, supervised_contrastive  # noqa: E402
from discovery_agents.mlinfra.train import Trainer, set_seed  # noqa: E402


def test_supcon_lower_when_same_label_clustered() -> None:
    # Two labels; clustered embeddings (same label identical, different orthogonal) -> low loss.
    clustered = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1, 1])
    scattered = torch.nn.functional.normalize(
        torch.randn(4, 2, generator=torch.Generator().manual_seed(0)), dim=-1
    )
    assert supervised_contrastive(clustered, labels) < supervised_contrastive(scattered, labels)


def test_supcon_gradients_flow() -> None:
    base = torch.randn(6, 8, requires_grad=True)  # leaf tensor carries .grad
    z = torch.nn.functional.normalize(base, dim=-1)
    loss = supervised_contrastive(z, torch.tensor([0, 0, 1, 1, 2, 2]))
    loss.backward()
    assert base.grad is not None and torch.isfinite(base.grad).all()


def test_label_batch_sampler_guarantees_positives() -> None:
    labels = [i % 5 for i in range(100)]  # 5 classes, 20 each
    sampler = LabelBatchSampler(labels, classes_per_batch=3, samples_per_class=4, steps=10, seed=1)
    batches = list(sampler)
    assert len(batches) == 10
    for batch in batches:
        assert len(batch) == 12  # 3 classes x 4 samples
        counts: dict[int, int] = {}
        for idx in batch:
            counts[labels[idx]] = counts.get(labels[idx], 0) + 1
        # every class in the batch has >= 2 members -> in-batch positives exist
        assert all(c >= 2 for c in counts.values())
    assert list(LabelBatchSampler(labels, steps=3, seed=1)) == list(
        LabelBatchSampler(labels, steps=3, seed=1)
    )  # deterministic


def test_fit_supervised_reduces_loss() -> None:
    # Tiny synthetic labeled token data: 6 classes, distinct token signatures.
    set_seed(0)
    seq_len, n_per = 8, 12
    rows, labels = [], []
    for cls in range(6):
        for _ in range(n_per):
            rows.append([cls + 1] * seq_len)  # class-distinctive tokens
            labels.append(cls)
    tokens = torch.tensor(rows, dtype=torch.long)
    label_t = torch.tensor(labels, dtype=torch.long)

    model = TextEncoder(
        ModelConfig(dim=32, num_layers=1, num_heads=4, max_seq_len=seq_len, vocab_size=16)
    )
    config = TrainConfig(lr=1e-3, warmup_steps=5, checkpoint_every=0, checkpoint_dir="/tmp/sup")
    trainer = Trainer(model, config, device="cpu")
    result = trainer.fit_supervised(
        tokens, label_t, classes_per_batch=4, samples_per_class=4, steps=60
    )
    losses = result["losses"]
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10  # SupCon loss drops
