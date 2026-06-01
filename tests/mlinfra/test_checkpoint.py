"""Checkpoint fidelity: restoring {model, optimizer, scheduler, RNG} reproduces the next
step exactly *given the same batch*; SIGTERM checkpoints. (Sampler/data-stream offset is
not yet checkpointed, so full data-stream resumability is future work -- see the spec.)
"""

from __future__ import annotations

import signal

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
from discovery_agents.mlinfra.train import (  # noqa: E402
    Trainer,
    install_sigterm_checkpoint,
    load_checkpoint,
    set_seed,
)

SEQ_LEN = 16


def _make_model(vocab_size: int) -> TextEncoder:
    return TextEncoder(
        ModelConfig(dim=32, num_layers=2, num_heads=4, max_seq_len=SEQ_LEN, vocab_size=vocab_size)
    )


def _fixture(tmp_path):
    docs = generate_corpus(96, seed=7)
    vocab = WordVocab.build(docs)
    store = open_store("numpy", str(tmp_path / "corpus"), mode="w")
    CorpusCurator(vocab, SEQ_LEN, LocalExecutor()).run(docs, store)
    loader = build_dataloader(ArrayStoreDataset(store), batch_size=16, seed=0)
    return len(vocab), next(iter(loader))


def test_resume_reproduces_next_step(tmp_path) -> None:
    vocab_size, batch = _fixture(tmp_path)
    config = TrainConfig(
        steps=5, lr=1e-3, warmup_steps=2, checkpoint_every=0, checkpoint_dir=str(tmp_path / "ckpt")
    )

    set_seed(0)
    trainer = Trainer(_make_model(vocab_size), config, device="cpu")
    for _ in range(5):
        trainer.train_step(batch)
    trainer.save()
    loss_a = trainer.train_step(batch)
    params_a = [p.detach().clone() for p in trainer.module.parameters()]

    # Resume into a fresh trainer under a DIFFERENT ambient seed: only the checkpoint
    # (model + optimizer + scheduler + RNG) should determine the next step.
    set_seed(999)
    resumed = Trainer(_make_model(vocab_size), config, device="cpu")
    step = load_checkpoint(
        config.checkpoint_dir,
        model=resumed.module,
        optimizer=resumed.optimizer,
        scheduler=resumed.scheduler,
    )
    resumed.step = step
    loss_b = resumed.train_step(batch)
    params_b = [p.detach().clone() for p in resumed.module.parameters()]

    assert step == 5
    assert abs(loss_a - loss_b) < 1e-5
    for a, b in zip(params_a, params_b):
        assert torch.allclose(a, b, atol=1e-6)


def test_atomic_save_leaves_no_temp_files(tmp_path) -> None:
    vocab_size, _ = _fixture(tmp_path)
    config = TrainConfig(checkpoint_dir=str(tmp_path / "ckpt"))
    set_seed(0)
    Trainer(_make_model(vocab_size), config, device="cpu").save()
    files = list((tmp_path / "ckpt").iterdir())
    assert any(f.name == "latest.pt" for f in files)
    assert not any(f.suffix == ".tmp" for f in files)  # temp file was renamed away


def test_sigterm_handler_checkpoints_then_exits(tmp_path, monkeypatch) -> None:
    calls: list[int] = []
    install_sigterm_checkpoint(lambda: calls.append(1))
    monkeypatch.setattr("discovery_agents.mlinfra.train.checkpoint.os.kill", lambda *a, **k: None)
    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler)
    handler(signal.SIGTERM, None)  # type: ignore[call-arg]
    signal.signal(signal.SIGTERM, signal.SIG_DFL)  # restore for other tests
    assert calls == [1]
