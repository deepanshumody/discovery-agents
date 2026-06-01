"""TorchEmbedder implements the retrieval Embedder protocol and powers RAG."""

from __future__ import annotations

import json
from dataclasses import asdict

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
from discovery_agents.mlinfra.embedder import TorchEmbedder  # noqa: E402
from discovery_agents.mlinfra.model import TextEncoder  # noqa: E402
from discovery_agents.mlinfra.store import open_store  # noqa: E402
from discovery_agents.mlinfra.tokenizer import WordVocab  # noqa: E402
from discovery_agents.mlinfra.train import Trainer, set_seed  # noqa: E402
from discovery_agents.retrieval import EvidenceIndex, HashingEmbedder, get_embedder  # noqa: E402
from discovery_agents.retrieval.embeddings import Embedder  # noqa: E402
from discovery_agents.sample_data import SAMPLE_EVIDENCE  # noqa: E402

SEQ_LEN = 16


def _make_artifacts(tmp_path) -> str:
    docs = generate_corpus(32, seed=7)
    vocab = WordVocab.build(docs)
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    vocab.save(ckpt / "vocab.json")

    model_config = ModelConfig(
        dim=32, num_layers=2, num_heads=4, max_seq_len=SEQ_LEN, vocab_size=len(vocab)
    )
    set_seed(0)
    model = TextEncoder(model_config)
    store = open_store("numpy", str(tmp_path / "corpus"), mode="w")
    CorpusCurator(vocab, SEQ_LEN, LocalExecutor()).run(docs, store)
    loader = build_dataloader(ArrayStoreDataset(store), batch_size=8, seed=0)
    config = TrainConfig(steps=3, warmup_steps=1, checkpoint_every=0, checkpoint_dir=str(ckpt))
    trainer = Trainer(model, config, device="cpu")
    trainer.fit(loader, steps=3)
    trainer.save()
    (ckpt / "model_config.json").write_text(
        json.dumps({**asdict(model_config), "seq_len": SEQ_LEN}), encoding="utf-8"
    )
    return str(ckpt)


def test_from_pretrained_embeds_and_normalizes(tmp_path) -> None:
    embedder = TorchEmbedder.from_pretrained(_make_artifacts(tmp_path))
    assert embedder.dim == 32
    vec = embedder.embed("enterprise agent workflow")
    assert len(vec) == 32
    assert abs(sum(x * x for x in vec) ** 0.5 - 1.0) < 1e-4  # L2-normalized

    batch = embedder.embed_batch(["alignment decision", "handoff requirements"])
    assert len(batch) == 2 and all(len(v) == 32 for v in batch)
    assert embedder.embed("same text") == embedder.embed("same text")  # eval determinism


def test_torch_embedder_satisfies_embedder_protocol(tmp_path) -> None:
    embedder = TorchEmbedder.from_pretrained(_make_artifacts(tmp_path))
    assert isinstance(embedder, Embedder)


def test_evidence_index_with_torch_embedder(tmp_path) -> None:
    embedder = TorchEmbedder.from_pretrained(_make_artifacts(tmp_path))
    index = EvidenceIndex.from_evidence(SAMPLE_EVIDENCE, embedder=embedder)
    hits = index.search("implementation handoff requirements", k=2)
    assert len(hits) == 2
    assert all(isinstance(h.chunk.id, str) for h in hits)


def test_get_embedder_fallbacks() -> None:
    assert isinstance(get_embedder("hashing"), HashingEmbedder)
    # torch kind with a missing artifact dir must fall back, not raise.
    assert isinstance(get_embedder("torch", artifact_dir="/nonexistent/path"), HashingEmbedder)


def test_pipeline_runs_with_torch_embedder(tmp_path) -> None:
    from discovery_agents import ProductDiscoveryPipeline, RunConfig
    from discovery_agents.sample_data import SAMPLE_BRIEF

    config = RunConfig(embedder="torch", ml_artifact_dir=_make_artifacts(tmp_path))
    run = ProductDiscoveryPipeline(config).run(SAMPLE_BRIEF, SAMPLE_EVIDENCE)
    assert run.selected_direction_id is not None
    assert len(run.directions) >= 5
