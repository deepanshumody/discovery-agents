"""Retrieval-quality benchmark: does the trained embedder beat the lexical baseline?

Embeds the Banking77 test split (queries) against the train split (pool), where
relevant = same intent, and reports recall@k / MRR / mAP for the lexical
HashingEmbedder, the supervised-contrastive TorchEmbedder, and (optionally) an
off-the-shelf sentence-transformers reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from ..retrieval.embeddings import HashingEmbedder
from .config import ModelConfig, TrainConfig
from .datasets import Banking77, LabeledSplit, load_banking77
from .embedder import TorchEmbedder
from .model.encoder import TextEncoder
from .tokenizer import WordVocab
from .train import Trainer, set_seed


class _EmbedderLike(Protocol):
    dim: int

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


def embed_texts(embedder: _EmbedderLike, texts: list[str], batch_size: int = 256) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embedder.embed_batch(texts[start : start + batch_size]))
    array = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)  # cosine via dot product


def retrieval_metrics(
    query_vecs: np.ndarray,
    query_labels: np.ndarray,
    pool_vecs: np.ndarray,
    pool_labels: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    sims = query_vecs @ pool_vecs.T  # (Nq, Np) cosine similarities
    max_k = max(ks)
    rows = np.arange(sims.shape[0])[:, None]
    part = np.argpartition(-sims, max_k - 1, axis=1)[:, :max_k]
    order = np.argsort(-sims[rows, part], axis=1)
    topk = part[rows, order]  # (Nq, max_k) pool indices, best-first

    relevant = pool_labels[topk] == query_labels[:, None]  # (Nq, max_k)
    # bincount must cover query labels too (a query intent may be absent from the pool).
    n_labels = int(max(np.max(pool_labels, initial=0), np.max(query_labels, initial=0))) + 1
    counts = np.bincount(pool_labels, minlength=n_labels)
    total_rel = counts[query_labels]  # relevant items in the pool per query

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = float(relevant[:, :k].any(axis=1).mean())

    reciprocal = np.zeros(len(query_labels))
    average_precision = np.zeros(len(query_labels))
    for i in range(len(query_labels)):
        hit_positions = np.nonzero(relevant[i])[0]
        if hit_positions.size:
            reciprocal[i] = 1.0 / (hit_positions[0] + 1)
            precision_at = np.cumsum(relevant[i]) / (np.arange(max_k) + 1)
            denom = min(int(total_rel[i]), max_k)
            average_precision[i] = (precision_at * relevant[i]).sum() / max(denom, 1)
    metrics["mrr"] = float(reciprocal.mean())
    metrics["map"] = float(average_precision.mean())
    return metrics


def evaluate_embedder(
    embedder: _EmbedderLike,
    queries: LabeledSplit,
    pool: LabeledSplit,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    query_vecs = embed_texts(embedder, queries.texts)
    pool_vecs = embed_texts(embedder, pool.texts)
    return retrieval_metrics(
        query_vecs, np.asarray(queries.labels), pool_vecs, np.asarray(pool.labels), ks
    )


class SentenceTransformerEmbedder:
    """Optional off-the-shelf reference embedder (requires the [st] extra)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [row.tolist() for row in vectors]


def train_embedder(
    data: Banking77,
    *,
    seq_len: int = 32,
    dim: int = 128,
    steps: int = 800,
    seed: int = 7,
) -> TorchEmbedder:
    set_seed(seed)
    vocab = WordVocab.build(data.train.texts)
    tokens = torch.tensor(
        [vocab.encode_padded(t, seq_len) for t in data.train.texts], dtype=torch.long
    )
    labels = torch.tensor(data.train.labels, dtype=torch.long)
    model = TextEncoder(
        ModelConfig(dim=dim, num_layers=2, num_heads=4, max_seq_len=seq_len, vocab_size=len(vocab))
    )
    config = TrainConfig(steps=steps, lr=1e-3, warmup_steps=max(5, steps // 20), checkpoint_every=0)
    trainer = Trainer(model, config, device="cpu")
    trainer.fit_supervised(tokens, labels, classes_per_batch=24, samples_per_class=4, steps=steps)
    return TorchEmbedder(trainer.module, vocab, seq_len=seq_len)


def run_benchmark(
    *, full: bool = False, with_st: bool = False, steps: int | None = None, seed: int = 7
) -> dict[str, Any]:
    data = load_banking77(full=full)
    steps = steps if steps is not None else (800 if full else 150)

    embedders: dict[str, _EmbedderLike] = {
        "hashing (lexical baseline)": HashingEmbedder(),
        "torch (supervised contrastive)": train_embedder(data, steps=steps, seed=seed),
    }
    if with_st:
        embedders["sentence-transformers (reference)"] = SentenceTransformerEmbedder()

    results: dict[str, dict[str, float]] = {}
    for name, embedder in embedders.items():
        results[name] = evaluate_embedder(embedder, data.test, data.train)
    return {
        "dataset": "banking77" + ("" if full else " (committed sample)"),
        "split": {"queries": len(data.test), "pool": len(data.train), "intents": data.num_labels},
        "train_steps": steps,
        "results": results,
    }


def write_results(report: dict[str, Any], out_dir: str = "benchmark") -> None:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    metric_cols = ["recall@1", "recall@5", "recall@10", "mrr", "map"]
    header = "| embedder | " + " | ".join(metric_cols) + " |"
    sep = "|" + "---|" * (len(metric_cols) + 1)
    lines = [
        "# Retrieval benchmark — Banking77 intent retrieval",
        "",
        f"Dataset: **{report['dataset']}** · queries (test): {report['split']['queries']} · "
        f"pool (train): {report['split']['pool']} · intents: {report['split']['intents']} · "
        f"train steps: {report['train_steps']}.",
        "",
        "Relevant = same intent. Reproduce: `python -m discovery_agents.mlinfra.cli benchmark --full`.",
        "",
        header,
        sep,
    ]
    for name, metrics in report["results"].items():
        cells = " | ".join(f"{metrics[c]:.4f}" for c in metric_cols)
        lines.append(f"| {name} | {cells} |")
    lines.append("")
    (directory / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
