"""Tests for retrieval metrics and the benchmark runner (sample, fast)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from discovery_agents.mlinfra.retrieval_eval import retrieval_metrics  # noqa: E402


def test_perfect_retrieval_scores_one() -> None:
    pool = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    pool_labels = np.array([0, 0, 1, 1])
    queries = np.array([[1.0, 0.0], [0.0, 1.0]])
    query_labels = np.array([0, 1])
    m = retrieval_metrics(queries, query_labels, pool, pool_labels, ks=(1, 2))
    assert m["hit@1"] == 1.0
    assert m["mrr"] == 1.0
    assert m["map"] == 1.0


def test_no_relevant_scores_zero() -> None:
    pool = np.array([[1.0, 0.0], [1.0, 0.0]])
    pool_labels = np.array([0, 0])
    queries = np.array([[0.0, 1.0]])  # orthogonal, but label 1 has no pool members
    query_labels = np.array([1])
    m = retrieval_metrics(queries, query_labels, pool, pool_labels, ks=(1,))
    assert m["hit@1"] == 0.0
    assert m["mrr"] == 0.0


def test_metrics_clamp_k_to_pool_size() -> None:
    # ks larger than the pool must clamp, not crash (np.argpartition kth bound).
    pool = np.array([[1.0, 0.0], [0.0, 1.0]])
    pool_labels = np.array([0, 1])
    queries = np.array([[1.0, 0.0]])
    m = retrieval_metrics(queries, np.array([0]), pool, pool_labels, ks=(1, 5, 10))
    assert m["hit@1"] == 1.0 and m["hit@10"] == 1.0  # k clamped to pool size of 2


def test_run_benchmark_sample_structure() -> None:
    pytest.importorskip("torch")
    from discovery_agents.mlinfra.retrieval_eval import run_benchmark

    report = run_benchmark(full=False, steps=20)
    assert {"hashing (lexical baseline)", "torch (supervised contrastive)"} <= set(
        report["results"]
    )
    for metrics in report["results"].values():
        for key in ("hit@1", "hit@5", "hit@10", "mrr", "map"):
            assert 0.0 <= metrics[key] <= 1.0
