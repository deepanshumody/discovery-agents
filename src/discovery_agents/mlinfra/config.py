"""Configuration for the ML-infra platform (torch-free, so it imports anywhere)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Small-by-default sentence encoder; scale the dims up for real runs."""

    dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 256
    max_seq_len: int = 64
    dropout: float = 0.1
    vocab_size: int = 4096  # upper bound; the real vocab size is set after building it


@dataclass
class DataConfig:
    corpus_size: int = 512  # number of synthetic documents
    seq_len: int = 64
    batch_size: int = 32
    store_backend: str = "numpy"  # numpy | zarr | hdf5
    store_path: str = "outputs/ml/corpus"
    executor: str = "local"  # local | dask
    seed: int = 7


@dataclass
class TrainConfig:
    steps: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 20
    grad_clip: float = 1.0
    temperature: float = 0.05
    log_every: int = 10
    checkpoint_every: int = 50
    checkpoint_dir: str = "outputs/ml/checkpoints"
    backend: str = "gloo"  # gloo (CPU) | nccl (GPU)
    world_size: int = 1
    seed: int = 7

    @classmethod
    def smoke(cls) -> TrainConfig:
        """A tiny, fast configuration for CI/smoke runs."""
        return cls(steps=2, warmup_steps=1, log_every=1, checkpoint_every=2)


def model_config_from_env() -> ModelConfig:
    return ModelConfig(
        dim=int(os.environ.get("ML_DIM", "128")),
        num_layers=int(os.environ.get("ML_LAYERS", "2")),
        max_seq_len=int(os.environ.get("ML_SEQ_LEN", "64")),
    )


def train_config_from_env() -> TrainConfig:
    return TrainConfig(
        steps=int(os.environ.get("ML_STEPS", "200")),
        lr=float(os.environ.get("ML_LR", "1e-3")),
        backend=os.environ.get("ML_BACKEND", "gloo"),
        world_size=int(os.environ.get("ML_WORLD_SIZE", "1")),
    )
