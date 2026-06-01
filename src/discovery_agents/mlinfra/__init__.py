"""ML-infrastructure platform: a trainable neural retrieval embedder.

This subpackage adds a production training/data platform to discovery-agents:
distributed PyTorch training (DDP) with resumable checkpointing, a multi-backend
tensor archive (Numpy/Zarr/HDF5) with a GPU-native data loader, distributed
corpus curation (Local/Dask), experiment tracking, and an I/O benchmark. The
trained model is exposed as a ``TorchEmbedder`` that implements the existing
``discovery_agents.retrieval.embeddings.Embedder`` protocol.

It is optional and isolated behind the ``[ml]`` extra; the agentic core never
imports it. Only lightweight, torch-free modules (``config``) are imported here
so ``import discovery_agents.mlinfra`` does not require torch.
"""

from __future__ import annotations

from .config import DataConfig, ModelConfig, TrainConfig

__all__ = ["DataConfig", "ModelConfig", "TrainConfig"]
