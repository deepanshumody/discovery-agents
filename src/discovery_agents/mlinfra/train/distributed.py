"""Distributed training helpers (DDP).

`gloo` on CPU, `nccl` on GPU. Single-process runs are a no-op (``maybe_ddp`` returns
the model unchanged), so the same training code runs locally and across a cluster.
FSDP is the documented multi-GPU sharding path (future work).
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main() -> bool:
    return get_rank() == 0


def setup(rank: int, world_size: int, backend: str = "gloo") -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup() -> None:
    if is_distributed():
        dist.destroy_process_group()


def maybe_ddp(model: nn.Module) -> nn.Module:
    """Wrap in DistributedDataParallel when a process group is initialized."""
    if is_distributed():
        return DistributedDataParallel(model)
    return model


def unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module whether or not it is DDP-wrapped."""
    return model.module if isinstance(model, DistributedDataParallel) else model


def spawn(worker: Callable[..., None], world_size: int, *args: object) -> None:
    """Launch ``world_size`` processes running ``worker(rank, world_size, *args)``."""
    import torch.multiprocessing as mp

    mp.spawn(worker, args=(world_size, *args), nprocs=world_size, join=True)
