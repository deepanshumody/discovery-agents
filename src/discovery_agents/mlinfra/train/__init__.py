"""Training: custom DDP loop + fault-tolerant resumable checkpointing."""

from __future__ import annotations

from .checkpoint import (
    has_checkpoint,
    install_sigterm_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from .distributed import (
    cleanup,
    get_rank,
    get_world_size,
    is_main,
    maybe_ddp,
    set_seed,
    setup,
    spawn,
)
from .loop import Trainer

__all__ = [
    "Trainer",
    "cleanup",
    "get_rank",
    "get_world_size",
    "has_checkpoint",
    "install_sigterm_checkpoint",
    "is_main",
    "load_checkpoint",
    "maybe_ddp",
    "save_checkpoint",
    "set_seed",
    "setup",
    "spawn",
]
