"""Fault-tolerant, resumable checkpointing.

Captures model + optimizer + scheduler + step + RNG state, writes atomically
(temp file then ``os.replace``) so a crash mid-write never corrupts ``latest.pt``,
and installs a SIGTERM handler so a preempted run checkpoints before exiting — the
"continuity of large training runs" the role calls for.
"""

from __future__ import annotations

import os
import random
import signal
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import torch
from torch import nn

from .distributed import unwrap

CHECKPOINT_NAME = "latest.pt"


def _rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def has_checkpoint(directory: str) -> bool:
    return (Path(directory) / CHECKPOINT_NAME).exists()


def save_checkpoint(
    directory: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    step: int,
) -> str:
    Path(directory).mkdir(parents=True, exist_ok=True)
    payload = {
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "rng": _rng_state(),
    }
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)
    torch.save(payload, tmp)
    os.replace(tmp, Path(directory) / CHECKPOINT_NAME)  # atomic on POSIX
    return str(Path(directory) / CHECKPOINT_NAME)


def load_checkpoint(
    directory: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str = "cpu",
) -> int:
    payload = torch.load(
        Path(directory) / CHECKPOINT_NAME, map_location=map_location, weights_only=False
    )
    unwrap(model).load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    _set_rng_state(payload["rng"])
    return int(payload["step"])


def install_sigterm_checkpoint(save_fn: Callable[[], None]) -> None:
    """Register a SIGTERM handler that checkpoints then re-raises the default action."""

    def _handler(signum: int, frame: FrameType | None) -> None:
        save_fn()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)

    signal.signal(signal.SIGTERM, _handler)
