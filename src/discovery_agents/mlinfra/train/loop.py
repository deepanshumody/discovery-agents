"""Custom training loop for the SimCSE contrastive embedder.

One step: encode the batch twice (two independent dropout masks => SimCSE positive
pairs) -> InfoNCE -> backward -> grad-clip -> optimizer + warmup-scheduler step. The
loop is DDP-aware, logs metrics through a Tracker, and checkpoints periodically.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import TrainConfig
from ..data.loader import to_device
from ..model.losses import info_nce
from . import distributed as dist_utils
from .checkpoint import save_checkpoint


def _warmup_factor(step: int, warmup_steps: int) -> float:
    return min(1.0, (step + 1) / max(1, warmup_steps))


class Trainer:
    """Owns the model, optimizer, warmup scheduler, and step counter."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        *,
        device: str = "cpu",
        tracker: Any | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.module = model.to(self.device)  # underlying encoder (optimizer + checkpoint target)
        self.model = dist_utils.maybe_ddp(self.module)  # DDP-wrapped when distributed
        self.tracker = tracker
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.module.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, functools.partial(_warmup_factor, warmup_steps=config.warmup_steps)
        )
        self.step = 0

    def train_step(self, batch: torch.Tensor) -> float:
        self.model.train()
        batch = to_device(batch, self.device)
        z1 = self.model(batch)
        z2 = self.model(batch)  # second view: independent dropout mask
        loss = info_nce(z1, z2, self.config.temperature)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.module.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return float(loss.detach().item())

    def fit(self, dataloader: DataLoader[torch.Tensor], steps: int | None = None) -> dict[str, Any]:
        steps = steps or self.config.steps
        losses: list[float] = []
        batches = self._cycle(dataloader)
        for _ in range(steps):
            loss = self.train_step(next(batches))
            losses.append(loss)
            if (
                self.step % self.config.log_every == 0
                and self.tracker is not None
                and dist_utils.is_main()
            ):
                self.tracker.log_metrics(
                    {"loss": loss, "lr": self.scheduler.get_last_lr()[0]}, self.step
                )
            if (
                self.config.checkpoint_every
                and self.step % self.config.checkpoint_every == 0
                and dist_utils.is_main()
            ):
                self.save()
        return {"losses": losses, "final_loss": losses[-1] if losses else None, "step": self.step}

    def save(self) -> str:
        return save_checkpoint(
            self.config.checkpoint_dir,
            model=self.module,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
        )

    def _cycle(self, dataloader: DataLoader[torch.Tensor]) -> Iterator[torch.Tensor]:
        while True:
            yield from dataloader
