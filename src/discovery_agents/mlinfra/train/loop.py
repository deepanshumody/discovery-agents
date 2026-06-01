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
        self.last_lr = float(self.optimizer.param_groups[0]["lr"])

    def train_step(self, batch: torch.Tensor) -> float:
        self.model.train()
        batch = to_device(batch, self.device)
        z1 = self.model(batch)
        z2 = self.model(batch)  # second view: independent dropout mask
        loss = info_nce(z1, z2, self.config.temperature)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.module.parameters(), self.config.grad_clip)
        # Capture the LR actually applied to this step BEFORE the scheduler advances,
        # so logged lr pairs with the loss of the same step.
        self.last_lr = float(self.optimizer.param_groups[0]["lr"])
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
                self.tracker.log_metrics({"loss": loss, "lr": self.last_lr}, self.step)
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

    def supervised_step(self, tokens: torch.Tensor, labels: torch.Tensor) -> float:
        """One supervised-contrastive step over a label-aware batch."""
        from ..model.losses import supervised_contrastive

        self.model.train()
        tokens = to_device(tokens, self.device)
        labels = to_device(labels, self.device)
        embeddings = self.model(tokens)
        loss = supervised_contrastive(embeddings, labels, self.config.temperature)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.module.parameters(), self.config.grad_clip)
        self.last_lr = float(self.optimizer.param_groups[0]["lr"])
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return float(loss.detach().item())

    def fit_supervised(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        *,
        classes_per_batch: int = 16,
        samples_per_class: int = 4,
        steps: int | None = None,
        seed: int = 7,
    ) -> dict[str, Any]:
        """Train with supervised contrastive loss using a label-aware batch sampler."""
        from ..data.sampler import LabelBatchSampler

        steps = steps or self.config.steps
        sampler = LabelBatchSampler(
            labels.tolist(),
            classes_per_batch=classes_per_batch,
            samples_per_class=samples_per_class,
            steps=steps,
            seed=seed,
        )
        losses: list[float] = []
        for batch_indices in sampler:
            index = torch.tensor(batch_indices, dtype=torch.long)
            losses.append(self.supervised_step(tokens[index], labels[index]))
            if self.tracker is not None and self.step % self.config.log_every == 0:
                self.tracker.log_metrics({"loss": losses[-1], "lr": self.last_lr}, self.step)
        return {"losses": losses, "final_loss": losses[-1] if losses else None, "step": self.step}

    def _cycle(self, dataloader: DataLoader[torch.Tensor]) -> Iterator[torch.Tensor]:
        while True:
            yield from dataloader
