"""A small Transformer sentence encoder producing L2-normalized embeddings."""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from ..tokenizer import PAD_ID


class TextEncoder(nn.Module):
    """Token + positional embeddings -> TransformerEncoder -> masked mean pool -> L2 norm."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.dim, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.dim)

    @property
    def dim(self) -> int:
        return self.config.dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device).unsqueeze(0).expand(batch, length)
        hidden = self.token_emb(tokens) + self.pos_emb(positions)

        pad_mask = tokens == PAD_ID  # (B, T) True where padding
        hidden = self.encoder(hidden, src_key_padding_mask=pad_mask)
        hidden = self.norm(hidden)

        keep = (~pad_mask).unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
        summed = (hidden * keep).sum(dim=1)
        counts = keep.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return torch.nn.functional.normalize(pooled, dim=-1)
