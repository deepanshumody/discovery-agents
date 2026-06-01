"""TorchEmbedder: serve the trained encoder through the retrieval Embedder protocol.

Implements ``discovery_agents.retrieval.embeddings.Embedder`` (``dim`` / ``embed`` /
``embed_batch``) so the agent pipeline's RAG layer can use the learned embedder in
place of the deterministic HashingEmbedder.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import ModelConfig
from .model.encoder import TextEncoder
from .tokenizer import WordVocab
from .train.checkpoint import load_checkpoint


class TorchEmbedder:
    """Embeds text with a trained TextEncoder (eval mode, no grad, L2-normalized)."""

    def __init__(
        self,
        encoder: TextEncoder,
        vocab: WordVocab,
        seq_len: int = 64,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device).eval()
        self.vocab = vocab
        self.seq_len = seq_len
        self.dim = encoder.config.dim

    @classmethod
    def from_pretrained(cls, artifact_dir: str, device: str = "cpu") -> TorchEmbedder:
        """Load vocab.json + model_config.json + latest.pt from an artifact directory."""
        directory = Path(artifact_dir)
        vocab = WordVocab.load(directory / "vocab.json")
        raw = json.loads((directory / "model_config.json").read_text(encoding="utf-8"))
        seq_len = int(raw.pop("seq_len", 64))
        config = ModelConfig(**raw)
        config.vocab_size = len(vocab)
        encoder = TextEncoder(config)
        load_checkpoint(str(directory), model=encoder, map_location=device)
        return cls(encoder, vocab, seq_len=seq_len, device=device)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        rows = [self.vocab.encode_padded(t, self.seq_len) for t in texts]
        tokens = torch.tensor(rows, dtype=torch.long, device=self.device)
        with torch.no_grad():
            vectors = self.encoder(tokens)
        # Guard against all-PAD rows (empty/OOV text) producing NaN in attention.
        vectors = torch.nan_to_num(vectors)
        return vectors.cpu().tolist()
