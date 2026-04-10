"""
Conformational Propensity Prior
================================

A 1D sequence-based model that predicts a kinase's *intrinsic* preference
for DFG-in vs DFG-out based solely on its 85-residue KLIFS pocket sequence.

This is NOT a replacement for the 3D Spiking EGNN.  Instead it serves as
a Bayesian prior:

    P(state | sequence)   ← this model
    P(state | structure)  ← the 3D EGNN

The combined system mirrors how AlphaFold uses MSAs (evolutionary signal)
alongside structural templates (3D evidence).

Usage
-----
    prior = ConformationalPrior.from_pretrained("checkpoints/conf_prior.pt")
    propensity = prior.predict(pocket_sequence)
    # propensity ≈ 0.8  → "this kinase family prefers DFG-out 80% of the time"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Amino acid encoding (matches KLIFS convention)
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY-X"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}


@dataclass
class ConformationalPriorConfig:
    """Configuration for the conformational propensity model."""
    pocket_size: int = 85
    vocab_size: int = 22       # 20 AA + gap + unknown
    embed_dim: int = 32
    hidden_dims: tuple = (256, 128, 64)
    dropout: float = 0.2


class ConformationalPrior(nn.Module):
    """Predicts P(DFG-out | pocket_sequence) from sequence alone.

    Architecture: Embedding → flatten → MLP → sigmoid.

    This captures the evolutionary landscape: kinases whose sequences
    are enriched in residues that stabilise DFG-out (e.g. bulky
    gatekeeper) will have higher propensity scores.
    """

    def __init__(self, config: ConformationalPriorConfig | None = None):
        super().__init__()
        if config is None:
            config = ConformationalPriorConfig()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        layers = []
        in_dim = config.pocket_size * config.embed_dim
        for h in config.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, pocket_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        pocket_idx : Tensor [batch, 85]
            Integer-encoded pocket sequence.

        Returns
        -------
        Tensor [batch]
            Logits for P(DFG-out).  Apply sigmoid for probability.
        """
        emb = self.embedding(pocket_idx)          # [batch, 85, embed_dim]
        flat = emb.reshape(emb.size(0), -1)       # [batch, 85 * embed_dim]
        return self.mlp(flat).squeeze(-1)          # [batch]

    def predict(self, pocket_sequence: str) -> float:
        """Convenience: sequence string → P(DFG-out) probability."""
        idx = encode_pocket(pocket_sequence)
        idx_t = torch.tensor(idx, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logit = self.forward(idx_t.to(next(self.parameters()).device))
        return torch.sigmoid(logit).item()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config.__dict__,
        }, path)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> 'ConformationalPrior':
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        config = ConformationalPriorConfig(**ckpt['config'])
        model = cls(config)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return model


def encode_pocket(sequence: str) -> np.ndarray:
    """Encode an 85-char KLIFS pocket string to integer array."""
    seq = sequence[:85].ljust(85, '-')
    return np.array([AA_TO_IDX.get(aa.upper(), 21) for aa in seq], dtype=np.int64)
