"""Trainable scoring head on top of frozen CLIP embeddings.

Small MLP: embedding_dim -> hidden -> 1, meant to be trained via
regression (MSE) to output a continuous "horseness" score — not a
binary class label. The backbone (embed.py) is never trained; only
this head gets retrained during drift cycles, which is what keeps
retraining cheap.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from horse.embed import EMBEDDING_DIM

# 8 hidden units (was 64, until 2026-10-19): small enough that votes on a few
# images move the scores of similar unseen ones rather than being memorized
# -- measured with horse.tune on the real votes. See CLAUDE.md "Model design".
DEFAULT_HIDDEN_DIM = 8


class HorseHead(nn.Module):
    """embedding -> horseness score (unbounded scalar; not squashed to [0,1])."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, hidden_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # hidden_dim 0 = a plain linear head: least able to memorize single
        # images, so votes have to move the score of everything similar.
        self.net = (
            nn.Sequential(nn.Linear(embedding_dim, 1))
            if hidden_dim == 0
            else nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding).squeeze(-1)

    def save(self, path: str | Path) -> None:
        """Write the checkpoint atomically: to a temp file beside it, then
        swapped into place. A save cut short (process killed mid-write)
        leaves the previous head.pt intact rather than a corrupt one --
        which matters because each retrain warm-starts from it."""
        tmp = Path(f"{path}.tmp")
        torch.save(
            {
                "state_dict": self.state_dict(),
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
            },
            tmp,
        )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "HorseHead":
        checkpoint = torch.load(path, map_location="cpu")
        head = cls(
            embedding_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"],
        )
        head.load_state_dict(checkpoint["state_dict"])
        head.eval()
        return head
