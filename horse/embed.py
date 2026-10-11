"""CLIP-based image embedding extractor (frozen backbone).

Wraps OpenCLIP to turn an image into a fixed-length embedding vector.
The encoder is frozen — it is never trained. Only the head in model.py
learns, on top of embeddings produced here.

Use ``get_embedder()`` rather than constructing ``ClipEmbedder`` directly:
when ``HORSE_FAKE_EMBEDDER`` is set it returns a deterministic stand-in
(``FakeEmbedder``) that needs no model download. Tests rely on this, and
it also lets the whole app run locally without pulling ~600 MB of CLIP
weights.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image

# ViT-B-32 gives a 512-d embedding and is cheap enough to run on CPU,
# which matters since this runs per-upload, not just at training time.
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
EMBEDDING_DIM = 512


class ClipEmbedder:
    """Loads a frozen CLIP image encoder and embeds images with it."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        pretrained: str = PRETRAINED,
        device: str | None = None,
    ):
        import open_clip  # heavy imports kept out of module import time
        import torch

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model.eval()
        model.requires_grad_(False)  # frozen: gradients never flow into the backbone
        self.model = model.to(self.device)
        self.preprocess = preprocess

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed a single PIL image -> (EMBEDDING_DIM,) float32, L2-normalized."""
        with self._torch.no_grad():
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_path(self, path: str | Path) -> np.ndarray:
        """Load an image file from disk and embed it."""
        image = Image.open(path).convert("RGB")
        return self.embed_image(image)


class FakeEmbedder:
    """Deterministic stand-in for ClipEmbedder -- no model, no download.

    Maps an image to a fixed (EMBEDDING_DIM,) L2-normalized vector by
    seeding an RNG with a hash of the pixel data. Same image -> same
    embedding, different images -> near-orthogonal embeddings, which is all
    the head training / gate / feed math needs to be exercised end to end.
    Enabled via HORSE_FAKE_EMBEDDER (see get_embedder).
    """

    def __init__(self, *_args, device: str | None = None, **_kwargs):
        self.device = device or "cpu"

    def embed_image(self, image: Image.Image) -> np.ndarray:
        digest = hashlib.sha256(image.convert("RGB").tobytes()).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def embed_path(self, path: str | Path) -> np.ndarray:
        return self.embed_image(Image.open(path).convert("RGB"))


def _use_fake_embedder() -> bool:
    return os.environ.get("HORSE_FAKE_EMBEDDER", "").lower() in {"1", "true", "yes", "on"}


def get_embedder(device: str | None = None):
    """Return a ClipEmbedder, or a FakeEmbedder when HORSE_FAKE_EMBEDDER is set."""
    if _use_fake_embedder():
        return FakeEmbedder(device=device)
    return ClipEmbedder(device=device)
