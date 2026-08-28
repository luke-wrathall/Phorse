"""CLIP-based image embedding extractor (frozen backbone).

Wraps OpenCLIP to turn an image into a fixed-length embedding vector.
The encoder is frozen — it is never trained. Only the head in model.py
learns, on top of embeddings produced here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open_clip
import torch
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model.eval()
        model.requires_grad_(False)  # frozen: gradients never flow into the backbone
        self.model = model.to(self.device)
        self.preprocess = preprocess

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed a single PIL image -> (EMBEDDING_DIM,) float32, L2-normalized."""
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_path(self, path: str | Path) -> np.ndarray:
        """Load an image file from disk and embed it."""
        image = Image.open(path).convert("RGB")
        return self.embed_image(image)
