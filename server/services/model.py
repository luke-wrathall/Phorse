"""Lazy, process-wide singletons for the frozen CLIP encoder and the head.

Loading CLIP is slow (seconds) and the head checkpoint should be read once,
not per request -- same reasoning as webui/app.py's STATE dict, just scoped
to the server package. First call to any function here triggers the load.

If no trained checkpoint exists at config HEAD_PATH, an untrained (random)
head is used and `head_loaded()` returns False so the API/UI can warn that
scores are meaningless until `python -m horse.train` has run.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch
from flask import current_app
from PIL import Image

from horse.embed import ClipEmbedder
from horse.model import HorseHead

_lock = threading.Lock()
_state: dict = {"embedder": None, "head": None, "head_loaded": False}


def _ensure_loaded() -> dict:
    if _state["embedder"] is None:
        with _lock:
            if _state["embedder"] is None:
                head_path = Path(current_app.config["HEAD_PATH"])
                if head_path.exists():
                    _state["head"] = HorseHead.load(head_path)
                    _state["head_loaded"] = True
                else:
                    head = HorseHead()
                    head.eval()
                    _state["head"] = head
                _state["embedder"] = ClipEmbedder()
    return _state


def embed_image(image: Image.Image) -> np.ndarray:
    return _ensure_loaded()["embedder"].embed_image(image)


def score_embedding(embedding: np.ndarray) -> float:
    head = _ensure_loaded()["head"]
    with torch.no_grad():
        return head(torch.from_numpy(embedding).unsqueeze(0)).item()


def head_loaded() -> bool:
    return _ensure_loaded()["head_loaded"]
