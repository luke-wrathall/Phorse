"""Lazy, process-wide singletons for the frozen CLIP encoder and the head.

Loading CLIP is slow (seconds) and the head checkpoint should be read once,
not per request -- same reasoning as webui/app.py's STATE dict, just scoped
to the server package. First call to any function here triggers the load.

If no trained checkpoint exists at config HEAD_PATH, an untrained (random)
head is used and `head_loaded()` returns False so the API/UI can warn that
scores are meaningless until a retrain has run.

The head is hot-reloaded when the checkpoint file changes on disk: the
retrain scheduler (server.scheduler) runs in a separate process and writes
a new HEAD_PATH, and `score_embedding` picks it up via an mtime check.
`reload_head()` forces the same, and is what an in-process retrain calls.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch
from flask import current_app
from PIL import Image

from horse.embed import get_embedder
from horse.model import DEFAULT_HIDDEN_DIM, HorseHead

_lock = threading.Lock()
_state: dict = {"embedder": None, "head": None, "head_loaded": False, "head_mtime": None}


def _head_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _load_head(path: Path) -> None:
    """Set _state['head'] from `path` if it exists, else an untrained head.

    A failed load (e.g. a torch.save half-written by the scheduler process)
    leaves the currently held head in place and retries on the next mtime tick.
    """
    if path.exists():
        try:
            _state["head"] = HorseHead.load(path)
        except Exception:  # noqa: BLE001 -- transient partial read; keep the old head
            return
        _state["head_loaded"] = True
        _state["head_mtime"] = _head_mtime(path)
    elif _state["head"] is None:
        head = HorseHead(hidden_dim=current_app.config.get("HEAD_HIDDEN", DEFAULT_HIDDEN_DIM))
        head.eval()
        _state["head"] = head
        _state["head_loaded"] = False
        _state["head_mtime"] = None


def _ensure_loaded() -> dict:
    if _state["embedder"] is None:
        with _lock:
            if _state["embedder"] is None:
                _load_head(Path(current_app.config["HEAD_PATH"]))
                _state["embedder"] = get_embedder()
    return _state


def warm_up(app) -> threading.Thread:
    """Load CLIP + the head in a background thread so the first upload
    doesn't pay the ~25s cold start. A request that arrives mid-load just
    waits on `_lock` in `_ensure_loaded` rather than loading twice."""

    def _load() -> None:
        with app.app_context():
            _ensure_loaded()

    thread = threading.Thread(target=_load, name="model-warm-up", daemon=True)
    thread.start()
    return thread


def _maybe_reload_head() -> None:
    """Reload the head if HEAD_PATH is newer than the copy we hold."""
    path = Path(current_app.config["HEAD_PATH"])
    current = _head_mtime(path)
    if current is not None and current != _state["head_mtime"]:
        with _lock:
            if current != _state["head_mtime"]:
                _load_head(path)


def reload_head() -> bool:
    """Force a reload of the head checkpoint. Returns head_loaded()."""
    _ensure_loaded()
    with _lock:
        _load_head(Path(current_app.config["HEAD_PATH"]))
    return _state["head_loaded"]


def embed_image(image: Image.Image) -> np.ndarray:
    return _ensure_loaded()["embedder"].embed_image(image)


def score_embedding(embedding: np.ndarray) -> float:
    _ensure_loaded()
    _maybe_reload_head()
    head = _state["head"]
    tensor = torch.from_numpy(np.array(embedding, dtype=np.float32))  # copy: frombuffer is read-only
    with torch.no_grad():
        return head(tensor.unsqueeze(0)).item()


def head_loaded() -> bool:
    return _ensure_loaded()["head_loaded"]
