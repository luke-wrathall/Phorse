"""One retrain cycle, orchestrated: ingest -> train -> reload -> re-score.

This is what closes the drift loop. Called on an interval by
server.scheduler (a separate process) and on demand by `python -m
server.retrain`. Must run inside a Flask app context (it reads config and
the app database).

Steps:
  1. horse.ingest -- copy app ratings + embeddings into the training store.
  2. horse.train  -- retrain the head from anchors + those ratings,
                     warm-started from the current head.pt.
  3. model.reload_head -- swap the new checkpoint into this process.
  4. re-score every stored image from its cached embedding so the feed
     reflects the drift. The gate decision (`passed`) is NOT re-evaluated
     -- admission is a point-in-time event (re-gating is an open question,
     see CLAUDE.md).

One cycle at a time, across processes: a cycle holds an OS file lock
(`<HEAD_PATH>.lock`), so a manual `server.retrain` started while the
scheduler is mid-cycle raises RetrainBusy instead of training alongside
it and racing to write head.pt. The OS drops the lock if the process
dies, so a crash never leaves it stuck. One machine is all this covers --
retraining across several would need a real job queue.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from flask import current_app

from horse.embed import EMBEDDING_DIM
from horse.ingest import ingest
from horse.train import train_from_store

from ..db import get_db
from . import model

log = logging.getLogger(__name__)


class RetrainBusy(Exception):
    """Another retrain cycle holds the lock."""


@contextmanager
def _exclusive(lock_path: Path):
    """Hold a non-blocking exclusive lock on `lock_path`, or raise RetrainBusy."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as f:
        try:
            if os.name == "nt":
                import msvcrt

                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RetrainBusy("a retrain is already running") from None
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _rescore_images() -> int:
    """Recompute images.score for every image using the current head."""
    db = get_db()
    rows = db.execute("SELECT id, embedding FROM images").fetchall()
    updates = []
    for row in rows:
        embedding = np.frombuffer(row["embedding"], dtype=np.float32)
        if embedding.shape != (EMBEDDING_DIM,):
            continue
        updates.append((float(model.score_embedding(embedding)), row["id"]))
    db.executemany("UPDATE images SET score = ? WHERE id = ?", updates)
    db.commit()
    return len(updates)


def run_retrain(fresh: bool = False) -> dict:
    """Run one full retrain cycle. Returns a summary dict.
    Raises RetrainBusy if another process is mid-cycle.

    `fresh` trains from scratch instead of warm-starting from head.pt --
    for after an anchor-set change, which the prox pull would otherwise
    let in only slowly. Every vote is still trained on, so the community's
    say carries over; the next scheduled cycle warm-starts from the result."""
    with _exclusive(Path(f"{current_app.config['HEAD_PATH']}.lock")):
        return _run_retrain(fresh)


def _run_retrain(fresh: bool = False) -> dict:
    cfg = current_app.config
    started = time.time()

    ingested = ingest(cfg["DB_PATH"], cfg["TRAINING_DB"])

    head_path = cfg["HEAD_PATH"]
    train_summary = train_from_store(
        cfg["TRAINING_DB"],
        cfg["ANCHOR_CACHE"],
        head_path,
        head_in=head_path if Path(head_path).exists() and not fresh else None,
        lambda_anchor=cfg["LAMBDA_ANCHOR"],
        lambda_decay=cfg["LAMBDA_DECAY"],
        community_weight_floor=cfg["COMMUNITY_WEIGHT_FLOOR"],
        epochs=cfg["RETRAIN_EPOCHS"],
        fresh_epochs=cfg["RETRAIN_FRESH_EPOCHS"],
        hidden_dim=cfg["HEAD_HIDDEN"],
        prox_weight=cfg["RETRAIN_PROX_WEIGHT"],
    )

    model.reload_head()
    rescored = _rescore_images()

    summary = {
        "ratings_synced": ingested["ratings_synced"],
        "images_rescored": rescored,
        "anchor_loss": train_summary["anchor_loss"],
        "community_loss": train_summary["community_loss"],
        "final_loss": train_summary["final_loss"],
        "fresh": train_summary["fresh"],
        "seconds": round(time.time() - started, 2),
    }
    log.info("retrain complete: %s", summary)
    return summary
