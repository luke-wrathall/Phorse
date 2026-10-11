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
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from flask import current_app

from horse.embed import EMBEDDING_DIM
from horse.ingest import ingest
from horse.train import train_from_store

from ..db import get_db
from . import model

log = logging.getLogger(__name__)


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


def run_retrain() -> dict:
    """Run one full retrain cycle. Returns a summary dict."""
    cfg = current_app.config
    started = time.time()

    ingested = ingest(cfg["DB_PATH"], cfg["TRAINING_DB"])

    head_path = cfg["HEAD_PATH"]
    train_summary = train_from_store(
        cfg["TRAINING_DB"],
        cfg["ANCHOR_CACHE"],
        head_path,
        head_in=head_path if Path(head_path).exists() else None,
        lambda_anchor=cfg["LAMBDA_ANCHOR"],
        lambda_decay=cfg["LAMBDA_DECAY"],
        community_weight_floor=cfg["COMMUNITY_WEIGHT_FLOOR"],
    )

    model.reload_head()
    rescored = _rescore_images()

    summary = {
        "ratings_synced": ingested["ratings_synced"],
        "images_rescored": rescored,
        "anchor_loss": train_summary["anchor_loss"],
        "community_loss": train_summary["community_loss"],
        "final_loss": train_summary["final_loss"],
        "seconds": round(time.time() - started, 2),
    }
    log.info("retrain complete: %s", summary)
    return summary
