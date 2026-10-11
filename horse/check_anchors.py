"""Sanity-check what an anchor set teaches the head, before it goes live.

For each anchor cache given, trains an anchors-only head from scratch
(averaged over a few seeds) and reports:

- gate pass rates on held-out PACS images (rows NOT used as anchors),
  per style, for horses and non-horses separately;
- scores for every image in a folder (default tests/test_horse/);
- scores for uploads already stored in the app db (cached embeddings).

Compare the live cache against a candidate:

    python -m horse.check_anchors --anchor-cache horse/data/anchor_cache.npz \\
        --anchor-cache path/to/candidate.npz
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import torch

from horse.anchors import DEFAULT_CACHE_PATH, AnchorSet, PACSAnchorSource
from horse.embed import get_embedder
from horse.train import train_head

GATE_THRESHOLD = 2.5
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _anchors_only_heads(anchor_set: AnchorSet, seeds: int) -> list:
    empty = np.zeros((0, anchor_set.embeddings.shape[1]), dtype=np.float32)
    heads = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        head, _ = train_head(
            anchor_set.embeddings, anchor_set.targets, empty,
            np.zeros(0, np.float32), np.zeros(0, np.float32),
        )
        heads.append(head)
    return heads


def _score(heads: list, embeddings: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(embeddings, dtype=np.float32))
    with torch.no_grad():
        return np.mean([h(x).numpy().ravel() for h in heads], axis=0)


def _pacs_holdout(embedder, per_group: int) -> tuple[np.ndarray, list[tuple[str, bool]]]:
    """Up to `per_group` held-out images per (domain, is_horse), embedded."""
    counts: dict[tuple[str, bool], int] = {}
    embeddings, groups = [], []
    for _id, image, target, domain in PACSAnchorSource().held_out():
        key = (domain, target > 0)
        if counts.get(key, 0) >= per_group:
            continue
        counts[key] = counts.get(key, 0) + 1
        embeddings.append(embedder.embed_image(image))
        groups.append(key)
    return np.stack(embeddings), groups


def _folder(embedder, folder: Path) -> tuple[np.ndarray, list[str]]:
    from PIL import Image

    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        return np.zeros((0, 512), np.float32), []
    return np.stack([embedder.embed_path(p) for p in paths]), [p.name for p in paths]


def _uploads(app_db: Path) -> tuple[np.ndarray, list[str]]:
    conn = sqlite3.connect(app_db)
    rows = conn.execute("SELECT id, caption, embedding FROM images ORDER BY created_at").fetchall()
    conn.close()
    rows = [r for r in rows if r[2] and len(r[2]) == 512 * 4]
    if not rows:
        return np.zeros((0, 512), np.float32), []
    names = [f"{r[0][:8]} {r[1]!r}" if r[1] else r[0][:8] for r in rows]
    return np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows]), names


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check anchor sets.")
    parser.add_argument("--anchor-cache", action="append", default=None,
                        help="Anchor cache (.npz). Repeat to compare several.")
    parser.add_argument("--images", default="tests/test_horse", help="Folder of images to score.")
    parser.add_argument("--app-db", default="server/data/app.db", help="Score stored uploads too.")
    parser.add_argument("--pacs-holdout", type=int, default=40,
                        help="Held-out PACS images per (style, horse/not). 0 skips.")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    caches = args.anchor_cache or [str(DEFAULT_CACHE_PATH)]
    sets = {Path(c).name: AnchorSet.load(c) for c in caches}
    heads = {name: _anchors_only_heads(s, args.seeds) for name, s in sets.items()}
    names = list(sets)
    embedder = get_embedder()

    def header(title: str) -> None:
        print(f"\n== {title}\n{'':38}" + "".join(f"{n[:22]:>24}" for n in names))

    if args.pacs_holdout:
        emb, groups = _pacs_holdout(embedder, args.pacs_holdout)
        header(f"held-out PACS: % passing the gate (>= {GATE_THRESHOLD}) -- want horse high, not-horse low")
        scores = {n: _score(heads[n], emb) for n in names}
        for key in sorted(set(groups)):
            mask = np.array([g == key for g in groups])
            label = f"{key[0]} {'horse' if key[1] else 'not-horse'} (n={mask.sum()})"
            cells = "".join(f"{100 * np.mean(scores[n][mask] >= GATE_THRESHOLD):>23.0f}%" for n in names)
            print(f"{label:38}{cells}")

    for title, (emb, labels) in (
        (f"images in {args.images}", _folder(embedder, Path(args.images)) if Path(args.images).is_dir() else (None, [])),
        (f"uploads in {args.app_db}", _uploads(Path(args.app_db)) if Path(args.app_db).exists() else (None, [])),
    ):
        if not labels:
            continue
        header(f"{title}: score (gate {GATE_THRESHOLD})")
        scores = {n: _score(heads[n], emb) for n in names}
        for i, label in enumerate(labels):
            print(f"{label[:38]:38}" + "".join(
                f"{scores[n][i]:>20.2f} {'ok' if scores[n][i] >= GATE_THRESHOLD else '--'}" for n in names))


if __name__ == "__main__":
    main()
