"""Anchor (seed) set for the drift head's anchor loss.

The anchor set is fixed reference data included in every retrain with a
small but constant weight (lambda_anchor) -- it's what keeps the drift
head from collapsing no matter how far community ratings wander. The
operator controls this set directly; the community never rates or edits
it (see CLAUDE.md "Training / drift mechanism").

It's built from one or more AnchorSource objects, each of which yields
(example_id, image, target_score) triples on a "horse out of 5" scale.
Swapping in a new source -- e.g. a horse2zebra set of real photos with
zebras as a thematic hard negative -- means adding one AnchorSource
subclass and listing it in DEFAULT_SOURCES; nothing else here or in
train.py needs to change.

Run directly to build (or rebuild) the cached anchor embeddings:

    python -m horse.anchors [--cache path] [--rebuild]
"""

from __future__ import annotations

import abc
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from PIL import Image

from horse.embed import ClipEmbedder, get_embedder

# Target scores on the same 0-5 "horse out of 5" scale community ratings use.
HORSE_TARGET = 5.0
NOT_HORSE_TARGET = 0.0

DEFAULT_CACHE_PATH = Path("horse/data/anchor_cache.npz")


class AnchorSource(abc.ABC):
    """Something that yields labeled (example_id, image, target_score) anchor examples.

    Subclass this to add a new anchor source (e.g. a curated image
    manifest, or horse2zebra). Each yielded image is embedded once by
    build_anchor_set() and never re-fetched after that.
    """

    name: str

    @abc.abstractmethod
    def __iter__(self) -> Iterator[tuple[str, Image.Image, float]]:
        """Yield (example_id, RGB image, target_score) triples."""


class CIFAR10AnchorSource(AnchorSource):
    """CIFAR-10's `horse` class as positives, a capped sample of the other
    nine classes as negatives.

    Auto-downloads via torchvision on first use (cached under `root`).
    Images are 32x32, upsampled by CLIP's preprocessing to 224x224 --
    fine for a small fixed anchor set, not a source of realism. That's
    what a future real-photo source (horse2zebra) would improve on.
    """

    name = "cifar10"
    HORSE_LABEL = 7  # torchvision CIFAR10 class ordering: horse is index 7

    def __init__(
        self,
        root: str | Path = "horse/data/cifar10",
        n_horse: int = 200,
        n_negative: int = 200,
        train: bool = True,
    ):
        self.root = root
        self.n_horse = n_horse
        self.n_negative = n_negative
        self.train = train

    def __iter__(self) -> Iterator[tuple[str, Image.Image, float]]:
        import torchvision  # imported lazily: only needed when building anchors

        dataset = torchvision.datasets.CIFAR10(
            root=str(self.root), train=self.train, download=True
        )
        horse_count = 0
        negative_count = 0
        for idx, (image, label) in enumerate(dataset):
            if horse_count >= self.n_horse and negative_count >= self.n_negative:
                break
            if label == self.HORSE_LABEL:
                if horse_count >= self.n_horse:
                    continue
                horse_count += 1
                yield f"cifar10-horse-{idx}", image.convert("RGB"), HORSE_TARGET
            else:
                if negative_count >= self.n_negative:
                    continue
                negative_count += 1
                yield f"cifar10-neg-{idx}", image.convert("RGB"), NOT_HORSE_TARGET


# Future thematic hard-negative source: real horse photos vs. zebra photos
# (visually horse-like but not a horse -- a good stress test for the
# boundary). To add it: implement
#
#     class Horse2ZebraAnchorSource(AnchorSource): ...
#
# (download the CycleGAN horse2zebra zip, yield HORSE_TARGET for trainA,
# NOT_HORSE_TARGET for trainB) and append an instance to DEFAULT_SOURCES.
# build_anchor_set() and train.py are already source-agnostic.
DEFAULT_SOURCES: list[AnchorSource] = [CIFAR10AnchorSource()]


@dataclass
class AnchorSet:
    """Pre-embedded anchor examples, ready to feed straight into the head."""

    ids: list[str]
    embeddings: np.ndarray  # (N, EMBEDDING_DIM) float32
    targets: np.ndarray  # (N,) float32

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=np.array(self.ids), embeddings=self.embeddings, targets=self.targets)

    @classmethod
    def load(cls, path: str | Path) -> "AnchorSet":
        data = np.load(path, allow_pickle=False)
        return cls(
            ids=list(data["ids"]),
            embeddings=data["embeddings"].astype(np.float32),
            targets=data["targets"].astype(np.float32),
        )


def build_anchor_set(
    sources: Iterable[AnchorSource] | None = None,
    embedder: ClipEmbedder | None = None,
) -> AnchorSet:
    """Embed every image from every source into one fixed AnchorSet."""
    sources = list(sources) if sources is not None else DEFAULT_SOURCES
    embedder = embedder or get_embedder()

    ids: list[str] = []
    embeddings: list[np.ndarray] = []
    targets: list[float] = []
    for source in sources:
        for example_id, image, target in source:
            ids.append(example_id)
            embeddings.append(embedder.embed_image(image))
            targets.append(target)

    if not ids:
        raise ValueError("no anchor examples produced by any AnchorSource")

    return AnchorSet(
        ids=ids,
        embeddings=np.stack(embeddings).astype(np.float32),
        targets=np.array(targets, dtype=np.float32),
    )


def load_or_build_anchor_set(
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    sources: Iterable[AnchorSource] | None = None,
    embedder: ClipEmbedder | None = None,
    force_rebuild: bool = False,
) -> AnchorSet:
    """Load the cached anchor embeddings, or build (and cache) them if missing.

    The anchor set only needs re-embedding when the sources change (new
    curated images, a new dataset) -- not on every retrain -- so it's
    cached to disk separately from the per-cycle head training.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_rebuild:
        return AnchorSet.load(cache_path)

    anchor_set = build_anchor_set(sources=sources, embedder=embedder)
    anchor_set.save(cache_path)
    return anchor_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Build (or rebuild) the cached anchor set.")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH), help="Path to the anchor cache (.npz).")
    parser.add_argument("--rebuild", action="store_true", help="Force re-embedding even if the cache exists.")
    args = parser.parse_args()

    anchor_set = load_or_build_anchor_set(cache_path=args.cache, force_rebuild=args.rebuild)
    n_horse = int(np.sum(anchor_set.targets == HORSE_TARGET))
    n_negative = len(anchor_set.ids) - n_horse
    print(f"[anchors] {len(anchor_set.ids)} examples ({n_horse} horse, {n_negative} negative) -> {args.cache}")


if __name__ == "__main__":
    main()
