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


class PACSAnchorSource(AnchorSource):
    """PACS (Photo / Art painting / Cartoon / Sketch): horses as positives,
    the other six classes (dog, elephant, giraffe, guitar, house, person) as
    negatives -- sampled per domain so every style has matched negatives.

    This is what lets *drawn* horses through the gate: without it the head
    only ever saw photos and learned "photo of a horse", so users with no
    horse to photograph had no way in. Negatives must share each style, or
    the head would learn "drawing = horse" and pass any doodle.

    Fetched from the Hugging Face Hub (`flwrlabs/pacs`, one ~190 MB parquet,
    cached by huggingface_hub). Needs `pyarrow` (only when building anchors).
    Sampling is seeded, so rebuilds are reproducible and `held_out()` yields
    the rows NOT used as anchors, for validation.
    """

    name = "pacs"
    REPO_ID = "flwrlabs/pacs"
    FILENAME = "data/train-00000-of-00001.parquet"
    HORSE_LABEL = 4
    DOMAINS = ("photo", "art_painting", "cartoon", "sketch")

    def __init__(
        self,
        domains: Iterable[str] = DOMAINS,
        n_horse_per_domain: int = 100,
        n_negative_per_domain: int = 100,
        seed: int = 0,
    ):
        self.domains = tuple(domains)
        self.n_horse = n_horse_per_domain
        self.n_negative = n_negative_per_domain
        self.seed = seed

    def _table(self):
        import pyarrow.parquet as pq  # imported lazily: only needed when building anchors
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(self.REPO_ID, self.FILENAME, repo_type="dataset")
        return pq.read_table(path)

    def _split(self, domains: list[str], labels: list[int]) -> set[int]:
        """Row indices chosen as anchors: per domain, n horses + n negatives
        spread round-robin across the non-horse classes."""
        rng = np.random.default_rng(self.seed)
        chosen: set[int] = set()
        for domain in self.domains:
            by_label: dict[int, list[int]] = {}
            for i, (d, label) in enumerate(zip(domains, labels)):
                if d == domain:
                    by_label.setdefault(label, []).append(i)
            for rows in by_label.values():
                rng.shuffle(rows)
            chosen.update(by_label.get(self.HORSE_LABEL, [])[: self.n_horse])
            negative_labels = sorted(label for label in by_label if label != self.HORSE_LABEL)
            taken = 0
            depth = 0
            while taken < self.n_negative and any(len(by_label[l]) > depth for l in negative_labels):
                for label in negative_labels:
                    if taken < self.n_negative and len(by_label[label]) > depth:
                        chosen.add(by_label[label][depth])
                        taken += 1
                depth += 1
        return chosen

    def _rows(self, anchors: bool) -> Iterator[tuple[str, Image.Image, float, str]]:
        import io

        table = self._table()
        domains = table.column("domain").to_pylist()
        labels = table.column("label").to_pylist()
        chosen = self._split(domains, labels)
        images = table.column("image")
        for i, (domain, label) in enumerate(zip(domains, labels)):
            if domain not in self.domains or (i in chosen) != anchors:
                continue
            image = Image.open(io.BytesIO(images[i].as_py()["bytes"])).convert("RGB")
            target = HORSE_TARGET if label == self.HORSE_LABEL else NOT_HORSE_TARGET
            yield f"pacs-{domain}-{label}-{i}", image, target, domain

    def __iter__(self) -> Iterator[tuple[str, Image.Image, float]]:
        for example_id, image, target, _domain in self._rows(anchors=True):
            yield example_id, image, target

    def held_out(self) -> Iterator[tuple[str, Image.Image, float, str]]:
        """(id, image, target, domain) for every row not used as an anchor."""
        return self._rows(anchors=False)


class QuickDrawAnchorSource(AnchorSource):
    """Google Quick, Draw! doodles: horses as positives, other doodles as
    negatives -- the MS-Paint / biro-scribble style PACS's sketches don't
    cover (real uploads like that scored 1.4-2.3 against a 2.5 gate).

    Negatives are mostly other four-legged animals, so the head has to
    learn "horse-shaped" rather than "doodle of an animal", plus a few
    everyday doodles (the gate let no car through, but nothing taught it
    not to). Zebra and donkey are deliberately left out: whether they count
    is the community's call, not the anchor set's.

    Drawings are stroke data (0-255 box), rendered here at 256 px with a
    seeded random pen width, black on white. Only drawings the game itself
    recognized are used. Each category file is ~100 MB, so only the first
    `max_bytes` are fetched (an HTTP range request) and cached under `root`.
    Dataset licence: CC BY 4.0. `held_out()` yields the cached drawings not
    used as anchors, for validation.
    """

    name = "quickdraw"
    URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"
    NEGATIVES = (
        "cow", "dog", "cat", "pig", "sheep", "camel", "giraffe", "elephant", "lion",
        "tiger", "bear", "kangaroo", "rabbit", "squirrel", "raccoon", "mouse", "dragon",
        "bird", "car", "house", "tree", "bicycle",
    )
    PEN_WIDTHS = (2, 3, 4, 6, 8)

    def __init__(
        self,
        root: str | Path = "horse/data/quickdraw",
        n_horse: int = 200,
        n_negative: int = 200,
        seed: int = 0,
        max_bytes: int = 1_000_000,
    ):
        self.root = Path(root)
        self.n_horse = n_horse
        self.n_negative = n_negative
        self.seed = seed
        self.max_bytes = max_bytes

    def _drawings(self, category: str) -> list[dict]:
        import json
        import urllib.parse
        import urllib.request

        path = self.root / f"{category}.ndjson"
        if not path.exists():
            request = urllib.request.Request(
                self.URL.format(urllib.parse.quote(category)),
                headers={"Range": f"bytes=0-{self.max_bytes - 1}"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data[: data.rfind(b"\n") + 1])  # drop the cut-off last line
        lines = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        return [d for d in lines if d.get("recognized")]

    @classmethod
    def render(cls, strokes: list, pen: int) -> Image.Image:
        from PIL import ImageDraw

        image = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(image)
        for xs, ys in strokes:
            # the drawing fills 0-255; shrink it into a margin so thick pens aren't clipped
            points = [(16 + x * 224 / 255, 16 + y * 224 / 255) for x, y in zip(xs, ys)]
            if len(points) == 1:
                points = points * 2
            draw.line(points, fill="black", width=pen, joint="curve")
        return image

    def _rows(self, anchors: bool, per_category: int | None = None):
        rng = np.random.default_rng(self.seed)
        # negatives split evenly across categories, the first few taking the remainder
        each, extra = divmod(self.n_negative, len(self.NEGATIVES))
        plan = [("horse", self.n_horse, HORSE_TARGET)] + [
            (c, each + (k < extra), NOT_HORSE_TARGET) for k, c in enumerate(self.NEGATIVES)
        ]
        for category, n, target in plan:
            drawings = self._drawings(category)
            order = rng.permutation(len(drawings))
            chosen = set(order[:n].tolist())
            rows = [i for i in range(len(drawings)) if (i in chosen) == anchors]
            for i in rows[:per_category]:
                drawing = drawings[i]
                pen = self.PEN_WIDTHS[int(drawing["key_id"]) % len(self.PEN_WIDTHS)]
                yield (f"quickdraw-{category.replace(' ', '_')}-{drawing['key_id']}",
                       self.render(drawing["drawing"], pen), target, "doodle")

    def __iter__(self) -> Iterator[tuple[str, Image.Image, float]]:
        for example_id, image, target, _style in self._rows(anchors=True):
            yield example_id, image, target

    def held_out(self, per_category: int | None = None) -> Iterator[tuple[str, Image.Image, float, str]]:
        """(id, image, target, "doodle") for cached drawings not used as
        anchors, at most `per_category` from each category."""
        return self._rows(anchors=False, per_category=per_category)


class CuratedFolderAnchorSource(AnchorSource):
    """Operator-curated anchors from a folder -- the manual escape hatch.

        horse/data/anchors_curated/
            horse/        -> target 5
            not_horse/    -> target 0
            3.5/          -> any folder named as a number is that target

    A missing folder yields nothing, so this is safe to leave in
    DEFAULT_SOURCES. Rebuild the cache after changing it:
    `python -m horse.anchors --rebuild`.
    """

    name = "curated"
    EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

    def __init__(self, root: str | Path = "horse/data/anchors_curated"):
        self.root = Path(root)

    @staticmethod
    def _target(folder: str) -> float | None:
        if folder == "horse":
            return HORSE_TARGET
        if folder == "not_horse":
            return NOT_HORSE_TARGET
        try:
            return float(folder)
        except ValueError:
            return None

    def __iter__(self) -> Iterator[tuple[str, Image.Image, float]]:
        if not self.root.is_dir():
            return
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir()):
            target = self._target(folder.name)
            if target is None:
                continue
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in self.EXTENSIONS:
                    yield f"curated-{folder.name}-{path.name}", Image.open(path).convert("RGB"), target


# Future thematic hard-negative source: real horse photos vs. zebra photos
# (visually horse-like but not a horse -- a good stress test for the
# boundary). To add it: implement
#
#     class Horse2ZebraAnchorSource(AnchorSource): ...
#
# (download the CycleGAN horse2zebra zip, yield HORSE_TARGET for trainA,
# NOT_HORSE_TARGET for trainB) and append an instance to DEFAULT_SOURCES.
# build_anchor_set() and train.py are already source-agnostic.
DEFAULT_SOURCES: list[AnchorSource] = [
    CIFAR10AnchorSource(),
    PACSAnchorSource(),
    QuickDrawAnchorSource(),
    CuratedFolderAnchorSource(),
]


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
