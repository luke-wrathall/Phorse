"""End-to-end scoring: image -> CLIP embedding -> head -> horseness score.

This is the smallest working slice of the Stage 2 pipeline — no training
loop, no storage, no Stage 1 precheck. It exists to prove embed -> head ->
score runs end to end before anything else gets layered on. Run directly:

    python -m horse.score path/to/image.jpg [more images...]

Without a trained checkpoint (--head), the head is randomly initialized,
so scores are meaningless numbers — this only proves the pipeline runs.
"""

from __future__ import annotations

import argparse

import torch

from horse.embed import ClipEmbedder
from horse.model import HorseHead


def score_images(paths: list[str], head_path: str | None = None) -> dict[str, float]:
    """Embed each image and run it through the head. Loads CLIP once, reused across images."""
    embedder = ClipEmbedder()
    head = HorseHead.load(head_path) if head_path else HorseHead()
    head.eval()

    scores: dict[str, float] = {}
    for path in paths:
        embedding = embedder.embed_path(path)
        with torch.no_grad():
            score = head(torch.from_numpy(embedding).unsqueeze(0)).item()
        scores[path] = score
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Score images for horseness (Stage 2 MVP).")
    parser.add_argument("images", nargs="+", help="Path(s) to image file(s).")
    parser.add_argument(
        "--head",
        default=None,
        help="Path to a trained head checkpoint (.pt). Omit for an untrained (random) head.",
    )
    args = parser.parse_args()

    if args.head is None:
        print("[warn] no --head checkpoint given — using an untrained (random) head; scores are meaningless.\n")

    scores = score_images(args.images, head_path=args.head)
    for path, score in scores.items():
        print(f"{score:8.4f}  {path}")


if __name__ == "__main__":
    main()
