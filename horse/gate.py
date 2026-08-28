"""Gate decision for Stage 2: score >= threshold -> pass / reject.

Deliberately thin: there is no separate gate model. CLAUDE.md's "System
architecture" decided against a decoupled gate + rating pair specifically
so that community-shaped culture and the actual rules of entry can't
diverge -- this reads the exact same head/score that feeds the rating
signal (score.py), and just thresholds it.

The threshold is a fixed constant for MVP (the "leaning toward starting
with a fixed threshold" open question in CLAUDE.md) -- revisit once real
score distributions exist and normalizing against them starts to matter.

Run:

    python -m horse.gate path/to/image.jpg [more images...] [--head path] [--threshold 2.5]
"""

from __future__ import annotations

import argparse

from horse.score import score_images

# Midpoint of the 0-5 "horse out of 5" scale ratings and anchors use.
DEFAULT_THRESHOLD = 2.5


def passes_gate(score: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    return score >= threshold


def gate_images(
    paths: list[str],
    head_path: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, tuple[float, bool]]:
    """Score each image and apply the gate. Returns {path: (score, passed)}."""
    scores = score_images(paths, head_path=head_path)
    return {path: (score, passes_gate(score, threshold)) for path, score in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score images and apply the Stage 2 gate.")
    parser.add_argument("images", nargs="+", help="Path(s) to image file(s).")
    parser.add_argument("--head", default=None, help="Path to a trained head checkpoint (.pt).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    if args.head is None:
        print("[warn] no --head checkpoint given -- using an untrained (random) head; scores are meaningless.\n")

    results = gate_images(args.images, head_path=args.head, threshold=args.threshold)
    for path, (score, passed) in results.items():
        verdict = "PASS" if passed else "REJECT"
        print(f"{score:8.4f}  {verdict:6s}  {path}")


if __name__ == "__main__":
    main()
