"""Tune the drift knobs offline, against a snapshot of real site data.

Read-only: point it at a *copy* of the server's app.db (and head.pt); it
never writes anything. Three sections:

1. the data -- votes per value, per image, per person;
2. the model now -- the snapshot head's score next to what people said,
   for every rated image, plus the uploads the gate turned away;
3. a knob sweep -- replays the retrain loop from a fresh anchors-only head
   (where the live head started) and reports after 1, 6 and 24 cycles (10
   min, 1 h, 4 h at the default interval). The first value of each knob
   list is the baseline (the live defaults); the others are tried one knob
   at a time, or every combination with --grid. Each cycle is real work
   (~1 s), so a run is a few minutes:

     fit       RMS gap between the model's score and the people's (recency-
               weighted) mean, over rated images. Lower = the community
               moved the model further.
     held-out  the same gap for images the run never saw ratings for (k-fold
               over images). Lower than `start` = rating some horses taught
               the model about others; the same as `start` = it only
               memorized the rated ones.
     anchors   % of horse anchors still passing / non-horse anchors still
               failing the gate, and the worst anchor style. The rubber band.
     unrated   mean score shift on images nobody has rated, and how many of
               them would now get the other gate decision.

Run (after copying the server's data down, see `--help`):

    python -m horse.tune --app-db tune/app.db --head tune/head.pt
    python -m horse.tune --app-db tune/app.db --lambda-anchor 0.1,0.25,1 --prox 1,3 --grid
"""

from __future__ import annotations

import argparse
import itertools
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from horse.anchors import DEFAULT_CACHE_PATH, AnchorSet
from horse.embed import EMBEDDING_DIM
from horse.model import HorseHead
from horse.train import DEFAULT_COMMUNITY_WEIGHT_FLOOR, DEFAULT_EPOCHS, recency_weights, train_head

GATE_THRESHOLD = 2.5


@dataclass
class Snapshot:
    image_ids: list[str]
    labels: list[str]          # "owner: caption" for the per-image table
    embeddings: np.ndarray     # (n_images, 512)
    passed: np.ndarray         # (n_images,) bool, the gate decision at upload
    created_at: np.ndarray     # (n_images,)
    vote_image: np.ndarray     # (n_votes,) index into image_ids
    vote_user: np.ndarray      # (n_votes,)
    vote_value: np.ndarray     # (n_votes,) float32
    vote_time: np.ndarray      # (n_votes,) unix seconds


def load_snapshot(app_db: Path) -> Snapshot:
    # mode=ro: never create or touch the file
    conn = sqlite3.connect(f"file:{app_db.as_posix()}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT i.id, u.username, i.caption, i.embedding, i.passed, i.created_at "
        "FROM images i JOIN users u ON u.id = i.owner_id ORDER BY i.created_at"
    ).fetchall()
    rows = [r for r in rows if r[3] and len(r[3]) == EMBEDDING_DIM * 4]
    index = {r[0]: n for n, r in enumerate(rows)}
    votes = [
        v for v in conn.execute("SELECT image_id, user_id, rating, created_at FROM ratings").fetchall()
        if v[0] in index
    ]
    conn.close()
    return Snapshot(
        image_ids=[r[0] for r in rows],
        labels=[f"{r[1]}: {r[2]}" if r[2] else f"{r[1]}: ({r[0][:8]})" for r in rows],
        embeddings=np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
        if rows else np.zeros((0, EMBEDDING_DIM), np.float32),
        passed=np.array([bool(r[4]) for r in rows]),
        created_at=np.array([r[5] for r in rows], dtype=np.float64),
        vote_image=np.array([index[v[0]] for v in votes], dtype=np.int64),
        vote_user=np.array([v[1] for v in votes], dtype=np.int64),
        vote_value=np.array([v[2] for v in votes], dtype=np.float32),
        vote_time=np.array([v[3] for v in votes], dtype=np.float64),
    )


def score(head: HorseHead, embeddings: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return head(torch.from_numpy(np.ascontiguousarray(embeddings))).numpy()


def image_means(snap: Snapshot, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per image: recency-weighted mean vote (nan if unrated), vote count."""
    n = len(snap.image_ids)
    wsum = np.bincount(snap.vote_image, weights=weights, minlength=n)
    vsum = np.bincount(snap.vote_image, weights=weights * snap.vote_value, minlength=n)
    count = np.bincount(snap.vote_image, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, vsum / wsum, np.nan), count


def anchor_groups(anchors: AnchorSet) -> dict[str, np.ndarray]:
    """'cifar10 horse', 'pacs-sketch not-horse', ... -> mask over anchors."""
    keys = []
    for anchor_id, target in zip(anchors.ids, anchors.targets):
        source = "-".join(anchor_id.split("-")[:2]) if anchor_id.startswith("pacs") else anchor_id.split("-")[0]
        keys.append(f"{source} {'horse' if target > 0 else 'not-horse'}")
    keys = np.array(keys)
    return {k: keys == k for k in sorted(set(keys))}


def anchors_only_head(anchors: AnchorSet, hidden_dim: int, epochs: int, seed: int = 0) -> HorseHead:
    """Where a head of this size starts: trained to convergence on the
    anchors alone. Small heads need far more than one cycle's epochs for
    this (a linear head is still near 50% anchor accuracy after 200)."""
    torch.manual_seed(seed)
    empty = np.zeros((0, anchors.embeddings.shape[1]), dtype=np.float32)
    head, _ = train_head(anchors.embeddings, anchors.targets, empty,
                         np.zeros(0, np.float32), np.zeros(0, np.float32),
                         hidden_dim=hidden_dim, epochs=epochs)
    return head


def clone(head: HorseHead) -> HorseHead:
    copy = HorseHead(head.embedding_dim, head.hidden_dim)
    copy.load_state_dict(head.state_dict())
    return copy


def run_cycles(base: HorseHead, anchors: AnchorSet, emb, targets, weights, *, cycles: list[int],
               lambda_anchor: float, prox_weight: float, floor: float, epochs: int):
    """The live loop: each cycle warm-starts from the last head and is pulled
    toward it. Yields (cycle, head) at each checkpoint in `cycles` (0 = the
    starting head)."""
    head = clone(base)
    if 0 in cycles:
        yield 0, head
    for cycle in range(1, max(cycles) + 1):
        head, _ = train_head(anchors.embeddings, anchors.targets, emb, targets, weights,
                             lambda_anchor=lambda_anchor, community_weight_floor=floor,
                             epochs=epochs, head=head, prox_weight=prox_weight)
        if cycle in cycles:
            yield cycle, head


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def print_data(snap: Snapshot, weights: np.ndarray, floor: float) -> None:
    n_votes = len(snap.vote_value)
    rated = len(set(snap.vote_image.tolist()))
    people = len(set(snap.vote_user.tolist()))
    print(f"== the data\n{len(snap.image_ids)} uploads ({snap.passed.sum()} passed the gate), "
          f"{n_votes} votes on {rated} images from {people} people")
    if not n_votes:
        return
    values, counts = np.unique(snap.vote_value, return_counts=True)
    print("votes by value:  " + "   ".join(f"{v:g}: {c}" for v, c in zip(values, counts)))
    per_image = np.bincount(np.bincount(snap.vote_image))[1:]
    print("images by votes: " + "   ".join(f"{k}: {c}" for k, c in enumerate(per_image, 1) if c))
    _, per_person = np.unique(snap.vote_user, return_counts=True)
    print(f"votes per person: median {np.median(per_person):g}, max {per_person.max()} "
          f"(top person cast {100 * per_person.max() / n_votes:.0f}% of all votes)")
    zeros = snap.vote_time[snap.vote_value == 0]
    if len(zeros):
        early_ones = int(np.sum((snap.vote_value == 1) & (snap.vote_time < zeros.min())))
        if early_ones:
            print(f"{early_ones} votes of 1 were cast before anyone voted 0 -- likely 'not a horse' "
                  f"on the old 1-5 buttons")
    else:
        print("nobody has voted 0 yet")
    age = (time.time() - snap.vote_time) / 86400
    print(f"vote age: newest {age.min():.1f} days, oldest {age.max():.1f} days; total recency weight "
          f"{weights.sum():.1f} (floor {floor:g} {'BINDS' if weights.sum() < floor else 'inert'})")


def print_now(snap: Snapshot, head: HorseHead, means, counts) -> None:
    scores = score(head, snap.embeddings)
    rated = counts > 0
    print(f"\n== the model now (snapshot head) vs what people said")
    if rated.any():
        spread = np.zeros(len(snap.image_ids))
        for i in np.flatnonzero(rated):
            spread[i] = snap.vote_value[snap.vote_image == i].std()
        print(f"{'image':40} {'votes':>5} {'people':>7} {'spread':>7} {'model':>6} {'gap':>6}")
        for i in sorted(np.flatnonzero(rated), key=lambda i: means[i] - scores[i]):
            flag = "  <- people say not a horse" if means[i] < GATE_THRESHOLD <= scores[i] else ""
            print(f"{snap.labels[i][:40]:40} {counts[i]:>5} {means[i]:>7.2f} {spread[i]:>7.2f} "
                  f"{scores[i]:>6.2f} {means[i] - scores[i]:>+6.2f}{flag}")
        r = np.corrcoef(means[rated], scores[rated])[0, 1] if rated.sum() > 2 else float("nan")
        print(f"RMS gap {rms(means[rated], scores[rated]):.2f}, mean gap "
              f"{np.mean(means[rated] - scores[rated]):+.2f} (+ = people rate higher than the model), "
              f"correlation {r:.2f}")
    failed = np.flatnonzero(~snap.passed)
    if len(failed):
        print(f"\nturned away by the gate (score now; it was < {GATE_THRESHOLD} at upload):")
        for i in sorted(failed, key=lambda i: -scores[i]):
            print(f"  {snap.labels[i][:40]:40} {scores[i]:6.2f}")
    hist, edges = np.histogram(scores, bins=[-np.inf, 0, 1, 2, 2.5, 3, 4, 5, np.inf])
    names = ["<0", "0-1", "1-2", "2-2.5", "2.5-3", "3-4", "4-5", ">5"]
    print("all uploads by model score: " + "  ".join(f"{n}: {c}" for n, c in zip(names, hist)))


# Sweep runs are independent and each cycle is real compute (~1 s for the
# 64-hidden head), so they go to a process pool. Workers get the arrays once.
_W: dict = {}


def _init_worker(anchors: AnchorSet, base_states: dict, emb, values, threads: int) -> None:
    torch.set_num_threads(threads)
    bases = {}
    for hidden_dim, state in base_states.items():
        bases[hidden_dim] = HorseHead(hidden_dim=hidden_dim)
        bases[hidden_dim].load_state_dict(state)
    _W.update(anchors=anchors, bases=bases, emb=emb, values=values)


def _run_job(hidden_dim: int, keep: np.ndarray, weights: np.ndarray, knobs: dict, score_on: np.ndarray):
    """One run of the loop on the votes in `keep`; at each checkpoint, the
    scores of `score_on` and of the anchors."""
    anchors = _W["anchors"]
    out = {}
    for c, h in run_cycles(_W["bases"][hidden_dim], anchors, _W["emb"][keep], _W["values"][keep],
                           weights[keep], **knobs):
        out[c] = (score(h, score_on), score(h, anchors.embeddings))
    return out


def knob_grid(knobs: list[list], every_combination: bool) -> list[tuple]:
    """The first value of each knob is the baseline. One at a time: the
    baseline, then each other value with the rest held at baseline."""
    if every_combination:
        return list(itertools.product(*knobs))
    baseline = tuple(k[0] for k in knobs)
    grid = [baseline]
    for i, values in enumerate(knobs):
        grid += [baseline[:i] + (v,) + baseline[i + 1:] for v in values[1:]]
    return grid


class _InProcess:
    """--workers 1: run each job on the spot, no pool."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        from concurrent.futures import Future

        future = Future()
        future.set_result(fn(*args))
        return future


def print_sweep(snap: Snapshot, anchors: AnchorSet, weights, means, counts, args) -> None:
    from concurrent.futures import ProcessPoolExecutor
    import os

    rated_images = np.flatnonzero(counts > 0)
    if not len(rated_images):
        print("\nno votes yet: nothing to sweep")
        return
    cycles = sorted(int(c) for c in args.cycles.split(","))
    unrated = np.flatnonzero(counts == 0)
    groups = anchor_groups(anchors)
    horse_anchor = anchors.targets > 0

    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(rated_images), min(args.folds, len(rated_images)))
    folds = folds if len(folds) > 1 else []

    grid = knob_grid([floats(args.lambda_decay), floats(args.lambda_anchor), floats(args.prox),
                      [int(h) for h in floats(args.hidden)]], args.grid)
    bases = {h: anchors_only_head(anchors, h, args.base_epochs) for h in sorted({g[3] for g in grid})}
    base_scores = {h: score(b, snap.embeddings) for h, b in bases.items()}
    every_vote = np.ones(len(snap.vote_value), dtype=bool)
    workers = args.workers or max(1, (os.cpu_count() or 2) // 2)
    initargs = (anchors, {h: b.state_dict() for h, b in bases.items()},
                snap.embeddings[snap.vote_image], snap.vote_value, 2)
    if workers == 1:
        _init_worker(*initargs)
        pool = _InProcess()
    else:
        pool = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=initargs)
    jobs = {}
    with pool:
        for config in grid:
            d, a, p, h = config
            w = recency_weights(snap.vote_time, d)
            knobs = dict(cycles=cycles, lambda_anchor=a, prox_weight=p, floor=args.floor, epochs=args.epochs)
            jobs[config, None] = pool.submit(_run_job, h, every_vote, w, knobs, snap.embeddings)
            for f, fold in enumerate(folds):
                jobs[config, f] = pool.submit(_run_job, h, ~np.isin(snap.vote_image, fold), w, knobs,
                                              snap.embeddings[fold])
        print(f"\n== knob sweep, from a fresh anchors-only head ({args.base_epochs} epochs), then "
              f"{args.epochs} epochs/cycle; {len(folds)}-fold held-out over images; "
              f"{len(jobs)} runs on {workers} processes. Cycle 0 is the starting head.")
        print(f"{'hidden':>6} {'anchor':>6} {'prox':>5} {'decay':>6} {'cycle':>5} | {'fit':>5} {'held-out':>8} | "
              f"{'horse ok':>8} {'not ok':>6} {'worst anchor style':>28} | {'unrated shift':>13} {'flips':>5}")

        for config in grid:
            d, a, p, h = config
            m, _ = image_means(snap, recency_weights(snap.vote_time, d))
            full = jobs[config, None].result()
            held = {c: np.full(len(snap.image_ids), np.nan) for c in cycles}
            for f, fold in enumerate(folds):
                for c, (s, _) in jobs[config, f].result().items():
                    held[c][fold] = s
            for c in cycles:
                s, anchor_scores = full[c]
                passes = anchor_scores >= GATE_THRESHOLD
                ok = {k: np.mean(passes[g] == horse_anchor[g]) for k, g in groups.items()}
                worst = min(ok, key=ok.get)
                start = base_scores[h][unrated]
                flips = int(np.sum((s[unrated] >= GATE_THRESHOLD) != (start >= GATE_THRESHOLD)))
                shift = np.mean(np.abs(s[unrated] - start)) if len(unrated) else float("nan")
                held_rms = rms(m[rated_images], held[c][rated_images]) if folds else float("nan")
                print(f"{h:>6} {a:>6g} {p:>5g} {d:>6g} {c:>5} | "
                      f"{rms(m[rated_images], s[rated_images]):>5.2f} {held_rms:>8.2f} | "
                      f"{100 * passes[horse_anchor].mean():>7.0f}% {100 * (~passes[~horse_anchor]).mean():>5.0f}% "
                      f"{worst[:21]:>21} {100 * ok[worst]:>5.0f}% | {shift:>13.2f} {flips:>5}", flush=True)
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline knob tuning against a snapshot of site data (read-only).",
        epilog="Get a snapshot from the server (run on your own machine, from the repo):\n"
               "  ssh root@SERVER 'systemctl start phorse-backup'\n"
               "  scp root@SERVER:/var/backups/phorse/$(date +%%F)/app.db tune/\n"
               "  scp root@SERVER:/var/backups/phorse/$(date +%%F)/head.pt tune/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--app-db", required=True, help="A COPY of the server's app.db.")
    parser.add_argument("--head", default=None, help="The server's head.pt, for 'the model now'.")
    parser.add_argument("--anchor-cache", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--lambda-anchor", default="0.1,0.25,1",
                        help="Comma list to sweep; the first is the baseline.")
    parser.add_argument("--prox", default="1,0.3,3", help="Prox weights to sweep; first is the baseline.")
    parser.add_argument("--lambda-decay", default="0.002", help="Per day; first is the baseline.")
    parser.add_argument("--hidden", default="8,64,16,0",
                        help="Head sizes (hidden units, 0 = linear); first is the baseline.")
    parser.add_argument("--base-epochs", type=int, default=3000,
                        help="Epochs for each size's anchors-only starting head.")
    parser.add_argument("--grid", action="store_true", help="Every combination, not one knob at a time.")
    parser.add_argument("--floor", type=float, default=DEFAULT_COMMUNITY_WEIGHT_FLOOR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--cycles", default="0,1,6,24", help="Report after these many retrain cycles.")
    parser.add_argument("--folds", type=int, default=3, help="Held-out folds over rated images.")
    parser.add_argument("--workers", type=int, default=0, help="Processes (default half the CPUs; 1 = none).")
    parser.add_argument("--no-sweep", action="store_true")
    args = parser.parse_args()

    snap = load_snapshot(Path(args.app_db))
    anchors = AnchorSet.load(args.anchor_cache)
    weights = recency_weights(snap.vote_time, floats(args.lambda_decay)[0])
    means, counts = image_means(snap, weights)

    print_data(snap, weights, args.floor)
    if args.head:
        print_now(snap, HorseHead.load(args.head), means, counts)
    if not args.no_sweep:
        print_sweep(snap, anchors, weights, means, counts, args)


if __name__ == "__main__":
    main()
