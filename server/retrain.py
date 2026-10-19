"""One-shot retrain (manual runs / debugging the drift loop):

    python -m server.retrain [--fresh]

Runs a single ingest -> train -> reload -> re-score cycle and prints the
summary. server.scheduler does the same on an interval. --fresh trains
the head from scratch (anchors + every vote) instead of warm-starting:
run it once after changing the anchor set.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import create_app
from .services.retrain import RetrainBusy, run_retrain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one retrain cycle now.")
    parser.add_argument("--fresh", action="store_true",
                        help="Train from scratch, not from head.pt (after an anchor-set change).")
    args = parser.parse_args(argv)
    app = create_app()
    with app.app_context():
        try:
            summary = run_retrain(fresh=args.fresh)
        except RetrainBusy:
            print("a retrain is already running (the scheduler?) -- try again in a minute",
                  file=sys.stderr)
            return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
