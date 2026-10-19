"""One-shot retrain (manual runs / debugging the drift loop):

    python -m server.retrain

Runs a single ingest -> train -> reload -> re-score cycle and prints the
summary. server.scheduler does the same on an interval.
"""

from __future__ import annotations

import json
import sys

from . import create_app
from .services.retrain import RetrainBusy, run_retrain


def main() -> int:
    app = create_app()
    with app.app_context():
        try:
            summary = run_retrain()
        except RetrainBusy:
            print("a retrain is already running (the scheduler?) -- try again in a minute",
                  file=sys.stderr)
            return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
