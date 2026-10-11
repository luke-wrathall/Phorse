"""One-shot retrain (manual runs / debugging the drift loop):

    python -m server.retrain

Runs a single ingest -> train -> reload -> re-score cycle and prints the
summary. server.scheduler does the same on an interval.
"""

from __future__ import annotations

import json

from . import create_app
from .services.retrain import run_retrain


def main() -> None:
    app = create_app()
    with app.app_context():
        summary = run_retrain()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
