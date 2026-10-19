"""Background retrain loop -- run alongside the API:

    python -m server.scheduler

Retrains every HORSE_RETRAIN_INTERVAL seconds (default 600). One retrain
runs immediately on startup so a fresh instance isn't stuck on a cold
head until the first interval elapses. Set HORSE_RETRAIN_INTERVAL=0 to
disable the loop entirely.

Deliberately a separate process, not a thread in the API: keeps the
retrain off the request path, sidesteps the Flask reloader double-start,
and makes "the retrain loop" a thing you can see and stop. The API
process picks up each new head.pt via an mtime check (server.services.model).
"""

from __future__ import annotations

import logging
import time

from . import create_app
from .services.retrain import RetrainBusy, run_retrain

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("server.scheduler")


def main() -> None:
    app = create_app()
    interval = app.config["RETRAIN_INTERVAL_SECONDS"]
    if interval <= 0:
        log.info("retrain scheduler disabled (HORSE_RETRAIN_INTERVAL=0)")
        return

    log.info("retrain scheduler up -- every %ss", interval)
    while True:
        try:
            with app.app_context():
                summary = run_retrain()
            log.info("retrain: %s", summary)
        except RetrainBusy:
            log.info("retrain skipped: another retrain (a manual run?) is in progress")
        except Exception:  # noqa: BLE001 -- keep the loop alive across failures
            log.exception("retrain failed")
        time.sleep(interval)


if __name__ == "__main__":
    main()
