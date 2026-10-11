"""server.scheduler: the interval loop that drives retraining."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _StopLoop(Exception):
    pass


def _fake_app(interval: int):
    from flask import Flask

    app = Flask(__name__)
    app.config["RETRAIN_INTERVAL_SECONDS"] = interval
    return app


def test_interval_zero_disables_the_loop(monkeypatch):
    from server import scheduler

    monkeypatch.setattr(scheduler, "create_app", lambda: _fake_app(0))
    monkeypatch.setattr(scheduler, "run_retrain", lambda: pytest.fail("should not retrain"))

    scheduler.main()


def test_loop_retrains_immediately_and_survives_a_failed_cycle(monkeypatch):
    from server import scheduler

    calls, sleeps = [], []

    def run_retrain():
        calls.append(len(sleeps))
        if len(calls) == 1:
            raise RuntimeError("boom")
        return {}

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise _StopLoop

    monkeypatch.setattr(scheduler, "create_app", lambda: _fake_app(42))
    monkeypatch.setattr(scheduler, "run_retrain", run_retrain)
    monkeypatch.setattr(scheduler, "time", SimpleNamespace(sleep=sleep))

    with pytest.raises(_StopLoop):
        scheduler.main()

    assert calls == [0, 1], "first retrain runs before any sleep; a failure doesn't stop the loop"
    assert sleeps == [42, 42]
