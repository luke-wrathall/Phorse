"""JWT secret guard: a real WSGI server must never serve with the public dev
secret (anyone could forge a token for any user)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from server.config import DEV_SECRET_KEY
from server.security import MIN_SECRET_KEY_LEN, check_secret_key

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "secret", [None, "", DEV_SECRET_KEY, "x" * (MIN_SECRET_KEY_LEN - 1)]
)
def test_unfit_secrets_are_refused(secret):
    with pytest.raises(RuntimeError, match="HORSE_SECRET_KEY"):
        check_secret_key(secret)


def test_long_random_secret_is_accepted():
    check_secret_key("x" * MIN_SECRET_KEY_LEN)


def _import_wsgi(tmp_path, secret: str | None) -> subprocess.CompletedProcess:
    """Import server.wsgi the way gunicorn does, in a clean process."""
    env = {k: v for k, v in os.environ.items() if k != "HORSE_SECRET_KEY"}
    env.update(
        {
            "HORSE_FAKE_EMBEDDER": "1",
            "HORSE_APP_DB": str(tmp_path / "app.db"),
            "HORSE_UPLOAD_DIR": str(tmp_path / "uploads"),
            "HORSE_HEAD_PATH": str(tmp_path / "head.pt"),
        }
    )
    if secret is not None:
        env["HORSE_SECRET_KEY"] = secret
    return subprocess.run(
        [sys.executable, "-c", "import server.wsgi"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )


def test_wsgi_import_refuses_without_a_secret(tmp_path):
    result = _import_wsgi(tmp_path, secret=None)

    assert result.returncode != 0
    assert "HORSE_SECRET_KEY is not set" in result.stderr
    assert not (tmp_path / "app.db").exists(), "must fail before creating the app"


def test_wsgi_import_starts_with_a_proper_secret(tmp_path):
    result = _import_wsgi(tmp_path, secret="s" * 64)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "app.db").exists()
