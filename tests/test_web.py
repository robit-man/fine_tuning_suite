from __future__ import annotations

import pytest

flask = pytest.importorskip("flask")

from training_suite.core.jobs import JobRunner
from training_suite.core.state import StateStore
from training_suite.web import create_app


def test_dashboard_loads(tmp_path) -> None:
    store = StateStore(tmp_path / "suite.sqlite3")
    app = create_app(store=store, runner=JobRunner(store))
    app.testing = True

    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    assert b"Model Dashboard" in response.data
