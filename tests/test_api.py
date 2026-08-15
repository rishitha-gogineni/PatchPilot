from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from patchpilot.api import create_app


def make_repository(root: Path) -> Path:
    (root / "pyproject.toml").write_text("[project]\nname = 'sample'\n", encoding="utf-8")
    (root / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "test_sample.py").write_text(
        "from target import VALUE\n\n\ndef test_value():\n    assert VALUE == 2\n",
        encoding="utf-8",
    )
    proposal = root / "proposal.json"
    proposal.write_text(
        json.dumps({
            "proposal": {
                "path": "target.py",
                "new_content": "VALUE = 2\n",
                "explanation": "Update the target value.",
                "risks": [],
                "test_command": "python -m pytest",
            }
        }),
        encoding="utf-8",
    )
    return proposal


def test_api_resumes_thread_after_app_recreation(tmp_path: Path) -> None:
    proposal = make_repository(tmp_path)
    database = tmp_path / "api-checkpoints.sqlite"
    payload = {
        "thread_id": "api-restart-test",
        "repository": str(tmp_path),
        "proposal_file": str(proposal),
    }

    with TestClient(create_app(database)) as client:
        started = client.post("/v1/workflows", json=payload)
        assert started.status_code == 200
        assert started.json()["pending_approval"] is True
        assert started.json()["status"] == "awaiting_approval"

    with TestClient(create_app(database)) as client:
        resumed = client.post("/v1/workflows/api-restart-test/resume", json={"approved": True})
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "completed"
        assert resumed.json()["state"]["test_success"] is True

    assert (tmp_path / "target.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_api_rejects_invalid_thread_id(tmp_path: Path) -> None:
    proposal = make_repository(tmp_path)
    database = tmp_path / "api-checkpoints.sqlite"
    with TestClient(create_app(database)) as client:
        response = client.post(
            "/v1/workflows",
            json={
                "thread_id": "bad/thread",
                "repository": str(tmp_path),
                "proposal_file": str(proposal),
            },
        )
    assert response.status_code == 422
