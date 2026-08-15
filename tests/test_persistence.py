from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph.checkpoint.sqlite")

from langgraph.types import Command

from patchpilot.graph_workflow import build_sqlite_graph, initial_state, pending_interrupts


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


def test_sqlite_checkpoint_resumes_after_reopening_graph(tmp_path: Path) -> None:
    proposal = make_repository(tmp_path)
    database = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "restart-test"}}

    graph, connection = build_sqlite_graph(database)
    paused = graph.invoke(initial_state(tmp_path, proposal), config=config)
    assert paused["status"] == "awaiting_approval"
    assert pending_interrupts(graph, config)
    connection.close()

    reopened_graph, reopened_connection = build_sqlite_graph(database)
    try:
        completed = reopened_graph.invoke(Command(resume={"approved": True}), config=config)
        assert completed["status"] == "completed"
        assert completed["test_success"] is True
        assert (tmp_path / "target.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    finally:
        reopened_connection.close()
