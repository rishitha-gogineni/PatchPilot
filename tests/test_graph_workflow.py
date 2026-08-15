from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from langgraph.types import Command

from patchpilot.graph_workflow import build_apply_graph, initial_state


def make_repository(root: Path) -> None:
    (root / "pyproject.toml").write_text("[project]\nname = 'sample'\n", encoding="utf-8")
    (root / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "test_sample.py").write_text(
        "from target import VALUE\n\n\ndef test_value():\n    assert VALUE == 2\n",
        encoding="utf-8",
    )


def write_proposal(path: Path, content: str) -> None:
    path.write_text(
        json.dumps(
            {
                "proposal": {
                    "path": "target.py",
                    "new_content": content,
                    "explanation": "Update the target value.",
                    "risks": [],
                    "test_command": "python -m pytest",
                }
            }
        ),
        encoding="utf-8",
    )


def test_graph_interrupts_before_edit_and_resumes_after_approval(tmp_path: Path) -> None:
    make_repository(tmp_path)
    proposal_file = tmp_path / "proposal.json"
    write_proposal(proposal_file, "VALUE = 2\n")
    graph = build_apply_graph()
    config = {"configurable": {"thread_id": "approval-test"}}

    paused = graph.invoke(initial_state(tmp_path, proposal_file), config=config)

    snapshot = graph.get_state(config)
    assert any(getattr(task, "interrupts", ()) for task in snapshot.tasks)
    assert (tmp_path / "target.py").read_text(encoding="utf-8") == "VALUE = 1\n"

    completed = graph.invoke(Command(resume={"approved": True}), config=config)

    assert completed["status"] == "completed"
    assert completed["test_success"] is True
    assert (tmp_path / "target.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_graph_rejection_leaves_repository_unchanged(tmp_path: Path) -> None:
    make_repository(tmp_path)
    proposal_file = tmp_path / "proposal.json"
    write_proposal(proposal_file, "VALUE = 2\n")
    graph = build_apply_graph()
    config = {"configurable": {"thread_id": "reject-test"}}

    paused = graph.invoke(initial_state(tmp_path, proposal_file), config=config)
    snapshot = graph.get_state(config)
    assert any(getattr(task, "interrupts", ()) for task in snapshot.tasks)

    rejected = graph.invoke(Command(resume={"approved": False}), config=config)

    assert rejected["status"] == "rejected"
    assert rejected["applied"] is False
    assert (tmp_path / "target.py").read_text(encoding="utf-8") == "VALUE = 1\n"
