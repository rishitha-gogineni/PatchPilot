import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from langgraph.types import Command

from patchpilot.multi_agent import MultiAgentAgents, build_multi_agent_graph, initial_multi_agent_state
from patchpilot.tools import ToolPermissionError


def make_repository(root: Path) -> None:
    (root / "pyproject.toml").write_text("[project]\nname = 'sample'\n", encoding="utf-8")
    (root / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "test_sample.py").write_text(
        "from target import VALUE\n\n\ndef test_value():\n    assert VALUE == 2\n",
        encoding="utf-8",
    )


def fake_agents() -> MultiAgentAgents:
    def planner(state, tools):
        inspection = tools.call("planner", "inspect_repository")
        assert "target.py" in inspection.files
        with pytest.raises(ToolPermissionError):
            tools.call("planner", "apply_edit", path="target.py", new_content="VALUE = 2\n")
        return {"plan": {"goal": state["task"]}, "selected_files": ["target.py"]}

    def proposer(state, tools):
        current = tools.call("implementer", "read_file", path="target.py")
        assert current == "VALUE = 1\n"
        return {
            "proposal": {
                "path": "target.py",
                "new_content": "VALUE = 2\n",
                "test_command": "python -m pytest",
            },
            "diff": "-VALUE = 1\n+VALUE = 2\n",
        }

    def reviewer(state, tools):
        tools.call("reviewer", "preview_edit", path="target.py", new_content="VALUE = 2\n")
        if state.get("revision_count", 0) == 0:
            return {"approved": False, "issues": ["Need one revision"], "suggestions": ["Re-check the test"]}
        return {"approved": True, "issues": [], "suggestions": [], "confidence": 0.95}

    return MultiAgentAgents(planner, proposer, reviewer)


def test_multi_agent_revision_then_approval_and_apply(tmp_path: Path) -> None:
    make_repository(tmp_path)
    graph = build_multi_agent_graph(fake_agents())
    config = {"configurable": {"thread_id": "multi-agent-approval"}}
    paused = graph.invoke(initial_multi_agent_state(tmp_path, "Update the value"), config=config)
    snapshot = graph.get_state(config)
    assert any(getattr(task, "interrupts", ()) for task in snapshot.tasks)
    assert paused["revision_count"] == 1
    assert (tmp_path / "target.py").read_text(encoding="utf-8") == "VALUE = 1\n"

    completed = graph.invoke(Command(resume={"approved": True}), config=config)
    assert completed["status"] == "completed"
    assert completed["test_success"] is True
    assert (tmp_path / "target.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_multi_agent_stops_after_revision_budget(tmp_path: Path) -> None:
    make_repository(tmp_path)

    def always_reject(state, tools):
        return {"approved": False, "issues": ["unsafe"], "suggestions": []}

    agents = fake_agents()
    bounded = MultiAgentAgents(agents.planner, agents.proposer, always_reject)
    graph = build_multi_agent_graph(bounded)
    result = graph.invoke(initial_multi_agent_state(tmp_path, "Update the value", max_revisions=1), config={"configurable": {"thread_id": "budget"}})
    assert result["status"] == "rejected"
    assert result["revision_count"] == 1
    assert (tmp_path / "target.py").read_text(encoding="utf-8") == "VALUE = 1\n"
