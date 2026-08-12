from pathlib import Path

from patchpilot.models import Inspection
from patchpilot.planner import make_plan


def test_plan_requires_approval_and_includes_tests() -> None:
    inspection = Inspection(Path("/repo"), "python", ("pyproject.toml",), ("python -m pytest",), ("pyproject.toml",))
    plan = make_plan("fix parser", inspection)
    assert plan.requires_approval is True
    assert len(plan.steps) == 4
    assert "python -m pytest" in plan.steps[2].action
