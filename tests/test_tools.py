from pathlib import Path

import pytest

from patchpilot.tools import ToolPermissionError, ToolRegistry, build_repository_tool_registry


def test_registry_enforces_role_permissions(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='sample'\n", encoding="utf-8")
    (tmp_path / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    registry = build_repository_tool_registry(tmp_path)
    assert registry.call("planner", "read_file", path="target.py") == "VALUE = 1\n"
    with pytest.raises(ToolPermissionError):
        registry.call("planner", "apply_edit", path="target.py", new_content="VALUE = 2\n")
    assert {spec.name for spec in registry.describe("executor")} == {"apply_edit", "preview_edit", "run_tests"}


def test_registry_rejects_duplicate_tools() -> None:
    registry = ToolRegistry()
    registry.register("read", lambda: None, roles=("planner",))
    with pytest.raises(ValueError):
        registry.register("read", lambda: None, roles=("planner",))
