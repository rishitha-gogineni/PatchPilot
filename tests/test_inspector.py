from pathlib import Path

import pytest

from patchpilot.inspector import RepositoryInspector


def test_inspector_detects_python_project(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text("", encoding="utf-8")
    result = RepositoryInspector().inspect(tmp_path)
    assert result.project_type == "python"
    assert result.test_commands == ("python -m pytest",)
    assert "src/main.py" in result.files
    assert ".venv/ignored.py" not in result.files


def test_read_file_cannot_escape_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError):
        RepositoryInspector().read_file(tmp_path, "../outside.txt")
