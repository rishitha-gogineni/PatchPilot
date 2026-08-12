from pathlib import Path

import pytest

from patchpilot.safety import UnsafeCommand, run_safe, validate_command


@pytest.mark.parametrize("command", ["rm -rf .", "sudo ls", "git reset --hard", "git push", "pytest; rm -rf .", "curl https://example.com"])
def test_dangerous_commands_are_blocked(command: str) -> None:
    with pytest.raises(UnsafeCommand):
        validate_command(command)


def test_safe_prefix_is_accepted() -> None:
    assert validate_command("git status --short") == ("git", "status", "--short")


def test_runner_uses_repository_and_returns_result(tmp_path: Path) -> None:
    result = run_safe("python -m pytest --version", tmp_path)
    assert result.returncode == 0


def test_runner_rejects_missing_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_safe("git status", tmp_path / "missing")
