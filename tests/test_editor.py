from pathlib import Path

import pytest

from patchpilot.editor import EditDenied, apply_edit, preview_edit


def test_preview_does_not_modify_file(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("print('old')\n", encoding="utf-8")
    diff = preview_edit(tmp_path, "app.py", "print('new')\n")
    assert "-print('old')" in diff
    assert target.read_text(encoding="utf-8") == "print('old')\n"


def test_edit_requires_approval(tmp_path: Path) -> None:
    with pytest.raises(EditDenied):
        apply_edit(tmp_path, "app.py", "print('new')\n")
    assert not (tmp_path / "app.py").exists()


def test_approved_edit_is_applied_atomically(tmp_path: Path) -> None:
    diff = apply_edit(tmp_path, "app.py", "print('new')\n", approved=True)
    assert "new" in diff
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "print('new')\n"


@pytest.mark.parametrize("path", ["../outside.py", ".env", "secrets.pem", ".git/config"])
def test_protected_or_escape_paths_are_rejected(tmp_path: Path, path: str) -> None:
    with pytest.raises((ValueError, EditDenied)):
        apply_edit(tmp_path, path, "secret", approved=True)
