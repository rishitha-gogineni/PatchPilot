from __future__ import annotations

import difflib
import os
import tempfile
from pathlib import Path


class EditDenied(PermissionError):
    """Raised when an edit is not explicitly approved or is protected."""


_PROTECTED_NAMES = {".git", ".ssh", ".aws"}
_MAX_EDIT_BYTES = 500_000


def _safe_target(repository: Path, relative_path: str) -> Path:
    root = repository.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"repository does not exist: {root}")
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError("edit path must be a non-empty relative path")
    target = (root / relative_path).resolve()
    if root not in target.parents:
        raise ValueError("edit path escapes repository root")
    if any(part in _PROTECTED_NAMES for part in target.relative_to(root).parts):
        raise EditDenied("protected paths cannot be edited")
    if target.name.startswith(".env") or target.name.endswith(('.pem', '.key')):
        raise EditDenied("secret and credential files cannot be edited")
    if target.exists() and target.is_dir():
        raise ValueError("edit target must be a file")
    if not target.parent.is_dir():
        raise ValueError("edit parent directory must already exist")
    return target


def preview_edit(repository: Path, relative_path: str, new_content: str) -> str:
    if len(new_content.encode("utf-8")) > _MAX_EDIT_BYTES:
        raise ValueError("edited content exceeds the size limit")
    target = _safe_target(repository, relative_path)
    old_content = target.read_text(encoding="utf-8") if target.exists() else ""
    return "".join(difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=relative_path,
        tofile=relative_path,
    ))


def apply_edit(repository: Path, relative_path: str, new_content: str, *, approved: bool = False) -> str:
    if not approved:
        raise EditDenied("edit requires explicit approval")
    if len(new_content.encode("utf-8")) > _MAX_EDIT_BYTES:
        raise ValueError("edited content exceeds the size limit")
    target = _safe_target(repository, relative_path)
    diff = preview_edit(repository, relative_path, new_content)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        handle.write(new_content)
        temporary = Path(handle.name)
    os.replace(temporary, target)
    return diff
