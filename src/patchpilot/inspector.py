from __future__ import annotations

from pathlib import Path

from .models import Inspection


_IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}


class RepositoryInspector:
    def inspect(self, repository: Path) -> Inspection:
        root = repository.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"repository does not exist: {root}")
        files: list[str] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or any(part in _IGNORED_DIRS for part in path.parts):
                continue
            relative = path.relative_to(root)
            if relative.name.startswith(".") and relative.name not in {".gitignore"}:
                continue
            files.append(relative.as_posix())
        markers = tuple(name for name in ("pyproject.toml", "requirements.txt", "package.json", "go.mod", "Cargo.toml", "Makefile") if (root / name).exists())
        project_type = self._project_type(markers, files)
        return Inspection(root, project_type, tuple(files), self._test_commands(markers), markers)

    @staticmethod
    def _project_type(markers: tuple[str, ...], files: list[str]) -> str:
        if "pyproject.toml" in markers or "requirements.txt" in markers or any(f.endswith(".py") for f in files):
            return "python"
        if "package.json" in markers:
            return "javascript"
        if "go.mod" in markers:
            return "go"
        if "Cargo.toml" in markers:
            return "rust"
        return "unknown"

    @staticmethod
    def _test_commands(markers: tuple[str, ...]) -> tuple[str, ...]:
        commands: list[str] = []
        if "pyproject.toml" in markers or "requirements.txt" in markers:
            commands.append("python -m pytest")
        if "package.json" in markers:
            commands.append("npm test")
        if "go.mod" in markers:
            commands.append("go test ./...")
        if "Cargo.toml" in markers:
            commands.append("cargo test")
        return tuple(commands)

    def read_file(self, repository: Path, relative_path: str, max_bytes: int = 100_000) -> str:
        root = repository.expanduser().resolve()
        target = (root / relative_path).resolve()
        if root not in target.parents and target != root:
            raise ValueError("path escapes repository root")
        if not target.is_file():
            raise FileNotFoundError(relative_path)
        return target.read_text(encoding="utf-8")[:max_bytes]
