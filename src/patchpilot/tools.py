"""Permissioned repository tools used by PatchPilot agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class ToolPermissionError(PermissionError):
    """Raised when an agent requests a tool outside its role permissions."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    roles: frozenset[str]


class ToolRegistry:
    """Small allow-list registry; agents never receive raw filesystem handles."""

    def __init__(self) -> None:
        self._handlers: dict[str, tuple[ToolSpec, Callable[..., Any]]] = {}

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        *,
        roles: frozenset[str] | set[str] | tuple[str, ...],
    ) -> None:
        if not name or name in self._handlers:
            raise ValueError("tool names must be non-empty and unique")
        allowed_roles = frozenset(roles)
        if not allowed_roles:
            raise ValueError("a tool must grant at least one role")
        self._handlers[name] = (ToolSpec(name, allowed_roles), handler)

    def call(self, role: str, name: str, /, **kwargs: Any) -> Any:
        try:
            spec, handler = self._handlers[name]
        except KeyError as exc:
            raise ToolPermissionError(f"unknown tool: {name}") from exc
        if role not in spec.roles:
            raise ToolPermissionError(f"role '{role}' cannot call tool '{name}'")
        return handler(**kwargs)

    def describe(self, role: str | None = None) -> tuple[ToolSpec, ...]:
        specs = [spec for spec, _ in self._handlers.values()]
        if role is not None:
            specs = [spec for spec in specs if role in spec.roles]
        return tuple(sorted(specs, key=lambda spec: spec.name))


def build_repository_tool_registry(repository: Path, *, timeout: float = 60.0) -> ToolRegistry:
    """Build the least-privilege tool set for one repository run."""
    from .editor import apply_edit, preview_edit
    from .inspector import RepositoryInspector
    from .orchestrator import run_test_loop
    from .retrieval import retrieve_repository_context

    root = repository.expanduser().resolve()
    inspector = RepositoryInspector()
    inspection = inspector.inspect(root)
    registry = ToolRegistry()
    read_roles = ("planner", "retriever", "implementer", "reviewer")
    registry.register("inspect_repository", lambda: inspection, roles=read_roles)
    registry.register(
        "read_file",
        lambda path: inspector.read_file(root, path),
        roles=read_roles,
    )
    registry.register(
        "retrieve_context",
        lambda query, max_results=8: retrieve_repository_context(
            root, query, inspection.files, max_results=max_results,
        ),
        roles=("retriever", "planner", "reviewer"),
    )
    registry.register(
        "preview_edit",
        lambda path, new_content: preview_edit(root, path, new_content),
        roles=("reviewer", "executor"),
    )
    registry.register(
        "apply_edit",
        lambda path, new_content: apply_edit(root, path, new_content, approved=True),
        roles=("executor",),
    )
    registry.register(
        "run_tests",
        lambda command: run_test_loop(root, command, timeout=timeout),
        roles=("reviewer", "executor"),
    )
    return registry
