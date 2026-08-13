from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Inspection:
    root: Path
    project_type: str
    files: tuple[str, ...]
    test_commands: tuple[str, ...]
    markers: tuple[str, ...]


@dataclass(frozen=True)
class PlanStep:
    order: int
    action: str
    rationale: str


@dataclass(frozen=True)
class TaskPlan:
    task: str
    repository: Path
    steps: tuple[PlanStep, ...]
    requires_approval: bool = True


@dataclass(frozen=True)
class ModelPlan:
    goal: str
    files_to_inspect: tuple[str, ...]
    proposed_changes: tuple[str, ...]
    test_command: str | None
    risks: tuple[str, ...]


@dataclass(frozen=True)
class PlannerResult:
    plan: ModelPlan
    model: str
    usage: dict[str, int]


@dataclass(frozen=True)
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class TestRunSummary:
    initial: CommandResult
    recovery: CommandResult | None = None
    recovery_attempted: bool = False

    @property
    def success(self) -> bool:
        result = self.recovery if self.recovery_attempted and self.recovery is not None else self.initial
        return result.returncode == 0 and not result.timed_out


@dataclass
class RunLog:
    """In-memory event container; JSONL persistence is a later milestone."""

    events: list[dict[str, object]] = field(default_factory=list)

    def add(self, event: str, **fields: object) -> None:
        self.events.append({"event": event, **fields})
