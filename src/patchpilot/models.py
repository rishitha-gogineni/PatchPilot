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
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass
class RunLog:
    """In-memory event container; JSONL persistence is a later milestone."""

    events: list[dict[str, object]] = field(default_factory=list)

    def add(self, event: str, **fields: object) -> None:
        self.events.append({"event": event, **fields})
