from __future__ import annotations

from pathlib import Path
from typing import Callable

from .models import CommandResult, TestRunSummary
from .safety import run_safe


def run_test_loop(
    repository: Path,
    test_command: str,
    *,
    timeout: float = 60.0,
    recovery: Callable[[], bool] | None = None,
    logger: Callable[..., None] | None = None,
) -> TestRunSummary:
    """Run tests and allow at most one approved recovery callback."""
    if logger:
        logger("test_started", command=test_command, attempt=1)
    initial = run_safe(test_command, repository, timeout)
    if logger:
        logger("test_completed", command=test_command, attempt=1, returncode=initial.returncode, timed_out=initial.timed_out)
    if initial.returncode == 0 and not initial.timed_out:
        return TestRunSummary(initial)
    if recovery is None:
        if logger:
            logger("recovery_skipped", reason="no approved recovery callback")
        return TestRunSummary(initial)
    try:
        recovered = recovery()
    except Exception as exc:  # recovery must not crash the outer run
        if logger:
            logger("recovery_failed", reason=type(exc).__name__)
        return TestRunSummary(initial)
    if not recovered:
        if logger:
            logger("recovery_skipped", reason="recovery callback declined")
        return TestRunSummary(initial)
    if logger:
        logger("recovery_started", attempt=1)
    retry = run_safe(test_command, repository, timeout)
    if logger:
        logger("test_completed", command=test_command, attempt=2, returncode=retry.returncode, timed_out=retry.timed_out)
    return TestRunSummary(initial, retry, recovery_attempted=True)
