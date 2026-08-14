from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .editor import apply_edit, preview_edit
from .inspector import RepositoryInspector
from .llm import PlannerError, validate_edit_proposal
from .logging import JsonlRunLogger
from .models import EditProposal, TestRunSummary
from .orchestrator import run_test_loop
from .safety import validate_command


@dataclass(frozen=True)
class ApplyProposalResult:
    proposal: EditProposal
    diff: str
    approved: bool
    applied: bool
    test_command: str | None = None
    test_summary: TestRunSummary | None = None


def _proposal_payload(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read proposal file: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("proposal file must contain a JSON object")
    payload = raw.get("proposal", raw)
    if not isinstance(payload, dict):
        raise ValueError("proposal field must contain a JSON object")
    return payload


def _load_validated_proposal(repository: Path, proposal_file: Path) -> EditProposal:
    inspector = RepositoryInspector()
    inspection = inspector.inspect(repository)
    payload = _proposal_payload(proposal_file)
    path = payload.get("path")
    if not isinstance(path, str) or not path.strip():
        raise PlannerError("proposal path must be a non-empty string")
    return validate_edit_proposal(payload, inspection, (path,))


def apply_proposal(
    repository: Path,
    proposal_file: Path,
    *,
    approved: bool = False,
    test_command: str | None = None,
    timeout: float = 60.0,
    recovery_proposal_file: Path | None = None,
    recovery_approved: bool = False,
    logger: Callable[..., None] | None = None,
) -> ApplyProposalResult:
    """Review, optionally apply, and test one complete-file proposal.

    Proposal and recovery contents are validated before any write. The initial
    approval is separate from the recovery approval, and the existing
    orchestrator limits recovery to one test retry.
    """
    repository = repository.expanduser().resolve()
    proposal = _load_validated_proposal(repository, proposal_file)
    diff = preview_edit(repository, proposal.path, proposal.new_content)
    active_test = test_command.strip() if test_command and test_command.strip() else proposal.test_command

    recovery_proposal: EditProposal | None = None
    if recovery_proposal_file is not None:
        if not recovery_approved:
            raise ValueError("recovery requires --approve-recovery")
        if not active_test:
            raise ValueError("recovery requires a test command")
        recovery_proposal = _load_validated_proposal(repository, recovery_proposal_file)
    elif recovery_approved:
        raise ValueError("--approve-recovery requires --recovery-proposal-file")

    if active_test:
        validate_command(active_test)

    if logger:
        logger("proposal_reviewed", path=proposal.path, changed=bool(diff), approved=approved)
    if not approved:
        if logger:
            logger("proposal_awaiting_approval", path=proposal.path)
        return ApplyProposalResult(proposal, diff, approved=False, applied=False, test_command=active_test)

    applied_diff = apply_edit(repository, proposal.path, proposal.new_content, approved=True)
    if logger:
        logger("proposal_applied", path=proposal.path, changed=bool(applied_diff))

    if not active_test:
        if logger:
            logger("test_skipped", reason="proposal did not provide a test command")
        return ApplyProposalResult(proposal, applied_diff, approved=True, applied=True)

    def recovery() -> bool:
        if recovery_proposal is None:
            return False
        recovery_diff = apply_edit(
            repository,
            recovery_proposal.path,
            recovery_proposal.new_content,
            approved=True,
        )
        if logger:
            logger("recovery_proposal_applied", path=recovery_proposal.path, changed=bool(recovery_diff))
        return True

    summary = run_test_loop(
        repository,
        active_test,
        timeout=timeout,
        recovery=recovery if recovery_proposal is not None else None,
        logger=logger,
    )
    if logger:
        logger("proposal_workflow_completed", path=proposal.path, success=summary.success, recovery_attempted=summary.recovery_attempted)
    return ApplyProposalResult(
        proposal,
        applied_diff,
        approved=True,
        applied=True,
        test_command=active_test,
        test_summary=summary,
    )
