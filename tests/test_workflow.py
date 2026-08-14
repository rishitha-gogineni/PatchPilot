from __future__ import annotations

import json
from pathlib import Path

import pytest

from patchpilot.workflow import apply_proposal


TEST_COMMAND = "python -m pytest"


def make_repository(root: Path) -> None:
    (root / "pyproject.toml").write_text("[project]\nname = 'sample'\n", encoding="utf-8")
    (root / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "test_sample.py").write_text(
        "from target import VALUE\n\n\ndef test_value():\n    assert VALUE == 2\n",
        encoding="utf-8",
    )


def write_proposal(path: Path, content: str, *, test_command: str | None = TEST_COMMAND) -> None:
    path.write_text(
        json.dumps(
            {
                "model": "test-model",
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "proposal": {
                    "path": "target.py",
                    "new_content": content,
                    "explanation": "Update the target value.",
                    "risks": [],
                    "test_command": test_command,
                },
            }
        ),
        encoding="utf-8",
    )


def test_apply_proposal_requires_approval_without_writing(tmp_path: Path) -> None:
    make_repository(tmp_path)
    proposal_file = tmp_path / "proposal.json"
    write_proposal(proposal_file, "VALUE = 2\n")

    result = apply_proposal(tmp_path, proposal_file)

    assert result.approved is False
    assert result.applied is False
    assert result.test_summary is None
    assert (tmp_path / "target.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_approved_proposal_runs_tests(tmp_path: Path) -> None:
    make_repository(tmp_path)
    proposal_file = tmp_path / "proposal.json"
    write_proposal(proposal_file, "VALUE = 2\n")

    result = apply_proposal(tmp_path, proposal_file, approved=True)

    assert result.applied is True
    assert result.test_summary is not None
    assert result.test_summary.success is True
    assert result.test_summary.recovery_attempted is False
    assert (tmp_path / "target.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_failed_test_can_use_separately_approved_recovery(tmp_path: Path) -> None:
    make_repository(tmp_path)
    proposal_file = tmp_path / "proposal.json"
    recovery_file = tmp_path / "recovery.json"
    write_proposal(proposal_file, "VALUE = 3\n")
    write_proposal(recovery_file, "VALUE = 2\n# recovered\n")

    result = apply_proposal(
        tmp_path,
        proposal_file,
        approved=True,
        recovery_proposal_file=recovery_file,
        recovery_approved=True,
    )

    assert result.test_summary is not None
    assert result.test_summary.success is True
    assert result.test_summary.recovery_attempted is True
    assert (tmp_path / "target.py").read_text(encoding="utf-8") == "VALUE = 2\n# recovered\n"


def test_recovery_requires_its_own_approval(tmp_path: Path) -> None:
    make_repository(tmp_path)
    proposal_file = tmp_path / "proposal.json"
    recovery_file = tmp_path / "recovery.json"
    write_proposal(proposal_file, "VALUE = 2\n")
    write_proposal(recovery_file, "VALUE = 2\n")

    with pytest.raises(ValueError, match="approve-recovery"):
        apply_proposal(tmp_path, proposal_file, approved=True, recovery_proposal_file=recovery_file)
    assert (tmp_path / "target.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_unsafe_test_override_is_rejected_before_writing(tmp_path: Path) -> None:
    make_repository(tmp_path)
    proposal_file = tmp_path / "proposal.json"
    write_proposal(proposal_file, "VALUE = 2\n", test_command=None)

    with pytest.raises(ValueError, match="shell operators"):
        apply_proposal(tmp_path, proposal_file, approved=True, test_command="python -m pytest && echo unsafe")
    assert (tmp_path / "target.py").read_text(encoding="utf-8") == "VALUE = 1\n"
