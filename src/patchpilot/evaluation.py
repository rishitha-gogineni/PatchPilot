"""Deterministic coding-task evaluation for PatchPilot's execution boundary."""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .editor import apply_edit
from .inspector import RepositoryInspector
from .llm import PlannerError, validate_edit_proposal
from .orchestrator import run_test_loop


@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    description: str
    files: dict[str, str]
    proposal: dict[str, Any]
    test_command: str


def load_tasks(path: Path) -> list[EvaluationTask]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("evaluation fixture must be a non-empty JSON list")
    tasks: list[EvaluationTask] = []
    for record in payload:
        if not isinstance(record, dict):
            raise ValueError("each evaluation task must be an object")
        task_id = record.get("task_id")
        files = record.get("files")
        proposal = record.get("proposal")
        command = record.get("test_command")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(files, dict) or not files or any(not isinstance(k, str) or not isinstance(v, str) for k, v in files.items()):
            raise ValueError(f"{task_id}: files must map paths to text")
        if not isinstance(proposal, dict) or not isinstance(command, str):
            raise ValueError(f"{task_id}: proposal and test_command are required")
        tasks.append(EvaluationTask(task_id, str(record.get("description", "")), files, proposal, command))
    return tasks


def evaluate_tasks(path: Path) -> dict[str, Any]:
    tasks = load_tasks(path)
    inspector = RepositoryInspector()
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for task in tasks:
        task_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"patchpilot-{task.task_id}-") as directory:
            repository = Path(directory)
            for relative, content in task.files.items():
                target = repository / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            inspection = inspector.inspect(repository)
            proposal_valid = False
            proposal_error = None
            test_passed = False
            try:
                proposal = validate_edit_proposal(
                    task.proposal,
                    inspection,
                    (str(task.proposal.get("path", "")),),
                )
                proposal_valid = True
                apply_edit(repository, proposal.path, proposal.new_content, approved=True)
                summary = run_test_loop(repository, task.test_command, timeout=30.0)
                test_passed = summary.success
            except (PlannerError, ValueError, OSError) as exc:
                proposal_error = str(exc)
            results.append({
                "task_id": task.task_id,
                "proposal_valid": proposal_valid,
                "test_passed": test_passed,
                "success": proposal_valid and test_passed,
                "error": proposal_error,
                "latency_ms": round((time.perf_counter() - task_started) * 1000, 2),
            })
    total = len(results)
    valid = sum(1 for result in results if result["proposal_valid"])
    passed = sum(1 for result in results if result["test_passed"])
    successful = sum(1 for result in results if result["success"])
    return {
        "fixture": str(path),
        "mode": "deterministic_proposals",
        "tasks": total,
        "proposal_validity_rate": valid / total,
        "test_pass_rate": passed / total,
        "task_success_rate": successful / total,
        "recovery_success_rate": None,
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "results": results,
    }
