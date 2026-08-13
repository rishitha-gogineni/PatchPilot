from __future__ import annotations

import argparse
import json
from pathlib import Path

from .editor import EditDenied, apply_edit, preview_edit
from .inspector import RepositoryInspector
from .logging import JsonlRunLogger
from .llm import OpenAIPlanner, PlannerError, plan_to_dict
from .orchestrator import run_test_loop
from .planner import make_plan
from .safety import UnsafeCommand, run_safe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="patchpilot", description="Safety-first coding assistant foundation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect", help="inspect a local repository")
    inspect.add_argument("repository", type=Path)
    plan = subparsers.add_parser("plan", help="create a deterministic review plan")
    plan.add_argument("task")
    plan.add_argument("--repo", type=Path, default=Path.cwd())
    check = subparsers.add_parser("run", help="run one allow-listed command")
    check.add_argument("command_text")
    check.add_argument("--repo", type=Path, default=Path.cwd())
    check.add_argument("--timeout", type=float, default=30.0)
    edit = subparsers.add_parser("edit", help="preview and optionally apply one approved file edit")
    edit.add_argument("file")
    edit.add_argument("--content-file", type=Path, required=True)
    edit.add_argument("--repo", type=Path, default=Path.cwd())
    edit.add_argument("--approve", action="store_true")
    logs = subparsers.add_parser("logs", help="show recent structured run events")
    logs.add_argument("--repo", type=Path, default=Path.cwd())
    logs.add_argument("--limit", type=int, default=20)
    test = subparsers.add_parser("test", help="run a test command with one optional recovery callback")
    test.add_argument("test_command")
    test.add_argument("--repo", type=Path, default=Path.cwd())
    test.add_argument("--timeout", type=float, default=60.0)
    test.add_argument("--recovery-content-file", type=Path)
    test.add_argument("--recovery-path")
    test.add_argument("--approve-recovery", action="store_true")
    ai_plan = subparsers.add_parser("llm-plan", help="generate a structured plan with the configured model")
    ai_plan.add_argument("task")
    ai_plan.add_argument("--repo", type=Path, default=Path.cwd())
    ai_plan.add_argument("--model")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inspector = RepositoryInspector()
    if args.command == "inspect":
        result = inspector.inspect(args.repository)
        print(f"Repository: {result.root}\nProject type: {result.project_type}")
        print(f"Markers: {', '.join(result.markers) or 'none'}")
        print(f"Files ({len(result.files)}):")
        print("\n".join(f"  {name}" for name in result.files) or "  (none)")
        print("Test commands: " + (", ".join(result.test_commands) or "none detected"))
        return 0
    if args.command == "plan":
        inspection = inspector.inspect(args.repo)
        task_plan = make_plan(args.task, inspection)
        print(f"Task: {task_plan.task}\nRepository: {task_plan.repository}\nApproval required: yes")
        for step in task_plan.steps:
            print(f"{step.order}. {step.action} — {step.rationale}")
        return 0
    if args.command == "llm-plan":
        try:
            inspection = inspector.inspect(args.repo)
            result = OpenAIPlanner(model=args.model).create_plan(args.task, inspection)
        except (PlannerError, ValueError) as exc:
            print(f"LLM PLAN BLOCKED: {exc}")
            return 2
        JsonlRunLogger.for_repository(args.repo).record(
            "model_plan_created", model=result.model, **result.usage,
        )
        print(json.dumps(plan_to_dict(result), indent=2, sort_keys=True))
        print("Approval required before any edit or command execution.")
        return 0
    logger = JsonlRunLogger.for_repository(args.repo) if args.command in {"edit", "test"} else None
    if args.command == "edit":
        try:
            new_content = args.content_file.read_text(encoding="utf-8")
            diff = preview_edit(args.repo, args.file, new_content)
            print(diff or "No changes.")
            if not args.approve:
                print("Approval required; no files changed.")
                return 3
            applied_diff = apply_edit(args.repo, args.file, new_content, approved=True)
            logger.record("edit_applied", path=args.file, changed=bool(applied_diff))
            print("Edit applied.")
            return 0
        except (EditDenied, ValueError, OSError) as exc:
            if logger:
                logger.record("edit_blocked", path=args.file, reason=str(exc))
            print(f"EDIT BLOCKED: {exc}")
            return 2
    if args.command == "logs":
        for event in JsonlRunLogger.for_repository(args.repo).tail(args.limit):
            print(json.dumps(event, sort_keys=True))
        return 0
    if args.command == "test":
        recovery = None
        if args.recovery_content_file or args.recovery_path or args.approve_recovery:
            if not (args.recovery_content_file and args.recovery_path and args.approve_recovery):
                print("TEST BLOCKED: recovery requires --recovery-content-file, --recovery-path, and --approve-recovery")
                logger.record("recovery_blocked", reason="missing recovery approval or target")
                return 2

            def recovery() -> bool:
                content = args.recovery_content_file.read_text(encoding="utf-8")
                diff = apply_edit(args.repo, args.recovery_path, content, approved=True)
                logger.record("recovery_edit_applied", path=args.recovery_path, changed=bool(diff))
                return True

        try:
            summary = run_test_loop(args.repo, args.test_command, timeout=args.timeout, recovery=recovery, logger=logger.record)
        except (UnsafeCommand, ValueError) as exc:
            logger.record("test_blocked", command=args.test_command, reason=str(exc))
            print(f"TEST BLOCKED: {exc}")
            return 2
        final = summary.recovery if summary.recovery_attempted else summary.initial
        print(final.stdout, end="")
        if final.stderr:
            print(final.stderr, end="")
        print(f"Test success: {'yes' if summary.success else 'no'}")
        print(f"Recovery attempted: {'yes' if summary.recovery_attempted else 'no'}")
        return final.returncode
    try:
        result = run_safe(args.command_text, args.repo, args.timeout)
    except (UnsafeCommand, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    return result.returncode
